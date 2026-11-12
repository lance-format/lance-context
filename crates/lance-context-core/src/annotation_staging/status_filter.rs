//! Status filtering of one pinned materialized annotation snapshot.
use super::*;
use lance::index::DatasetIndexExt;
use std::collections::BTreeSet;

pub const MAX_STATUS_MATCHES: usize = 1024;
const SCAN_ROWS: usize = 1024;
const MIN_SCAN_ROWS: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize)]
pub struct StatusMatches {
    pub scope: Scope,
    pub base_version: u64,
    /// WAL positions scoped by `scope`, not mutation counts or wall-clock lag.
    pub materialized_through: u64,
    pub applied_through_at_capture: u64,
    pub indexed_fragments: usize,
    pub unindexed_fragments: usize,
    /// Includes deleted rows, if any. None means native row counts are unknown.
    pub unindexed_physical_rows: Option<u64>,
    /// Unordered matches in this snapshot. No cross-request pagination promise.
    pub ids: Vec<String>,
    pub truncated: bool,
}

/// Keeps the selected native files protected by the same registry as point
/// reads, through publication and physical collection. It does not fence an
/// owner or establish a distributed barrier; the serving layer must do that.
pub struct StatusRead {
    base: Arc<views::Base>,
    matches: StatusMatches,
    status: Option<String>,
    limit: usize,
    response_bytes: usize,
    value_bytes: usize,
    scan_rows: usize,
    _reader: OwnedSemaphorePermit,
}

impl AnnotationStaging {
    /// Capture base and progress under one lock. Active/frozen annotations are
    /// intentionally absent from this materialized filter, but remain visible
    /// to point reads. A later publication cannot change this captured result.
    pub async fn capture_status(
        &self,
        status: Option<String>,
        limit: usize,
        response_bytes: usize,
    ) -> Result<StatusRead> {
        self.capture_status_inner(status, limit, response_bytes, false)
            .await
    }

    /// Wait for reader admission before capturing base and progress. Callers
    /// must bound concurrent waiters and include admission in their deadline.
    /// Waiting holds no staging lock or base snapshot; cancellation removes
    /// the semaphore waiter.
    pub async fn capture_status_when_available(
        &self,
        status: Option<String>,
        limit: usize,
        response_bytes: usize,
    ) -> Result<StatusRead> {
        self.capture_status_inner(status, limit, response_bytes, true)
            .await
    }

    async fn capture_status_inner(
        &self,
        status: Option<String>,
        limit: usize,
        response_bytes: usize,
        wait_for_reader: bool,
    ) -> Result<StatusRead> {
        if !(1..=MAX_STATUS_MATCHES).contains(&limit)
            || status
                .as_ref()
                .is_some_and(|s| s.len() > self.limits.value_bytes)
            || response_bytes < 1024
        {
            return Err(invalid("invalid status filter bounds"));
        }
        let reader = if wait_for_reader {
            self.readers
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| invalid("annotation read admission closed"))?
        } else {
            reserve(&self.readers, 1)?
        };
        let state = self.state.read().await;
        Ok(StatusRead {
            base: state.base.clone(),
            matches: StatusMatches {
                scope: self.scope.clone(),
                base_version: state.base.version_id(),
                materialized_through: state.materialized,
                applied_through_at_capture: state.applied,
                indexed_fragments: 0,
                unindexed_fragments: 0,
                unindexed_physical_rows: Some(0),
                ids: Vec::new(),
                truncated: false,
            },
            status,
            limit,
            response_bytes,
            value_bytes: self.limits.value_bytes,
            scan_rows: SCAN_ROWS,
            _reader: reader,
        })
    }
}

impl StatusRead {
    /// Reduce the scan batch when the caller's total workspace budget cannot
    /// accommodate the preferred batch. Keep the previous 32-row minimum so
    /// existing small budgets can still serve queries without tiny scan batches.
    pub fn cap_scan_workspace(&mut self, budget: usize) -> Result<()> {
        let rows = self
            .response_bytes
            .checked_add(64 << 10)
            .and_then(|fixed| budget.checked_sub(fixed))
            .zip(self.value_bytes.checked_mul(2))
            .and_then(|(bytes, row_bytes)| bytes.checked_div(row_bytes))
            .unwrap_or(0);
        if rows < MIN_SCAN_ROWS {
            return Err(invalid("status filter workspace exceeds budget"));
        }
        self.scan_rows = self.scan_rows.min(rows);
        Ok(())
    }

    /// Admission estimate for response storage and narrow scanner vectors.
    /// This is not a hard cap on native index/query allocations or total I/O.
    pub fn workspace_bytes(&self) -> Result<usize> {
        self.value_bytes
            .checked_mul(self.scan_rows * 2)
            .and_then(|n| n.checked_add(self.response_bytes))
            .and_then(|n| n.checked_add(64 << 10))
            .ok_or_else(|| invalid("status filter workspace overflow"))
    }

    /// Native filtering covers indexed fragments and the unindexed Lance tail.
    /// Only IDs are projected; this never reads a WAL tail or turn body.
    pub async fn resolve(&self) -> Result<StatusMatches> {
        let mut result = self.matches.clone();
        let field = self.base.schema().field("status").unwrap().id;
        let mut indexed = BTreeSet::new();
        for index in self.base.load_indices().await?.iter() {
            if index.fields == [field]
                && index
                    .index_details
                    .as_ref()
                    .is_some_and(|d| d.type_url.ends_with("BTreeIndexDetails"))
            {
                if let Some(coverage) = &index.fragment_bitmap {
                    indexed.extend(coverage.iter());
                }
            }
        }
        for fragment in self.base.fragments().iter() {
            if u32::try_from(fragment.id)
                .ok()
                .is_some_and(|id| indexed.contains(&id))
            {
                result.indexed_fragments += 1;
            } else {
                result.unindexed_fragments += 1;
                result.unindexed_physical_rows = result
                    .unindexed_physical_rows
                    .zip(fragment.physical_rows)
                    .and_then(|(total, rows)| total.checked_add(rows as u64));
            }
        }
        let mut charged = ::serde_json::to_vec(&result)
            .map_err(|error| invalid(error.to_string()))?
            .len();
        if charged > self.response_bytes {
            return Err(invalid("status filter metadata exceeds response bound"));
        }
        let mut scan = self.base.scan();
        scan.project(&["id"])?;
        scan.filter_expr(self.status.as_ref().map_or_else(
            || col("status").is_null(),
            |status| col("status").eq(lit(status.clone())),
        ));
        scan.limit(Some(self.limit as i64 + 1), None)?;
        scan.batch_size(self.scan_rows)
            .batch_readahead(1)
            .fragment_readahead(1);
        let mut stream = scan.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| invalid("invalid annotation id"))?;
            for id in ids.iter() {
                let id = id
                    .filter(|s| !s.is_empty() && s.len() <= self.value_bytes)
                    .ok_or_else(|| invalid("invalid status filter turn key"))?;
                // JSON escaping can expand each byte to six bytes. Charge the
                // quotes/comma before cloning a result, including one lookahead.
                let next = id
                    .len()
                    .checked_mul(6)
                    .and_then(|n| n.checked_add(3))
                    .and_then(|n| charged.checked_add(n))
                    .ok_or_else(|| invalid("status filter result size overflow"))?;
                if result.ids.len() == self.limit || next > self.response_bytes {
                    result.truncated = true;
                    return Ok(result);
                }
                charged = next;
                result.ids.push(id.to_owned());
            }
        }
        Ok(result)
    }
}
