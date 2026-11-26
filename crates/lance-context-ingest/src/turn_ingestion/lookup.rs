//! Bounded composite-key probes against the full body dataset.
use super::*;
use futures::TryStreamExt;
use lance::Dataset;
use lance_context_core::{MergeMemoryBudget, MergeReservation};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex,
    time::Instant,
};

const MAX_KEY_BYTES: usize = 64 << 10;
const SCAN_BATCH_ROWS: usize = 2048;

#[cfg(test)]
mod tests;

/// One admitted lookup window. Projection counters describe returned key
/// batches; they are not physical object-store or HTTP bytes. Lance execution
/// counters are optional because a cancelled scan may not report its summary.
#[derive(Clone, Debug, Default)]
pub struct TurnLookupObservation {
    pub requested_keys: usize,
    pub scans_started: u64,
    pub projected_rows: u64,
    pub projected_array_bytes: u64,
    pub elapsed_us: u64,
    pub lance_bytes_read: Option<usize>,
    pub lance_requests: Option<usize>,
    pub lance_indices_loaded: Option<usize>,
    pub fragments_scanned: Option<usize>,
    pub ranges_scanned: Option<usize>,
    pub rows_scanned: Option<usize>,
    /// Native task-stream wait time in nanoseconds; this is not CPU time.
    pub task_wait_time_ns: Option<usize>,
}

/// Key-only results for one exact captured base. A miss is meaningful only for
/// a queried key, and only while applying against that same base. This does not
/// retain historical key state or carry a publication/GC decision.
pub struct TurnLookup {
    base: Arc<Dataset>,
    keys: BTreeMap<String, bool>,
    observation: TurnLookupObservation,
    _workspace: MergeReservation,
}

impl TurnLookup {
    pub fn matches(&self, base: &Arc<Dataset>) -> bool {
        Arc::ptr_eq(&self.base, base)
    }

    pub fn get(&self, key: &str) -> Option<bool> {
        self.keys.get(key).copied()
    }

    pub fn observation(&self) -> &TurnLookupObservation {
        &self.observation
    }
}

impl TurnIngestionSpec {
    pub(crate) fn lookup_workspace(&self, lengths: impl Iterator<Item = usize>) -> Result<usize> {
        let names = self
            .primary_key
            .iter()
            .try_fold(0usize, |total, name| {
                total.checked_add(name.len().checked_mul(8)?)
            })
            .context("body lookup field names overflow")?;
        let nodes = self
            .primary_key
            .len()
            .checked_add(1)
            .and_then(|columns| columns.checked_mul(512))
            .context("body lookup key columns overflow")?;
        lengths
            .into_iter()
            .try_fold(names, |total, length| {
                total
                    .checked_add(length.checked_mul(8)?)?
                    .checked_add(nodes)
            })
            .context("body lookup workspace overflow")
    }
    /// Resolve the entire admitted key window with one scan of one snapshot.
    /// The caller accumulates unresolved keys across incoming frames before
    /// calling; do not loop over arbitrarily small windows on a large base.
    ///
    /// Per-column membership predicates let Lance use available indices while
    /// avoiding a deep OR expression for composite keys. Their cross product
    /// can include other keys, so exact composite membership is checked below.
    /// Without an index this is one key-only streaming scan, irrespective of
    /// the number of requested keys. All hits end the scan early; misses require
    /// exhausting it. This existence query is not a historical uniqueness audit.
    ///
    /// Membership alone does not deduplicate concurrent writes. The caller must
    /// retain its protected read view and validate [`TurnLookup::matches`] before
    /// ordered application; this helper does not fence publication or file GC.
    /// Native scan errors propagate rather than becoming negative lookups.
    ///
    /// `None` means workspace was not admitted, not that keys are absent. It
    /// never waits for admission while ordered apply holds other resources.
    /// Lance scan/decoder workspace requires separate caller admission.
    pub async fn lookup_existing(
        &self,
        base: Arc<Dataset>,
        keys: &[&str],
        workspace: &Arc<MergeMemoryBudget>,
    ) -> Result<Option<TurnLookup>> {
        if keys.is_empty() {
            return Ok(None);
        }
        let started = Instant::now();
        let actual = arrow_schema::Schema::from(base.schema());
        ensure!(
            actual == *self.body_schema
                || actual == *crate::annotations::single_body::storage_schema(self)?,
            "body lookup schema differs from its ingestion specification"
        );
        ensure!(
            keys.iter()
                .all(|key| !key.is_empty() && key.len() <= MAX_KEY_BYTES),
            "body lookup key exceeds limit"
        );
        let bytes = self.lookup_workspace(keys.iter().map(|key| key.len()))?;
        if bytes > workspace.limit() {
            return Ok(None);
        }
        let Some(reservation) = workspace.try_reserve(bytes) else {
            return Ok(None);
        };
        let mut requested = keys
            .iter()
            .map(|key| ((*key).to_owned(), false))
            .collect::<BTreeMap<_, _>>();
        let mut column_values = vec![BTreeSet::new(); self.primary_key.len()];
        for key in requested.keys() {
            for (values, value) in column_values.iter_mut().zip(self.key_values(key)?) {
                values.insert(value.into_owned());
            }
        }
        let predicate = self
            .primary_key
            .iter()
            .zip(column_values)
            .map(|(name, values)| {
                let values = values
                    .into_iter()
                    .map(|value| format!("'{}'", value.replace('\'', "''")))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("`{}` IN ({values})", name.replace('`', "``"))
            })
            .collect::<Vec<_>>()
            .join(" AND ");
        let mut scan = base.scan();
        scan.project(
            &self
                .primary_key
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )?;
        scan.filter(&predicate)?;
        scan.batch_size(SCAN_BATCH_ROWS)
            .batch_readahead(1)
            .fragment_readahead(1);
        let summary = Arc::new(Mutex::new(None));
        let observed = summary.clone();
        scan.scan_stats_callback(Arc::new(move |stats| {
            *observed.lock().unwrap() = Some(stats.clone());
        }));
        let mut observation = TurnLookupObservation {
            requested_keys: requested.len(),
            scans_started: 1,
            ..Default::default()
        };
        let mut remaining = requested.len();
        let mut stream = scan.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            observation.projected_rows = observation
                .projected_rows
                .saturating_add(batch.num_rows() as u64);
            observation.projected_array_bytes = observation
                .projected_array_bytes
                .saturating_add(batch.get_array_memory_size() as u64);
            let columns = batch
                .columns()
                .iter()
                .map(|column| {
                    column
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .context("invalid body key projection")
                })
                .collect::<Result<Vec<_>>>()?;
            for row in 0..batch.num_rows() {
                ensure!(
                    columns.iter().all(|column| !column.is_null(row)),
                    "null materialized body key"
                );
                let values = columns
                    .iter()
                    .map(|column| column.value(row))
                    .collect::<Vec<_>>();
                let key = self.business_key(&values, MAX_KEY_BYTES)?;
                if let Some(found) = requested.get_mut(&key) {
                    if !*found {
                        *found = true;
                        remaining -= 1;
                    }
                }
            }
            if remaining == 0 {
                break;
            }
        }
        drop(stream);
        if let Some(stats) = summary.lock().unwrap().take() {
            observation.lance_bytes_read = Some(stats.bytes_read);
            observation.lance_requests = Some(stats.requests);
            observation.lance_indices_loaded = Some(stats.indices_loaded);
            observation.fragments_scanned = stats.all_counts.get("fragments_scanned").copied();
            observation.ranges_scanned = stats.all_counts.get("ranges_scanned").copied();
            observation.rows_scanned = stats.all_counts.get("rows_scanned").copied();
            observation.task_wait_time_ns = stats.all_times.get("task_wait_time").copied();
        }
        observation.elapsed_us = started.elapsed().as_micros().try_into().unwrap_or(u64::MAX);
        Ok(Some(TurnLookup {
            base,
            keys: requested,
            observation,
            _workspace: reservation,
        }))
    }
}
