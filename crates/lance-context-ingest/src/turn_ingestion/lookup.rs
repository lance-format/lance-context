//! Bounded composite-key probes against the full body dataset.
use super::*;
use futures::TryStreamExt;
use lance::{dataset::scanner::DatasetRecordBatchStream, Dataset};
use lance_context_core::{MergeMemoryBudget, MergeReservation};
use lance_datafusion::exec::{
    collect_execution_metrics, execute_plan, ExecutionSummaryCounts, LanceExecutionOptions,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

const MAX_KEY_BYTES: usize = 64 << 10;
const SCAN_BATCH_ROWS: usize = 2048;
const ADMITTED_SCAN_READAHEAD: usize = 8;

/// One admitted lookup window. Projection counters describe returned key
/// batches; they are not physical object-store or HTTP bytes. Native counters
/// snapshot successful scans at lookup completion, including early exits.
/// They exclude work finishing later; staging-only resolutions have no scan.
#[derive(Clone, Debug, Default)]
pub struct TurnLookupObservation {
    pub requested_keys: usize,
    pub staged_keys: usize,
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

/// A bounded input window, before capturing the base and doing any native I/O.
pub(crate) struct PreparedTurnLookup {
    pub(crate) keys: BTreeMap<String, bool>,
    workspace: MergeReservation,
    started: Instant,
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
    /// `None` means workspace was not admitted, not that keys are absent. It
    /// never waits for admission while ordered apply holds other resources.
    /// Lance scan/decoder workspace requires separate caller admission. A
    /// caller-supplied native memory pool permits bounded concurrent reads;
    /// its reservations follow read/decode buffers and refusals remain errors.
    pub async fn lookup_existing(
        &self,
        base: Arc<Dataset>,
        keys: &[&str],
        workspace: &Arc<MergeMemoryBudget>,
    ) -> Result<Option<TurnLookup>> {
        let Some(prepared) = self.prepare_lookup(keys, workspace)? else {
            return Ok(None);
        };
        Ok(Some(self.lookup_prepared(base, prepared).await?))
    }

    pub(crate) fn prepare_lookup(
        &self,
        keys: &[&str],
        workspace: &Arc<MergeMemoryBudget>,
    ) -> Result<Option<PreparedTurnLookup>> {
        if keys.is_empty() {
            return Ok(None);
        }
        let started = Instant::now();
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
        Ok(Some(PreparedTurnLookup {
            keys: keys.iter().map(|key| ((*key).to_owned(), false)).collect(),
            workspace: reservation,
            started,
        }))
    }

    pub(crate) async fn lookup_prepared(
        &self,
        base: Arc<Dataset>,
        prepared: PreparedTurnLookup,
    ) -> Result<TurnLookup> {
        ensure!(
            arrow_schema::Schema::from(base.schema()) == *self.body_schema
                || arrow_schema::Schema::from(base.schema())
                    == *crate::annotations::single_body::storage_schema(self)?,
            "body lookup schema differs from its ingestion specification"
        );
        let PreparedTurnLookup {
            keys: mut requested,
            workspace: reservation,
            started,
        } = prepared;
        let staged_keys = requested.values().filter(|present| **present).count();
        let mut remaining = requested.len() - staged_keys;
        let mut observation = TurnLookupObservation {
            requested_keys: requested.len(),
            staged_keys,
            ..Default::default()
        };
        if remaining == 0 {
            observation.elapsed_us = started.elapsed().as_micros().try_into().unwrap_or(u64::MAX);
            return Ok(TurnLookup {
                base,
                keys: requested,
                observation,
                _workspace: reservation,
            });
        }
        let mut column_values = vec![BTreeSet::new(); self.primary_key.len()];
        for (key, present) in &requested {
            if *present {
                continue;
            }
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
        let readahead = if base.session().native_memory().is_some() {
            // These are scanner targets, not a replacement for admission of
            // actual native pages. Keep the caller's pool and shared runtime.
            scan.batch_size_bytes(1 << 20).io_buffer_size(8 << 20);
            ADMITTED_SCAN_READAHEAD
        } else {
            1
        };
        scan.batch_size(SCAN_BATCH_ROWS)
            .batch_readahead(readahead)
            .fragment_readahead(readahead);
        observation.scans_started = 1;
        // Keep the scanner's plan so an all-hit early exit can report work
        // without polling to EOF just to trigger its completion callback.
        // Preserve Scanner::execution_options and the logical schema adapter.
        let plan = scan.create_plan().await?;
        let mut stream = DatasetRecordBatchStream::new(execute_plan(
            plan.clone(),
            LanceExecutionOptions {
                runtime_env: base.session().runtime_env(),
                batch_size: Some(SCAN_BATCH_ROWS),
                ..Default::default()
            },
        )?);
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
        // One snapshot for either completion path. Do not wait for cancelled
        // background work: these are observed plan metrics, not all store I/O.
        let mut stats = ExecutionSummaryCounts::default();
        collect_execution_metrics(plan.as_ref(), &mut stats);
        drop(plan);
        observation.lance_bytes_read = Some(stats.bytes_read);
        observation.lance_requests = Some(stats.requests);
        observation.lance_indices_loaded = Some(stats.indices_loaded);
        observation.fragments_scanned = stats.all_counts.get("fragments_scanned").copied();
        observation.ranges_scanned = stats.all_counts.get("ranges_scanned").copied();
        observation.rows_scanned = stats.all_counts.get("rows_scanned").copied();
        observation.task_wait_time_ns = stats.all_times.get("task_wait_time").copied();
        observation.elapsed_us = started.elapsed().as_micros().try_into().unwrap_or(u64::MAX);
        Ok(TurnLookup {
            base,
            keys: requested,
            observation,
            _workspace: reservation,
        })
    }
}
