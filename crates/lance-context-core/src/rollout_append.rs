//! Parallel, immutable rollout file preparation with one fenced metadata committer.
//!
//! The caller must hold the table's maintenance claim across planning and commit.
//! Workers need no write ownership: staging never publishes a table/WAL manifest.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch, StringArray};
use arrow_schema::Schema;
use arrow_select::filter::filter_record_batch;
use datafusion::prelude::{col, lit};
use futures::{StreamExt, TryStreamExt};
use lance::dataset::{
    mem_wal::{DatasetMemWalExt, ShardManifestStore},
    transaction::{Operation, Transaction},
    CommitBuilder, Dataset, InsertBuilder, WriteMode, WriteParams,
};
use lance::index::DatasetIndexExt;
use lance_index::mem_wal::{MemWalIndexDetails, MergedGeneration, MEM_WAL_INDEX_NAME};
use lance_table::format::{pb, Fragment};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::store_base::{align_batch_to_schema, derive_shard_id, StorageBase};
use crate::{LanceError as Error, MergeMemoryBudget, Session};
use lance::Result;

const CUTOVER_PREFIX: &str = "lance-context.rollout-append.cutover.";
pub const MAX_PLAN_GENERATIONS: usize = 256;
pub const MAX_STAGE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Generation {
    pub number: u64,
    pub path: String,
}

/// A bounded, immutable prefix of one shard. URIs/credentials are never supplied
/// by an RPC caller: each process resolves the rollout name in its own registry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppendPlan {
    pub base_version: u64,
    pub shard: Uuid,
    pub generations: Vec<Generation>,
    pub max_bytes: usize,
    pub legacy_through: u64,
}

impl AppendPlan {
    pub fn validate(&self) -> Result<()> {
        if self.base_version == 0
            || self.generations.is_empty()
            || self.generations.len() > MAX_PLAN_GENERATIONS
            || self.max_bytes == 0
            || self.max_bytes > MAX_STAGE_BYTES
            || self
                .generations
                .windows(2)
                .any(|g| g[0].number >= g[1].number)
            || self.generations.iter().any(|g| {
                g.path.is_empty() || g.path.contains('/') || g.path.contains('\\') || g.path == ".."
            })
        {
            return Err(Error::invalid_input("invalid rollout append plan"));
        }
        Ok(())
    }
}

/// Only file metadata crosses the worker/master boundary. No Arrow payloads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StagedAppend {
    pub plan: AppendPlan,
    pub completed: usize,
    pub fragments: Vec<Fragment>,
    pub rows: usize,
}

/// Streaming RPC keeps completed work distinct from a live connection.
#[derive(Debug, Serialize, Deserialize)]
pub enum StageEvent {
    Progress(u64),
    Complete(StagedAppend),
    Failed(String),
}

pub(crate) async fn watermarks(dataset: &Dataset) -> Result<HashMap<Uuid, u64>> {
    let indices = dataset.load_indices().await?;
    let Some(index) = indices.iter().find(|i| i.name == MEM_WAL_INDEX_NAME) else {
        return Ok(HashMap::new());
    };
    let details = index
        .index_details
        .as_ref()
        .ok_or_else(|| Error::io("MemWAL index has no details"))?;
    let details = MemWalIndexDetails::try_from(details.to_msg::<pb::MemWalIndexDetails>()?)?;
    Ok(details
        .merged_generations
        .into_iter()
        .map(|g| (g.shard_id, g.generation))
        .collect())
}

async fn shard_store(dataset: &Dataset, shard: Uuid) -> Result<ShardManifestStore> {
    Ok(ShardManifestStore::new(
        dataset.object_store(None).await?,
        &dataset.branch_location().path,
        shard,
        16,
    ))
}

/// A previous append can be visible while its separate shard drain is not.
/// Reconcile solely from the atomic base watermark, without reading any WAL data.
pub(crate) async fn reconcile_shard(dataset: &Dataset, shard: Uuid) -> Result<usize> {
    let Some(high) = watermarks(dataset).await?.get(&shard).copied() else {
        return Ok(0);
    };
    let store = shard_store(dataset, shard).await?;
    let Some(manifest) = store.read_latest().await? else {
        return Ok(0);
    };
    let generations: HashSet<_> = manifest
        .flushed_generations
        .iter()
        .filter(|g| g.generation <= high)
        .map(|g| g.generation)
        .collect();
    let count = generations.len();
    if count > 0 {
        let paths: Vec<_> = manifest
            .flushed_generations
            .iter()
            .filter(|g| generations.contains(&g.generation))
            .map(|g| g.path.clone())
            .collect();
        crate::merge_write_scope::drain_generations(store, manifest.writer_epoch, generations)
            .await?;
        let object_store = dataset.object_store(None).await?;
        let root = dataset
            .branch_location()
            .path
            .join("_mem_wal")
            .join(shard.to_string().as_str());
        tokio::spawn(async move {
            futures::stream::iter(paths)
                .for_each_concurrent(16, |path| {
                    let object_store = object_store.clone();
                    let path = root.clone().join(path.as_str());
                    async move {
                        if let Err(error) = object_store.remove_dir_all(path).await {
                            tracing::warn!(%error, "merged rollout WAL directory cleanup failed");
                        }
                    }
                })
                .await;
        });
    }
    Ok(count)
}

/// The only publisher. Open this inside the existing MergeWriteScope so both
/// table commits and shard drains retain ownership checks and version fencing.
pub struct AppendCoordinator {
    dataset: Dataset,
}

impl AppendCoordinator {
    pub async fn open(uri: &str, session: Option<Arc<Session>>) -> Result<Self> {
        // Do not open a resident shard writer or evolve schema on a staging worker.
        let dataset = StorageBase::load_with_options(uri, None, session).await?;
        let actual: Schema = dataset.schema().into();
        let expected = crate::rollout_schema();
        if actual != expected {
            return Err(Error::invalid_input(
                "rollout append requires the current rollout schema",
            ));
        }
        Ok(Self { dataset })
    }

    pub fn version(&self) -> u64 {
        self.dataset.version().version
    }

    pub async fn plan(
        &mut self,
        shards: &[String],
        max_generations: usize,
        max_bytes: usize,
    ) -> Result<(Vec<AppendPlan>, usize)> {
        if max_generations == 0
            || max_generations > MAX_PLAN_GENERATIONS
            || max_bytes == 0
            || max_bytes > MAX_STAGE_BYTES
            || shards.len() > 256
        {
            return Err(Error::invalid_input("invalid rollout append limits"));
        }
        self.dataset.checkout_latest().await?;
        let mut unique = HashSet::new();
        for name in shards {
            if !unique.insert(derive_shard_id(Some(name))) {
                return Err(Error::invalid_input("duplicate rollout append shard"));
            }
        }
        // Metadata-only shard-directory enumeration also finds retired workers.
        // An unavailable ingestion worker must not strand its immutable WAL.
        unique.extend(self.dataset.list_mem_wal_latest_shard_ids().await?);
        if unique.len() > 256 {
            return Err(Error::invalid_input(
                "rollout append supports at most 256 shards per table",
            ));
        }
        let mut shard_ids: Vec<_> = unique.into_iter().collect();
        shard_ids.sort();
        let mut pending = Vec::new();
        let mut cutovers = HashMap::new();
        let mut reclaimed = 0;
        for shard in shard_ids {
            reclaimed += reconcile_shard(&self.dataset, shard).await?;
            let Some(manifest) = shard_store(&self.dataset, shard)
                .await?
                .read_latest()
                .await?
            else {
                continue;
            };
            crate::merge_write_scope::checkpoint();
            let key = format!("{CUTOVER_PREFIX}{shard}");
            let cutoff = match self.dataset.metadata().get(&key) {
                Some(value) => value
                    .parse::<u64>()
                    .map_err(|_| Error::io("invalid rollout cutover"))?,
                None => {
                    // The historical prefix can contain a legacy append whose
                    // drain failed. Only that prefix needs a base ID lookup.
                    let high = manifest
                        .flushed_generations
                        .iter()
                        .map(|g| g.generation)
                        .max()
                        .unwrap_or(0);
                    cutovers.insert(key, high.to_string());
                    high
                }
            };
            let mut generations: Vec<_> = manifest
                .flushed_generations
                .into_iter()
                .map(|g| Generation {
                    number: g.generation,
                    path: g.path,
                })
                .collect();
            generations.sort_by_key(|g| g.number);
            generations.truncate(max_generations);
            if !generations.is_empty() {
                pending.push((shard, generations, cutoff));
            }
        }
        if !cutovers.is_empty() {
            self.dataset
                .update_metadata(cutovers.iter().map(|(k, v)| (k.as_str(), v.as_str())))
                .await?;
        }
        let version = self.version();
        let plans = pending
            .into_iter()
            .map(|(shard, generations, legacy_through)| AppendPlan {
                base_version: version,
                shard,
                generations,
                max_bytes,
                legacy_through,
            })
            .collect();
        Ok((plans, reclaimed))
    }

    /// Publication is a single transaction: immutable fragments + per-shard
    /// merged prefix. No ID index maintenance, delete, or target payload scan.
    /// Do not retry this Transaction blindly after an uncertain response: call
    /// this method again so the freshly read watermarks decide what remains.
    pub async fn commit(&mut self, staged: Vec<StagedAppend>) -> Result<usize> {
        self.dataset.checkout_latest().await?;
        let marks = watermarks(&self.dataset).await?;
        let mut shards = HashSet::new();
        let mut fragments = Vec::new();
        let mut merged = Vec::new();
        for part in staged {
            part.plan.validate()?;
            if part.completed == 0
                || part.completed > part.plan.generations.len()
                || !shards.insert(part.plan.shard)
            {
                return Err(Error::invalid_input("invalid or duplicate staged shard"));
            }
            let selected = &part.plan.generations[..part.completed];
            let high = selected.last().unwrap().number;
            if let Some(old) = marks.get(&part.plan.shard) {
                if high <= *old {
                    continue;
                }
                if selected[0].number <= *old {
                    return Err(Error::invalid_input(
                        "partially stale staged prefix; replan",
                    ));
                }
            }
            // Require the version used to encode fields to have the same schema.
            let snapshot = self
                .dataset
                .checkout_version(part.plan.base_version)
                .await?;
            // Dictionary values/offsets are loaded lazily and are file-local
            // in V2. Raw Lance Schema equality compares that cache state too.
            let options = lance::datatypes::SchemaCompareOptions {
                compare_metadata: true,
                compare_field_ids: true,
                ..Default::default()
            };
            if Schema::from(snapshot.schema()) != Schema::from(self.dataset.schema())
                || snapshot
                    .schema()
                    .check_compatible(self.dataset.schema(), &options)
                    .is_err()
            {
                return Err(Error::invalid_input(
                    "schema changed during rollout staging; replan",
                ));
            }
            let manifest = shard_store(&self.dataset, part.plan.shard)
                .await?
                .read_latest()
                .await?
                .ok_or_else(|| Error::io("staged shard disappeared"))?;
            let mut pending: Vec<_> = manifest.flushed_generations.iter().collect();
            pending.sort_by_key(|g| g.generation);
            if pending.len() < selected.len()
                || pending
                    .iter()
                    .zip(selected)
                    .any(|(a, b)| a.generation != b.number || a.path != b.path)
            {
                return Err(Error::invalid_input("staged WAL prefix changed; replan"));
            }
            if part
                .fragments
                .iter()
                .map(|f| f.physical_rows.unwrap_or(0))
                .sum::<usize>()
                != part.rows
            {
                return Err(Error::invalid_input("staged row count mismatch"));
            }
            fragments.extend(part.fragments);
            merged.push(MergedGeneration::new(part.plan.shard, high));
        }
        if !merged.is_empty() {
            commit_files(&mut self.dataset, fragments, merged).await?;
        }
        let mut reclaimed = 0;
        for shard in shards {
            reclaimed += reconcile_shard(&self.dataset, shard).await?;
        }
        Ok(reclaimed)
    }
}

/// Keep legacy fallback publications recoverable after an append-mode cutover.
pub(crate) fn has_cutover(dataset: &Dataset, shard: Uuid) -> bool {
    dataset
        .metadata()
        .contains_key(&format!("{CUTOVER_PREFIX}{shard}"))
}

pub(crate) async fn commit_files(
    dataset: &mut Dataset,
    fragments: Vec<Fragment>,
    merged: Vec<MergedGeneration>,
) -> Result<()> {
    let operation = Operation::Update {
        removed_fragment_ids: Vec::new(),
        updated_fragments: Vec::new(),
        new_fragments: fragments,
        fields_modified: Vec::new(),
        merged_generations: merged,
        fields_for_preserving_frag_bitmap: Vec::new(),
        update_mode: None,
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    // Even zero-retry Lance transactions can rebase before their first write.
    let version = dataset.version().version;
    crate::merge_write_scope::at_base_version(
        version + 1,
        CommitBuilder::new(Arc::new(dataset.clone()))
            .with_max_retries(0)
            .execute(Transaction::new(version, operation, None)),
    )
    .await?;
    dataset.checkout_latest().await?;
    Ok(())
}

/// Encode files only. Holds the shared worker memory reservation until encoding
/// finishes, then returns small metadata. No shard epoch or manifest is changed.
pub async fn stage(
    uri: &str,
    plan: AppendPlan,
    budget: Arc<MergeMemoryBudget>,
    session: Option<Arc<Session>>,
) -> Result<StagedAppend> {
    plan.validate()?;
    let dataset = StorageBase::load_with_options(uri, None, session.clone())
        .await?
        .checkout_version(plan.base_version)
        .await?;
    let schema: Arc<Schema> = Arc::new(dataset.schema().into());
    if *schema != crate::rollout_schema() {
        return Err(Error::invalid_input("staging requires a rollout schema"));
    }
    let key = format!("{CUTOVER_PREFIX}{}", plan.shard);
    if dataset.metadata().get(&key) != Some(&plan.legacy_through.to_string()) {
        return Err(Error::invalid_input(
            "staging plan does not match durable cutover",
        ));
    }
    let mut reservation = budget.reserve(plan.max_bytes.min(budget.limit())).await;
    let mut batches = Vec::new();
    let mut bytes = 0usize;
    let mut completed = 0;
    'generations: for generation in &plan.generations {
        let path = format!(
            "{}/_mem_wal/{}/{}",
            uri.trim_end_matches('/'),
            plan.shard,
            generation.path
        );
        let source = StorageBase::load_with_options(&path, None, session.clone()).await?;
        let mut stream = source.scan().try_into_stream().await?;
        let mut current = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let batch = align_batch_to_schema(batch, schema.clone())?;
            bytes = bytes.saturating_add(batch.get_array_memory_size());
            if !reservation.try_grow_to(bytes.saturating_mul(2)) {
                if completed == 0 {
                    return Err(Error::io(
                        "merge memory budget busy while reading first generation; retry",
                    ));
                }
                break 'generations;
            }
            current.push(batch);
            crate::merge_write_scope::checkpoint();
        }
        batches.extend(current);
        completed += 1;
        if bytes >= plan.max_bytes {
            break;
        }
    }
    // Rollout IDs are immutable. Dedup identical retries within this batch;
    // during migration also exclude IDs already published by a legacy merge.
    let mut seen = HashSet::new();
    if plan.generations[0].number <= plan.legacy_through {
        let mut ids = HashSet::new();
        for batch in &batches {
            let column = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::invalid_input("rollout id must be Utf8"))?;
            ids.extend(column.iter().flatten().map(str::to_owned));
        }
        let ids: Vec<_> = ids.into_iter().collect();
        for chunk in ids.chunks(1024) {
            let mut scanner = dataset.scan();
            scanner.project(&["id"])?;
            scanner.filter_expr(
                col("id").in_list(chunk.iter().map(|id| lit(id.clone())).collect(), false),
            );
            let mut rows = scanner.try_into_stream().await?;
            while let Some(batch) = rows.try_next().await? {
                let column = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                seen.extend(column.iter().flatten().map(str::to_owned));
                crate::merge_write_scope::checkpoint();
            }
        }
    }
    let mut output = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        if ids.null_count() > 0 {
            return Err(Error::invalid_input("rollout id cannot be null"));
        }
        let keep = BooleanArray::from(
            ids.iter()
                .map(|id| seen.insert(id.unwrap().to_owned()))
                .collect::<Vec<_>>(),
        );
        if keep.true_count() == batch.num_rows() {
            output.push(batch);
        } else if keep.true_count() > 0 {
            output.push(filter_record_batch(&batch, &keep)?);
        }
    }
    let rows = output.iter().map(RecordBatch::num_rows).sum();
    let fragments = if output.is_empty() {
        Vec::new()
    } else {
        let params = WriteParams {
            mode: WriteMode::Append,
            max_bytes_per_file: plan.max_bytes,
            write_progress: Some(crate::merge_write_scope::write_progress()),
            ..Default::default()
        };
        let tx = InsertBuilder::new(Arc::new(dataset))
            .with_params(&params)
            .execute_uncommitted(output)
            .await?;
        let Operation::Append { fragments } = tx.operation else {
            return Err(Error::io("staging produced a non-append transaction"));
        };
        fragments
    };
    drop(reservation);
    crate::merge_write_scope::checkpoint();
    Ok(StagedAppend {
        plan,
        completed,
        fragments,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_write_scope::{CommitAuthorizer, MergeWriteScope};
    use crate::{RolloutRecord, RolloutStore, RolloutStoreOptions, ROLE_ARTIFACT};
    use chrono::{TimeZone, Utc};
    use serde_json::json;
    use std::{
        future::Future,
        pin::Pin,
        sync::atomic::{AtomicBool, Ordering},
    };
    fn artifact_record(id: &str, bytes: &[u8]) -> RolloutRecord {
        RolloutRecord {
            id: id.to_string(),
            rollout_id: "rollout-1".to_string(),
            problem_id: "problem-1".to_string(),
            dataset: None,
            sequence_order: 1,
            role: ROLE_ARTIFACT.to_string(),
            created_at: Utc.timestamp_micros(1_700_000_000_500_000).unwrap(),
            content: None,
            content_type: "application/octet-stream".to_string(),
            model_input_string: None,
            model_output_string: None,
            rationale: None,
            problem_text: None,
            user_metadata: None,
            input_tokens: None,
            output_tokens: None,
            num_input_tokens: None,
            num_output_tokens: None,
            output_logprobs: None,
            input_logprobs: None,
            ref_logprobs: None,
            loss_mask: None,
            advantage: None,
            reward: None,
            raw_reward: None,
            grader_id: None,
            score: None,
            include_in_training: None,
            exclude_reason: None,
            policy_version: None,
            relationships: Vec::new(),
            binary_payload: Some(bytes.to_vec()),
            payload_size: Some(bytes.len() as i64),
            payload_checksum: Some("sha256:cafef00d".to_string()),
            artifact_type: Some("excel_grade_screenshot".to_string()),
            metadata: Some(json!({"filename": "trace.bin"})),
        }
    }

    async fn writer(uri: &str, shard: &str) -> RolloutStore {
        RolloutStore::open_with_options(
            uri,
            RolloutStoreOptions {
                shard_id: Some(shard.into()),
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap()
    }
    async fn put(store: &RolloutStore, id: &str, bytes: usize) {
        store
            .add(&[artifact_record(id, &vec![42; bytes])])
            .await
            .unwrap();
        store.flush().await.unwrap();
    }
    async fn rows(uri: &str) -> usize {
        Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap()
    }
    fn budget() -> Arc<MergeMemoryBudget> {
        MergeMemoryBudget::new(8 * 1024 * 1024)
    }

    #[tokio::test]
    async fn parallel_files_publish_once_and_preserve_new_writes() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let a = writer(uri, "a").await;
        let b = writer(uri, "b").await;
        put(&a, "a1", 4096).await;
        put(&b, "b1", 4096).await;
        let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
        let (plans, _) = commit
            .plan(&["a".into(), "b".into()], 64, 1024 * 1024)
            .await
            .unwrap();
        let before = commit.version();
        let memory = budget();
        let (a_part, b_part) = tokio::try_join!(
            stage(uri, plans[0].clone(), memory.clone(), None),
            stage(uri, plans[1].clone(), memory.clone(), None)
        )
        .unwrap();
        assert_eq!(rows(uri).await, 0, "staging must not publish rows");
        assert_eq!(memory.reserved(), 0);
        put(&a, "a2", 4096).await;
        assert_eq!(
            commit
                .commit(vec![a_part.clone(), b_part.clone()])
                .await
                .unwrap(),
            2
        );
        assert_eq!(commit.version(), before + 1, "one version for both workers");
        assert_eq!(rows(uri).await, 2);
        // Duplicate responses and retries do not create versions or rows.
        assert_eq!(commit.commit(vec![a_part, b_part]).await.unwrap(), 0);
        assert_eq!(commit.version(), before + 1);
        let fresh = RolloutStore::open_existing_with_options(uri, RolloutStoreOptions::default())
            .await
            .unwrap();
        assert_eq!(fresh.get_blob("a1").await.unwrap(), Some(vec![42; 4096]));
        assert_eq!(fresh.get_blob("b1").await.unwrap(), Some(vec![42; 4096]));
        assert_eq!(fresh.get_blob("a2").await.unwrap(), Some(vec![42; 4096]));
        let (plans, _) = commit
            .plan(&["a".into(), "b".into()], 64, 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(plans.len(), 1);
        let part = stage(uri, plans[0].clone(), memory, None).await.unwrap();
        commit.commit(vec![part]).await.unwrap();
        assert_eq!(rows(uri).await, 3);
        assert_eq!(
            commit
                .dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .filter(|i| i.name != MEM_WAL_INDEX_NAME)
                .count(),
            0,
            "append merge must not build an ID index"
        );
    }

    #[derive(Debug)]
    struct FailDrain(AtomicBool);
    impl CommitAuthorizer for FailDrain {
        fn authorize<'a>(
            &'a self,
            resource: &'a str,
            _version: u64,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                if resource.starts_with("shard:") && self.0.load(Ordering::SeqCst) {
                    Err(Error::io("injected failure after base commit"))
                } else {
                    Ok(())
                }
            })
        }
    }

    #[tokio::test]
    async fn restart_after_base_commit_repairs_drain_without_duplicate_append() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let a = writer(uri, "a").await;
        put(&a, "a1", 4096).await;
        let guard = Arc::new(FailDrain(AtomicBool::new(true)));
        let scope = MergeWriteScope::with_pinned_authorizer(guard.clone());
        let (part, version) = scope
            .run(async {
                let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
                let (plans, _) = commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap();
                let part = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
                assert!(commit
                    .commit(vec![part.clone()])
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("injected"));
                (part, Dataset::open(uri).await.unwrap().version().version)
            })
            .await;
        scope.drain().await;
        assert_eq!(rows(uri).await, 1);
        put(&a, "a2", 4096).await;
        // A fresh coordinator has no in-memory record of the first execution.
        let mut restarted = AppendCoordinator::open(uri, None).await.unwrap();
        assert_eq!(restarted.commit(vec![part]).await.unwrap(), 1);
        assert_eq!(restarted.version(), version);
        assert_eq!(rows(uri).await, 1);
        let (plans, _) = restarted
            .plan(&["a".into()], 64, 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(plans[0].generations.len(), 1);
        let staged = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
        restarted.commit(vec![staged]).await.unwrap();
        assert_eq!(rows(uri).await, 2);
    }

    #[tokio::test]
    async fn legacy_append_without_drain_is_not_duplicated_at_cutover() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut a = writer(uri, "a").await;
        put(&a, "a1", 4096).await;
        let guard = Arc::new(FailDrain(AtomicBool::new(true)));
        let scope = MergeWriteScope::with_pinned_authorizer(guard);
        // Open inside the scope so the legacy publisher uses its guard.
        scope
            .run(async {
                a = writer(uri, "a").await;
                assert!(a.cleanup_own_shard().await.is_err());
            })
            .await;
        scope.drain().await;
        assert_eq!(rows(uri).await, 1);
        put(&a, "a2", 4096).await;
        let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
        let (plans, _) = commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap();
        let part = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
        assert_eq!(part.rows, 1, "base ID lookup removes legacy retry");
        assert_eq!(commit.commit(vec![part]).await.unwrap(), 2);
        assert_eq!(rows(uri).await, 2);
    }

    #[tokio::test]
    async fn legacy_merge_honors_committed_append_watermark() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let a = writer(uri, "a").await;
        put(&a, "a1", 4096).await;
        let scope =
            MergeWriteScope::with_pinned_authorizer(Arc::new(FailDrain(AtomicBool::new(true))));
        scope
            .run(async {
                let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
                let (plans, _) = commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap();
                let part = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
                assert!(commit.commit(vec![part]).await.is_err());
            })
            .await;
        scope.drain().await;
        put(&a, "a2", 4096).await;
        let mut legacy = writer(uri, "a").await;
        legacy.cleanup_own_shard().await.unwrap();
        assert_eq!(rows(uri).await, 2);
        let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
        assert!(commit
            .plan(&["a".into()], 64, 1024 * 1024)
            .await
            .unwrap()
            .0
            .is_empty());
    }

    #[tokio::test]
    async fn partial_stale_prefix_rejected_and_byte_cap_preserves_tail() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let a = writer(uri, "a").await;
        for i in 0..3 {
            put(&a, &format!("a{i}"), 128 * 1024).await;
        }
        let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
        let (plans, _) = commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap();
        let large = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
        let mut small_plan = plans[0].clone();
        small_plan.max_bytes = 64 * 1024;
        let small = stage(uri, small_plan, budget(), None).await.unwrap();
        assert_eq!(small.completed, 1);
        commit.commit(vec![small]).await.unwrap();
        assert!(commit
            .commit(vec![large])
            .await
            .unwrap_err()
            .to_string()
            .contains("partially stale"));
        assert_eq!(rows(uri).await, 1);
        assert_eq!(
            commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap().0[0]
                .generations
                .len(),
            2
        );
    }
    #[tokio::test]
    async fn changed_schema_rejects_staged_files() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let a = writer(uri, "a").await;
        put(&a, "a1", 4096).await;
        let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
        let (plans, _) = commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap();
        let part = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
        let mut dataset = Dataset::open(uri).await.unwrap();
        dataset
            .update_schema_metadata([("changed", "true")])
            .await
            .unwrap();
        assert!(commit
            .commit(vec![part])
            .await
            .unwrap_err()
            .to_string()
            .contains("schema changed"));
        assert_eq!(rows(uri).await, 0);
    }

    #[tokio::test]
    async fn exact_version_guard_rejects_lances_initial_rebase() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let _a = writer(uri, "a").await;
        let old = Dataset::open(uri).await.unwrap();
        let mut current = old.clone();
        current
            .update_metadata([("concurrent", "commit")])
            .await
            .unwrap();
        let version = current.version().version;
        let handler = lance_table::io::commit::commit_handler_from_url(uri, &None)
            .await
            .unwrap();
        let handler = Arc::new(crate::merge_write_scope::GuardedCommit::new(handler));
        let expected = old.version().version + 1;
        let transaction = Transaction::new(
            old.version().version,
            Operation::Append {
                fragments: Vec::new(),
            },
            None,
        );
        assert!(crate::merge_write_scope::at_base_version(
            expected,
            CommitBuilder::new(Arc::new(old))
                .with_commit_handler(handler)
                .with_max_retries(0)
                .execute(transaction)
        )
        .await
        .is_err());
        assert_eq!(Dataset::open(uri).await.unwrap().version().version, version);
    }

    #[tokio::test]
    async fn revoked_owner_cannot_publish_prepared_files() {
        #[derive(Debug)]
        struct Reject;
        impl CommitAuthorizer for Reject {
            fn authorize<'a>(
                &'a self,
                _resource: &'a str,
                _version: u64,
            ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
                Box::pin(async { Err(Error::io("owner revoked")) })
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let a = writer(uri, "a").await;
        put(&a, "a1", 4096).await;
        let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
        let (plans, _) = commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap();
        let part = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
        let version = commit.version();
        let scope = MergeWriteScope::with_pinned_authorizer(Arc::new(Reject));
        let mut publisher = scope.run(AppendCoordinator::open(uri, None)).await.unwrap();
        // A captured owner guard must remain effective after leaving its task-local scope.
        let error = publisher.commit(vec![part.clone()]).await.unwrap_err();
        assert!(error.to_string().contains("owner revoked"));
        assert_eq!(Dataset::open(uri).await.unwrap().version().version, version);
        assert_eq!(rows(uri).await, 0);
        // A replacement owner may reuse the immutable files.
        commit.commit(vec![part]).await.unwrap();
        assert_eq!(rows(uri).await, 1);
    }
    /// Opt-in synthetic benchmark. Never scans an existing table: every fixture
    /// gets a UUID path beneath the supplied scratch root.
    #[tokio::test]
    #[ignore = "opt-in rollout append benchmark"]
    async fn benchmark_rollout_parallel_append() {
        if std::env::var("ROLLOUT_APPEND_BENCH").as_deref() != Ok("1") {
            return;
        }
        for repeat in 0..2 {
            for parallel in if repeat == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let dir = tempfile::tempdir().unwrap();
                let root = std::env::var("ROLLOUT_APPEND_BENCH_ROOT")
                    .unwrap_or_else(|_| dir.path().to_string_lossy().into());
                let uri = format!(
                    "{}/rollout-append-bench-{}",
                    root.trim_end_matches('/'),
                    Uuid::new_v4()
                );
                let mut writers = Vec::new();
                let mut shards = Vec::new();
                for i in 0..4 {
                    let shard = format!("worker-{i}");
                    let store = writer(&uri, &shard).await;
                    for generation in 0..8 {
                        let records: Vec<_> = (0..16)
                            .map(|row| {
                                artifact_record(&format!("{i}-{generation}-{row}"), &vec![42; 8192])
                            })
                            .collect();
                        store.add(&records).await.unwrap();
                        store.flush().await.unwrap();
                    }
                    writers.push(store);
                    shards.push(shard);
                }
                let initial = Dataset::open(&uri).await.unwrap().version().version;
                let start = std::time::Instant::now();
                if parallel {
                    let mut commit = AppendCoordinator::open(&uri, None).await.unwrap();
                    let (plans, _) = commit.plan(&shards, 64, 2 * 1024 * 1024).await.unwrap();
                    let memory = MergeMemoryBudget::new(64 * 1024 * 1024);
                    let results = futures::future::try_join_all(
                        plans
                            .into_iter()
                            .map(|plan| stage(&uri, plan, memory.clone(), None)),
                    )
                    .await
                    .unwrap();
                    assert_eq!(commit.commit(results).await.unwrap(), 32);
                    assert_eq!(memory.reserved(), 0);
                } else {
                    for store in &mut writers {
                        store.cleanup_own_shard().await.unwrap();
                    }
                }
                let seconds = start.elapsed().as_secs_f64();
                let dataset = Dataset::open(&uri).await.unwrap();
                assert_eq!(dataset.count_rows(None).await.unwrap(), 512);
                let mut scan = dataset.scan();
                scan.project(&["id", "binary_payload"]).unwrap();
                let mut rows = scan.try_into_stream().await.unwrap();
                let mut ids = HashSet::new();
                while let Some(batch) = rows.try_next().await.unwrap() {
                    let keys = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();
                    let blobs = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<arrow_array::LargeBinaryArray>()
                        .unwrap();
                    for row in 0..batch.num_rows() {
                        assert!(ids.insert(keys.value(row).to_owned()));
                        assert_eq!(blobs.value(row), vec![42; 8192]);
                    }
                }
                println!(
                    "APPEND_BENCH {}",
                    json!({"parallel":parallel,"repeat":repeat,"seconds":seconds,
                    "generations_per_second":32.0/seconds,"versions":dataset.version().version-initial,"rows":ids.len(),"uri":uri})
                );
            }
        }
    }
    #[tokio::test]
    async fn disable_then_reenable_preserves_fallback_commit_watermarks() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let a = writer(uri, "a").await;
        put(&a, "a1", 4096).await;
        let mut commit = AppendCoordinator::open(uri, None).await.unwrap();
        let (plans, _) = commit.plan(&["a".into()], 64, 1024 * 1024).await.unwrap();
        let part = stage(uri, plans[0].clone(), budget(), None).await.unwrap();
        commit.commit(vec![part]).await.unwrap();
        put(&a, "a2", 4096).await;
        let scope =
            MergeWriteScope::with_pinned_authorizer(Arc::new(FailDrain(AtomicBool::new(true))));
        scope
            .run(async {
                let mut fallback = writer(uri, "a").await;
                assert!(fallback.cleanup_own_shard().await.is_err());
            })
            .await;
        scope.drain().await;
        assert_eq!(rows(uri).await, 2);
        let mut restarted = AppendCoordinator::open(uri, None).await.unwrap();
        let (plans, reclaimed) = restarted
            .plan(&["a".into()], 64, 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(reclaimed, 1);
        assert!(plans.is_empty());
        assert_eq!(rows(uri).await, 2);
    }
}
