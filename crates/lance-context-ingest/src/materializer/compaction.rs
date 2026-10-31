//! Rewrite bounded small-file groups through the existing managed publication.
//! Old files remain available to captured snapshots; this does not run object GC.
use super::*;
use lance::dataset::{
    optimize::{CompactionMode, CompactionOptions, CompactionTask, RewriteResult, TaskData},
    transaction::{Operation, RewriteGroup},
};
use lance::index::DatasetIndexExt;
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{Fragment, IndexMetadata};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SmallFileCompaction {
    pub target_file_bytes: usize,
    pub min_fragments: usize,
}
impl Default for SmallFileCompaction {
    fn default() -> Self {
        Self {
            target_file_bytes: 128 << 20,
            min_fragments: 4,
        }
    }
}
impl SmallFileCompaction {
    fn workspace(&self, record_bytes: usize) -> Result<usize> {
        // One task admits at most twice the target on disk. Also allow for
        // four late-materialization batches and decode/encoding workspace.
        // These are targets, not an allocator/RSS cap on Lance internals.
        self.target_file_bytes
            .checked_mul(6)
            .and_then(|bytes| {
                record_bytes
                    .checked_mul(12)
                    .and_then(|rows| bytes.checked_add(rows))
            })
            .context("compaction workspace overflow")
    }
    pub(super) fn validate(&self, record_bytes: usize, budget: usize) -> Result<()> {
        ensure!(
            (64 << 10..=1 << 30).contains(&self.target_file_bytes)
                && (2..=64).contains(&self.min_fragments),
            "invalid small-file compaction bounds"
        );
        ensure!(
            self.workspace(record_bytes)? <= budget,
            "compaction workspace exceeds shared budget"
        );
        Ok(())
    }
    fn fragments(
        &self,
        dataset: &Dataset,
        indices: &[IndexMetadata],
        row_limit: Option<usize>,
    ) -> Option<Vec<Fragment>> {
        // A rewrite must have uniform coverage in every native index segment.
        let mut groups: BTreeMap<Vec<bool>, (Vec<Fragment>, usize, usize)> = BTreeMap::new();
        for fragment in dataset.manifest().fragments.iter() {
            // Managed ingest creates append-only, sized fragments. Unknown
            // sizes and deletion rewrites need separate planning policies.
            if fragment.deletion_file.is_some() || fragment.physical_rows.is_none_or(|n| n == 0) {
                continue;
            }
            let size = fragment.files.iter().try_fold(0usize, |n, file| {
                n.checked_add(usize::try_from(file.file_size_bytes.get()?.get()).ok()?)
            });
            let Some(size) = size.filter(|n| *n > 0 && *n < self.target_file_bytes / 2) else {
                continue;
            };
            let coverage = indices
                .iter()
                .map(|index| {
                    index
                        .fragment_bitmap
                        .as_ref()
                        .unwrap()
                        .contains(fragment.id as u32)
                })
                .collect();
            let (selected, bytes, rows) = groups.entry(coverage).or_default();
            let count = fragment.physical_rows.unwrap();
            if size > self.target_file_bytes * 2 - *bytes {
                continue;
            }
            if row_limit.is_some_and(|limit| count > limit.saturating_sub(*rows)) {
                continue;
            }
            selected.push(fragment.clone());
            *bytes += size;
            *rows = rows.checked_add(count)?;
            if selected.len() == 64 {
                return Some(std::mem::take(selected));
            }
        }
        groups
            .into_values()
            .map(|(fragments, _, _)| fragments)
            .find(|fragments| fragments.len() >= self.min_fragments)
    }
}

async fn supported_indices(dataset: &Dataset) -> Result<Arc<Vec<IndexMetadata>>> {
    let indices = dataset.load_indices().await?;
    ensure!(
        indices.is_empty() || dataset.manifest().uses_stable_row_ids(),
        "indexed compaction requires stable row IDs"
    );
    ensure!(
        indices.iter().all(|i| i.fragment_bitmap.is_some()),
        "compaction index coverage is unknown"
    );
    Ok(indices)
}

pub(crate) async fn read_compaction_base(publisher: &Publisher, version: u64) -> Result<Dataset> {
    if matches!(
        publisher.ingress.scope.contract.policy,
        WritePolicy::TurnAnnotations | WritePolicy::TurnIngestion
    ) {
        // Annotation bases contain the original turn schema plus narrow columns;
        // they do not have the immutable rollout ingress's exact Arrow schema.
        // This byte target belongs only to maintenance readers. The live read
        // view and the existing publication workspace reservation are unchanged.
        let mut read_params = lance::dataset::ReadParams::default();
        read_params
            .file_reader_options
            .get_or_insert_with(Default::default)
            .batch_size_bytes = Some(u64::from(publisher.ingress.scope.contract.max_record_bytes));
        Ok((*crate::annotations::checkpoint::load_with_read_params(
            &publisher.ingress,
            &publisher.storage,
            version,
            read_params,
        )
        .await?)
            .clone())
    } else {
        publisher.read_dataset(Some(version)).await
    }
}

pub(crate) async fn has_work(publisher: &Publisher, version: u64) -> Result<bool> {
    let Some(options) = &publisher.options.compaction else {
        return Ok(false);
    };
    let dataset = read_compaction_base(publisher, version).await?;
    let indices = supported_indices(&dataset).await?;
    Ok(options
        .fragments(
            &dataset,
            &indices,
            publisher
                .options
                .turn_id_index
                .as_ref()
                .map(|_| turn_id_index::MAX_BUILD_ROWS),
        )
        .is_some())
}

pub(crate) async fn has_work_on(dataset: &Dataset, options: &SmallFileCompaction) -> Result<bool> {
    let indices = supported_indices(dataset).await?;
    Ok(options.fragments(dataset, &indices, None).is_some())
}

pub(crate) async fn plan(publisher: &Publisher, view: &p::Table) -> Result<Option<p::Plan>> {
    let version = view.progress.minimum_read_version;
    if !has_work(publisher, version).await? {
        return Ok(None);
    }
    Ok(Some(p::Plan {
        transaction: Uuid::new_v4(),
        read_version: version,
        sources: BTreeMap::new(),
        next_group: publisher.maintenance_group()?,
    }))
}

pub(super) async fn prepare(
    publisher: &Publisher,
    pending: &p::PendingPlan,
) -> Result<p::Descriptors> {
    let options = publisher
        .options
        .compaction
        .as_ref()
        .context("pending compaction needs configured limits")?;
    let dataset = read_compaction_base(publisher, pending.plan.read_version).await?;
    let indices = supported_indices(&dataset).await?;
    let fragments = options
        .fragments(
            &dataset,
            &indices,
            publisher
                .options
                .turn_id_index
                .as_ref()
                .map(|_| turn_id_index::MAX_BUILD_ROWS),
        )
        .context("pending compaction has no eligible fragments under configured limits")?;
    let rows = fragments
        .iter()
        .try_fold(0usize, |n, f| n.checked_add(f.physical_rows?))
        .context("compaction row count overflow")?;
    let record_bytes = publisher.ingress.scope.contract.max_record_bytes as usize;
    // V1, blob materialization and overlays may collect rows before byte-based
    // output splitting. Keep the prior row ceiling for those paths and for
    // non-annotation readers, which do not configure the byte target above.
    let mut single_row = !matches!(
        publisher.ingress.scope.contract.policy,
        WritePolicy::TurnAnnotations | WritePolicy::TurnIngestion
    ) || dataset.schema().fields_pre_order().any(|f| f.is_blob())
        || fragments.iter().any(|f| !f.overlays.is_empty());
    for file in fragments.iter().flat_map(|f| f.referenced_lance_files()) {
        single_row |= file.file_version()? == ConcreteFileVersion::V1;
    }
    let task = CompactionTask {
        task: TaskData { fragments },
        read_version: pending.plan.read_version,
        options: CompactionOptions {
            target_rows_per_fragment: rows,
            max_bytes_per_file: Some(options.target_file_bytes),
            num_threads: Some(1),
            batch_size: single_row.then_some(1),
            io_buffer_size: Some(record_bytes as u64),
            compaction_mode: Some(CompactionMode::TryBinaryCopy),
            binary_copy_read_batch_bytes: Some(record_bytes),
            ..Default::default()
        },
    };
    let reservation = publisher
        .phases
        .measure(Phase::Admission, async {
            Ok::<_, anyhow::Error>(
                publisher
                    .budget
                    .reserve(options.workspace(record_bytes)?)
                    .await,
            )
        })
        .await?;
    let phases = publisher.phases.clone();
    let selection = crate::annotation_publication::Selection {
        table: publisher.authority.table,
        incarnation: publisher.ingress.scope.incarnation,
        pending: pending.clone(),
    };
    let storage = publisher.storage.clone();
    let maximum = publisher.ingress.client.max_descriptor_bytes();
    let annotations = matches!(
        publisher.ingress.scope.contract.policy,
        WritePolicy::TurnAnnotations | WritePolicy::TurnIngestion
    );
    // The detached task retains admission if the scheduling caller is dropped.
    // No prepared output is deleted on an error or an ambiguous later commit.
    tokio::spawn(async move {
        let _reservation = reservation;
        let mut result = phases
            .measure(Phase::BlobWrite, task.execute(&dataset))
            .await?;
        if !indices.is_empty() {
            // The catalog fixes the read and target versions and fences other
            // writers. Persist real IDs now so bitmap maintenance sees the same
            // IDs as the selected manifest, without a separate reservation commit.
            for (ordinal, fragment) in result.new_fragments.iter_mut().enumerate() {
                ensure!(
                    fragment.id == 0,
                    "native compaction assigned unexpected fragment IDs"
                );
                fragment.id = next_fragment_id(&dataset, ordinal)?;
            }
        }
        validate_result(&dataset, &result).await?;
        let bytes = serde_json::to_vec(&result)?;
        if annotations && bytes.len() as u64 > maximum {
            crate::annotation_publication::compaction::descriptors(
                &dataset, result, &selection, &storage, maximum,
            )
            .await
        } else {
            p::Descriptors::lance_compaction(Bytes::from(bytes))
        }
    })
    .await?
}

/// Keep the returned admission guard through native body selection and
/// descriptor persistence; returning the rewrite does not release its budget.
pub(crate) async fn prepare_body_dataset(
    dataset: Dataset,
    options: SmallFileCompaction,
    record_bytes: usize,
    budget: Arc<MergeMemoryBudget>,
    phases: Arc<PhaseCounters>,
) -> Result<(RewriteResult, MergeReservation)> {
    options.validate(record_bytes, budget.limit())?;
    let indices = supported_indices(&dataset).await?;
    let fragments = options
        .fragments(&dataset, &indices, None)
        .context("pending compaction has no eligible fragments under configured limits")?;
    let rows = fragments
        .iter()
        .try_fold(0usize, |n, f| n.checked_add(f.physical_rows?))
        .context("compaction row count overflow")?;
    let task = CompactionTask {
        task: TaskData { fragments },
        read_version: dataset.version_id(),
        options: CompactionOptions {
            target_rows_per_fragment: rows,
            max_bytes_per_file: Some(options.target_file_bytes),
            num_threads: Some(1),
            batch_size: Some(1),
            io_buffer_size: Some(record_bytes as u64),
            compaction_mode: Some(CompactionMode::TryBinaryCopy),
            binary_copy_read_batch_bytes: Some(record_bytes),
            ..Default::default()
        },
    };
    let reservation = phases
        .measure(Phase::Admission, async {
            Ok::<_, anyhow::Error>(budget.reserve(options.workspace(record_bytes)?).await)
        })
        .await?;
    // The detached task retains admission if the scheduling caller is dropped.
    // No prepared output is deleted on an error or an ambiguous later commit.
    tokio::spawn(async move {
        let mut result = phases
            .measure(Phase::BlobWrite, task.execute(&dataset))
            .await?;
        // The catalog fixes the read and target versions and fences other
        // writers. Persist real IDs now so bitmap maintenance sees the same
        // IDs as the selected manifest, without a separate reservation commit.
        for (ordinal, fragment) in result.new_fragments.iter_mut().enumerate() {
            ensure!(
                fragment.id == 0,
                "native compaction assigned unexpected fragment IDs"
            );
            fragment.id = next_fragment_id(&dataset, ordinal)?;
        }
        validate_result(&dataset, &result).await?;
        Ok((result, reservation))
    })
    .await?
}

pub(crate) async fn validate_result(dataset: &Dataset, result: &RewriteResult) -> Result<()> {
    ensure!(
        result.read_version == dataset.version().version
            && result.original_fragments.len() >= 2
            && !result.new_fragments.is_empty()
            && result.new_fragments.len() < result.original_fragments.len()
            && result.row_addrs.is_none(),
        "invalid or ineffective compaction result"
    );
    let indices = supported_indices(dataset).await?;
    if !indices.is_empty() {
        for (ordinal, fragment) in result.new_fragments.iter().enumerate() {
            ensure!(
                fragment.id == next_fragment_id(dataset, ordinal)?,
                "compaction fragment IDs differ from fixed snapshot"
            );
        }
        for index in indices.iter() {
            let bitmap = index.fragment_bitmap.as_ref().unwrap();
            let covered = bitmap.contains(result.original_fragments[0].id as u32);
            ensure!(
                result
                    .original_fragments
                    .iter()
                    .all(|f| bitmap.contains(f.id as u32) == covered),
                "compaction mixes index coverage"
            );
        }
    }
    let mut ids = BTreeSet::new();
    for fragment in &result.original_fragments {
        ensure!(
            ids.insert(fragment.id)
                && fragment.deletion_file.is_none()
                && dataset.manifest().fragments.iter().any(|f| f == fragment),
            "compaction source differs from fixed snapshot"
        );
    }
    let rows = |fragments: &[Fragment]| {
        fragments
            .iter()
            .try_fold(0usize, |n, f| n.checked_add(f.physical_rows?))
    };
    ensure!(
        rows(&result.original_fragments).is_some()
            && rows(&result.original_fragments) == rows(&result.new_fragments)
            && result
                .new_fragments
                .iter()
                .all(|f| f.deletion_file.is_none() && !f.files.is_empty()),
        "compaction changed row count"
    );
    Ok(())
}

fn next_fragment_id(dataset: &Dataset, ordinal: usize) -> Result<u64> {
    let base = dataset
        .manifest()
        .max_fragment_id
        .context("missing fragment high-water mark")?;
    let id = base
        .checked_add(1)
        .and_then(|id| id.checked_add(u32::try_from(ordinal).ok()?))
        .context("compaction fragment ID space exhausted")?;
    Ok(u64::from(id))
}

pub(super) async fn operation(
    dataset: &Dataset,
    descriptors: &p::Descriptors,
) -> Result<Operation> {
    ensure!(
        descriptors.format == 2,
        "unexpected compaction descriptor format"
    );
    let result: RewriteResult = serde_json::from_slice(&descriptors.bytes)?;
    validate_result(dataset, &result).await?;
    Ok(Operation::Rewrite {
        groups: vec![RewriteGroup {
            old_fragments: result.original_fragments,
            new_fragments: result.new_fragments,
        }],
        rewritten_indices: Vec::new(),
        frag_reuse_index: None,
    })
}
