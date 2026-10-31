//! Bounded native BTree segments, published by the existing fenced transaction.
use super::*;
use lance::{dataset::transaction::Operation, index::DatasetIndexExt};
use lance_index::{scalar::ScalarIndexParams, IndexType};
use lance_table::format::{pb, Fragment, IndexMetadata};
use prost::Message;
use std::collections::BTreeSet;

const NAME: &str = "lance_context_rollout_id";

#[derive(Clone, Copy)]
pub(super) enum Key {
    Session,
    Turn,
    Status,
}
impl Key {
    pub(super) fn column(self) -> &'static str {
        match self {
            Self::Session => "rollout_id",
            Self::Turn => "id",
            Self::Status => "status",
        }
    }
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Session => NAME,
            Self::Turn => "lance_context_turn_id",
            Self::Status => "lance_context_annotation_status",
        }
    }
}
const MAX_ROWS: usize = 1 << 20;
pub(super) const WORKSPACE: usize = 512 << 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionIndex {
    pub min_fragments: usize,
    pub max_fragments: usize,
    /// Consolidate native index files before adding another segment at this bound.
    pub max_segments: usize,
}
impl Default for SessionIndex {
    fn default() -> Self {
        Self {
            min_fragments: 16,
            max_fragments: 128,
            max_segments: 16,
        }
    }
}
impl SessionIndex {
    pub(super) fn validate(&self, budget: usize) -> Result<()> {
        ensure!(
            self.min_fragments > 0
                && self.min_fragments <= self.max_fragments
                && self.max_fragments <= 512
                && (2..=64).contains(&self.max_segments),
            "invalid session index fragment bounds"
        );
        // Admission is a workspace estimate, not a cap on native allocations.
        ensure!(
            budget >= WORKSPACE,
            "session index workspace exceeds shared budget"
        );
        Ok(())
    }

    async fn action(&self, dataset: &Dataset) -> Result<Option<Action>> {
        ensure!(
            dataset.manifest().uses_stable_row_ids(),
            "session index maintenance requires stable row IDs"
        );
        let indices = dataset.load_indices().await?;
        let mut covered = BTreeSet::new();
        for index in indices.iter().filter(|i| i.name == NAME) {
            validate_metadata(dataset, index, Key::Session)?;
            covered.extend(index.fragment_bitmap.as_ref().unwrap().iter());
        }
        let segments: Vec<_> = indices.iter().filter(|i| i.name == NAME).cloned().collect();
        if segments.len() >= self.max_segments {
            return Ok(Some(Action::Merge(segments)));
        }
        let mut selected = Vec::new();
        let mut rows = 0usize;
        for fragment in dataset.manifest().fragments.iter() {
            let id = u32::try_from(fragment.id)?;
            if covered.contains(&id) || fragment.deletion_file.is_some() {
                continue;
            }
            let Some(count) = fragment.physical_rows.filter(|n| *n > 0) else {
                continue;
            };
            if count > MAX_ROWS - rows {
                continue;
            }
            selected.push(id);
            rows += count;
            if selected.len() == self.max_fragments {
                break;
            }
        }
        Ok((selected.len() >= self.min_fragments).then_some(Action::Build(selected)))
    }
}

pub(super) enum Action {
    Build(Vec<u32>),
    BuildReplacingEmpty {
        fragments: Vec<u32>,
        empty_sources: Vec<Uuid>,
    },
    Merge(Vec<IndexMetadata>),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Change {
    read_version: u64,
    index: Vec<u8>,
    removed: Vec<Uuid>,
}

pub(super) fn validate_metadata(dataset: &Dataset, index: &IndexMetadata, key: Key) -> Result<()> {
    let field = dataset
        .schema()
        .field(key.column())
        .context("missing session column")?;
    ensure!(
        index.name == key.name()
            && index.fields == vec![field.id]
            && index.covering_fields.is_empty()
            && !index.uuid.is_nil()
            && index.base_id.is_none()
            && index.dataset_version <= dataset.version().version
            && index
                .index_details
                .as_ref()
                .is_some_and(|d| d.type_url.ends_with("BTreeIndexDetails"))
            && index
                .fragment_bitmap
                .as_ref()
                .is_some_and(|b| !b.is_empty() || matches!(key, Key::Status)),
        "incompatible managed session index metadata"
    );
    Ok(())
}

/// Split one snapshot only when a known session index has an uncovered tail.
/// Lance's exact-index plan otherwise reads the tail's full payload projection
/// before filtering it. Keep native planning for absent or unfamiliar indexes.
pub(crate) async fn scan_fragments(dataset: &Dataset) -> Result<Option<[Vec<Fragment>; 2]>> {
    let indices = dataset.load_indices().await?;
    let mut coverage = BTreeSet::new();
    for index in indices.iter().filter(|index| index.name == NAME) {
        if validate_metadata(dataset, index, Key::Session).is_err() {
            return Ok(None);
        }
        coverage.extend(
            index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .map(u64::from),
        );
    }
    if coverage.is_empty() {
        return Ok(None);
    }
    let (covered, tail): (Vec<_>, Vec<_>) = dataset
        .fragments()
        .iter()
        .cloned()
        .partition(|fragment| coverage.contains(&fragment.id));
    Ok((!covered.is_empty() && !tail.is_empty()).then_some([covered, tail]))
}

pub(super) async fn plan(publisher: &Publisher, view: &p::Table) -> Result<Option<p::Plan>> {
    let Some(options) = &publisher.options.session_index else {
        return Ok(None);
    };
    let version = view.progress.minimum_read_version;
    let next_group = publisher.maintenance_group()?;
    let dataset = publisher.read_dataset(Some(version)).await?;
    if options.action(&dataset).await?.is_none() {
        return Ok(None);
    }
    Ok(Some(p::Plan {
        transaction: Uuid::new_v4(),
        read_version: version,
        sources: BTreeMap::new(),
        next_group,
    }))
}

pub(super) async fn prepare(
    publisher: &Publisher,
    pending: &p::PendingPlan,
) -> Result<Option<p::Descriptors>> {
    let Some(options) = &publisher.options.session_index else {
        return Ok(None);
    };
    let dataset = publisher
        .read_dataset(Some(pending.plan.read_version))
        .await?;
    let Some(action) = options.action(&dataset).await? else {
        return Ok(None);
    };
    Ok(Some(
        prepare_action(publisher, dataset, action, Key::Session).await?,
    ))
}

pub(super) async fn prepare_action(
    publisher: &Publisher,
    mut dataset: Dataset,
    action: Action,
    key: Key,
) -> Result<p::Descriptors> {
    let reservation = publisher.budget.reserve(WORKSPACE).await;
    let phases = publisher.phases.clone();
    // Cancellation cannot release admission while a native index build still runs.
    let descriptors = tokio::spawn(async move {
        let _reservation = reservation;
        let is_merge = matches!(&action, Action::Merge(_));
        let (coverage, removed): (BTreeSet<_>, Vec<_>) = match &action {
            Action::Build(ids) => (ids.iter().copied().collect(), Vec::new()),
            Action::BuildReplacingEmpty {
                fragments,
                empty_sources,
            } => (fragments.iter().copied().collect(), empty_sources.clone()),
            Action::Merge(segments) => (
                segments
                    .iter()
                    .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                    .collect(),
                segments.iter().map(|i| i.uuid).collect(),
            ),
        };
        let index = phases
            .measure(Phase::BlobWrite, async {
                match action {
                    Action::Build(ids) | Action::BuildReplacingEmpty { fragments: ids, .. } => {
                        dataset
                            .create_index_builder(
                                &[key.column()],
                                IndexType::BTree,
                                &ScalarIndexParams::default(),
                            )
                            .name(key.name().to_owned())
                            .replace(true)
                            .fragments(ids)
                            .execute_uncommitted()
                            .await
                    }
                    Action::Merge(segments) => {
                        dataset.merge_existing_index_segments(segments).await
                    }
                }
            })
            .await?;
        validate_metadata(&dataset, &index, key)?;
        ensure!(
            is_merge || index.dataset_version == dataset.version().version,
            "index snapshot changed"
        );
        ensure!(
            index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<BTreeSet<_>>()
                == coverage,
            "index coverage differs from selected fragments"
        );
        let change = Change {
            read_version: dataset.version().version,
            index: pb::IndexMetadata::from(&index).encode_to_vec(),
            removed,
        };
        p::Descriptors::lance_index(Bytes::from(serde_json::to_vec(&change)?))
    })
    .await??;
    Ok(descriptors)
}

pub(super) async fn operation(
    dataset: &Dataset,
    descriptors: &p::Descriptors,
) -> Result<Operation> {
    operation_for(dataset, descriptors, Key::Session).await
}

pub(super) async fn operation_for(
    dataset: &Dataset,
    descriptors: &p::Descriptors,
    key: Key,
) -> Result<Operation> {
    decode_operation(dataset, descriptors, Some(key)).await
}

/// Recover a selected annotation index operation from its native metadata,
/// including when the current owner's maintenance configuration has changed.
pub(super) async fn annotation_operation(
    dataset: &Dataset,
    descriptors: &p::Descriptors,
) -> Result<Operation> {
    decode_operation(dataset, descriptors, None).await
}

async fn decode_operation(
    dataset: &Dataset,
    descriptors: &p::Descriptors,
    expected: Option<Key>,
) -> Result<Operation> {
    ensure!(
        descriptors.format == 3 && dataset.manifest().uses_stable_row_ids(),
        "unsupported index publication"
    );
    let change: Change = serde_json::from_slice(&descriptors.bytes)?;
    ensure!(
        change.read_version == dataset.version().version,
        "index snapshot changed"
    );
    let index = IndexMetadata::try_from(pb::IndexMetadata::decode(change.index.as_slice())?)?;
    let key = match expected {
        Some(key) => key,
        None if index.name == Key::Turn.name() => Key::Turn,
        None if index.name == Key::Status.name() => Key::Status,
        None => anyhow::bail!("unsupported annotation index"),
    };
    validate_metadata(dataset, &index, key)?;
    ensure!(
        !change.removed.is_empty() || index.dataset_version == dataset.version().version,
        "index snapshot changed"
    );
    let covered = index.fragment_bitmap.as_ref().unwrap();
    ensure!(!covered.is_empty(), "index result has no coverage");
    let live: BTreeSet<_> = dataset.manifest().fragments.iter().map(|f| f.id).collect();
    ensure!(
        covered.iter().all(|id| live.contains(&u64::from(id))),
        "index covers foreign fragments"
    );
    let remove: BTreeSet<_> = change.removed.iter().copied().collect();
    ensure!(
        remove.len() == change.removed.len() && remove.len() <= 64,
        "invalid index consolidation sources"
    );
    let mut removed_indices = Vec::new();
    let mut removed_coverage = BTreeSet::new();
    for old in dataset
        .load_indices()
        .await?
        .iter()
        .filter(|i| i.name == key.name())
    {
        validate_metadata(dataset, old, key)?;
        ensure!(
            old.uuid != index.uuid,
            "index reused an existing segment identity"
        );
        if remove.contains(&old.uuid) {
            removed_coverage.extend(old.fragment_bitmap.as_ref().unwrap().iter());
            removed_indices.push(old.clone());
        } else {
            ensure!(
                old.fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .iter()
                    .all(|id| !covered.contains(id)),
                "index segment overlaps existing coverage"
            );
        }
    }
    ensure!(
        removed_indices.len() == remove.len(),
        "index consolidation source changed"
    );
    if !remove.is_empty() && removed_coverage.is_empty() {
        // Updating the last indexed fragment retains an empty definition.
        // CreateIndex does not retire it implicitly: a fresh catch-up build
        // may remove only proven-empty status sources at this exact snapshot.
        ensure!(
            matches!(key, Key::Status) && index.dataset_version == dataset.version().version,
            "invalid empty status index replacement"
        );
    } else if !remove.is_empty() {
        ensure!(
            remove.len() >= 2 && covered.iter().collect::<BTreeSet<_>>() == removed_coverage,
            "index consolidation changed coverage"
        );
    }
    Ok(Operation::CreateIndex {
        new_indices: vec![index],
        removed_indices,
    })
}
