//! Bounded ID indexing for the current annotation table, using native BTree files.
use super::*;
use lance::index::DatasetIndexExt;
use lance_table::format::IndexMetadata;
use session_index::{Action, Key};
use std::collections::BTreeSet;

pub(crate) const MAX_BUILD_ROWS: usize = 1 << 20;
const MAX_MERGE_ROWS: usize = 16 * MAX_BUILD_ROWS;
const MERGE_FAN_IN: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TurnIdIndex {
    /// Leave a smaller cold tail to native filtered scans until it reaches this
    /// many rows. This bounds when to start a build, not native allocator usage.
    pub min_rows: usize,
}
impl Default for TurnIdIndex {
    fn default() -> Self {
        Self { min_rows: 64 << 10 }
    }
}

impl TurnIdIndex {
    pub(super) fn validate(&self, budget: usize) -> Result<()> {
        ensure!(
            (1..=MAX_BUILD_ROWS).contains(&self.min_rows) && budget >= session_index::WORKSPACE,
            "invalid turn ID index threshold or insufficient workspace"
        );
        Ok(())
    }

    pub(crate) fn validate_base(dataset: &Dataset) -> Result<()> {
        ensure!(
            dataset.manifest().uses_stable_row_ids()
                && dataset.fragments().iter().all(|f| f.deletion_file.is_none()
                    && f.physical_rows.is_some_and(|n| n <= MAX_BUILD_ROWS)),
            "turn ID indexing requires stable IDs and fragments of at most 1048576 physical rows without deletions"
        );
        Ok(())
    }

    // Merge four similarly sized segments, never a large historical segment
    // with every new tail. A merge covers at most sixteen million rows. Large
    // shards can still accumulate capped segments; this is not a scale-out policy.
    fn merge(segments: Vec<(IndexMetadata, usize)>) -> Option<Action> {
        let mut tiers: BTreeMap<usize, Vec<IndexMetadata>> = BTreeMap::new();
        for (index, rows) in segments {
            let mut ceiling = 1;
            while ceiling < rows && ceiling <= MAX_MERGE_ROWS / MERGE_FAN_IN {
                ceiling *= MERGE_FAN_IN;
            }
            if rows > ceiling || ceiling > MAX_MERGE_ROWS / MERGE_FAN_IN {
                continue;
            }
            let tier = tiers.entry(ceiling).or_default();
            tier.push(index);
            if tier.len() == MERGE_FAN_IN {
                return Some(Action::Merge(std::mem::take(tier)));
            }
        }
        None
    }

    async fn action(&self, dataset: &Dataset) -> Result<Option<Action>> {
        Self::validate_base(dataset)?;
        let fragments: BTreeMap<_, _> = dataset
            .fragments()
            .iter()
            .map(|f| Ok((u32::try_from(f.id)?, f.physical_rows.unwrap())))
            .collect::<Result<_>>()?;
        let mut covered = BTreeSet::new();
        let mut segments = Vec::new();
        for index in dataset
            .load_indices()
            .await?
            .iter()
            .filter(|i| i.name == Key::Turn.name())
        {
            session_index::validate_metadata(dataset, index, Key::Turn)?;
            let mut rows = 0usize;
            for id in index.fragment_bitmap.as_ref().unwrap().iter() {
                ensure!(
                    covered.insert(id),
                    "turn ID segments have overlapping coverage"
                );
                rows = rows
                    .checked_add(
                        *fragments
                            .get(&id)
                            .context("turn ID index covers a foreign fragment")?,
                    )
                    .context("turn ID segment row count overflow")?;
            }
            segments.push((index.clone(), rows));
        }
        if let Some(action) = Self::merge(segments) {
            return Ok(Some(action));
        }
        let mut selected = Vec::new();
        let mut rows = 0;
        for (id, count) in fragments {
            if covered.contains(&id) || count == 0 || count > MAX_BUILD_ROWS - rows {
                continue;
            }
            selected.push(id);
            rows += count;
            if selected.len() == 128 {
                break;
            }
        }
        Ok((rows >= self.min_rows || selected.len() == 128).then_some(Action::Build(selected)))
    }
}

pub(crate) async fn has_work(publisher: &Publisher, version: u64) -> Result<bool> {
    let Some(options) = &publisher.options.turn_id_index else {
        return Ok(false);
    };
    let dataset = compaction::read_compaction_base(publisher, version).await?;
    Ok(options.action(&dataset).await?.is_some())
}

pub(super) async fn plan(publisher: &Publisher, view: &p::Table) -> Result<Option<p::Plan>> {
    let version = view.progress.minimum_read_version;
    let next_group = publisher.maintenance_group()?;
    Ok(has_work(publisher, version).await?.then(|| p::Plan {
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
    let Some(options) = &publisher.options.turn_id_index else {
        return Ok(None);
    };
    let dataset = compaction::read_compaction_base(publisher, pending.plan.read_version).await?;
    let Some(action) = options.action(&dataset).await? else {
        return Ok(None);
    };
    Ok(Some(
        session_index::prepare_action(publisher, dataset, action, Key::Turn).await?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The policy consumes native segment metadata and known physical row
    // counts. No index file contents are consulted while choosing merge work.
    fn segment(id: u128, rows: usize) -> (IndexMetadata, usize) {
        (
            IndexMetadata {
                uuid: Uuid::from_u128(id),
                fields: vec![0],
                covering_fields: vec![],
                name: Key::Turn.name().into(),
                dataset_version: 1,
                fragment_bitmap: None,
                index_details: None,
                index_version: 0,
                created_at: None,
                base_id: None,
                files: None,
            },
            rows,
        )
    }

    #[test]
    fn turn_id_merge_excludes_large_history_and_caps_each_operation() {
        let large = segment(1, MAX_MERGE_ROWS);
        let small = (2..=5)
            .map(|id| segment(id, MAX_BUILD_ROWS))
            .collect::<Vec<_>>();
        let Some(Action::Merge(selected)) = TurnIdIndex::merge(
            std::iter::once(large.clone())
                .chain(small.clone())
                .collect(),
        ) else {
            panic!("four comparable tail segments must merge");
        };
        assert_eq!(
            selected.iter().map(|i| i.uuid).collect::<Vec<_>>(),
            small.iter().map(|(i, _)| i.uuid).collect::<Vec<_>>()
        );
        assert!(TurnIdIndex::merge(vec![
            large,
            segment(2, 4 * MAX_BUILD_ROWS),
            segment(3, MAX_BUILD_ROWS),
            segment(4, MAX_BUILD_ROWS),
            segment(5, MAX_BUILD_ROWS)
        ])
        .is_none());
        assert!(matches!(
            TurnIdIndex::merge((1..=4).map(|id| segment(id, MAX_MERGE_ROWS / 4)).collect()),
            Some(Action::Merge(_))
        ));
        assert!(TurnIdIndex::merge(
            (1..=4)
                .map(|id| segment(id, MAX_MERGE_ROWS / 4 + 1))
                .collect()
        )
        .is_none());
    }
}
