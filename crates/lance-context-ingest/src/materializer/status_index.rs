//! Incremental status indexing of the selected narrow annotation snapshot.
use super::*;
use lance::index::DatasetIndexExt;
use lance_table::format::IndexMetadata;
use session_index::{Action, Key};
use std::collections::BTreeSet;

const FAN_IN: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StatusIndex {
    /// Physical input bytes per native merge, including masked old values.
    /// This bounds admitted I/O, not native allocator use or total index size.
    pub max_merge_bytes: u64,
}

impl Default for StatusIndex {
    fn default() -> Self {
        Self {
            max_merge_bytes: 64 << 20,
        }
    }
}

impl StatusIndex {
    pub(super) fn validate(&self, budget: usize) -> Result<()> {
        ensure!(
            (1 << 20..=1 << 30).contains(&self.max_merge_bytes)
                && budget >= session_index::WORKSPACE,
            "invalid status index merge budget or insufficient workspace"
        );
        Ok(())
    }

    fn physical_bytes(index: &IndexMetadata) -> Result<u64> {
        let files = index
            .files
            .as_ref()
            .context("status index has no file costs")?;
        let bytes = files.iter().try_fold(0u64, |sum, file| {
            sum.checked_add(file.size_bytes)
                .context("status index file cost overflow")
        })?;
        ensure!(bytes > 0, "status index has empty file costs");
        Ok(bytes)
    }

    fn merge(&self, segments: Vec<IndexMetadata>) -> Result<Option<Action>> {
        let mut tiers: BTreeMap<u32, Vec<IndexMetadata>> = BTreeMap::new();
        for index in segments {
            let bytes = Self::physical_bytes(&index)?;
            if bytes > self.max_merge_bytes / FAN_IN as u64 {
                continue;
            }
            // Merge similarly sized physical files. Live coverage may have
            // shrunk to one fragment while the old file remains very large.
            let tier = tiers.entry(bytes.ilog2()).or_default();
            tier.push(index);
            if tier.len() == FAN_IN {
                return Ok(Some(Action::Merge(std::mem::take(tier))));
            }
        }
        Ok(None)
    }

    async fn action(&self, dataset: &Dataset) -> Result<Option<Action>> {
        TurnIdIndex::validate_base(dataset)?;
        let fragments: BTreeMap<_, _> = dataset
            .fragments()
            .iter()
            .map(|f| Ok((u32::try_from(f.id)?, f.physical_rows.unwrap())))
            .collect::<Result<_>>()?;
        let mut covered = BTreeSet::new();
        let mut segments = Vec::new();
        let mut empty_sources = Vec::new();
        for index in dataset
            .load_indices()
            .await?
            .iter()
            .filter(|i| i.name == Key::Status.name())
        {
            session_index::validate_metadata(dataset, index, Key::Status)?;
            let coverage = index.fragment_bitmap.as_ref().unwrap();
            for id in coverage {
                ensure!(
                    fragments.contains_key(&id),
                    "status index covers a foreign fragment"
                );
                ensure!(covered.insert(id), "status index segments overlap");
            }
            // CreateIndex requires explicit removal of empty definitions;
            // do not read or merge their obsolete physical pages.
            if coverage.is_empty() {
                if empty_sources.len() < 64 {
                    empty_sources.push(index.uuid);
                }
            } else {
                segments.push(index.clone());
            }
        }
        if let Some(action) = self.merge(segments)? {
            return Ok(Some(action));
        }
        let mut selected = Vec::new();
        let mut rows = 0;
        for (id, count) in fragments {
            if covered.contains(&id) || count == 0 || count > turn_id_index::MAX_BUILD_ROWS - rows {
                continue;
            }
            selected.push(id);
            rows += count;
            if selected.len() == 128 {
                break;
            }
        }
        // Even a single idle update must catch up without another write to
        // invalidate the owner's cached no-work decision for this version.
        Ok(
            (!selected.is_empty()).then_some(if empty_sources.is_empty() {
                Action::Build(selected)
            } else {
                Action::BuildReplacingEmpty {
                    fragments: selected,
                    empty_sources,
                }
            }),
        )
    }
}

pub(crate) async fn has_work(publisher: &Publisher, version: u64) -> Result<bool> {
    let Some(options) = &publisher.options.status_index else {
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
    let Some(options) = &publisher.options.status_index else {
        return Ok(None);
    };
    let dataset = compaction::read_compaction_base(publisher, pending.plan.read_version).await?;
    let Some(action) = options.action(&dataset).await? else {
        return Ok(None);
    };
    Ok(Some(
        session_index::prepare_action(publisher, dataset, action, Key::Status).await?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_table::format::IndexFile;

    fn segment(id: u128, bytes: u64) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::from_u128(id),
            fields: vec![0],
            covering_fields: vec![],
            name: Key::Status.name().into(),
            dataset_version: 1,
            // All sources have one live fragment even though their physical
            // files may retain very different amounts of withdrawn data.
            fragment_bitmap: Some([id as u32].into_iter().collect()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: Some(vec![IndexFile {
                path: "page_data.lance".into(),
                size_bytes: bytes,
            }]),
        }
    }

    #[test]
    fn status_merge_charges_masked_physical_files_and_rejects_unknown_costs() -> Result<()> {
        let options = StatusIndex {
            max_merge_bytes: 1 << 20,
        };
        let large = segment(1, 8 << 20);
        let small = (2..=5).map(|id| segment(id, 64 << 10)).collect::<Vec<_>>();
        let Some(Action::Merge(selected)) = options.merge(
            std::iter::once(large.clone())
                .chain(small.clone())
                .collect(),
        )?
        else {
            panic!("small physical sources must consolidate");
        };
        assert_eq!(selected, small);
        assert!(
            selected
                .iter()
                .map(|i| StatusIndex::physical_bytes(i).unwrap())
                .sum::<u64>()
                <= options.max_merge_bytes
        );
        assert!(options
            .merge(vec![
                large,
                segment(2, 64 << 10),
                segment(3, 64 << 10),
                segment(4, 64 << 10)
            ])?
            .is_none());
        let mut unknown = segment(1, 1);
        unknown.files = None;
        assert!(options.merge(vec![unknown]).is_err());
        let mut overflow = segment(1, u64::MAX);
        overflow.files.as_mut().unwrap().push(IndexFile {
            path: "page_lookup.lance".into(),
            size_bytes: 1,
        });
        assert!(options.merge(vec![overflow]).is_err());
        Ok(())
    }
}
