//! Bounded application results bound to the same native transaction as values.
use super::*;
use crate::{annotation_publication, materializer};
use lance::Dataset;
use lance_context_core::annotation_staging::{
    FrozenAnnotations, PreparedAnnotations, RejectionReason, MAX_RESULTS_BYTES, RESULTS_PROPERTY,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Encoded {
    version: u8,
    slots: u32,
    // Sorted slot, accepted index, rejected row, rejection code. At most 37
    // JSON bytes per entry even at the largest legal field values: all 16,384
    // producers fit below 640 KiB, leaving room in the 1 MiB native descriptor cap.
    rejected: Vec<(u32, u64, u32, u8)>,
}

pub(crate) struct Snapshot {
    pub through: u64,
    encoded: Encoded,
}

impl Snapshot {
    fn empty(slots: u32) -> Self {
        Self {
            through: 0,
            encoded: Encoded {
                version: 1,
                slots,
                rejected: Vec::new(),
            },
        }
    }

    pub(crate) fn decode(value: &str, through: u64) -> Result<Self> {
        ensure!(
            value.len() <= MAX_RESULTS_BYTES && through > 0 && through < u64::MAX,
            "invalid annotation result snapshot size/cut"
        );
        let encoded: Encoded = serde_json::from_str(value)?;
        ensure!(
            encoded.version == 1
                && (1..=producer::MAX_SLOTS).contains(&encoded.slots)
                && encoded.rejected.len() <= encoded.slots as usize
                && encoded.rejected.windows(2).all(|w| w[0].0 < w[1].0)
                && encoded.rejected.iter().all(|&(slot, index, row, reason)| {
                    slot < encoded.slots
                        && index > 0
                        && index <= through
                        && row < 65_536
                        && reason < 5
                }),
            "invalid annotation result snapshot"
        );
        Ok(Self { through, encoded })
    }

    fn rejected(&self, slot: usize) -> Option<(u64, Rejection)> {
        let row = self
            .encoded
            .rejected
            .binary_search_by_key(&(slot as u32), |e| e.0)
            .ok()?;
        let (_, index, row, reason) = self.encoded.rejected[row];
        Some((
            index,
            Rejection {
                row: row as usize,
                reason: match reason {
                    0 => RejectionReason::InvalidPatch,
                    1 => RejectionReason::InvalidTurn,
                    2 => RejectionReason::MissingTurn,
                    3 => RejectionReason::ValueTooLarge,
                    4 => RejectionReason::BatchTooLarge,
                    _ => unreachable!("validated result snapshot"),
                },
            },
        ))
    }
}

pub(super) struct Results {
    pub base: Arc<Snapshot>,
    pub latest: Vec<Option<Completed>>,
}
impl Results {
    pub fn resolve(&self, accepted: &AcceptedAppend, slot: usize) -> Result<Completed> {
        if let Some(result) = &self.latest[slot] {
            if result.index == accepted.position.index && result.meta == accepted.meta {
                return Ok(result.clone());
            }
        }
        // accepted came from a successful exact WAL append/retry, never a bare
        // caller position. The proven cut covers every application's outcome.
        ensure!(
            accepted.position.index <= self.base.through,
            "annotation request outcome expired; request must not be re-executed"
        );
        let result = match self.base.rejected(slot) {
            Some((index, rejected)) if index == accepted.position.index => {
                ensure!(
                    rejected.row < accepted.rows as usize,
                    "rejection row is outside the original request"
                );
                MutationResult::Rejected(rejected)
            }
            _ => MutationResult::Applied,
        };
        Ok(Completed {
            index: accepted.position.index,
            meta: accepted.meta.clone(),
            result,
        })
    }

    fn snapshot(&self, through: u64) -> Result<Snapshot> {
        ensure!(through >= self.base.through, "result cut regressed");
        let mut rejected = Vec::new();
        for (slot, latest) in self.latest.iter().enumerate() {
            let rejection = match latest.as_ref().filter(|r| r.index > self.base.through) {
                Some(result) => {
                    ensure!(result.index <= through, "result cut precedes application");
                    match &result.result {
                        MutationResult::Applied => None,
                        MutationResult::Rejected(rejected) => {
                            Some((result.index, rejected.clone()))
                        }
                    }
                }
                None => self.base.rejected(slot),
            };
            if let Some((index, rejection)) = rejection {
                let code = match rejection.reason {
                    RejectionReason::InvalidPatch => 0,
                    RejectionReason::InvalidTurn => 1,
                    RejectionReason::MissingTurn => 2,
                    RejectionReason::ValueTooLarge => 3,
                    RejectionReason::BatchTooLarge => 4,
                };
                rejected.push((slot as u32, index, u32::try_from(rejection.row)?, code));
            }
        }
        Ok(Snapshot {
            through,
            encoded: Encoded {
                version: 1,
                slots: self.latest.len() as u32,
                rejected,
            },
        })
    }
}

pub struct FrozenCheckpoint {
    origin: Arc<Shared>,
    annotations: FrozenAnnotations,
    results: Arc<Snapshot>,
}
pub struct PreparedCheckpoint {
    origin: Arc<Shared>,
    annotations: PreparedAnnotations,
    results: Arc<Snapshot>,
}
impl FrozenCheckpoint {
    pub fn through(&self) -> u64 {
        self.annotations.through()
    }
    pub fn identity(&self) -> &str {
        self.annotations.identity()
    }
    pub async fn prepare(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        batch_rows: usize,
    ) -> Result<PreparedCheckpoint> {
        Ok(PreparedCheckpoint {
            origin: self.origin.clone(),
            annotations: self.annotations.prepare(workspace, batch_rows).await?,
            results: self.results.clone(),
        })
    }
}
impl PreparedCheckpoint {
    pub fn annotations(&self) -> &PreparedAnnotations {
        &self.annotations
    }
}

impl Service {
    /// Capture results and values before another committed batch can apply.
    /// The caller must select/pin exactly this complete source prefix. This
    /// method does not authorize publication, confirmation or source closure.
    pub async fn freeze(&self, transaction: Uuid) -> Result<Option<FrozenCheckpoint>> {
        let _ordered = self.shared.ordered.lock().await;
        let through = self.shared.progress.borrow().through;
        let snapshot = {
            let results = self.shared.results.lock().unwrap();
            if !results
                .latest
                .iter()
                .flatten()
                .any(|r| r.index > results.base.through)
            {
                return Ok(None);
            }
            results.snapshot(through)?
        };
        let encoded = serde_json::to_string(&snapshot.encoded)?;
        Snapshot::decode(&encoded, through)?;
        let annotations = self
            .shared
            .staging
            .freeze_with_results(transaction, Some(encoded))
            .await?
            .context("application results have no frozen generation")?;
        ensure!(
            annotations.through() == through,
            "value/result freeze cut differs"
        );
        Ok(Some(FrozenCheckpoint {
            origin: self.shared.clone(),
            annotations,
            results: Arc::new(snapshot),
        }))
    }

    /// Install only after the caller resolves the original managed publication.
    /// Preserve later outcomes and values; retire only this frozen generation.
    pub async fn install_published(
        &self,
        prepared: &PreparedCheckpoint,
        selected: Arc<Dataset>,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.shared, &prepared.origin),
            "foreign application checkpoint"
        );
        let _ordered = self.shared.ordered.lock().await;
        self.shared
            .staging
            .install_published(&prepared.annotations, selected)
            .await?;
        self.shared.results.lock().unwrap().base = prepared.results.clone();
        Ok(())
    }
}

pub(super) async fn open(
    ingress: &Ingress,
    storage: &Storage,
) -> Result<(Arc<Dataset>, Arc<Snapshot>)> {
    let query::Value::Publisher(Some(view)) = materializer::query(
        ingress,
        ingress.client.catalog_group(),
        query::Query::Publisher {
            table: ingress.scope.table,
        },
    )
    .await?
    else {
        bail!("annotation catalog unavailable");
    };
    let manifests = view
        .manifests
        .as_ref()
        .context("annotation manifests unavailable")?;
    ensure!(
        view.destination == ingress.scope.destination
            && view.ingest == ingress.scope.contract
            && manifests.incarnation == ingress.scope.incarnation,
        "foreign annotation catalog binding"
    );
    let record = &manifests.versions[&manifests.head];
    let base = load(ingress, storage, record.version).await?;
    let native = base
        .read_transaction()
        .await?
        .context("annotation base has no native transaction")?;
    ensure!(
        native.uuid == record.transaction.to_string(),
        "native annotation transaction differs from catalog"
    );
    let properties = native.transaction_properties.as_ref();
    let cut = properties.and_then(|p| p.get("lance-context.annotation-cut"));
    let covered = view
        .progress
        .checkpoint
        .get(&ingress.group(0)?)
        .copied()
        .unwrap_or(0);
    let Some(cut) = cut else {
        ensure!(
            covered == 0 && record.plan_index == 0,
            "materialized annotations have no application cut"
        );
        return Ok((
            base,
            Arc::new(Snapshot::empty(ingress.client.producer_slots())),
        ));
    };
    let (scope, through): (StagingScope, u64) = serde_json::from_str(cut)?;
    ensure!(
        scope
            == (StagingScope {
                table_instance: ingress.scope.incarnation.to_string(),
                source_group: ingress.group(0)?,
                shard: 0
            }),
        "foreign annotation application cut"
    );
    let results = properties
        .and_then(|p| p.get(RESULTS_PROPERTY))
        .context("annotation publication has no request results")?;
    let snapshot = Snapshot::decode(results, through)?;
    ensure!(
        snapshot.encoded.slots == ingress.client.producer_slots(),
        "annotation producer capacity changed"
    );
    if let Some(pending) = view
        .progress
        .pending
        .as_ref()
        .filter(|p| p.plan.target_version().ok() == Some(base.version_id()))
    {
        let prepared = view
            .progress
            .prepared
            .as_ref()
            .context("selected publication has no preparation")?;
        let original = load(ingress, storage, pending.plan.read_version).await?;
        let selected = annotation_publication::selected_cut(
            &base,
            &original,
            &prepared.descriptors,
            &annotation_publication::Selection {
                table: ingress.scope.table,
                incarnation: ingress.scope.incarnation,
                pending: (**pending).clone(),
            },
            record,
        )
        .await?;
        ensure!(selected == through, "selected request result cut differs");
    } else {
        ensure!(
            covered == through && view.progress.minimum_read_version == base.version_id(),
            "annotation application cut is not covered by completed publication"
        );
    }
    Ok((base, Arc::new(snapshot)))
}

async fn load(ingress: &Ingress, storage: &Storage, version: u64) -> Result<Arc<Dataset>> {
    let handler = ManagedManifest::reader(
        Arc::new(ingress.client.manifest_catalog()),
        storage.base.clone(),
        &ingress.layout,
        ingress.scope.table,
        ingress.scope.incarnation,
    )?;
    Ok(Arc::new(
        DatasetBuilder::from_uri(&storage.uri)
            .with_version(version)
            .with_storage_options(storage.options.clone())
            .with_session(storage.session.clone())
            .with_commit_handler(Arc::new(handler))
            .load()
            .await?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maximum_producer_rejections_fit_without_repeating_receipt_metadata() -> Result<()> {
        let snapshot = Encoded {
            version: 1,
            slots: producer::MAX_SLOTS,
            rejected: (0..producer::MAX_SLOTS)
                .map(|slot| (slot, u64::MAX - 1, 65_535, 4))
                .collect(),
        };
        let encoded = serde_json::to_string(&snapshot)?;
        assert!(encoded.len() <= MAX_RESULTS_BYTES);
        assert_eq!(
            Snapshot::decode(&encoded, u64::MAX - 1)?
                .encoded
                .rejected
                .len(),
            producer::MAX_SLOTS as usize
        );
        Ok(())
    }
}
