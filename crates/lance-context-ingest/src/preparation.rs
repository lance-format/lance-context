//! Distributed file preparation without distributed manifest authority.
//! Requests carry immutable plan references, never payloads or storage locations.
use super::*;
use crate::materializer::{observation::Counters, Options, Storage};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use lance_context_wal::replication::publication as p;
use lance_table::format::Fragment;
use std::collections::BTreeMap;

mod pool;
mod service;
pub use pool::Pool;
pub use service::{Observation, Service};

pub const ROUTE: &str = "/api/v1/preparation";
const IDENTITY: &str = "x-lance-preparation";
const CONTENT_TYPE: &str = "application/vnd.lance.fragments+json";

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    cluster: Uuid,
    layout: [u8; 32],
    table: Uuid,
    incarnation: Uuid,
    plan_index: u64,
    transaction: Uuid,
    read_version: u64,
    part: usize,
    parts: usize,
}
impl Task {
    fn new(ingress: &Ingress, pending: &p::PendingPlan, part: usize, parts: usize) -> Self {
        Self {
            cluster: ingress.scope.cluster,
            layout: ingress.scope.layout,
            table: ingress.scope.table,
            incarnation: ingress.scope.incarnation,
            plan_index: pending.index,
            transaction: pending.plan.transaction,
            read_version: pending.plan.read_version,
            part,
            parts,
        }
    }
    fn validate(&self, ingress: &Ingress) -> Result<()> {
        ensure!(
            self.cluster == ingress.scope.cluster
                && self.layout == ingress.scope.layout
                && self.table == ingress.scope.table
                && self.incarnation == ingress.scope.incarnation
                && (1..=8).contains(&self.parts)
                && self.part < self.parts
                && self.plan_index > 0
                && self.read_version > 0
                && !self.transaction.is_nil(),
            "foreign preparation scope or invalid partition"
        );
        Ok(())
    }
    fn identity(&self) -> Result<String> {
        Ok(URL_SAFE_NO_PAD.encode(
            ring::digest::digest(&ring::digest::SHA256, &serde_json::to_vec(self)?).as_ref(),
        ))
    }
    fn select(&self, pending: &p::PendingPlan) -> Result<Vec<(u64, u64, BatchMeta)>> {
        ensure!(
            pending.index == self.plan_index
                && pending.plan.transaction == self.transaction
                && pending.plan.read_version == self.read_version,
            "preparation plan changed"
        );
        let inputs = pending
            .plan
            .sources
            .iter()
            .flat_map(|(&group, range)| {
                range
                    .records
                    .iter()
                    .map(move |(index, meta)| (group, *index, meta.clone()))
            })
            .collect::<Vec<_>>();
        ensure!(self.parts <= inputs.len(), "empty preparation partition");
        Ok(inputs
            .into_iter()
            .enumerate()
            .filter_map(|(i, record)| (i % self.parts == self.part).then_some(record))
            .collect())
    }
}

fn fragments(descriptors: &p::Descriptors, maximum: u64) -> Result<Vec<Fragment>> {
    descriptors.validate(maximum)?;
    let fragments: Vec<Fragment> = serde_json::from_slice(&descriptors.bytes)?;
    ensure!(!fragments.is_empty(), "empty preparation result");
    for fragment in &fragments {
        ensure!(
            fragment.id == 0
                && !fragment.files.is_empty()
                && fragment.physical_rows.is_some_and(|n| n > 0)
                && fragment.deletion_file.is_none()
                && fragment.overlays.is_empty()
                && fragment.row_id_meta.is_none(),
            "invalid uncommitted fragment"
        );
        for file in &fragment.files {
            ensure!(
                file.base_id.is_none()
                    && !file.path.contains(':')
                    && file
                        .path
                        .split('/')
                        .all(|p| !p.is_empty() && p != "." && p != ".."),
                "preparer returned an external or invalid data path"
            );
        }
    }
    Ok(fragments)
}
