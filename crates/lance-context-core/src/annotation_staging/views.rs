//! Local base lifetimes for coordinated reclamation. No per-read registration
//! or consensus operation: a cold reader shares its base generation's Arc.
use super::*;
use std::{ops::Deref, sync::Weak};

pub(super) struct Base {
    pub dataset: Arc<Dataset>,
    // Only validated index-only publications retain this identity. Data
    // publication, compaction and another staging instance invalidate lookups.
    values_identity: Arc<()>,
}
impl Base {
    pub(super) fn same_values(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.values_identity, &other.values_identity)
    }
}
impl Deref for Base {
    type Target = Dataset;
    fn deref(&self) -> &Self::Target {
        &self.dataset
    }
}

pub(super) struct Registry(Vec<Weak<Base>>);
impl Registry {
    pub fn new(dataset: Arc<Dataset>) -> (Arc<Base>, Self) {
        let base = Arc::new(Base {
            dataset,
            values_identity: Arc::new(()),
        });
        let registry = Self(vec![Arc::downgrade(&base)]);
        (base, registry)
    }

    pub fn install(&mut self, dataset: Arc<Dataset>) -> Arc<Base> {
        self.install_with_values(dataset, Arc::new(()))
    }

    pub fn install_indexed(&mut self, dataset: Arc<Dataset>, source: &Base) -> Arc<Base> {
        self.install_with_values(dataset, source.values_identity.clone())
    }

    fn install_with_values(
        &mut self,
        dataset: Arc<Dataset>,
        values_identity: Arc<()>,
    ) -> Arc<Base> {
        self.0.retain(|base| base.strong_count() != 0);
        let base = Arc::new(Base {
            dataset,
            values_identity,
        });
        self.0.push(Arc::downgrade(&base));
        base
    }

    fn capture(&self) -> Vec<Arc<Base>> {
        self.0.iter().filter_map(Weak::upgrade).collect()
    }
}

/// Pins this staging instance's current base and bases retained by its cold
/// point reads or frozen/prepared generations. Dropping the snapshot releases
/// its pins; at most one such snapshot can be outstanding per staging instance.
///
/// This is only the LOCAL part of a GC root set, not a deletion permit. A caller
/// must also protect durable catalog/publication dependencies and any independent
/// readers (including separately connected services and cloned raw Datasets).
/// Serialize preparation and seal the closed deletion batch with current owner
/// authority before deleting anything. Newly prepared files are not listed here.
pub struct ProtectedBases {
    bases: Vec<Arc<Base>>,
    _permit: OwnedSemaphorePermit,
}
impl ProtectedBases {
    pub fn datasets(&self) -> impl Iterator<Item = &Dataset> {
        self.bases.iter().map(|base| base.dataset.as_ref())
    }
}

impl AnnotationStaging {
    /// Capture local protection under the same gate as reads and base switches.
    /// This does not pause subsequent point reads or publication.
    ///
    /// The registry keeps weak references, pruned at each base installation.
    /// Live old bases require admitted readers or generations charged to value
    /// or metadata budgets. The one outstanding snapshot may pin one previous
    /// bounded set, but callers cannot accumulate snapshots across publications.
    pub async fn protected_bases(&self) -> Result<ProtectedBases> {
        let permit = reserve(&self.root_snapshot, 1)?;
        let bases = self.state.read().await.views.capture();
        Ok(ProtectedBases {
            bases,
            _permit: permit,
        })
    }
}
