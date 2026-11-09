//! Type-aware storage access for the existing maintenance claim/fencing paths.
//! Task identity always retains `generic:`; only physical URI construction
//! strips the prefix. Opening never creates a missing table.

use lance::dataset::optimize::CompactionMetrics;
use lance_context_core::{
    CompactionConfig, GenericStore, PreparedCompaction, PreparedKeyIndex, RolloutStore,
};

use crate::scheduler::{parse_target, StoreKind};
use crate::state::MasterState;

pub(crate) fn uri(state: &MasterState, target: &str) -> String {
    let (kind, name) = parse_target(target);
    match kind {
        StoreKind::Rollout => state.rollout_uri(name),
        StoreKind::Generic => state.generic_uri(name),
    }
}

pub(crate) enum MaintenanceStore {
    Rollout(RolloutStore),
    Generic(GenericStore),
}

pub(crate) struct Observation {
    pub version: u64,
    pub row_count: i64,
    pub fragment_count: i64,
    pub last_updated: i64,
    pub pending_wal_generations: i64,
}

impl MaintenanceStore {
    pub async fn open(state: &MasterState, target: &str) -> lance::Result<Self> {
        let uri = uri(state, target);
        match parse_target(target).0 {
            StoreKind::Rollout => Ok(Self::Rollout(
                RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
                    .await?,
            )),
            StoreKind::Generic => Ok(Self::Generic(
                GenericStore::open_existing(&uri, state.generic_store_options()).await?,
            )),
        }
    }

    pub fn version(&self) -> u64 {
        match self {
            Self::Rollout(store) => store.version(),
            Self::Generic(store) => store.version(),
        }
    }

    pub async fn compact(
        &mut self,
        config: Option<CompactionConfig>,
    ) -> lance::Result<CompactionMetrics> {
        match self {
            Self::Rollout(store) => store.compact(config).await,
            Self::Generic(store) => store.compact(config).await,
        }
    }

    pub async fn prepare_compaction(
        &self,
        config: CompactionConfig,
    ) -> lance::Result<PreparedCompaction> {
        match self {
            Self::Rollout(store) => store.prepare_compaction(config).await,
            Self::Generic(store) => store.prepare_compaction(config).await,
        }
    }

    pub async fn commit_prepared_compaction(
        &mut self,
        prepared: PreparedCompaction,
    ) -> lance::Result<CompactionMetrics> {
        match self {
            Self::Rollout(store) => store.commit_prepared_compaction(prepared).await,
            Self::Generic(store) => store.commit_prepared_compaction(prepared).await,
        }
    }

    pub async fn create_id_key_index(&mut self) -> lance::Result<()> {
        match self {
            Self::Rollout(store) => store.create_id_key_index().await,
            Self::Generic(store) => store.create_id_key_index().await,
        }
    }

    pub async fn prepare_id_key_index(&self) -> lance::Result<PreparedKeyIndex> {
        match self {
            Self::Rollout(store) => store.prepare_id_key_index().await,
            Self::Generic(store) => store.prepare_id_key_index().await,
        }
    }

    pub async fn commit_prepared_id_key_index(
        &mut self,
        prepared: PreparedKeyIndex,
    ) -> lance::Result<()> {
        match self {
            Self::Rollout(store) => store.commit_prepared_id_key_index(prepared).await,
            Self::Generic(store) => store.commit_prepared_id_key_index(prepared).await,
        }
    }

    pub async fn observe(&self) -> lance::Result<Observation> {
        match self {
            Self::Rollout(store) => {
                let obs = store.observe().await?;
                Ok(Observation {
                    version: obs.version,
                    row_count: obs.row_count,
                    fragment_count: obs.fragment_count,
                    last_updated: obs.last_updated,
                    pending_wal_generations: obs.pending_wal_generations,
                })
            }
            Self::Generic(store) => {
                // Match the generic scanner: base rows and manifest-only WAL
                // counts. No record payload scan or writer initialization.
                let pending = store.pending_wal_generations().await?;
                Ok(Observation {
                    version: store.version(),
                    row_count: store.count_base_rows().await? as i64,
                    fragment_count: store.compaction_stats().total_fragments as i64,
                    last_updated: chrono::Utc::now().timestamp_millis(),
                    pending_wal_generations: pending as i64,
                })
            }
        }
    }
}
