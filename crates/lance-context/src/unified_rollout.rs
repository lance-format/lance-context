use lance_context_api::{
    AddRolloutRequest, AddRolloutsResponse, ContextError, ContextResult, RolloutRecordDto,
    RolloutStoreApi,
};
use lance_context_core::{RolloutStore as LocalStore, RolloutStoreOptions};

#[cfg(feature = "remote")]
use lance_context_client::RemoteRolloutStore;

/// A rollout store that is either an in-process Lance dataset (`Local`) or a
/// handle to a remote server (`Remote`). Mirrors [`crate::ContextStore`] but for
/// the RL rollout schema.
pub enum RolloutStore {
    Local(Box<LocalStore>),
    #[cfg(feature = "remote")]
    Remote(RemoteRolloutStore),
}

impl RolloutStore {
    pub async fn open(uri: &str) -> Result<Self, ContextError> {
        let store = LocalStore::open(uri)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Local(Box::new(store)))
    }

    pub async fn open_with_options(
        uri: &str,
        storage_options: Option<std::collections::HashMap<String, String>>,
    ) -> Result<Self, ContextError> {
        let options = RolloutStoreOptions {
            storage_options,
            // Embedded single-process use writes to the fallback shard. A
            // multi-writer embedded deployment should thread a per-writer id
            // through here (see `RolloutStoreOptions::shard_id`).
            shard_id: None,
            // Embedded use accumulates generations and unions at read time; a
            // caller that wants count-triggered self-merge opens the core
            // `RolloutStore` directly with `merge_after_generations` set.
            merge_after_generations: None,
            ..Default::default()
        };
        let store = LocalStore::open_with_options(uri, options)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Local(Box::new(store)))
    }

    #[cfg(feature = "remote")]
    pub async fn connect(base_url: &str, store_name: &str) -> Result<Self, ContextError> {
        let store = RemoteRolloutStore::connect(base_url, store_name)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }

    #[cfg(feature = "remote")]
    pub async fn connect_or_create(
        base_url: &str,
        req: &lance_context_api::CreateRolloutStoreRequest,
    ) -> Result<Self, ContextError> {
        let store = RemoteRolloutStore::connect_or_create(base_url, req)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }

    /// Append rollout rows and wait until they are visible to subsequent reads.
    ///
    /// Local stores seal their MemWAL memtable after the append. Remote stores
    /// send one append request with `?flush=true`, so the server instance that
    /// accepted the rows also makes them visible. Remote servers must be v0.6.5
    /// or newer.
    ///
    /// The append is durable before the flush starts. If the flush fails, this
    /// method returns an error even though the rows may already be durable.
    /// Callers that retry should reuse the same record ids. Sealing can also
    /// make earlier or concurrent pending appends visible.
    pub async fn add_with_flush(
        &mut self,
        records: &[AddRolloutRequest],
    ) -> ContextResult<AddRolloutsResponse> {
        if records.is_empty() {
            return Err(ContextError::InvalidRequest(
                "records must not be empty".to_string(),
            ));
        }

        match self {
            Self::Local(store) => {
                let response = RolloutStoreApi::add(store.as_mut(), records).await?;
                store
                    .flush()
                    .await
                    .map_err(|e| ContextError::Internal(e.to_string()))?;
                Ok(response)
            }
            #[cfg(feature = "remote")]
            Self::Remote(store) => store.add_with_flush(records).await,
        }
    }

    /// Seal the local MemWAL memtable so rows written by [`RolloutStoreApi::add`]
    /// become visible to subsequent reads on this handle.
    ///
    /// `add` is durable on return but *not* visible on return: visibility is
    /// driven by a periodic sweeper in the server. An embedded caller has no
    /// sweeper, so without an explicit `flush` a write-then-read sequence
    /// returns nothing. Call this before reading back rows you just wrote.
    ///
    /// A no-op for `Remote` stores. Use [`Self::add_with_flush`] when a remote
    /// append needs read-your-write visibility.
    pub async fn flush(&self) -> Result<(), ContextError> {
        match self {
            RolloutStore::Local(s) => s
                .flush()
                .await
                .map_err(|e| ContextError::Internal(e.to_string())),
            #[cfg(feature = "remote")]
            RolloutStore::Remote(_) => Ok(()),
        }
    }
}

macro_rules! dispatch_mut {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            RolloutStore::Local(s) => RolloutStoreApi::$method(s.as_mut() $(, $arg)*).await,
            #[cfg(feature = "remote")]
            RolloutStore::Remote(s) => RolloutStoreApi::$method(s $(, $arg)*).await,
        }
    };
}

macro_rules! dispatch_ref {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            RolloutStore::Local(s) => RolloutStoreApi::$method(s.as_ref() $(, $arg)*).await,
            #[cfg(feature = "remote")]
            RolloutStore::Remote(s) => RolloutStoreApi::$method(s $(, $arg)*).await,
        }
    };
}

macro_rules! dispatch_sync {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            RolloutStore::Local(s) => RolloutStoreApi::$method(s.as_ref() $(, $arg)*),
            #[cfg(feature = "remote")]
            RolloutStore::Remote(s) => RolloutStoreApi::$method(s $(, $arg)*),
        }
    };
}

impl RolloutStoreApi for RolloutStore {
    async fn add(&mut self, records: &[AddRolloutRequest]) -> ContextResult<AddRolloutsResponse> {
        dispatch_mut!(self, add, records)
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<RolloutRecordDto>> {
        dispatch_ref!(self, list, limit, offset)
    }

    async fn list_filtered(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<serde_json::Value>,
    ) -> ContextResult<Vec<RolloutRecordDto>> {
        dispatch_ref!(self, list_filtered, limit, offset, filters)
    }

    async fn get(&self, id: &str) -> ContextResult<Option<RolloutRecordDto>> {
        dispatch_ref!(self, get, id)
    }

    async fn get_blob(&self, id: &str) -> ContextResult<Option<Vec<u8>>> {
        dispatch_ref!(self, get_blob, id)
    }

    fn version(&self) -> u64 {
        dispatch_sync!(self, version)
    }

    async fn checkout(&mut self, version: u64) -> ContextResult<()> {
        dispatch_mut!(self, checkout, version)
    }
}
