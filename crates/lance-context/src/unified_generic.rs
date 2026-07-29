use std::collections::HashMap;

use lance_context_api::{
    AddRowsResponse, ContextError, ContextResult, GenericStoreApi, SchemaSpec,
};
use lance_context_core::{GenericStore as LocalStore, GenericStoreOptions};
use serde_json::{Map, Value};

#[cfg(feature = "remote")]
use lance_context_api::CreateGenericStoreRequest;
#[cfg(feature = "remote")]
use lance_context_client::RemoteGenericStore;

/// A store over a user-declared schema, either an in-process Lance dataset
/// (`Local`) or a handle to a remote server (`Remote`). Mirrors
/// [`crate::RolloutStore`] but for schemas the caller defines.
pub enum GenericStore {
    Local(Box<LocalStore>),
    #[cfg(feature = "remote")]
    Remote(RemoteGenericStore),
}

impl GenericStore {
    /// Open an embedded store at `uri`, creating it with `schema` if absent.
    pub async fn open(uri: &str, schema: SchemaSpec) -> Result<Self, ContextError> {
        Self::open_with_options(uri, schema, None, false).await
    }

    pub async fn open_with_options(
        uri: &str,
        schema: SchemaSpec,
        storage_options: Option<HashMap<String, String>>,
        seal_on_add: bool,
    ) -> Result<Self, ContextError> {
        let options = GenericStoreOptions {
            storage_options,
            // Embedded single-process use writes to the fallback shard; a
            // multi-writer embedded deployment threads a per-writer id through
            // the core `GenericStore` directly.
            shard_id: None,
            merge_after_generations: None,
            session: None,
            seal_on_add,
        };
        let store = LocalStore::open(uri, schema, options)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Local(Box::new(store)))
    }

    /// Open an existing embedded store, reading its schema from the dataset.
    pub async fn open_existing(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        seal_on_add: bool,
    ) -> Result<Self, ContextError> {
        let options = GenericStoreOptions {
            storage_options,
            shard_id: None,
            merge_after_generations: None,
            session: None,
            seal_on_add,
        };
        let store = LocalStore::open_existing(uri, options)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Local(Box::new(store)))
    }

    #[cfg(feature = "remote")]
    pub async fn connect(base_url: &str, store_name: &str) -> Result<Self, ContextError> {
        let store = RemoteGenericStore::connect(base_url, store_name)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }

    #[cfg(feature = "remote")]
    pub async fn connect_or_create(
        base_url: &str,
        req: &CreateGenericStoreRequest,
    ) -> Result<Self, ContextError> {
        let store = RemoteGenericStore::connect_or_create(base_url, req)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }

    /// Merge pending WAL generations into the base table. Local stores only —
    /// a remote store's server owns its own merge schedule.
    pub async fn cleanup_wal(&mut self) -> Result<usize, ContextError> {
        match self {
            Self::Local(store) => store
                .cleanup_wal()
                .await
                .map_err(|e| ContextError::Internal(e.to_string())),
            #[cfg(feature = "remote")]
            Self::Remote(_) => Err(ContextError::InvalidRequest(
                "cleanup_wal is not available on a remote store; the server merges its own shard"
                    .to_string(),
            )),
        }
    }
}

impl GenericStoreApi for GenericStore {
    fn spec(&self) -> &SchemaSpec {
        match self {
            Self::Local(store) => GenericStoreApi::spec(store.as_ref()),
            #[cfg(feature = "remote")]
            Self::Remote(store) => GenericStoreApi::spec(store),
        }
    }

    async fn add(&self, rows: &[Map<String, Value>]) -> ContextResult<AddRowsResponse> {
        match self {
            Self::Local(store) => GenericStoreApi::add(store.as_ref(), rows).await,
            #[cfg(feature = "remote")]
            Self::Remote(store) => GenericStoreApi::add(store, rows).await,
        }
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<Map<String, Value>>> {
        match self {
            Self::Local(store) => GenericStoreApi::list(store.as_ref(), limit, offset).await,
            #[cfg(feature = "remote")]
            Self::Remote(store) => GenericStoreApi::list(store, limit, offset).await,
        }
    }

    async fn list_filtered(
        &self,
        filter: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<Map<String, Value>>> {
        match self {
            Self::Local(store) => {
                GenericStoreApi::list_filtered(store.as_ref(), filter, limit, offset).await
            }
            #[cfg(feature = "remote")]
            Self::Remote(store) => {
                GenericStoreApi::list_filtered(store, filter, limit, offset).await
            }
        }
    }

    async fn get(
        &self,
        id: &str,
        columns: Option<&[String]>,
    ) -> ContextResult<Option<Map<String, Value>>> {
        match self {
            Self::Local(store) => GenericStoreApi::get(store.as_ref(), id, columns).await,
            #[cfg(feature = "remote")]
            Self::Remote(store) => GenericStoreApi::get(store, id, columns).await,
        }
    }

    async fn flush(&self) -> ContextResult<()> {
        match self {
            Self::Local(store) => GenericStoreApi::flush(store.as_ref()).await,
            #[cfg(feature = "remote")]
            Self::Remote(store) => GenericStoreApi::flush(store).await,
        }
    }

    fn version(&self) -> u64 {
        match self {
            Self::Local(store) => GenericStoreApi::version(store.as_ref()),
            #[cfg(feature = "remote")]
            Self::Remote(store) => GenericStoreApi::version(store),
        }
    }
}
