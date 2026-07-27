use lance_context_api::{
    AddDatagenEventsResponse, ContextError, ContextResult, DatagenEventDto, DatagenFailureDto,
    DatagenRootItemStatusesResponse, DatagenStoreApi, FoldedDatagenItemDto,
};
use lance_context_core::{DatagenStore as LocalStore, DatagenStoreOptions};

#[cfg(feature = "remote")]
use lance_context_client::RemoteDatagenStore;

/// A datagen checkpoint store that is either an in-process Lance dataset
/// (`Local`) or a handle to a remote server (`Remote`). Mirrors
/// [`crate::RolloutStore`] but for the datagen delta-log schema.
pub enum DatagenStore {
    Local(Box<LocalStore>),
    #[cfg(feature = "remote")]
    Remote(RemoteDatagenStore),
}

impl DatagenStore {
    pub async fn open(uri: &str) -> Result<Self, ContextError> {
        Self::open_with_options(uri, None).await
    }

    pub async fn open_with_options(
        uri: &str,
        storage_options: Option<std::collections::HashMap<String, String>>,
    ) -> Result<Self, ContextError> {
        let options = DatagenStoreOptions {
            storage_options,
            // Embedded single-process use writes to the fallback shard; a
            // multi-writer embedded deployment threads a per-writer id through
            // the core `DatagenStore` directly.
            shard_id: None,
            merge_after_generations: None,
            cleanup_interval_secs: None,
        };
        let store = LocalStore::open_with_options(uri, options)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Local(Box::new(store)))
    }

    #[cfg(feature = "remote")]
    pub async fn connect(base_url: &str, store_name: &str) -> Result<Self, ContextError> {
        let store = RemoteDatagenStore::connect(base_url, store_name)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }

    #[cfg(feature = "remote")]
    pub async fn connect_or_create(
        base_url: &str,
        req: &lance_context_api::CreateDatagenStoreRequest,
    ) -> Result<Self, ContextError> {
        let store = RemoteDatagenStore::connect_or_create(base_url, req)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }
}

macro_rules! dispatch_mut {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            DatagenStore::Local(s) => DatagenStoreApi::$method(s.as_mut() $(, $arg)*).await,
            #[cfg(feature = "remote")]
            DatagenStore::Remote(s) => DatagenStoreApi::$method(s $(, $arg)*).await,
        }
    };
}

macro_rules! dispatch_ref {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            DatagenStore::Local(s) => DatagenStoreApi::$method(s.as_ref() $(, $arg)*).await,
            #[cfg(feature = "remote")]
            DatagenStore::Remote(s) => DatagenStoreApi::$method(s $(, $arg)*).await,
        }
    };
}

macro_rules! dispatch_sync {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            DatagenStore::Local(s) => DatagenStoreApi::$method(s.as_ref() $(, $arg)*),
            #[cfg(feature = "remote")]
            DatagenStore::Remote(s) => DatagenStoreApi::$method(s $(, $arg)*),
        }
    };
}

impl DatagenStoreApi for DatagenStore {
    async fn append(
        &mut self,
        events: &[DatagenEventDto],
    ) -> ContextResult<AddDatagenEventsResponse> {
        dispatch_mut!(self, append, events)
    }

    async fn append_checkpoint(
        &mut self,
        events: &[DatagenEventDto],
    ) -> ContextResult<AddDatagenEventsResponse> {
        dispatch_mut!(self, append_checkpoint, events)
    }

    async fn fold_item(&self, item_id: &str) -> ContextResult<Option<FoldedDatagenItemDto>> {
        dispatch_ref!(self, fold_item, item_id)
    }

    async fn root_item_statuses(
        &self,
        root_item_ids: &[String],
    ) -> ContextResult<DatagenRootItemStatusesResponse> {
        dispatch_ref!(self, root_item_statuses, root_item_ids)
    }

    async fn item_failures(&self, item_id: &str) -> ContextResult<Vec<DatagenFailureDto>> {
        dispatch_ref!(self, item_failures, item_id)
    }

    async fn get_blob(&self, event_id: &str) -> ContextResult<Option<Vec<u8>>> {
        dispatch_ref!(self, get_blob, event_id)
    }

    fn version(&self) -> u64 {
        dispatch_sync!(self, version)
    }
}
