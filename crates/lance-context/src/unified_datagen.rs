use lance_context_api::{
    AddDatagenEventsResponse, ContextError, ContextResult, DatagenEventDto, DatagenFailureDto,
    DatagenRootItemStatusesResponse, DatagenRunOverviewDto, DatagenStoreApi, FoldedDatagenItemDto,
};
use lance_context_core::{
    datagen_event_to_dto, datagen_events_from_dtos, fold_datagen_events, open_stream_events,
    DatagenEvent, DatagenItemId, DatagenItemTree, DatagenNewStream, DatagenStore as LocalStore,
    DatagenStoreOptions, DatagenStreamWriter, DatagenWriteContext,
};

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

    /// Assemble the item tree rooted at `root_item_id`. Works for both `Local`
    /// and `Remote`: raw events come from `events_for_root` and the fold/tree
    /// assembly is the single-source [`DatagenItemTree::build`], so remote and
    /// embedded produce identical trees.
    pub async fn item_tree(&self, root_item_id: &str) -> Result<DatagenItemTree, ContextError> {
        let dtos = self.events_for_root(root_item_id).await?;
        let events = datagen_events_from_dtos(&dtos)?;
        DatagenItemTree::build(&events).map_err(ContextError::InvalidRequest)
    }

    /// Open a fresh stream (Case 3): persist ITEM_CREATED and return a writer
    /// positioned to continue after it. The writer is a pure client-side state
    /// machine — its later events are appended by the caller — so this works for
    /// both `Local` and `Remote` with no writer-specific endpoint.
    pub async fn open_stream(
        &mut self,
        stream: &DatagenNewStream,
        context: &DatagenWriteContext,
    ) -> Result<DatagenStreamWriter, ContextError> {
        let opened = open_stream_events(stream, context);
        let created = datagen_event_to_dto(&opened.created_event);
        self.append(std::slice::from_ref(&created)).await?;
        Ok(opened.writer)
    }

    /// Rebuild a writer to resume an already-started item (Case 2). Pure — folds
    /// the item to find `last_item_seq`/`last_attempt`, emits nothing. Returns
    /// `None` if the item never started.
    pub async fn resume_stream(
        &self,
        item_id: &str,
        context: &DatagenWriteContext,
    ) -> Result<Option<DatagenStreamWriter>, ContextError> {
        let dtos = self.events_for_root(&item_id_root(item_id)?).await?;
        let events = datagen_events_from_dtos(&dtos)?;
        let item_events: Vec<DatagenEvent> = events
            .into_iter()
            .filter(|event| event.item_id == item_id)
            .collect();
        let folded = fold_datagen_events(&item_events).map_err(ContextError::InvalidRequest)?;
        Ok(folded.map(|item| item.resuming_writer(context)))
    }
}

fn item_id_root(item_id: &str) -> Result<String, ContextError> {
    Ok(DatagenItemId::parse(item_id)
        .map_err(ContextError::InvalidRequest)?
        .root()
        .to_string())
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

    async fn fold_item_with_blobs(
        &self,
        item_id: &str,
        load_blobs: bool,
    ) -> ContextResult<Option<FoldedDatagenItemDto>> {
        dispatch_ref!(self, fold_item_with_blobs, item_id, load_blobs)
    }

    async fn root_item_statuses(
        &self,
        root_item_ids: &[String],
    ) -> ContextResult<DatagenRootItemStatusesResponse> {
        dispatch_ref!(self, root_item_statuses, root_item_ids)
    }

    async fn overview(&self) -> ContextResult<DatagenRunOverviewDto> {
        dispatch_ref!(self, overview)
    }

    async fn item_failures(&self, item_id: &str) -> ContextResult<Vec<DatagenFailureDto>> {
        dispatch_ref!(self, item_failures, item_id)
    }

    async fn events_for_root(&self, root_item_id: &str) -> ContextResult<Vec<DatagenEventDto>> {
        dispatch_ref!(self, events_for_root, root_item_id)
    }

    async fn get_blob(&self, event_id: &str) -> ContextResult<Option<Vec<u8>>> {
        dispatch_ref!(self, get_blob, event_id)
    }

    fn version(&self) -> u64 {
        dispatch_sync!(self, version)
    }
}
