use std::collections::HashSet;

use lance_context_api::{
    AddRecordRequest, AddRecordsResponse, CompactRequest, CompactResponse, CompactStatsResponse,
    ContextError, ContextResult, ContextStoreApi, DeleteRecordResponse, RecordDto, RetrieveRequest,
    RetrieveResultDto, SearchRequest, SearchResultDto, UpdateRecordRequest, UpdateRecordResponse,
    UpsertRecordRequest, UpsertRecordResponse, UpsertRecordsRequest, UpsertRecordsResponse,
};
use lance_context_core::{
    ContextStore as LocalStore, ContextStoreOptions, DistanceMetric, IdIndexType,
};

#[cfg(feature = "remote")]
use lance_context_client::RemoteContextStore;

pub enum ContextStore {
    Local(Box<LocalStore>),
    #[cfg(feature = "remote")]
    Remote(RemoteContextStore),
}

impl ContextStore {
    pub async fn open(uri: &str) -> Result<Self, ContextError> {
        let store = LocalStore::open(uri)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Local(Box::new(store)))
    }

    pub async fn open_with_options(
        uri: &str,
        storage_options: Option<std::collections::HashMap<String, String>>,
        id_index_type: Option<&str>,
        blob_columns: Option<Vec<String>>,
        distance_metric: Option<&str>,
    ) -> Result<Self, ContextError> {
        let id_idx = match id_index_type {
            Some("btree") => IdIndexType::BTree,
            Some("zonemap") => IdIndexType::ZoneMap,
            Some("none") | None => IdIndexType::None,
            Some(other) => {
                return Err(ContextError::InvalidRequest(format!(
                    "Invalid id_index_type: '{other}'"
                )));
            }
        };
        let metric = match distance_metric {
            Some(value) => Some(
                DistanceMetric::parse(value)
                    .map_err(|e| ContextError::InvalidRequest(e.to_string()))?,
            ),
            None => None,
        };
        let options = ContextStoreOptions {
            storage_options,
            blob_columns: blob_columns
                .unwrap_or_default()
                .into_iter()
                .collect::<HashSet<_>>(),
            id_index_type: id_idx,
            distance_metric: metric,
            ..Default::default()
        };
        let store = LocalStore::open_with_options(uri, options)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Local(Box::new(store)))
    }

    #[cfg(feature = "remote")]
    pub async fn connect(base_url: &str, context_name: &str) -> Result<Self, ContextError> {
        let store = RemoteContextStore::connect(base_url, context_name)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }

    #[cfg(feature = "remote")]
    pub async fn connect_or_create(
        base_url: &str,
        req: &lance_context_api::CreateContextRequest,
    ) -> Result<Self, ContextError> {
        let store = RemoteContextStore::connect_or_create(base_url, req)
            .await
            .map_err(|e| ContextError::Internal(e.to_string()))?;
        Ok(Self::Remote(store))
    }
}

macro_rules! dispatch_mut {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ContextStore::Local(s) => ContextStoreApi::$method(s.as_mut() $(, $arg)*).await,
            #[cfg(feature = "remote")]
            ContextStore::Remote(s) => ContextStoreApi::$method(s $(, $arg)*).await,
        }
    };
}

macro_rules! dispatch_ref {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ContextStore::Local(s) => ContextStoreApi::$method(s.as_ref() $(, $arg)*).await,
            #[cfg(feature = "remote")]
            ContextStore::Remote(s) => ContextStoreApi::$method(s $(, $arg)*).await,
        }
    };
}

macro_rules! dispatch_sync {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ContextStore::Local(s) => ContextStoreApi::$method(s.as_ref() $(, $arg)*),
            #[cfg(feature = "remote")]
            ContextStore::Remote(s) => ContextStoreApi::$method(s $(, $arg)*),
        }
    };
}

impl ContextStoreApi for ContextStore {
    async fn add(&mut self, records: &[AddRecordRequest]) -> ContextResult<AddRecordsResponse> {
        dispatch_mut!(self, add, records)
    }

    async fn upsert(
        &mut self,
        request: &UpsertRecordRequest,
    ) -> ContextResult<UpsertRecordResponse> {
        dispatch_mut!(self, upsert, request)
    }

    async fn upsert_many(
        &mut self,
        request: &UpsertRecordsRequest,
    ) -> ContextResult<UpsertRecordsResponse> {
        dispatch_mut!(self, upsert_many, request)
    }

    async fn update(
        &mut self,
        request: &UpdateRecordRequest,
    ) -> ContextResult<UpdateRecordResponse> {
        dispatch_mut!(self, update, request)
    }

    async fn get(&self, id: &str) -> ContextResult<Option<RecordDto>> {
        dispatch_ref!(self, get, id)
    }

    async fn get_by_external_id(&self, external_id: &str) -> ContextResult<Option<RecordDto>> {
        dispatch_ref!(self, get_by_external_id, external_id)
    }

    async fn delete_by_id(&mut self, id: &str) -> ContextResult<DeleteRecordResponse> {
        dispatch_mut!(self, delete_by_id, id)
    }

    async fn delete_by_external_id(
        &mut self,
        external_id: &str,
    ) -> ContextResult<DeleteRecordResponse> {
        dispatch_mut!(self, delete_by_external_id, external_id)
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<serde_json::Value>,
        include_expired: bool,
        include_retired: bool,
    ) -> ContextResult<Vec<RecordDto>> {
        dispatch_ref!(
            self,
            list,
            limit,
            offset,
            filters,
            include_expired,
            include_retired
        )
    }

    async fn related(
        &self,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        include_expired: bool,
        include_retired: bool,
    ) -> ContextResult<Vec<RecordDto>> {
        dispatch_ref!(
            self,
            related,
            target_id,
            relation,
            limit,
            include_expired,
            include_retired
        )
    }

    async fn search(&self, request: &SearchRequest) -> ContextResult<Vec<SearchResultDto>> {
        dispatch_ref!(self, search, request)
    }

    async fn retrieve(&self, request: &RetrieveRequest) -> ContextResult<Vec<RetrieveResultDto>> {
        dispatch_ref!(self, retrieve, request)
    }

    fn version(&self) -> u64 {
        dispatch_sync!(self, version)
    }

    async fn checkout(&mut self, version: u64) -> ContextResult<()> {
        dispatch_mut!(self, checkout, version)
    }

    async fn compact(&mut self, options: Option<CompactRequest>) -> ContextResult<CompactResponse> {
        dispatch_mut!(self, compact, options)
    }

    async fn compaction_stats(&self) -> ContextResult<CompactStatsResponse> {
        dispatch_ref!(self, compaction_stats)
    }
}
