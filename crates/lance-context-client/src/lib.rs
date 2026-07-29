use lance_context_api::*;
use reqwest::Client;
use serde_json::{Map, Value};

mod error;
pub use error::ClientError;

pub struct ContextClient {
    base_url: String,
    http: Client,
}

pub struct RemoteContextStore {
    client: ContextClient,
    context_name: String,
    cached_version: u64,
}

impl RemoteContextStore {
    pub async fn connect(base_url: &str, context_name: &str) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = client.get_context(context_name).await?;
        Ok(Self {
            client,
            context_name: context_name.to_string(),
            cached_version: info.version,
        })
    }

    pub async fn connect_or_create(
        base_url: &str,
        req: &CreateContextRequest,
    ) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = match client.get_context(&req.name).await {
            Ok(info) => info,
            Err(ClientError::Api { status: 404, .. }) => client.create_context(req).await?,
            Err(e) => return Err(e),
        };
        Ok(Self {
            client,
            context_name: req.name.clone(),
            cached_version: info.version,
        })
    }

    /// Resolve a record's external payload reference to its raw bytes via the
    /// server. Errors if no such record or the record has no external reference.
    pub async fn fetch_payload(&self, id: &str) -> ContextResult<Vec<u8>> {
        self.client
            .fetch_record_payload(&self.context_name, id)
            .await
            .map_err(to_ctx_err)
    }
}

impl ContextStoreApi for RemoteContextStore {
    async fn add(&mut self, records: &[AddRecordRequest]) -> ContextResult<AddRecordsResponse> {
        let req = AddRecordsRequest {
            records: records.to_vec(),
        };
        let resp = self
            .client
            .add_records(&self.context_name, &req)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn upsert(
        &mut self,
        request: &UpsertRecordRequest,
    ) -> ContextResult<UpsertRecordResponse> {
        let resp = self
            .client
            .upsert_record(&self.context_name, request)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn upsert_many(
        &mut self,
        request: &UpsertRecordsRequest,
    ) -> ContextResult<UpsertRecordsResponse> {
        let resp = self
            .client
            .upsert_records(&self.context_name, request)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn update(
        &mut self,
        request: &UpdateRecordRequest,
    ) -> ContextResult<UpdateRecordResponse> {
        let resp = self
            .client
            .update_record(&self.context_name, request)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn get(&self, id: &str) -> ContextResult<Option<RecordDto>> {
        let resp = self
            .client
            .get_record(&self.context_name, id)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.record)
    }

    async fn get_by_external_id(&self, external_id: &str) -> ContextResult<Option<RecordDto>> {
        let resp = self
            .client
            .get_record_by_external_id(&self.context_name, external_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.record)
    }

    async fn delete_by_id(&mut self, id: &str) -> ContextResult<DeleteRecordResponse> {
        let resp = self
            .client
            .delete_record(&self.context_name, id)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn delete_by_external_id(
        &mut self,
        external_id: &str,
    ) -> ContextResult<DeleteRecordResponse> {
        let resp = self
            .client
            .delete_record_by_external_id(&self.context_name, external_id)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<serde_json::Value>,
        include_expired: bool,
        include_retired: bool,
    ) -> ContextResult<Vec<RecordDto>> {
        let filters = filters
            .as_ref()
            .map(|value| {
                serde_json::to_string(value)
                    .map_err(|err| ContextError::InvalidRequest(err.to_string()))
            })
            .transpose()?;
        let resp = self
            .client
            .list_records(
                &self.context_name,
                limit,
                offset,
                filters.as_deref(),
                include_expired,
                include_retired,
            )
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.records)
    }

    async fn related(
        &self,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        include_expired: bool,
        include_retired: bool,
    ) -> ContextResult<Vec<RecordDto>> {
        let resp = self
            .client
            .related_records(
                &self.context_name,
                target_id,
                relation,
                limit,
                include_expired,
                include_retired,
            )
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.records)
    }

    async fn search(&self, request: &SearchRequest) -> ContextResult<Vec<SearchResultDto>> {
        let resp = self
            .client
            .search(&self.context_name, request)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.results)
    }

    async fn retrieve(&self, request: &RetrieveRequest) -> ContextResult<Vec<RetrieveResultDto>> {
        let resp = self
            .client
            .retrieve(&self.context_name, request)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.results)
    }

    fn version(&self) -> u64 {
        self.cached_version
    }

    async fn checkout(&mut self, version: u64) -> ContextResult<()> {
        let req = CheckoutRequest { version };
        let resp = self
            .client
            .checkout(&self.context_name, &req)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(())
    }

    async fn compact(&mut self, options: Option<CompactRequest>) -> ContextResult<CompactResponse> {
        let req = options.unwrap_or_default();
        let resp = self
            .client
            .compact(&self.context_name, &req)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp)
    }

    async fn compaction_stats(&self) -> ContextResult<CompactStatsResponse> {
        self.client
            .compact_stats(&self.context_name)
            .await
            .map_err(to_ctx_err)
    }
}

pub struct RemoteRolloutStore {
    client: ContextClient,
    store_name: String,
    cached_version: u64,
}

impl RemoteRolloutStore {
    pub async fn connect(base_url: &str, store_name: &str) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = client.get_rollout_store(store_name).await?;
        Ok(Self {
            client,
            store_name: store_name.to_string(),
            cached_version: info.version.unwrap_or(0),
        })
    }

    pub async fn connect_or_create(
        base_url: &str,
        req: &CreateRolloutStoreRequest,
    ) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = match client.get_rollout_store(&req.name).await {
            Ok(info) => info,
            Err(ClientError::Api { status: 404, .. }) => client.create_rollout_store(req).await?,
            Err(e) => return Err(e),
        };
        Ok(Self {
            client,
            store_name: req.name.clone(),
            cached_version: info.version.unwrap_or(0),
        })
    }
}

impl RolloutStoreApi for RemoteRolloutStore {
    async fn add(&mut self, records: &[AddRolloutRequest]) -> ContextResult<AddRolloutsResponse> {
        let resp = self
            .client
            .add_rollouts(&self.store_name, records)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<RolloutRecordDto>> {
        let resp = self
            .client
            .list_rollouts(&self.store_name, limit, offset)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.records)
    }

    async fn list_filtered(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<serde_json::Value>,
    ) -> ContextResult<Vec<RolloutRecordDto>> {
        let filters = filters
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| ContextError::InvalidRequest(err.to_string()))?;
        let resp = self
            .client
            .list_rollouts_filtered(&self.store_name, limit, offset, filters.as_deref())
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.records)
    }

    async fn get(&self, id: &str) -> ContextResult<Option<RolloutRecordDto>> {
        let resp = self
            .client
            .get_rollout(&self.store_name, id)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.record)
    }

    async fn get_blob(&self, id: &str) -> ContextResult<Option<Vec<u8>>> {
        self.client
            .fetch_rollout_blob(&self.store_name, id)
            .await
            .map_err(to_ctx_err)
    }

    fn version(&self) -> u64 {
        self.cached_version
    }

    async fn checkout(&mut self, version: u64) -> ContextResult<()> {
        let req = CheckoutRequest { version };
        let resp = self
            .client
            .checkout_rollout(&self.store_name, &req)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(())
    }
}

/// A [`GenericStoreApi`] backed by the REST server.
///
/// Caches the schema at connect: unlike the fixed-schema stores there is no
/// compile-time type describing the columns, and `GenericStoreApi::spec` is
/// synchronous, so it cannot be fetched on demand.
pub struct RemoteGenericStore {
    client: ContextClient,
    store_name: String,
    spec: SchemaSpec,
    cached_version: u64,
}

impl RemoteGenericStore {
    pub async fn connect(base_url: &str, store_name: &str) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = client.get_generic_store(store_name).await?;
        let spec = info.schema.ok_or_else(|| ClientError::Api {
            status: 500,
            code: "MISSING_SCHEMA".to_string(),
            message: "server did not report the store schema".to_string(),
        })?;
        Ok(Self {
            client,
            store_name: store_name.to_string(),
            spec,
            cached_version: info.version.unwrap_or(0),
        })
    }

    pub async fn connect_or_create(
        base_url: &str,
        req: &CreateGenericStoreRequest,
    ) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = match client.get_generic_store(&req.name).await {
            Ok(info) => info,
            Err(ClientError::Api { status: 404, .. }) => client.create_generic_store(req).await?,
            Err(e) => return Err(e),
        };
        Ok(Self {
            client,
            store_name: req.name.clone(),
            spec: info.schema.unwrap_or_else(|| req.schema.clone()),
            cached_version: info.version.unwrap_or(0),
        })
    }

    /// Seal the store's memtable so added rows become readable.
    pub async fn flush_remote(&self) -> Result<(), ClientError> {
        self.client.flush_generic_store(&self.store_name).await
    }
}

impl GenericStoreApi for RemoteGenericStore {
    fn spec(&self) -> &SchemaSpec {
        &self.spec
    }

    async fn add(&self, rows: &[Map<String, Value>]) -> ContextResult<AddRowsResponse> {
        self.client
            .add_rows(&self.store_name, rows)
            .await
            .map_err(to_ctx_err)
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<Map<String, Value>>> {
        Ok(self
            .client
            .list_rows(&self.store_name, None, limit, offset)
            .await
            .map_err(to_ctx_err)?
            .rows)
    }

    async fn list_filtered(
        &self,
        filter: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<Map<String, Value>>> {
        Ok(self
            .client
            .list_rows(&self.store_name, Some(filter), limit, offset)
            .await
            .map_err(to_ctx_err)?
            .rows)
    }

    async fn get(
        &self,
        id: &str,
        columns: Option<&[String]>,
    ) -> ContextResult<Option<Map<String, Value>>> {
        self.client
            .get_row(&self.store_name, id, columns)
            .await
            .map_err(to_ctx_err)
    }

    async fn flush(&self) -> ContextResult<()> {
        self.client
            .flush_generic_store(&self.store_name)
            .await
            .map_err(to_ctx_err)
    }

    fn version(&self) -> u64 {
        self.cached_version
    }
}

pub struct RemoteDatagenStore {
    client: ContextClient,
    store_name: String,
    cached_version: u64,
}

impl RemoteDatagenStore {
    pub async fn connect(base_url: &str, store_name: &str) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = client.get_datagen_store(store_name).await?;
        Ok(Self {
            client,
            store_name: store_name.to_string(),
            cached_version: info.version.unwrap_or(0),
        })
    }

    pub async fn connect_or_create(
        base_url: &str,
        req: &CreateDatagenStoreRequest,
    ) -> Result<Self, ClientError> {
        let client = ContextClient::new(base_url);
        let info = match client.get_datagen_store(&req.name).await {
            Ok(info) => info,
            Err(ClientError::Api { status: 404, .. }) => client.create_datagen_store(req).await?,
            Err(e) => return Err(e),
        };
        Ok(Self {
            client,
            store_name: req.name.clone(),
            cached_version: info.version.unwrap_or(0),
        })
    }
}

impl DatagenStoreApi for RemoteDatagenStore {
    async fn append(
        &mut self,
        events: &[DatagenEventDto],
    ) -> ContextResult<AddDatagenEventsResponse> {
        let resp = self
            .client
            .add_datagen_events(&self.store_name, events, false)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn append_checkpoint(
        &mut self,
        events: &[DatagenEventDto],
    ) -> ContextResult<AddDatagenEventsResponse> {
        let resp = self
            .client
            .add_datagen_events(&self.store_name, events, true)
            .await
            .map_err(to_ctx_err)?;
        self.cached_version = resp.version;
        Ok(resp)
    }

    async fn fold_item(&self, item_id: &str) -> ContextResult<Option<FoldedDatagenItemDto>> {
        let resp = self
            .client
            .fold_datagen_item(&self.store_name, item_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.item)
    }

    async fn fold_item_with_blobs(
        &self,
        item_id: &str,
        load_blobs: bool,
    ) -> ContextResult<Option<FoldedDatagenItemDto>> {
        let resp = self
            .client
            .fold_datagen_item_with_blobs(&self.store_name, item_id, load_blobs)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.item)
    }

    async fn root_item_statuses(
        &self,
        root_item_ids: &[String],
    ) -> ContextResult<DatagenRootItemStatusesResponse> {
        self.client
            .datagen_root_item_statuses(&self.store_name, root_item_ids)
            .await
            .map_err(to_ctx_err)
    }

    async fn overview(&self) -> ContextResult<DatagenRunOverviewDto> {
        self.client
            .datagen_overview(&self.store_name)
            .await
            .map_err(to_ctx_err)
    }

    async fn item_failures(&self, item_id: &str) -> ContextResult<Vec<DatagenFailureDto>> {
        let resp = self
            .client
            .datagen_item_failures(&self.store_name, item_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.failures)
    }

    async fn events_for_root(&self, root_item_id: &str) -> ContextResult<Vec<DatagenEventDto>> {
        let resp = self
            .client
            .datagen_events_for_root(&self.store_name, root_item_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(resp.events)
    }

    async fn get_blob(&self, event_id: &str) -> ContextResult<Option<Vec<u8>>> {
        self.client
            .fetch_datagen_blob(&self.store_name, event_id)
            .await
            .map_err(to_ctx_err)
    }

    fn version(&self) -> u64 {
        self.cached_version
    }
}

fn to_ctx_err(err: ClientError) -> ContextError {
    match err {
        ClientError::Api {
            status: 404,
            message,
            ..
        } => ContextError::NotFound(message),
        ClientError::Api {
            status: 409,
            code,
            message,
        } => {
            if code == "COMPACTION_IN_PROGRESS" {
                ContextError::CompactionInProgress
            } else {
                ContextError::AlreadyExists(message)
            }
        }
        ClientError::Api {
            status: 400,
            message,
            ..
        } => ContextError::InvalidRequest(message),
        ClientError::Api { message, .. } => ContextError::Internal(message),
        ClientError::Http(e) => ContextError::Internal(e.to_string()),
        ClientError::Serialize(e) => ContextError::InvalidRequest(e.to_string()),
    }
}

// --- Low-level client (still available for context lifecycle management) ---

impl ContextClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http: Client::new(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/api/v1{}", self.base_url, path)
    }

    pub async fn create_context(
        &self,
        req: &CreateContextRequest,
    ) -> Result<ContextInfo, ClientError> {
        let resp = self
            .http
            .post(self.url("/contexts"))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn list_contexts(&self) -> Result<ListContextsResponse, ClientError> {
        let resp = self.http.get(self.url("/contexts")).send().await?;
        Self::handle_response(resp).await
    }

    pub async fn get_context(&self, name: &str) -> Result<ContextInfo, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/contexts/{}", name)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn delete_context(&self, name: &str) -> Result<(), ClientError> {
        let resp = self
            .http
            .delete(self.url(&format!("/contexts/{}", name)))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    pub async fn add_records(
        &self,
        name: &str,
        req: &AddRecordsRequest,
    ) -> Result<AddRecordsResponse, ClientError> {
        let resp = self
            .http
            .post(self.url(&format!("/contexts/{}/records", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn upsert_record(
        &self,
        name: &str,
        req: &UpsertRecordRequest,
    ) -> Result<UpsertRecordResponse, ClientError> {
        let resp = self
            .http
            .put(self.url(&format!("/contexts/{}/records", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn upsert_records(
        &self,
        name: &str,
        req: &UpsertRecordsRequest,
    ) -> Result<UpsertRecordsResponse, ClientError> {
        let resp = self
            .http
            .put(self.url(&format!("/contexts/{}/records/batch", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn update_record(
        &self,
        name: &str,
        req: &UpdateRecordRequest,
    ) -> Result<UpdateRecordResponse, ClientError> {
        let resp = self
            .http
            .patch(self.url(&format!("/contexts/{}/records", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn get_record(&self, name: &str, id: &str) -> Result<GetRecordResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/contexts/{}/records/{}", name, id)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn get_record_by_external_id(
        &self,
        name: &str,
        external_id: &str,
    ) -> Result<GetRecordResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/contexts/{}/records/by-external-id", name)))
            .query(&[("external_id", external_id)])
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    /// Resolve a record's external payload reference to its raw bytes via the
    /// server, which fetches from object storage using the context's
    /// `storage_options`. Returns the raw payload bytes on success.
    pub async fn fetch_record_payload(&self, name: &str, id: &str) -> Result<Vec<u8>, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/contexts/{}/records/{}/payload", name, id)))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(resp.bytes().await?.to_vec())
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    pub async fn delete_record(
        &self,
        name: &str,
        id: &str,
    ) -> Result<DeleteRecordResponse, ClientError> {
        let resp = self
            .http
            .delete(self.url(&format!("/contexts/{}/records/{}", name, id)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn delete_record_by_external_id(
        &self,
        name: &str,
        external_id: &str,
    ) -> Result<DeleteRecordResponse, ClientError> {
        let resp = self
            .http
            .delete(self.url(&format!("/contexts/{}/records", name)))
            .query(&[("external_id", external_id)])
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn list_records(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<&str>,
        include_expired: bool,
        include_retired: bool,
    ) -> Result<ListRecordsResponse, ClientError> {
        let mut request = self
            .http
            .get(self.url(&format!("/contexts/{}/records", name)));
        if let Some(limit) = limit {
            request = request.query(&[("limit", limit)]);
        }
        if let Some(offset) = offset {
            request = request.query(&[("offset", offset)]);
        }
        if let Some(filters) = filters {
            request = request.query(&[("filters", filters)]);
        }
        if include_expired {
            request = request.query(&[("include_expired", include_expired)]);
        }
        if include_retired {
            request = request.query(&[("include_retired", include_retired)]);
        }

        let resp = request.send().await?;
        Self::handle_response(resp).await
    }

    pub async fn related_records(
        &self,
        name: &str,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        include_expired: bool,
        include_retired: bool,
    ) -> Result<ListRecordsResponse, ClientError> {
        let mut request = self
            .http
            .get(self.url(&format!("/contexts/{}/records/related", name)))
            .query(&[("target_id", target_id)]);
        if let Some(relation) = relation {
            request = request.query(&[("relation", relation)]);
        }
        if let Some(limit) = limit {
            request = request.query(&[("limit", limit)]);
        }
        if include_expired {
            request = request.query(&[("include_expired", include_expired)]);
        }
        if include_retired {
            request = request.query(&[("include_retired", include_retired)]);
        }

        let resp = request.send().await?;
        Self::handle_response(resp).await
    }

    pub async fn search(
        &self,
        name: &str,
        req: &SearchRequest,
    ) -> Result<SearchResponse, ClientError> {
        let resp = self
            .http
            .post(self.url(&format!("/contexts/{}/search", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn retrieve(
        &self,
        name: &str,
        req: &RetrieveRequest,
    ) -> Result<RetrieveResponse, ClientError> {
        let resp = self
            .http
            .post(self.url(&format!("/contexts/{}/retrieve", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn get_version(&self, name: &str) -> Result<VersionResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/contexts/{}/version", name)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn checkout(
        &self,
        name: &str,
        req: &CheckoutRequest,
    ) -> Result<VersionResponse, ClientError> {
        let resp = self
            .http
            .post(self.url(&format!("/contexts/{}/checkout", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn compact(
        &self,
        name: &str,
        req: &CompactRequest,
    ) -> Result<CompactResponse, ClientError> {
        let resp = self
            .http
            .post(self.url(&format!("/contexts/{}/compact", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn compact_stats(&self, name: &str) -> Result<CompactStatsResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/contexts/{}/compact/stats", name)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn create_rollout_store(
        &self,
        req: &CreateRolloutStoreRequest,
    ) -> Result<RolloutStoreInfo, ClientError> {
        let resp = self
            .http
            .post(self.url("/rollouts"))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn list_rollout_stores(&self) -> Result<ListRolloutStoresResponse, ClientError> {
        let resp = self.http.get(self.url("/rollouts")).send().await?;
        Self::handle_response(resp).await
    }

    pub async fn get_rollout_store(&self, name: &str) -> Result<RolloutStoreInfo, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/rollouts/{}", name)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn delete_rollout_store(&self, name: &str) -> Result<(), ClientError> {
        let resp = self
            .http
            .delete(self.url(&format!("/rollouts/{}", name)))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    /// Append rollout rows. When any record carries `binary_payload`, the request
    /// is sent as `multipart/form-data`: the first part, `metadata`, holds the
    /// records array with each `binary_payload` stripped to null; each record that
    /// carries a blob then contributes one raw binary part named for that record's
    /// zero-based index in the metadata array (`"0"`, `"1"`, ...). Naming by index
    /// keeps part names round-trip safe (record ids may contain arbitrary bytes and
    /// are not unique). The `metadata` part is sent first so the server can parse
    /// the manifest before matching binary parts. When no record carries bytes, a
    /// plain JSON body is sent instead.
    pub async fn add_rollouts(
        &self,
        name: &str,
        records: &[AddRolloutRequest],
    ) -> Result<AddRolloutsResponse, ClientError> {
        let url = self.url(&format!("/rollouts/{}/records", name));
        let has_blob = records.iter().any(|r| r.binary_payload.is_some());

        let resp = if has_blob {
            let stripped: Vec<AddRolloutRequest> = records
                .iter()
                .map(|r| {
                    let mut without_bytes = r.clone();
                    without_bytes.binary_payload = None;
                    without_bytes
                })
                .collect();
            let metadata = serde_json::to_string(&AddRolloutsRequest { records: stripped })?;

            // metadata must be the first part: multer parses parts sequentially and
            // the server needs the manifest before it can match binary parts by index.
            let mut form = reqwest::multipart::Form::new().text("metadata", metadata);
            for (idx, r) in records.iter().enumerate() {
                if let Some(bytes) = &r.binary_payload {
                    let part = reqwest::multipart::Part::bytes(bytes.clone())
                        .mime_str("application/octet-stream")?;
                    form = form.part(idx.to_string(), part);
                }
            }
            self.http.post(url).multipart(form).send().await?
        } else {
            let req = AddRolloutsRequest {
                records: records.to_vec(),
            };
            self.http.post(url).json(&req).send().await?
        };
        Self::handle_response(resp).await
    }

    pub async fn list_rollouts(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<ListRolloutsResponse, ClientError> {
        self.list_rollouts_filtered(name, limit, offset, None).await
    }

    pub async fn list_rollouts_filtered(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<&str>,
    ) -> Result<ListRolloutsResponse, ClientError> {
        let mut request = self
            .http
            .get(self.url(&format!("/rollouts/{}/records", name)));
        if let Some(limit) = limit {
            request = request.query(&[("limit", limit)]);
        }
        if let Some(offset) = offset {
            request = request.query(&[("offset", offset)]);
        }
        if let Some(filters) = filters {
            request = request.query(&[("filters", filters)]);
        }
        let resp = request.send().await?;
        Self::handle_response(resp).await
    }

    pub async fn get_rollout(
        &self,
        name: &str,
        id: &str,
    ) -> Result<GetRolloutResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/rollouts/{}/records/{}", name, id)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    /// Materialize a single artifact row's offloaded `binary_payload` bytes.
    /// Returns `None` when the row or its payload is absent (server 404).
    pub async fn fetch_rollout_blob(
        &self,
        name: &str,
        id: &str,
    ) -> Result<Option<Vec<u8>>, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/rollouts/{}/records/{}/blob", name, id)))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(Some(resp.bytes().await?.to_vec()))
        } else if resp.status().as_u16() == 404 {
            Ok(None)
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    pub async fn checkout_rollout(
        &self,
        name: &str,
        req: &CheckoutRequest,
    ) -> Result<VersionResponse, ClientError> {
        let resp = self
            .http
            .post(self.url(&format!("/rollouts/{}/checkout", name)))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    // ----------------------------------------------------------- generic

    pub async fn create_generic_store(
        &self,
        req: &CreateGenericStoreRequest,
    ) -> Result<GenericStoreInfo, ClientError> {
        let resp = self
            .http
            .post(self.url("/generic"))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn list_generic_stores(&self) -> Result<ListGenericStoresResponse, ClientError> {
        let resp = self.http.get(self.url("/generic")).send().await?;
        Self::handle_response(resp).await
    }

    pub async fn get_generic_store(&self, name: &str) -> Result<GenericStoreInfo, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/generic/{name}")))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn delete_generic_store(&self, name: &str) -> Result<(), ClientError> {
        let resp = self
            .http
            .delete(self.url(&format!("/generic/{name}")))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    /// Append rows. They are validated server-side against the store's schema.
    pub async fn add_rows(
        &self,
        name: &str,
        rows: &[Map<String, Value>],
    ) -> Result<AddRowsResponse, ClientError> {
        let req = AddRowsRequest {
            rows: rows.to_vec(),
        };
        let resp = self
            .http
            .post(self.url(&format!("/generic/{name}/rows")))
            .json(&req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    /// List rows. Blob columns are projected out server-side.
    pub async fn list_rows(
        &self,
        name: &str,
        filter: Option<&str>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<ListRowsResponse, ClientError> {
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(filter) = filter {
            query.push(("filter".to_string(), filter.to_string()));
        }
        if let Some(limit) = limit {
            query.push(("limit".to_string(), limit.to_string()));
        }
        if let Some(offset) = offset {
            query.push(("offset".to_string(), offset.to_string()));
        }
        let resp = self
            .http
            .get(self.url(&format!("/generic/{name}/rows")))
            .query(&query)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    /// Fetch one row. `columns` selects what to read; `None` omits blob
    /// columns. Name a blob column explicitly to fetch its bytes.
    pub async fn get_row(
        &self,
        name: &str,
        id: &str,
        columns: Option<&[String]>,
    ) -> Result<Option<Map<String, Value>>, ClientError> {
        let mut request = self
            .http
            .get(self.url(&format!("/generic/{name}/rows/{id}")));
        if let Some(columns) = columns {
            request = request.query(&[("columns", columns.join(","))]);
        }
        let resp = request.send().await?;
        if resp.status() == 404 {
            return Ok(None);
        }
        Self::handle_response(resp).await.map(Some)
    }

    /// Seal the store's active memtable so added rows become readable.
    pub async fn flush_generic_store(&self, name: &str) -> Result<(), ClientError> {
        let resp = self
            .http
            .post(self.url(&format!("/generic/{name}/flush")))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    pub async fn create_datagen_store(
        &self,
        req: &CreateDatagenStoreRequest,
    ) -> Result<DatagenStoreInfo, ClientError> {
        let resp = self
            .http
            .post(self.url("/datagen"))
            .json(req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn list_datagen_stores(&self) -> Result<ListDatagenStoresResponse, ClientError> {
        let resp = self.http.get(self.url("/datagen")).send().await?;
        Self::handle_response(resp).await
    }

    pub async fn get_datagen_store(&self, name: &str) -> Result<DatagenStoreInfo, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/datagen/{}", name)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn delete_datagen_store(&self, name: &str) -> Result<(), ClientError> {
        let resp = self
            .http
            .delete(self.url(&format!("/datagen/{}", name)))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    /// Append datagen events. `checkpoint = true` commits the batch as one atomic
    /// step boundary (`append_checkpoint`); `false` appends a raw generation.
    /// FIELD blobs are offloaded to a content-addressed artifact store before the
    /// event reaches the log, so events are small JSON and travel inline.
    pub async fn add_datagen_events(
        &self,
        name: &str,
        events: &[DatagenEventDto],
        checkpoint: bool,
    ) -> Result<AddDatagenEventsResponse, ClientError> {
        let req = AddDatagenEventsRequest {
            events: events.to_vec(),
        };
        let resp = self
            .http
            .post(self.url(&format!("/datagen/{}/events", name)))
            .query(&[("checkpoint", checkpoint)])
            .json(&req)
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn fold_datagen_item(
        &self,
        name: &str,
        item_id: &str,
    ) -> Result<GetFoldedDatagenItemResponse, ClientError> {
        self.fold_datagen_item_with_blobs(name, item_id, false)
            .await
    }

    /// Fold an item, choosing the blob projection: `load_blobs` materializes blob bytes inline
    /// instead of leaving them to a later `get_blob`.
    pub async fn fold_datagen_item_with_blobs(
        &self,
        name: &str,
        item_id: &str,
        load_blobs: bool,
    ) -> Result<GetFoldedDatagenItemResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/datagen/{}/items/{}", name, item_id)))
            .query(&[("load_blobs", load_blobs)])
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn datagen_item_failures(
        &self,
        name: &str,
        item_id: &str,
    ) -> Result<ListDatagenFailuresResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/datagen/{}/items/{}/failures", name, item_id)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    /// Aggregate the whole datagen log into a run overview.
    pub async fn datagen_overview(&self, name: &str) -> Result<DatagenRunOverviewDto, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/datagen/{}/overview", name)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    pub async fn datagen_root_item_statuses(
        &self,
        name: &str,
        root_item_ids: &[String],
    ) -> Result<DatagenRootItemStatusesResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/datagen/{}/root-status", name)))
            .query(&[("ids", root_item_ids.join(","))])
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    /// Fetch every raw event whose root item is `root_item_id`. The client
    /// folds these into a tree via `DatagenItemTree::build`; the server does no
    /// fold/tree work.
    pub async fn datagen_events_for_root(
        &self,
        name: &str,
        root_item_id: &str,
    ) -> Result<ListDatagenEventsResponse, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/datagen/{}/roots/{}/events", name, root_item_id)))
            .send()
            .await?;
        Self::handle_response(resp).await
    }

    /// Materialize one FIELD_* event's offloaded blob bytes by event id.
    /// Returns `None` when the event or its payload is absent (server 404).
    pub async fn fetch_datagen_blob(
        &self,
        name: &str,
        event_id: &str,
    ) -> Result<Option<Vec<u8>>, ClientError> {
        let resp = self
            .http
            .get(self.url(&format!("/datagen/{}/blobs/{}", name, event_id)))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(Some(resp.bytes().await?.to_vec()))
        } else if resp.status().as_u16() == 404 {
            Ok(None)
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    async fn handle_response<T: serde::de::DeserializeOwned>(
        resp: reqwest::Response,
    ) -> Result<T, ClientError> {
        if resp.status().is_success() {
            Ok(resp.json::<T>().await?)
        } else {
            Err(Self::extract_error(resp).await)
        }
    }

    async fn extract_error(resp: reqwest::Response) -> ClientError {
        let status = resp.status().as_u16();
        match resp.json::<ErrorResponse>().await {
            Ok(err_resp) => ClientError::Api {
                status,
                code: err_resp.error.code,
                message: err_resp.error.message,
            },
            Err(_) => ClientError::Api {
                status,
                code: "UNKNOWN".to_string(),
                message: "Failed to parse error response".to_string(),
            },
        }
    }
}
