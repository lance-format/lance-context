use lance_context_api::*;
use reqwest::Client;

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
            cached_version: info.version,
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
            cached_version: info.version,
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
        let mut request = self
            .http
            .get(self.url(&format!("/rollouts/{}/records", name)));
        if let Some(limit) = limit {
            request = request.query(&[("limit", limit)]);
        }
        if let Some(offset) = offset {
            request = request.query(&[("offset", offset)]);
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
