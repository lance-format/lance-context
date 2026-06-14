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
