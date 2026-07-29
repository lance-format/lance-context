use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

use lance_context_api::{
    AddDatagenEventsResponse, AddRecordRequest, AddRecordsResponse, AddRolloutRequest,
    AddRolloutsResponse, AddRowsResponse, CompactRequest, CompactResponse, CompactStatsResponse,
    ContextError, ContextResult, ContextStoreApi, DatagenEventDto, DatagenFailureBucketDto,
    DatagenFailureDto, DatagenFieldStateDto, DatagenRootItemStatusesResponse,
    DatagenRunOverviewDto, DatagenStepCursorDto, DatagenStoreApi, DatagenStreamPositionDto,
    DatagenValueDto, DeleteRecordResponse, FoldedDatagenItemDto, GenericStoreApi, RecordDto,
    RecordPatchDto, RelationshipDto, RetrieveRequest, RetrieveResultDto, RolloutRecordDto,
    RolloutStoreApi, SchemaSpec, SearchRequest, SearchResultDto, StateMetadataDto,
    UpdateRecordRequest, UpdateRecordResponse, UpsertRecordRequest, UpsertRecordResponse,
    UpsertRecordsRequest, UpsertRecordsResponse, UpsertResultDto,
};

use crate::datagen::{
    DatagenBlobProjection, DatagenBlobValue, DatagenEvent, DatagenEventType, DatagenFailure,
    DatagenFieldState, DatagenItemLookup, DatagenItemStatus, DatagenRootItemStatuses,
    DatagenRunOverview, DatagenStepCursor, DatagenStepKind, DatagenStreamPosition, DatagenValue,
    FoldedDatagenItem,
};
use crate::datagen_store::DatagenStore;
use crate::generic_codec::Row;
use crate::generic_store::GenericStore;
use crate::record::{
    ContextRecord, LifecycleQueryOptions, RecordFilters, RecordPatch, Relationship, StateMetadata,
    LIFECYCLE_ACTIVE,
};
use crate::rollout::RolloutRecord;
use crate::rollout_store::{RolloutFilters, RolloutStore};
use crate::store::{CompactionConfig, ContextStore};

impl ContextStoreApi for ContextStore {
    async fn add(&mut self, records: &[AddRecordRequest]) -> ContextResult<AddRecordsResponse> {
        let run_id = Uuid::new_v4().to_string();
        let mut ids = Vec::with_capacity(records.len());
        let mut core_records = Vec::with_capacity(records.len());

        for r in records {
            let id = Uuid::new_v4().to_string();
            ids.push(id.clone());
            core_records.push(record_from_add_request(r, id, run_id.clone()));
        }

        let count = core_records.len();
        // Disambiguate: the inherent `ContextStore::add`, not this trait method.
        let version = ContextStore::add(self, &core_records)
            .await
            .map_err(to_ctx_err)?;
        Ok(AddRecordsResponse {
            version,
            ids,
            count,
        })
    }

    async fn upsert(
        &mut self,
        request: &UpsertRecordRequest,
    ) -> ContextResult<UpsertRecordResponse> {
        if request.key != "external_id" {
            return Err(ContextError::InvalidRequest(format!(
                "upsert key '{}' is not supported; use 'external_id'",
                request.key
            )));
        }
        if request
            .record
            .external_id
            .as_deref()
            .is_none_or(str::is_empty)
        {
            return Err(ContextError::InvalidRequest(
                "upsert requires record.external_id".to_string(),
            ));
        }

        let record = record_from_add_request(
            &request.record,
            Uuid::new_v4().to_string(),
            Uuid::new_v4().to_string(),
        );
        let result = ContextStore::upsert_by_external_id(self, record)
            .await
            .map_err(to_ctx_err)?;
        Ok(UpsertRecordResponse {
            version: result.version,
            inserted: result.inserted,
            replaced_id: result.replaced_id,
            record: record_to_dto(result.record),
        })
    }

    async fn upsert_many(
        &mut self,
        request: &UpsertRecordsRequest,
    ) -> ContextResult<UpsertRecordsResponse> {
        if request.key != "external_id" {
            return Err(ContextError::InvalidRequest(format!(
                "upsert key '{}' is not supported; use 'external_id'",
                request.key
            )));
        }
        if request.records.is_empty() {
            return Err(ContextError::InvalidRequest(
                "upsert_many requires at least one record".to_string(),
            ));
        }
        for (index, record) in request.records.iter().enumerate() {
            if record.external_id.as_deref().is_none_or(str::is_empty) {
                return Err(ContextError::InvalidRequest(format!(
                    "upsert_many requires record.external_id (records[{index}])"
                )));
            }
        }

        let core_records: Vec<ContextRecord> = request
            .records
            .iter()
            .map(|r| {
                record_from_add_request(r, Uuid::new_v4().to_string(), Uuid::new_v4().to_string())
            })
            .collect();

        let results = ContextStore::upsert_many_by_external_id(self, core_records)
            .await
            .map_err(to_ctx_err)?;
        let version = results
            .last()
            .map(|r| r.version)
            .unwrap_or_else(|| ContextStore::version(self));
        Ok(UpsertRecordsResponse {
            version,
            results: results
                .into_iter()
                .map(|r| UpsertResultDto {
                    inserted: r.inserted,
                    replaced_id: r.replaced_id,
                    record: record_to_dto(r.record),
                })
                .collect(),
        })
    }

    async fn update(
        &mut self,
        request: &UpdateRecordRequest,
    ) -> ContextResult<UpdateRecordResponse> {
        if request.patch.is_empty() {
            return Err(ContextError::InvalidRequest(
                "update requires at least one patch field".to_string(),
            ));
        }

        let patch = patch_from_dto(&request.patch);
        let result = match (&request.id, &request.external_id) {
            (Some(id), None) => ContextStore::update_by_id(self, id, patch).await,
            (None, Some(external_id)) => {
                ContextStore::update_by_external_id(self, external_id, patch).await
            }
            (None, None) => {
                return Err(ContextError::InvalidRequest(
                    "update requires either id or external_id".to_string(),
                ));
            }
            (Some(_), Some(_)) => {
                return Err(ContextError::InvalidRequest(
                    "update accepts only one of id or external_id".to_string(),
                ));
            }
        }
        .map_err(to_ctx_err)?;

        Ok(match result {
            Some(result) => UpdateRecordResponse {
                version: result.version,
                updated: true,
                replaced_id: Some(result.replaced_id),
                record: Some(record_to_dto(result.record)),
            },
            None => UpdateRecordResponse {
                version: ContextStore::version(self),
                updated: false,
                replaced_id: None,
                record: None,
            },
        })
    }

    async fn get(&self, id: &str) -> ContextResult<Option<RecordDto>> {
        let record = ContextStore::get(self, id).await.map_err(to_ctx_err)?;
        Ok(record.map(record_to_dto))
    }

    async fn get_by_external_id(&self, external_id: &str) -> ContextResult<Option<RecordDto>> {
        let record = ContextStore::get_by_external_id(self, external_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(record.map(record_to_dto))
    }

    async fn delete_by_id(&mut self, id: &str) -> ContextResult<DeleteRecordResponse> {
        let deleted = ContextStore::delete_by_id(self, id)
            .await
            .map_err(to_ctx_err)?;
        Ok(DeleteRecordResponse {
            deleted,
            version: ContextStore::version(self),
        })
    }

    async fn delete_by_external_id(
        &mut self,
        external_id: &str,
    ) -> ContextResult<DeleteRecordResponse> {
        let deleted = ContextStore::delete_by_external_id(self, external_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(DeleteRecordResponse {
            deleted,
            version: ContextStore::version(self),
        })
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<Value>,
        include_expired: bool,
        include_retired: bool,
    ) -> ContextResult<Vec<RecordDto>> {
        let filters = filters
            .map(RecordFilters::from_json_value)
            .transpose()
            .map_err(ContextError::InvalidRequest)?;
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        let records = ContextStore::list_filtered_with_options(
            self,
            limit,
            offset,
            filters.as_ref(),
            options,
        )
        .await
        .map_err(to_ctx_err)?;
        Ok(records.into_iter().map(record_to_dto).collect())
    }

    async fn related(
        &self,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        include_expired: bool,
        include_retired: bool,
    ) -> ContextResult<Vec<RecordDto>> {
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        let records =
            ContextStore::list_related_with_options(self, target_id, relation, limit, options)
                .await
                .map_err(to_ctx_err)?;
        Ok(records.into_iter().map(record_to_dto).collect())
    }

    async fn search(&self, request: &SearchRequest) -> ContextResult<Vec<SearchResultDto>> {
        let filters = request
            .filters
            .clone()
            .map(RecordFilters::from_json_value)
            .transpose()
            .map_err(ContextError::InvalidRequest)?;
        let options = LifecycleQueryOptions::new(request.include_expired, request.include_retired);
        let results = ContextStore::search_filtered_with_options(
            self,
            &request.query,
            Some(request.limit),
            filters.as_ref(),
            options,
        )
        .await
        .map_err(to_ctx_err)?;
        Ok(results
            .into_iter()
            .map(|mut sr| {
                if !request.include_relationships {
                    sr.record.relationships.clear();
                }
                SearchResultDto {
                    record: record_to_dto(sr.record),
                    distance: sr.distance,
                }
            })
            .collect())
    }

    async fn retrieve(&self, request: &RetrieveRequest) -> ContextResult<Vec<RetrieveResultDto>> {
        if request.fusion != "rrf" {
            return Err(ContextError::InvalidRequest(
                "retrieve fusion currently supports only 'rrf'".to_string(),
            ));
        }

        let filters = request
            .filters
            .clone()
            .map(RecordFilters::from_json_value)
            .transpose()
            .map_err(ContextError::InvalidRequest)?;
        let options = LifecycleQueryOptions::new(request.include_expired, request.include_retired);
        let results = self
            .retrieve_filtered_with_options(
                request.text.as_deref(),
                request.vector.as_deref(),
                Some(request.limit),
                filters.as_ref(),
                options,
            )
            .await
            .map_err(to_ctx_err)?;

        Ok(results
            .into_iter()
            .map(|mut result| {
                if !request.include_relationships {
                    result.record.relationships.clear();
                }
                RetrieveResultDto {
                    record: record_to_dto(result.record),
                    score: result.score,
                    vector_distance: result.vector_distance,
                    text_score: result.text_score,
                    matched_channels: result.matched_channels,
                }
            })
            .collect())
    }

    fn version(&self) -> u64 {
        ContextStore::version(self)
    }

    async fn checkout(&mut self, version: u64) -> ContextResult<()> {
        ContextStore::checkout(self, version)
            .await
            .map_err(to_ctx_err)
    }

    async fn compact(&mut self, options: Option<CompactRequest>) -> ContextResult<CompactResponse> {
        let config = options.map(|req| {
            let mut c = CompactionConfig::default();
            if let Some(v) = req.target_rows_per_fragment {
                c.target_rows_per_fragment = v;
            }
            if let Some(v) = req.materialize_deletions {
                c.materialize_deletions = v;
            }
            c
        });

        let metrics = ContextStore::compact(self, config)
            .await
            .map_err(to_ctx_err)?;
        Ok(CompactResponse {
            fragments_removed: metrics.fragments_removed,
            fragments_added: metrics.fragments_added,
            files_removed: metrics.files_removed,
            files_added: metrics.files_added,
        })
    }

    async fn compaction_stats(&self) -> ContextResult<CompactStatsResponse> {
        let stats = ContextStore::compaction_stats(self)
            .await
            .map_err(to_ctx_err)?;
        Ok(CompactStatsResponse {
            total_fragments: stats.total_fragments,
            is_compacting: stats.is_compacting,
            last_compaction: stats.last_compaction,
            last_error: stats.last_error,
            total_compactions: stats.total_compactions,
        })
    }
}

impl GenericStoreApi for GenericStore {
    fn spec(&self) -> &SchemaSpec {
        GenericStore::spec(self)
    }

    async fn add(&self, rows: &[Row]) -> ContextResult<AddRowsResponse> {
        let count = rows.len();
        let version = GenericStore::add(self, rows).await.map_err(to_ctx_err)?;
        Ok(AddRowsResponse { version, count })
    }

    async fn list(&self, limit: Option<usize>, offset: Option<usize>) -> ContextResult<Vec<Row>> {
        GenericStore::list(self, limit, offset)
            .await
            .map_err(to_ctx_err)
    }

    async fn list_filtered(
        &self,
        filter: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<Row>> {
        GenericStore::list_filtered(self, filter, limit, offset)
            .await
            .map_err(to_ctx_err)
    }

    async fn get(&self, id: &str, columns: Option<&[String]>) -> ContextResult<Option<Row>> {
        GenericStore::get(self, id, columns)
            .await
            .map_err(to_ctx_err)
    }

    async fn flush(&self) -> ContextResult<()> {
        GenericStore::flush(self).await.map_err(to_ctx_err)
    }

    fn version(&self) -> u64 {
        GenericStore::version(self)
    }
}

impl RolloutStoreApi for RolloutStore {
    async fn add(&mut self, records: &[AddRolloutRequest]) -> ContextResult<AddRolloutsResponse> {
        let mut ids = Vec::with_capacity(records.len());
        let mut core_records = Vec::with_capacity(records.len());
        for r in records {
            ids.push(r.id.clone());
            core_records.push(rollout_record_from_add_request(r));
        }

        let count = core_records.len();
        let version = RolloutStore::add(self, &core_records)
            .await
            .map_err(to_ctx_err)?;
        Ok(AddRolloutsResponse {
            version,
            ids,
            count,
        })
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<RolloutRecordDto>> {
        let records = RolloutStore::list(self, limit, offset)
            .await
            .map_err(to_ctx_err)?;
        Ok(records.into_iter().map(rollout_record_to_dto).collect())
    }

    async fn list_filtered(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<Value>,
    ) -> ContextResult<Vec<RolloutRecordDto>> {
        let filters = filters
            .map(RolloutFilters::from_json_value)
            .transpose()
            .map_err(ContextError::InvalidRequest)?;
        let records = RolloutStore::list_with_filters(self, limit, offset, filters.as_ref())
            .await
            .map_err(to_ctx_err)?;
        Ok(records.into_iter().map(rollout_record_to_dto).collect())
    }

    async fn get_trajectory(&self, rollout_id: &str) -> ContextResult<Vec<RolloutRecordDto>> {
        let records = RolloutStore::get_trajectory(self, rollout_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(records.into_iter().map(rollout_record_to_dto).collect())
    }

    async fn get(&self, id: &str) -> ContextResult<Option<RolloutRecordDto>> {
        let record = RolloutStore::get_by_id(self, id)
            .await
            .map_err(to_ctx_err)?;
        Ok(record.map(rollout_record_to_dto))
    }

    async fn get_blob(&self, id: &str) -> ContextResult<Option<Vec<u8>>> {
        RolloutStore::get_blob(self, id).await.map_err(to_ctx_err)
    }

    fn version(&self) -> u64 {
        RolloutStore::version(self)
    }

    async fn checkout(&mut self, version: u64) -> ContextResult<()> {
        RolloutStore::checkout(self, version)
            .await
            .map_err(to_ctx_err)
    }
}

pub fn rollout_record_from_add_request(r: &AddRolloutRequest) -> RolloutRecord {
    RolloutRecord {
        id: r.id.clone(),
        rollout_id: r.rollout_id.clone(),
        problem_id: r.problem_id.clone().unwrap_or_else(|| r.rollout_id.clone()),
        dataset: r.dataset.clone(),
        sequence_order: r.sequence_order,
        role: r.role.clone(),
        created_at: r.created_at.unwrap_or_else(Utc::now),
        content: r.content.clone(),
        content_type: r.content_type.clone(),
        model_input_string: r.model_input_string.clone(),
        model_output_string: r.model_output_string.clone(),
        rationale: r.rationale.clone(),
        problem_text: r.problem_text.clone(),
        user_metadata: r.user_metadata.clone(),
        input_tokens: r.input_tokens.clone(),
        output_tokens: r.output_tokens.clone(),
        num_input_tokens: r.num_input_tokens,
        num_output_tokens: r.num_output_tokens,
        output_logprobs: r.output_logprobs.clone(),
        input_logprobs: r.input_logprobs.clone(),
        ref_logprobs: r.ref_logprobs.clone(),
        loss_mask: r.loss_mask.clone(),
        advantage: r.advantage,
        reward: r.reward,
        raw_reward: r.raw_reward,
        grader_id: r.grader_id.clone(),
        score: r.score,
        include_in_training: r.include_in_training,
        exclude_reason: r.exclude_reason.clone(),
        policy_version: r.policy_version.clone(),
        relationships: r
            .relationships
            .iter()
            .cloned()
            .map(dto_to_relationship)
            .collect(),
        binary_payload: r.binary_payload.clone(),
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum.clone(),
        artifact_type: r.artifact_type.clone(),
        metadata: r.metadata.clone(),
    }
}

#[must_use]
pub fn rollout_record_to_dto(r: RolloutRecord) -> RolloutRecordDto {
    RolloutRecordDto {
        id: r.id,
        rollout_id: r.rollout_id,
        problem_id: r.problem_id,
        dataset: r.dataset,
        sequence_order: r.sequence_order,
        role: r.role,
        created_at: r.created_at,
        content: r.content,
        content_type: r.content_type,
        model_input_string: r.model_input_string,
        model_output_string: r.model_output_string,
        rationale: r.rationale,
        problem_text: r.problem_text,
        user_metadata: r.user_metadata,
        input_tokens: r.input_tokens,
        output_tokens: r.output_tokens,
        num_input_tokens: r.num_input_tokens,
        num_output_tokens: r.num_output_tokens,
        output_logprobs: r.output_logprobs,
        input_logprobs: r.input_logprobs,
        ref_logprobs: r.ref_logprobs,
        loss_mask: r.loss_mask,
        advantage: r.advantage,
        reward: r.reward,
        raw_reward: r.raw_reward,
        grader_id: r.grader_id,
        score: r.score,
        include_in_training: r.include_in_training,
        exclude_reason: r.exclude_reason,
        policy_version: r.policy_version,
        relationships: r
            .relationships
            .into_iter()
            .map(relationship_to_dto)
            .collect(),
        binary_payload: r.binary_payload,
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum,
        artifact_type: r.artifact_type,
        metadata: r.metadata,
    }
}

pub fn dto_to_relationship(r: RelationshipDto) -> Relationship {
    Relationship {
        target_id: r.target_id,
        relation: r.relation,
        weight: r.weight,
    }
}

pub fn relationship_to_dto(r: Relationship) -> RelationshipDto {
    RelationshipDto {
        target_id: r.target_id,
        relation: r.relation,
        weight: r.weight,
    }
}

pub fn patch_from_dto(patch: &RecordPatchDto) -> RecordPatch {
    RecordPatch {
        bot_id: patch.bot_id.clone(),
        session_id: patch.session_id.clone(),
        tenant: patch.tenant.clone(),
        source: patch.source.clone(),
        state_metadata: patch.state_metadata.as_ref().map(|sm| StateMetadata {
            step: sm.step,
            active_plan_id: sm.active_plan_id.clone(),
            tokens_used: sm.tokens_used,
            custom: sm.custom.clone(),
        }),
        metadata: patch.metadata.clone(),
        relationships: patch.relationships.as_ref().map(|relationships| {
            relationships
                .iter()
                .cloned()
                .map(dto_to_relationship)
                .collect()
        }),
        expires_at: patch.expires_at,
        retention_policy: patch.retention_policy.clone(),
        lifecycle_status: patch.lifecycle_status.clone(),
        retired_at: patch.retired_at,
        retired_reason: patch.retired_reason.clone(),
        embedding: patch.embedding.clone(),
        payload_uri: patch.payload_uri.clone(),
        payload_size: patch.payload_size,
        payload_checksum: patch.payload_checksum.clone(),
    }
}

pub fn record_from_add_request(r: &AddRecordRequest, id: String, run_id: String) -> ContextRecord {
    ContextRecord {
        id,
        external_id: r.external_id.clone(),
        run_id,
        bot_id: r.bot_id.clone(),
        session_id: r.session_id.clone(),
        tenant: r.tenant.clone(),
        source: r.source.clone(),
        created_at: Utc::now(),
        role: r.role.clone(),
        state_metadata: r.state_metadata.as_ref().map(|sm| StateMetadata {
            step: sm.step,
            active_plan_id: sm.active_plan_id.clone(),
            tokens_used: sm.tokens_used,
            custom: sm.custom.clone(),
        }),
        metadata: r.metadata.clone(),
        relationships: r
            .relationships
            .iter()
            .cloned()
            .map(dto_to_relationship)
            .collect(),
        expires_at: r.expires_at,
        retention_policy: r.retention_policy.clone(),
        lifecycle_status: LIFECYCLE_ACTIVE.to_string(),
        retired_at: None,
        retired_reason: None,
        supersedes_id: r.supersedes_id.clone(),
        superseded_by_id: None,
        content_type: r.content_type.clone(),
        text_payload: r.text_payload.clone(),
        binary_payload: r.binary_payload.clone(),
        payload_uri: r.payload_uri.clone(),
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum.clone(),
        embedding: r.embedding.clone(),
    }
}

pub fn record_to_dto(r: ContextRecord) -> RecordDto {
    RecordDto {
        id: r.id,
        external_id: r.external_id,
        run_id: r.run_id,
        bot_id: r.bot_id,
        session_id: r.session_id,
        tenant: r.tenant,
        source: r.source,
        created_at: r.created_at,
        role: r.role,
        content_type: r.content_type,
        text_payload: r.text_payload,
        binary_payload: r.binary_payload,
        payload_uri: r.payload_uri,
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum,
        embedding: r.embedding,
        state_metadata: r.state_metadata.map(|sm| StateMetadataDto {
            step: sm.step,
            active_plan_id: sm.active_plan_id,
            tokens_used: sm.tokens_used,
            custom: sm.custom,
        }),
        metadata: r.metadata,
        relationships: r
            .relationships
            .into_iter()
            .map(relationship_to_dto)
            .collect(),
        expires_at: r.expires_at,
        retention_policy: r.retention_policy,
        lifecycle_status: r.lifecycle_status,
        retired_at: r.retired_at,
        retired_reason: r.retired_reason,
        supersedes_id: r.supersedes_id,
        superseded_by_id: r.superseded_by_id,
    }
}

fn to_ctx_err(err: lance::Error) -> ContextError {
    let msg = err.to_string();
    if msg.contains("already in progress") {
        ContextError::CompactionInProgress
    } else if msg.contains("not found") || msg.contains("DatasetNotFound") {
        ContextError::NotFound(msg)
    } else if msg.contains("Invalid") {
        ContextError::InvalidRequest(msg)
    } else {
        ContextError::Internal(msg)
    }
}

impl DatagenStoreApi for DatagenStore {
    async fn append(
        &mut self,
        events: &[DatagenEventDto],
    ) -> ContextResult<AddDatagenEventsResponse> {
        let core = datagen_events_from_dtos(events)?;
        let count = core.len();
        let version = DatagenStore::append(self, &core)
            .await
            .map_err(to_ctx_err)?;
        Ok(AddDatagenEventsResponse { version, count })
    }

    async fn append_checkpoint(
        &mut self,
        events: &[DatagenEventDto],
    ) -> ContextResult<AddDatagenEventsResponse> {
        let core = datagen_events_from_dtos(events)?;
        let count = core.len();
        let version = DatagenStore::append_checkpoint(self, &core)
            .await
            .map_err(to_ctx_err)?;
        Ok(AddDatagenEventsResponse { version, count })
    }

    async fn fold_item(&self, item_id: &str) -> ContextResult<Option<FoldedDatagenItemDto>> {
        let lookup = DatagenStore::fold_item(self, item_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(match lookup {
            DatagenItemLookup::NeverStarted => None,
            DatagenItemLookup::Found(item) => Some(folded_item_to_dto(&item)),
        })
    }

    async fn fold_item_with_blobs(
        &self,
        item_id: &str,
        load_blobs: bool,
    ) -> ContextResult<Option<FoldedDatagenItemDto>> {
        let blobs = if load_blobs {
            DatagenBlobProjection::Eager
        } else {
            DatagenBlobProjection::Lazy
        };
        let lookup = DatagenStore::fold_item_with(self, item_id, blobs)
            .await
            .map_err(to_ctx_err)?;
        Ok(match lookup {
            DatagenItemLookup::NeverStarted => None,
            DatagenItemLookup::Found(item) => Some(folded_item_to_dto(&item)),
        })
    }

    async fn root_item_statuses(
        &self,
        root_item_ids: &[String],
    ) -> ContextResult<DatagenRootItemStatusesResponse> {
        let ids: Vec<&str> = root_item_ids.iter().map(String::as_str).collect();
        let statuses = DatagenStore::root_item_statuses(self, &ids)
            .await
            .map_err(to_ctx_err)?;
        Ok(root_item_statuses_to_dto(&statuses))
    }

    async fn overview(&self) -> ContextResult<DatagenRunOverviewDto> {
        let overview = DatagenStore::overview(self).await.map_err(to_ctx_err)?;
        Ok(run_overview_to_dto(&overview))
    }

    async fn item_failures(&self, item_id: &str) -> ContextResult<Vec<DatagenFailureDto>> {
        let failures = DatagenStore::item_failures(self, item_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(failures.iter().map(failure_to_dto).collect())
    }

    async fn events_for_root(&self, root_item_id: &str) -> ContextResult<Vec<DatagenEventDto>> {
        let events = DatagenStore::events_for_root(self, root_item_id)
            .await
            .map_err(to_ctx_err)?;
        Ok(events.iter().map(datagen_event_to_dto).collect())
    }

    async fn get_blob(&self, event_id: &str) -> ContextResult<Option<Vec<u8>>> {
        DatagenStore::get_blob(self, event_id)
            .await
            .map_err(to_ctx_err)
    }

    fn version(&self) -> u64 {
        DatagenStore::version(self)
    }
}

pub fn datagen_events_from_dtos(events: &[DatagenEventDto]) -> ContextResult<Vec<DatagenEvent>> {
    events.iter().map(datagen_event_from_dto).collect()
}

fn datagen_event_from_dto(dto: &DatagenEventDto) -> ContextResult<DatagenEvent> {
    let event_type =
        DatagenEventType::parse(&dto.event_type).map_err(ContextError::InvalidRequest)?;
    let step_kind = dto
        .step_kind
        .as_deref()
        .map(DatagenStepKind::parse)
        .transpose()
        .map_err(ContextError::InvalidRequest)?;
    let status = dto
        .status
        .as_deref()
        .map(DatagenItemStatus::parse)
        .transpose()
        .map_err(ContextError::InvalidRequest)?;
    let value = dto.value.as_ref().map(datagen_value_from_dto).transpose()?;
    Ok(DatagenEvent {
        event_id: dto.event_id.clone(),
        item_id: dto.item_id.clone(),
        root_item_id: dto.root_item_id.clone(),
        parent_item_id: dto.parent_item_id.clone(),
        item_seq: dto.item_seq,
        checkpoint_id: dto.checkpoint_id.clone(),
        event_type,
        step_name: dto.step_name.clone(),
        step_kind,
        step_index: dto.step_index,
        enclosing_step: dto.enclosing_step.clone(),
        selector_step: dto.selector_step.clone(),
        attempt: dto.attempt,
        run_id: dto.run_id.clone(),
        writer_epoch: dto.writer_epoch.clone(),
        field_name: dto.field_name.clone(),
        field_type: dto.field_type.clone(),
        codec_version: dto.codec_version,
        value,
        query_tags: dto.query_tags.clone(),
        status,
        error_type: dto.error_type.clone(),
        error_dump: dto.error_dump.clone(),
        traceback: dto.traceback.clone(),
        event_ts: dto.event_ts.unwrap_or_else(Utc::now),
        schema_version: dto.schema_version,
    })
}

fn datagen_value_from_dto(dto: &DatagenValueDto) -> ContextResult<DatagenValue> {
    let missing = || {
        ContextError::InvalidRequest(format!(
            "datagen value of kind '{}' is missing 'value'",
            dto.kind
        ))
    };
    match dto.kind.as_str() {
        "int" => Ok(DatagenValue::Int(
            dto.value
                .as_ref()
                .and_then(Value::as_i64)
                .ok_or_else(missing)?,
        )),
        "float" => Ok(DatagenValue::Float(
            dto.value
                .as_ref()
                .and_then(Value::as_f64)
                .ok_or_else(missing)?,
        )),
        "bool" => Ok(DatagenValue::Bool(
            dto.value
                .as_ref()
                .and_then(Value::as_bool)
                .ok_or_else(missing)?,
        )),
        "str" => Ok(DatagenValue::Str(
            dto.value
                .as_ref()
                .and_then(Value::as_str)
                .ok_or_else(missing)?
                .to_string(),
        )),
        "json" => Ok(DatagenValue::Json(dto.value.clone().ok_or_else(missing)?)),
        "blob" => Ok(DatagenValue::Blob(DatagenBlobValue {
            bytes: dto.bytes.clone(),
            size: dto
                .size
                .unwrap_or_else(|| dto.bytes.as_ref().map(|b| b.len() as i64).unwrap_or(0)),
            checksum: dto.checksum.clone(),
        })),
        other => Err(ContextError::InvalidRequest(format!(
            "unsupported datagen value kind '{other}'"
        ))),
    }
}

fn datagen_value_to_dto(value: &DatagenValue) -> DatagenValueDto {
    let mut dto = DatagenValueDto {
        kind: value.kind().to_string(),
        value: None,
        bytes: None,
        size: None,
        checksum: None,
    };
    match value {
        DatagenValue::Int(inner) => dto.value = Some(Value::from(*inner)),
        DatagenValue::Float(inner) => dto.value = Some(Value::from(*inner)),
        DatagenValue::Bool(inner) => dto.value = Some(Value::from(*inner)),
        DatagenValue::Str(inner) => dto.value = Some(Value::from(inner.clone())),
        DatagenValue::Json(inner) => dto.value = Some(inner.clone()),
        DatagenValue::Blob(blob) => {
            dto.bytes = blob.bytes.clone();
            dto.size = Some(blob.size);
            dto.checksum = blob.checksum.clone();
        }
    }
    dto
}

pub fn datagen_event_to_dto(event: &DatagenEvent) -> DatagenEventDto {
    DatagenEventDto {
        event_id: event.event_id.clone(),
        item_id: event.item_id.clone(),
        root_item_id: event.root_item_id.clone(),
        parent_item_id: event.parent_item_id.clone(),
        item_seq: event.item_seq,
        checkpoint_id: event.checkpoint_id.clone(),
        event_type: event.event_type.as_str().to_string(),
        step_name: event.step_name.clone(),
        step_kind: event.step_kind.map(|kind| kind.as_str().to_string()),
        step_index: event.step_index,
        enclosing_step: event.enclosing_step.clone(),
        selector_step: event.selector_step.clone(),
        attempt: event.attempt,
        run_id: event.run_id.clone(),
        writer_epoch: event.writer_epoch.clone(),
        field_name: event.field_name.clone(),
        field_type: event.field_type.clone(),
        codec_version: event.codec_version,
        value: event.value.as_ref().map(datagen_value_to_dto),
        query_tags: event.query_tags.clone(),
        status: event.status.map(|status| status.as_str().to_string()),
        error_type: event.error_type.clone(),
        error_dump: event.error_dump.clone(),
        traceback: event.traceback.clone(),
        event_ts: Some(event.event_ts),
        schema_version: event.schema_version,
    }
}

pub fn folded_item_to_dto(item: &FoldedDatagenItem) -> FoldedDatagenItemDto {
    FoldedDatagenItemDto {
        item_id: item.item_id.to_string(),
        root_item_id: item.root_item_id.to_string(),
        parent_item_id: item.parent_item_id.as_ref().map(ToString::to_string),
        status: item.status.as_str().to_string(),
        last_item_seq: item.last_item_seq,
        last_attempt: item.last_attempt,
        fields: item
            .fields
            .iter()
            .map(|(name, state)| (name.clone(), field_state_to_dto(state)))
            .collect(),
        trajectory: item.trajectory.ordered.iter().map(cursor_to_dto).collect(),
        started: position_set_to_dto(&item.trajectory.started),
        completed: position_set_to_dto(&item.trajectory.completed),
        query_tags: item.query_tags.clone(),
        blob_event_ids: item.blob_event_ids.clone(),
    }
}

fn field_state_to_dto(state: &DatagenFieldState) -> DatagenFieldStateDto {
    match state {
        DatagenFieldState::Set(value) => DatagenFieldStateDto {
            mode: "set".to_string(),
            value: Some(datagen_value_to_dto(value)),
            values: Vec::new(),
        },
        DatagenFieldState::Appended(values) => DatagenFieldStateDto {
            mode: "append".to_string(),
            value: None,
            values: values.iter().map(datagen_value_to_dto).collect(),
        },
    }
}

fn cursor_to_dto(cursor: &DatagenStepCursor) -> DatagenStepCursorDto {
    DatagenStepCursorDto {
        step_name: cursor.position.step.name.clone(),
        step_kind: cursor.position.step.kind.as_str().to_string(),
        step_index: cursor.position.index,
        enclosing_step: cursor.position.enclosing.clone(),
        selector_step: cursor.position.selector.clone(),
        item_seq: cursor.item_seq,
    }
}

fn position_to_dto(position: &DatagenStreamPosition) -> DatagenStreamPositionDto {
    DatagenStreamPositionDto {
        step_name: position.step.name.clone(),
        step_kind: position.step.kind.as_str().to_string(),
        step_index: position.index,
        enclosing_step: position.enclosing.clone(),
        selector_step: position.selector.clone(),
    }
}

/// Project a fold position set to DTOs in a deterministic order (the sets are unordered).
fn position_set_to_dto(
    positions: &std::collections::HashSet<DatagenStreamPosition>,
) -> Vec<DatagenStreamPositionDto> {
    let mut dtos: Vec<DatagenStreamPositionDto> = positions.iter().map(position_to_dto).collect();
    dtos.sort_by(|a, b| {
        (
            &a.step_name,
            a.step_index,
            &a.enclosing_step,
            &a.selector_step,
        )
            .cmp(&(
                &b.step_name,
                b.step_index,
                &b.enclosing_step,
                &b.selector_step,
            ))
    });
    dtos
}

fn root_item_statuses_to_dto(
    statuses: &DatagenRootItemStatuses,
) -> DatagenRootItemStatusesResponse {
    DatagenRootItemStatusesResponse {
        statuses: statuses
            .iter()
            .map(|(id, status)| (id.clone(), status.as_str().to_string()))
            .collect(),
    }
}

fn run_overview_to_dto(overview: &DatagenRunOverview) -> DatagenRunOverviewDto {
    DatagenRunOverviewDto {
        items: overview.items,
        running: overview.running,
        completed: overview.completed,
        filtered: overview.filtered,
        failures: overview.failures,
        failures_by_error_type: overview.failures_by_error_type.clone(),
        failures_by_run: overview
            .failures_by_run
            .iter()
            .map(|(run_id, bucket)| {
                (
                    run_id.clone(),
                    DatagenFailureBucketDto {
                        failures: bucket.failures,
                        failures_by_error_type: bucket.failures_by_error_type.clone(),
                        sample_root_item_ids: bucket.sample_root_item_ids.clone(),
                    },
                )
            })
            .collect(),
        completed_steps: overview.completed_steps.clone(),
    }
}

fn failure_to_dto(failure: &DatagenFailure) -> DatagenFailureDto {
    DatagenFailureDto {
        at: cursor_to_dto(&failure.at),
        run_id: failure.run_id.clone(),
        attempt: failure.attempt,
        error_type: failure.error.error_type.clone(),
        error_dump: failure.error.error_dump.clone(),
        traceback: failure.error.traceback.clone(),
    }
}
