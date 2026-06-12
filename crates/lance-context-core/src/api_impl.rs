use chrono::Utc;
use uuid::Uuid;

use lance_context_api::{
    AddRecordRequest, AddRecordsResponse, CompactRequest, CompactResponse, CompactStatsResponse,
    ContextError, ContextResult, ContextStoreApi, RecordDto, RelationshipDto, RetrieveRequest,
    RetrieveResultDto, SearchResultDto, StateMetadataDto,
};

use crate::record::{
    ContextRecord, LifecycleQueryOptions, RecordFilters, Relationship, StateMetadata,
    LIFECYCLE_ACTIVE,
};
use crate::store::{CompactionConfig, ContextStore};

impl ContextStoreApi for ContextStore {
    async fn add(&mut self, records: &[AddRecordRequest]) -> ContextResult<AddRecordsResponse> {
        let run_id = Uuid::new_v4().to_string();
        let mut ids = Vec::with_capacity(records.len());
        let mut core_records = Vec::with_capacity(records.len());

        for r in records {
            let id = Uuid::new_v4().to_string();
            ids.push(id.clone());
            core_records.push(ContextRecord {
                id,
                external_id: r.external_id.clone(),
                run_id: run_id.clone(),
                bot_id: r.bot_id.clone(),
                session_id: r.session_id.clone(),
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
                embedding: r.embedding.clone(),
            });
        }

        let count = core_records.len();
        let version = self.add(&core_records).await.map_err(to_ctx_err)?;
        Ok(AddRecordsResponse {
            version,
            ids,
            count,
        })
    }

    async fn get(&self, id: &str) -> ContextResult<Option<RecordDto>> {
        let record = ContextStore::get(self, id).await.map_err(to_ctx_err)?;
        Ok(record.map(record_to_dto))
    }

    async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> ContextResult<Vec<RecordDto>> {
        let records = ContextStore::list(self, limit, offset)
            .await
            .map_err(to_ctx_err)?;
        Ok(records.into_iter().map(record_to_dto).collect())
    }

    async fn search(
        &self,
        query: &[f32],
        limit: Option<usize>,
        include_relationships: bool,
    ) -> ContextResult<Vec<SearchResultDto>> {
        let results = ContextStore::search(self, query, limit)
            .await
            .map_err(to_ctx_err)?;
        Ok(results
            .into_iter()
            .map(|mut sr| {
                if !include_relationships {
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

fn dto_to_relationship(r: RelationshipDto) -> Relationship {
    Relationship {
        target_id: r.target_id,
        relation: r.relation,
        weight: r.weight,
    }
}

fn relationship_to_dto(r: Relationship) -> RelationshipDto {
    RelationshipDto {
        target_id: r.target_id,
        relation: r.relation,
        weight: r.weight,
    }
}

fn record_to_dto(r: ContextRecord) -> RecordDto {
    RecordDto {
        id: r.id,
        external_id: r.external_id,
        run_id: r.run_id,
        bot_id: r.bot_id,
        session_id: r.session_id,
        created_at: r.created_at,
        role: r.role,
        content_type: r.content_type,
        text_payload: r.text_payload,
        binary_payload: r.binary_payload,
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
