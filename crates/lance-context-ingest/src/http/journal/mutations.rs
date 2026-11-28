//! Mutation transport over the same journal, lane fencing and exact-intent
//! state machine as immutable ingestion. An applied ACK includes
//! Applied/Rejected; a commit ACK only proves durable WAL commit and its
//! outcome is fetched with `wait_applied`. Neither is a selected Lance
//! publication or an external source checkpoint.
use super::*;
use crate::{
    annotations::Shard,
    http::annotations::{AppliedRequest, CommitReply, MutationReply},
    turn_ingestion::TurnIngestionSpec,
};
use lance_context_core::annotation_staging::Limits;

pub struct DurableMutationClient {
    inner: DurableClient,
}

impl DurableMutationClient {
    /// Full original-schema first-create writes, using the existing durable
    /// journal and mutation ACK protocol with the single-body schema binding.
    #[allow(clippy::too_many_arguments)]
    pub async fn open_full_body(
        config: Config,
        layout: &Layout,
        binding: TableBinding,
        spec: &TurnIngestionSpec,
        limits: &Limits,
        shard: Shard,
        client: reqwest::Client,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<Self> {
        ensure!(
            config.opens.iter().all(|open| open.bucket == shard.bucket),
            "mutation producer lane belongs to another physical shard"
        );
        let contract = crate::annotations::single_body::contract(&binding, spec, limits, shard)?;
        Ok(Self {
            inner: DurableClient::open_with_contract(
                config, layout, binding, contract, client, budget,
            )
            .await?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn open(
        config: Config,
        layout: &Layout,
        binding: TableBinding,
        spec: &TurnIngestionSpec,
        limits: &Limits,
        shard: Shard,
        client: reqwest::Client,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<Self> {
        ensure!(
            config.opens.iter().all(|open| open.bucket == shard.bucket),
            "mutation producer lane belongs to another physical shard"
        );
        let contract = shard.turn_contract(&binding, spec, limits)?;
        Ok(Self {
            inner: DurableClient::open_with_contract(
                config, layout, binding, contract, client, budget,
            )
            .await?,
        })
    }

    pub async fn stage(&self, lane: usize, body: Bytes) -> Result<u64> {
        self.inner.stage(lane, body).await
    }

    pub async fn append(&self, key: usize, body: Bytes) -> Result<MutationReply> {
        let mut lane = self.inner.lane(key)?.lock().await;
        self.inner.stage_locked(key, &mut lane, body).await?;
        Self::mutation(
            self.inner
                .resume_locked(key, &mut lane, false)
                .await?
                .context("staged mutation disappeared")?,
        )
    }

    /// Commit-ack append: persist the intent, send it, and return as soon as
    /// the owner reports a durable WAL commit. The lane lock is held only for
    /// stage + commit (one in-flight body per lane, as before) and is free for
    /// the next body immediately after. The returned reply holds no body.
    pub async fn commit(&self, key: usize, body: Bytes) -> Result<CommitReply> {
        let mut lane = self.inner.lane(key)?.lock().await;
        self.inner.stage_locked(key, &mut lane, body).await?;
        Self::committed(
            self.inner
                .resume_locked(key, &mut lane, true)
                .await?
                .context("staged mutation disappeared")?,
        )
    }

    /// Resend a recovered/staged intent in commit-ack mode.
    pub async fn resume_commit(&self, key: usize) -> Result<Option<CommitReply>> {
        let mut lane = self.inner.lane(key)?.lock().await;
        self.inner
            .resume_locked(key, &mut lane, true)
            .await?
            .map(Self::committed)
            .transpose()
    }

    /// Latest durable commit ACK persisted for this lane, if any.
    pub async fn last_committed(&self, key: usize) -> Result<Option<CommitReply>> {
        let lane = self.inner.lane(key)?.lock().await;
        ensure!(
            !lane.reopen,
            "local journal outcome unknown; reopen before continuing"
        );
        Ok(lane.saved.last_commit.clone())
    }

    /// Wait (one bounded request) for the applied outcome of a committed
    /// receipt. Takes no lane lock. `Ok(reply)` carries Applied or Rejected;
    /// an earlier rejected request of this producer returns HTTP 409
    /// `prior_request_rejected` as an `AppendResponseError`. A 503
    /// `applied_pending` means the applier has not reached the index yet.
    /// HTTP 409 `receipt_superseded` (slot reused by a newer generation) is
    /// returned as a `ReceiptSuperseded` failure, never Applied.
    pub async fn wait_applied(&self, key: usize, commit: &CommitReply) -> Result<MutationReply> {
        let open = self.inner.opens.get(key).context("client lane out of range")?;
        validate_commit(commit, &self.inner.scope, open)?;
        let response = self
            .inner
            .client
            .post(self.inner.endpoint("applied"))
            .timeout(self.inner.timeout)
            .json(&AppliedRequest {
                bucket: open.bucket,
                receipt: commit.receipt.clone(),
            })
            .send()
            .await?;
        let status = response.status();
        if status != reqwest::StatusCode::OK && status != reqwest::StatusCode::UNPROCESSABLE_ENTITY
        {
            let error = super::unresolved_append(response).await;
            // The slot was reused by a newer producer generation: this
            // receipt can never resolve. Typed so no caller retries it or
            // treats it as Applied.
            if status == reqwest::StatusCode::CONFLICT
                && error
                    .downcast_ref::<AppendResponseError>()
                    .is_some_and(|e| e.code == "receipt_superseded")
            {
                return Err(error.context(crate::annotations::ReceiptSuperseded {
                    receipt_generation: commit.meta.producer.as_u128() as u64,
                    slot_generation: 0,
                }));
            }
            return Err(error);
        }
        let result: MutationReply = super::bounded_json(response).await?;
        validate(&result, &self.inner.scope, open)?;
        ensure!(
            result.receipt == commit.receipt
                && result.completion.index == commit.index
                && result.completion.meta == commit.meta
                && (status == reqwest::StatusCode::OK)
                    == matches!(
                        result.completion.result,
                        crate::annotations::MutationResult::Applied
                    ),
            "foreign applied outcome"
        );
        Ok(result)
    }

    fn committed(value: Acknowledged) -> Result<CommitReply> {
        match value {
            Acknowledged::Commit(result) => Ok(result),
            _ => anyhow::bail!("non-commit result on commit-ack request"),
        }
    }

    pub async fn resume(&self, key: usize) -> Result<Option<MutationReply>> {
        let mut lane = self.inner.lane(key)?.lock().await;
        self.inner
            .resume_locked(key, &mut lane, false)
            .await?
            .map(Self::mutation)
            .transpose()
    }

    fn mutation(value: Acknowledged) -> Result<MutationReply> {
        match value {
            Acknowledged::Mutation(result) => Ok(result),
            _ => anyhow::bail!("non-applied result on mutation client"),
        }
    }

    pub async fn last_completed(&self, key: usize) -> Result<Option<MutationReply>> {
        let lane = self.inner.lane(key)?.lock().await;
        ensure!(
            !lane.reopen,
            "local journal outcome unknown; reopen before continuing"
        );
        Ok(lane.saved.last_mutation.clone())
    }

    pub fn statistics(&self) -> lance_context_wal::client_journal::Statistics {
        self.inner.statistics()
    }

    /// Fetch the exact managed body reference and covered WAL cut. This does
    /// not pin files or confirm any external source/association checkpoint.
    pub async fn selected_body(&self) -> Result<crate::annotations::mixed::SelectedBody> {
        let response = self
            .inner
            .client
            .post(self.inner.endpoint("selected-body"))
            .timeout(self.inner.timeout)
            .json(&super::super::annotations::SelectedBodyRequest {
                bucket: self.inner.opens[0].bucket,
            })
            .send()
            .await?
            .error_for_status()?;
        let selected: crate::annotations::mixed::SelectedBody = bounded_json(response).await?;
        let scope = &self.inner.scope;
        ensure!(
            selected.cluster == scope.cluster
                && selected.layout == scope.layout
                && selected.table == scope.table
                && selected.schema == scope.contract.schema
                && selected.source.table_instance == scope.incarnation.to_string()
                && selected.source.shard == self.inner.opens[0].bucket
                && selected.source.source_group == self.inner.groups[0]
                && selected.body.table_instance == selected.source.table_instance,
            "foreign selected body scope"
        );
        Ok(selected)
    }

    /// Exact selected full-body manifest and covered shard position. This does
    /// not grant retention, write authority, or an external checkpoint ACK.
    pub async fn selected_full_body(
        &self,
    ) -> Result<crate::annotations::single_body::SelectedBody> {
        let response = self
            .inner
            .client
            .post(self.inner.endpoint("selected-body"))
            .timeout(self.inner.timeout)
            .json(&super::super::annotations::SelectedBodyRequest {
                bucket: self.inner.opens[0].bucket,
            })
            .send()
            .await?
            .error_for_status()?;
        let selected: crate::annotations::single_body::SelectedBody =
            bounded_json(response).await?;
        let scope = &self.inner.scope;
        ensure!(
            selected.cluster == scope.cluster
                && selected.layout == scope.layout
                && selected.table == scope.table
                && selected.incarnation == scope.incarnation
                && selected.schema == scope.contract.schema
                && selected.source.table_instance == scope.incarnation.to_string()
                && selected.source.shard == self.inner.opens[0].bucket
                && selected.source.source_group == self.inner.groups[0]
                && selected.through < u64::MAX
                && !selected.uri.is_empty()
                && selected.uri.len() <= 8192
                && !selected.manifest.identity.is_nil()
                && !selected.manifest.transaction.is_nil(),
            "foreign selected full body scope"
        );
        selected
            .manifest
            .object
            .validate(selected.manifest.version)?;
        Ok(selected)
    }

    pub async fn shutdown(self) -> Result<()> {
        self.inner.shutdown().await
    }
}

pub(super) fn validate(value: &MutationReply, scope: &Scope, open: &ProducerOpen) -> Result<()> {
    let completed = &value.completion;
    let meta = &completed.meta;
    ensure!(
        completed.index > 0
            && completed.index < u64::MAX
            && meta.stream == scope.table
            && meta.bucket == open.bucket
            && meta.kind == lance_context_wal::record::RecordKind::Data
            && meta.producer == producer::producer_id(open.slot, open.expected_generation + 1)
            && meta.sequence > 0
            && meta.sequence < u64::MAX
            && meta.body_bytes > 0
            && meta.body_bytes <= scope.contract.max_record_bytes as u64
            && !value.receipt.is_empty()
            && value.receipt.len() <= MAX_TOKEN,
        "foreign mutation ACK scope"
    );
    Ok(())
}

pub(super) fn validate_commit(value: &CommitReply, scope: &Scope, open: &ProducerOpen) -> Result<()> {
    let meta = &value.meta;
    ensure!(
        value.index > 0
            && value.index < u64::MAX
            && meta.stream == scope.table
            && meta.bucket == open.bucket
            && meta.kind == lance_context_wal::record::RecordKind::Data
            && meta.producer == producer::producer_id(open.slot, open.expected_generation + 1)
            && meta.sequence > 0
            && meta.sequence < u64::MAX
            && meta.body_bytes > 0
            && meta.body_bytes <= scope.contract.max_record_bytes as u64
            && !value.receipt.is_empty()
            && value.receipt.len() <= MAX_TOKEN,
        "foreign mutation commit ACK scope"
    );
    Ok(())
}
