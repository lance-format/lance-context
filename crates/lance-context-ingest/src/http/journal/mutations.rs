//! Mutation transport over the same journal, lane fencing and exact-intent
//! state machine as immutable ingestion. An ACK includes Applied/Rejected;
//! it is not a selected Lance publication or an external source checkpoint.
use super::*;
use crate::{
    annotations::Shard, http::annotations::MutationReply, turn_ingestion::TurnIngestionSpec,
};
use lance_context_core::annotation_staging::Limits;

pub struct DurableMutationClient {
    inner: DurableClient,
}

impl DurableMutationClient {
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
                .resume_locked(key, &mut lane)
                .await?
                .context("staged mutation disappeared")?,
        )
    }

    pub async fn resume(&self, key: usize) -> Result<Option<MutationReply>> {
        let mut lane = self.inner.lane(key)?.lock().await;
        self.inner
            .resume_locked(key, &mut lane)
            .await?
            .map(Self::mutation)
            .transpose()
    }

    fn mutation(value: Acknowledged) -> Result<MutationReply> {
        match value {
            Acknowledged::Mutation(result) => Ok(result),
            Acknowledged::Append(_) => anyhow::bail!("immutable result on mutation client"),
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
