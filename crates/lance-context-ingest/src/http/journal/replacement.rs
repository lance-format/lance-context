//! Small scheduler-owned control intents, independent of an executor's journal.
use super::*;
use crate::{
    annotations::Shard,
    http::annotations::{ReplaceProducerReply, ReplaceProducerRequest},
    turn_ingestion::TurnIngestionSpec,
};
use lance_context_core::annotation_staging::Limits;

/// Persist the entire value in the scheduler before the first execute call.
/// Retrying preparation after replacement succeeds cannot recover the old
/// credential: retry execution with this saved intent instead.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplacementIntent {
    scope: [u8; 32],
    successor: ProducerOpen,
    request: ReplaceProducerRequest,
}

/// Uses the canonical managed-turn scope without opening a payload journal.
/// Install behind the same authorized table boundary as mutation requests.
pub struct ProducerControlClient {
    base: reqwest::Url,
    client: reqwest::Client,
    timeout: Duration,
    scope: [u8; 32],
    bucket: u32,
}

impl ProducerControlClient {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut base: reqwest::Url,
        client: reqwest::Client,
        timeout: Duration,
        layout: &Layout,
        binding: &TableBinding,
        spec: &TurnIngestionSpec,
        limits: &Limits,
        shard: Shard,
    ) -> Result<Self> {
        ensure!(
            matches!(base.scheme(), "http" | "https")
                && base.query().is_none()
                && base.fragment().is_none()
                && (Duration::from_millis(100)..=Duration::from_secs(120)).contains(&timeout)
                && !binding.incarnation.is_nil(),
            "invalid producer control configuration"
        );
        layout.group_for_bucket(shard.bucket)?;
        let scope = Scope {
            cluster: layout.cluster,
            layout: layout.fingerprint()?,
            table: binding.table,
            destination: binding.destination.clone(),
            incarnation: binding.incarnation,
            contract: shard.turn_contract(binding, spec, limits)?,
        };
        let scope = receipt::scope_hash(&scope)?;
        base.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid producer control base URL"))?
            .pop_if_empty()
            .extend(["api", "v1", "annotations", &binding.qualified_table]);
        Ok(Self {
            base,
            client,
            timeout,
            scope,
            bucket: shard.bucket,
        })
    }

    fn endpoint(&self, action: &str) -> reqwest::Url {
        let mut url = self.base.clone();
        url.path_segments_mut().unwrap().push(action);
        url
    }

    /// Recover the predecessor credential from its originally persisted open
    /// tuple. This only allocates/reobserves that predecessor; it never replaces
    /// it. Persist the returned intent before execute, and serialize scheduler
    /// ownership so only one executor can use the resulting successor journal.
    pub async fn prepare_replacement(
        &self,
        previous: &ProducerOpen,
        operation: Uuid,
    ) -> Result<ReplacementIntent> {
        ensure!(
            previous.bucket == self.bucket
                && previous.slot < producer::MAX_SLOTS
                && previous.expected_generation < u64::MAX - 1
                && !previous.operation.is_nil()
                && !operation.is_nil()
                && operation != previous.operation,
            "invalid predecessor or replacement operation"
        );
        let response = self
            .client
            .post(self.endpoint("producer"))
            .timeout(self.timeout)
            .json(previous)
            .send()
            .await?
            .error_for_status()?;
        let credential: Opened = bounded_json(response).await?;
        self.validate_credential(&credential, previous.expected_generation + 1)?;
        Ok(ReplacementIntent {
            scope: self.scope,
            successor: ProducerOpen {
                bucket: previous.bucket,
                slot: previous.slot,
                expected_generation: credential.generation,
                operation,
            },
            request: ReplaceProducerRequest {
                producer: credential.producer,
                operation,
            },
        })
    }

    /// One attempt, with no automatic retry. Waiting is explicit. On a lost or
    /// unknown response retain this exact intent. An Opened result may already
    /// have been used: it never authorizes reopening that generation at sequence
    /// one with an empty journal. Losing that executor needs a new replacement.
    /// Replacement does not resolve source obligations or authorize advancing a
    /// checkpoint; replay must use business-idempotent operations (first-create).
    pub async fn execute(&self, intent: &ReplacementIntent) -> Result<ReplaceProducerReply> {
        ensure!(
            intent.scope == self.scope
                && intent.successor.bucket == self.bucket
                && intent.successor.slot < producer::MAX_SLOTS
                && intent.successor.expected_generation > 0
                && intent.successor.expected_generation < u64::MAX
                && intent.successor.operation == intent.request.operation
                && !intent.request.operation.is_nil()
                && !intent.request.producer.is_empty()
                && intent.request.producer.len() <= MAX_TOKEN,
            "foreign or invalid replacement intent"
        );
        let response = self
            .client
            .post(self.endpoint("replace-producer"))
            .timeout(self.timeout)
            .json(&intent.request)
            .send()
            .await?;
        if response.status() == reqwest::StatusCode::CONFLICT {
            // Only a validated service decision is a definitive fence. A proxy
            // status alone is not evidence that the operation was rejected.
            let details: AppendResponseError = bounded_json(response).await?;
            if details.code == "producer_replacement_fenced"
                && details.operation_outcome == "rejected"
            {
                return Err(ProducerReplacementFenced.into());
            }
            anyhow::bail!("unresolved replacement conflict; retain original intent");
        }
        if !response.status().is_success() {
            return Err(unresolved_append(response).await);
        }
        let reply: ReplaceProducerReply = bounded_json(response).await?;
        if let ReplaceProducerReply::Opened { open, credential } = &reply {
            ensure!(*open == intent.successor, "foreign replacement allocation");
            self.validate_credential(credential, open.expected_generation + 1)?;
        }
        Ok(reply)
    }

    fn validate_credential(&self, credential: &Opened, generation: u64) -> Result<()> {
        ensure!(
            credential.scope == self.scope
                && credential.generation == generation
                && !credential.producer.is_empty()
                && credential.producer.len() <= MAX_TOKEN,
            "foreign producer credential"
        );
        Ok(())
    }
}
