//! Close commit admission before fencing the finite set of already admitted
//! manifest versions. Lease expiry and process identity are not storage fences.
use crate::{encode, execution_key, ClaimProof, Coordinator, Execution, Phase, Result};
use etcd_client::{Compare, CompareOp, Txn, TxnOp, TxnOpResponse};
use lance_context_metrics::etcd::{ObserveEtcd, Operation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CommitWatermarks {
    #[serde(default)]
    pub dataset_uri: Option<String>,
    pub versions: BTreeMap<String, u64>,
}

#[derive(Default)]
struct PermitState {
    revision: i64,
    watermarks: CommitWatermarks,
}

/// One publisher per execution scope, shared by its parallel manifest writers.
/// Only permission updates are serialized; storage I/O remains outside this lock.
/// Cached watermarks never authorize a write: every call CASes the exact owner.
pub struct CommitPermitPublisher {
    coordinator: Coordinator,
    execution: Execution,
    dataset_uri: String,
    state: tokio::sync::Mutex<PermitState>,
}

impl CommitPermitPublisher {
    pub async fn authorize(&self, resource: &str, version: u64) -> Result<()> {
        let running = &self.execution;
        if running.protocol != 2 || running.phase != Phase::Running {
            return Err("merge commit requires fencing protocol 2".into());
        }
        if resource != "base"
            && resource
                .strip_prefix("shard:")
                .is_none_or(|s| uuid::Uuid::parse_str(s).is_err())
        {
            return Err("invalid merge manifest resource".into());
        }
        let owner_key = execution_key(&self.coordinator.prefix, &running.target);
        let permit_key = self.coordinator.permits_key(running);
        let owner = encode(running);
        let mut state = self.state.lock().await;
        for _ in 0..16 {
            let mut next = state.watermarks.clone();
            if next
                .dataset_uri
                .as_deref()
                .is_some_and(|uri| uri != self.dataset_uri)
            {
                return Err("merge execution cannot write multiple datasets".into());
            }
            next.dataset_uri = Some(self.dataset_uri.clone());
            let watermark = next.versions.entry(resource.into()).or_default();
            *watermark = (*watermark).max(version);
            let response = self
                .coordinator
                .client
                .clone()
                .txn(
                    Txn::new()
                        .when(vec![
                            Compare::value(owner_key.clone(), CompareOp::Equal, owner.clone()),
                            Compare::mod_revision(
                                permit_key.clone(),
                                CompareOp::Equal,
                                state.revision,
                            ),
                        ])
                        .and_then(vec![TxnOp::put(
                            permit_key.clone(),
                            serde_json::to_vec(&next).unwrap(),
                            None,
                        )])
                        .or_else(vec![
                            TxnOp::get(owner_key.clone(), None),
                            TxnOp::get(permit_key.clone(), None),
                        ]),
                )
                .observed(Operation::MergeCommitWrite)
                .await
                .map_err(|e| e.to_string())?;
            if response.succeeded() {
                // An error or cancellation before this acknowledgement leaves the
                // old revision intact. The next CAS must reconcile any late write.
                let revision = response
                    .header()
                    .ok_or("commit permit publication missing revision")?
                    .revision();
                *state = PermitState {
                    revision,
                    watermarks: next,
                };
                return Ok(());
            }
            let responses = response.op_responses();
            let [TxnOpResponse::Get(observed_owner), TxnOpResponse::Get(permits)] =
                responses.as_slice()
            else {
                return Err("invalid commit permit publication conflict response".into());
            };
            if observed_owner.kvs().first().map(|kv| kv.value()) != Some(owner.as_slice()) {
                return Err("merge commit ownership revoked for recovery".into());
            }
            let previous = permits.kvs().first();
            let watermarks = previous
                .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
                .transpose()?
                .unwrap_or_default();
            *state = PermitState {
                revision: previous.map_or(0, |kv| kv.mod_revision()),
                watermarks,
            };
        }
        Err("merge commit permission contention exceeded limit".into())
    }
}

impl Coordinator {
    pub fn commit_permit_publisher(
        &self,
        execution: &Execution,
        dataset_uri: &str,
    ) -> CommitPermitPublisher {
        CommitPermitPublisher {
            coordinator: self.clone(),
            execution: execution.clone(),
            dataset_uri: dataset_uri.into(),
            state: Default::default(),
        }
    }

    fn permits_key(&self, execution: &Execution) -> String {
        format!(
            "{}/merge-commit-permits/{}",
            self.prefix.trim_end_matches('/'),
            execution.id
        )
    }

    /// Every manifest write calls this *after selecting its immutable version*
    /// and before issuing storage I/O. A concurrent freeze either observes this
    /// permit or prevents it; there is no check-then-write ownership gap.
    pub async fn authorize_commit(
        &self,
        running: &Execution,
        dataset_uri: &str,
        resource: &str,
        version: u64,
    ) -> Result<()> {
        if running.protocol != 2 || running.phase != Phase::Running {
            return Err("merge commit requires fencing protocol 2".into());
        }
        if resource != "base"
            && resource
                .strip_prefix("shard:")
                .is_none_or(|s| uuid::Uuid::parse_str(s).is_err())
        {
            return Err("invalid merge manifest resource".into());
        }
        let key = self.permits_key(running);
        for _ in 0..16 {
            let response = self
                .client
                .clone()
                .get(key.clone(), None)
                .observed(Operation::MergeCommitRead)
                .await
                .map_err(|e| e.to_string())?;
            let previous = response.kvs().first();
            let mut watermarks: CommitWatermarks = previous
                .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
                .transpose()?
                .unwrap_or_default();
            if watermarks
                .dataset_uri
                .as_deref()
                .is_some_and(|uri| uri != dataset_uri)
            {
                return Err("merge execution cannot write multiple datasets".into());
            }
            watermarks.dataset_uri = Some(dataset_uri.into());
            let watermark = watermarks.versions.entry(resource.into()).or_default();
            *watermark = (*watermark).max(version);
            if self
                .transact_observed(
                    Operation::MergeCommitWrite,
                    vec![
                        Compare::value(
                            execution_key(&self.prefix, &running.target),
                            CompareOp::Equal,
                            encode(running),
                        ),
                        Compare::mod_revision(
                            key.clone(),
                            CompareOp::Equal,
                            previous.map_or(0, |kv| kv.mod_revision()),
                        ),
                    ],
                    vec![TxnOp::put(
                        key.clone(),
                        serde_json::to_vec(&watermarks).unwrap(),
                        None,
                    )],
                )
                .await?
            {
                return Ok(());
            }
            if self.get(&running.target).await?.as_ref() != Some(running) {
                return Err("merge commit ownership revoked for recovery".into());
            }
        }
        Err("merge commit permission contention exceeded limit".into())
    }

    pub async fn freeze(&self, proof: &ClaimProof, old: &Execution) -> Result<Option<Execution>> {
        if old.protocol != 2
            || !matches!(
                old.phase,
                Phase::Running | Phase::Uncertain | Phase::Recovering
            )
        {
            return Err("execution lacks recoverable storage fencing protocol".into());
        }
        let mut frozen = old.clone();
        frozen.phase = Phase::Recovering;
        Ok(self
            .transact(
                vec![
                    Compare::value(proof.key.as_str(), CompareOp::Equal, proof.token.as_bytes()),
                    Compare::value(
                        execution_key(&self.prefix, &old.target),
                        CompareOp::Equal,
                        encode(old),
                    ),
                ],
                vec![TxnOp::put(
                    execution_key(&self.prefix, &old.target),
                    encode(&frozen),
                    None,
                )],
            )
            .await?
            .then_some(frozen))
    }

    pub async fn watermarks(&self, frozen: &Execution) -> Result<CommitWatermarks> {
        if frozen.phase != Phase::Recovering {
            return Err("commit admission must be frozen first".into());
        }
        let response = self
            .client
            .clone()
            .get(self.permits_key(frozen), None)
            .observed(Operation::MergeCommitRead)
            .await
            .map_err(|e| e.to_string())?;
        response
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()
            .map(|v| v.unwrap_or_default())
    }

    /// Only call after storage confirms that every permitted version is fenced.
    pub async fn finish_recovery(&self, proof: &ClaimProof, frozen: &Execution) -> Result<bool> {
        if frozen.phase != Phase::Recovering {
            return Err("execution is not recovering".into());
        }
        let cause = frozen
            .error
            .as_deref()
            .unwrap_or("merge execution deadline or lost worker");
        let cause = cause
            .strip_prefix("merge ownership unresolved: manifest commit result unknown; ")
            .or_else(|| cause.strip_prefix("merge ownership unresolved: "))
            .unwrap_or(cause);
        let mut recovered = frozen.finished(Err(format!(
            "merge storage fenced for recovery; original failure: {cause}"
        )));
        recovered.phase = Phase::Recovered;
        self.transact(
            vec![
                Compare::value(proof.key.as_str(), CompareOp::Equal, proof.token.as_bytes()),
                Compare::value(
                    execution_key(&self.prefix, &frozen.target),
                    CompareOp::Equal,
                    encode(frozen),
                ),
            ],
            vec![TxnOp::put(
                execution_key(&self.prefix, &frozen.target),
                encode(&recovered),
                None,
            )],
        )
        .await
    }

    pub(crate) fn remove_permits(&self, execution: &Execution) -> TxnOp {
        TxnOp::delete(self.permits_key(execution), None)
    }
}
