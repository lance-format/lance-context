//! Identity-bound retirement journal for pre-fencing worker executions.
//!
//! The master verifies process death and replacement admission before preparing
//! storage barriers. These transactions never release a target lock: the normal
//! scheduler must still acquire its claim and release the terminal execution.
use crate::{
    encode, execution_key, execution_owner, target_lock_key, Coordinator, Execution, Phase, Result,
};
use etcd_client::{Compare, CompareOp, TxnOp};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub namespace: String,
    pub pod: String,
    pub pod_uid: String,
    pub container: String,
    pub container_id: String,
    pub instance: String,
    pub writer_id: String,
}

/// A live process captured before a planned replacement, independently of
/// whether it currently has a merge execution. This is intent, not proof of
/// death, storage fencing, or permission to remove any target lock.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerRetirementIntent {
    pub id: String,
    pub endpoint: String,
    pub base_uri: String,
    pub process: ProcessIdentity,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClaimProof;

    async fn fixture() -> (Coordinator, ClaimProof, Execution, ProcessIdentity) {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let prefix = format!("/legacy-retirement-tests/{}", uuid::Uuid::new_v4());
        let lease = client.lease_grant(60, None).await.unwrap().id();
        let proof = ClaimProof {
            key: format!("{prefix}/claims/test"),
            token: "original".into(),
            lease_id: lease,
        };
        for key in [&proof.key, &target_lock_key(&prefix, "table")] {
            client
                .put(
                    key.as_str(),
                    proof.token.as_str(),
                    Some(etcd_client::PutOptions::new().with_lease(lease)),
                )
                .await
                .unwrap();
        }
        let coordinator = Coordinator::new(client, prefix);
        let execution = coordinator
            .admit_legacy(&proof, "table", "http://worker:3000")
            .await
            .unwrap();
        let process = ProcessIdentity {
            namespace: "test".into(),
            pod: "worker".into(),
            pod_uid: "uid".into(),
            container: "server".into(),
            container_id: "containerd://old".into(),
            instance: "instance-old".into(),
            writer_id: "worker".into(),
        };
        (coordinator, proof, execution, process)
    }

    fn plan() -> BTreeMap<String, u64> {
        BTreeMap::from([
            ("base".into(), 12),
            (format!("shard:{}", uuid::Uuid::nil()), 15),
        ])
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn worker_intent_without_execution_binds_late_target_without_releasing_ownership() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let prefix = format!("/worker-retirement-tests/{}", uuid::Uuid::new_v4());
        let coordinator = Coordinator::new(client.clone(), prefix.clone());
        let intent = WorkerRetirementIntent {
            id: uuid::Uuid::new_v4().to_string(),
            endpoint: "http://worker:3000".into(),
            base_uri: "az://data/root".into(),
            process: ProcessIdentity {
                namespace: "test".into(),
                pod: "worker".into(),
                pod_uid: "uid".into(),
                container: "server".into(),
                container_id: "containerd://old".into(),
                instance: "old-instance".into(),
                writer_id: "worker".into(),
            },
        };
        assert!(coordinator.get("late").await.unwrap().is_none());
        let captured = coordinator
            .prepare_worker_retirement(intent.clone())
            .await
            .unwrap();
        assert_eq!(captured, intent);
        assert_eq!(
            coordinator
                .worker_retirement_for_endpoint(&intent.endpoint)
                .await
                .unwrap(),
            Some(intent.clone())
        );
        assert_eq!(
            coordinator
                .prepare_worker_retirement(intent.clone())
                .await
                .unwrap(),
            intent
        );
        let mut altered = intent.clone();
        altered.process.container_id = "containerd://other".into();
        assert!(coordinator
            .prepare_worker_retirement(altered)
            .await
            .is_err());
        let mut second = intent.clone();
        second.id = uuid::Uuid::new_v4().to_string();
        second.process.container_id = "containerd://newer".into();
        assert!(coordinator
            .prepare_worker_retirement(second.clone())
            .await
            .is_err());
        assert!(coordinator
            .worker_retirement_intent(&second.id)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            coordinator
                .worker_retirement_for_endpoint(&intent.endpoint)
                .await
                .unwrap(),
            Some(intent.clone())
        );
        assert!(coordinator.get("late").await.unwrap().is_none());
        assert!(client
            .get(target_lock_key(&prefix, "late"), None)
            .await
            .unwrap()
            .kvs()
            .is_empty());

        let proof = ClaimProof {
            key: format!("{prefix}/claims/late"),
            token: "late-owner".into(),
            lease_id: 0,
        };
        for key in [&proof.key, &target_lock_key(&prefix, "late")] {
            client
                .put(key.as_str(), proof.token.as_str(), None)
                .await
                .unwrap();
        }
        let late = coordinator
            .admit_legacy(&proof, "late", &intent.endpoint)
            .await
            .unwrap();
        let mut invented = intent.clone();
        invented.process.instance = "not-captured".into();
        assert!(coordinator
            .prepare_legacy_retirement_from_worker(&late, &invented)
            .await
            .is_err());
        assert!(coordinator
            .legacy_retirement(&late.id)
            .await
            .unwrap()
            .is_none());
        let journal = coordinator
            .prepare_legacy_retirement_from_worker(&late, &intent)
            .await
            .unwrap();
        assert_eq!(journal.process, intent.process);
        assert!(journal.versions.is_none());
        assert!(coordinator
            .finish_legacy_retirement(&journal)
            .await
            .is_err());
        assert_eq!(coordinator.get("late").await.unwrap(), Some(late.clone()));
        let planned = coordinator
            .plan_legacy_retirement(&journal, plan())
            .await
            .unwrap();
        assert!(coordinator
            .finish_legacy_retirement(&planned)
            .await
            .unwrap());
        assert!(!client
            .get(target_lock_key(&prefix, "late"), None)
            .await
            .unwrap()
            .kvs()
            .is_empty());
        let terminal = coordinator.get("late").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &terminal).await.unwrap());
        let next = Execution::new("late", "new", "new", 600);
        assert!(coordinator.reserve(&proof, &next).await.unwrap());
        assert!(coordinator
            .prepare_legacy_retirement_from_worker(&next, &intent)
            .await
            .is_err());
        assert_eq!(coordinator.get("late").await.unwrap(), Some(next));
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn retirement_retains_lock_until_normal_claim_release_and_cannot_touch_successor() {
        let (coordinator, proof, execution, process) = fixture().await;
        let prepared = coordinator
            .prepare_legacy_retirement(&execution, process)
            .await
            .unwrap();
        assert!(coordinator
            .finish_legacy_retirement(&prepared)
            .await
            .is_err());
        assert_eq!(
            coordinator.get("table").await.unwrap(),
            Some(execution.clone())
        );
        let planned = coordinator
            .plan_legacy_retirement(&prepared, plan())
            .await
            .unwrap();
        assert!(coordinator
            .finish_legacy_retirement(&planned)
            .await
            .unwrap());
        let terminal = coordinator.get("table").await.unwrap().unwrap();
        assert_eq!(terminal.phase, Phase::Finished);
        assert!(!coordinator
            .reserve(&proof, &Execution::new("table", "new", "new", 600))
            .await
            .unwrap());
        let wrong_claim = ClaimProof {
            token: "wrong".into(),
            ..proof.clone()
        };
        assert!(!coordinator.release(&wrong_claim, &terminal).await.unwrap());
        assert!(coordinator.release(&proof, &terminal).await.unwrap());
        let next = Execution::new("table", "new", "new", 600);
        assert!(coordinator.reserve(&proof, &next).await.unwrap());
        assert!(!coordinator
            .finish_legacy_retirement(&planned)
            .await
            .unwrap());
        assert_eq!(coordinator.get("table").await.unwrap(), Some(next));
        assert!(
            coordinator
                .legacy_retirement(&execution.id)
                .await
                .unwrap()
                .unwrap()
                .completed
        );
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn restart_reuses_first_barrier_plan_and_rejects_different_process_evidence() {
        let (coordinator, _, execution, process) = fixture().await;
        let prepared = coordinator
            .prepare_legacy_retirement(&execution, process.clone())
            .await
            .unwrap();
        let planned = coordinator
            .plan_legacy_retirement(&prepared, plan())
            .await
            .unwrap();
        let mut newer = plan();
        newer.insert("base".into(), 1000);
        assert_eq!(
            coordinator
                .plan_legacy_retirement(&prepared, newer)
                .await
                .unwrap(),
            planned
        );
        let mut other = process;
        other.container_id = "containerd://another".into();
        assert!(coordinator
            .prepare_legacy_retirement(&execution, other)
            .await
            .is_err());
        assert_eq!(coordinator.get("table").await.unwrap(), Some(execution));
        assert_eq!(
            coordinator
                .legacy_retirement(&planned.execution.id)
                .await
                .unwrap(),
            Some(planned)
        );
    }
}

#[cfg(test)]
mod witness_tests {
    use super::*;
    use crate::ClaimProof;

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn late_admission_inherits_verified_process_but_not_other_target_barriers() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let prefix = format!("/retirement-witness-tests/{}", uuid::Uuid::new_v4());
        let proof = ClaimProof {
            key: format!("{prefix}/claims/test"),
            token: "owner".into(),
            lease_id: 0,
        };
        client
            .put(proof.key.as_str(), proof.token.as_str(), None)
            .await
            .unwrap();
        for target in ["first", "late"] {
            client
                .put(target_lock_key(&prefix, target), proof.token.as_str(), None)
                .await
                .unwrap();
        }
        let coordinator = Coordinator::new(client, prefix);
        let execution = coordinator
            .admit_legacy(&proof, "first", "http://worker:3000")
            .await
            .unwrap();
        let process = ProcessIdentity {
            namespace: "test".into(),
            pod: "worker".into(),
            pod_uid: "uid".into(),
            container: "server".into(),
            container_id: "containerd://old".into(),
            instance: "old".into(),
            writer_id: "worker".into(),
        };
        let prepared = coordinator
            .prepare_legacy_retirement(&execution, process.clone())
            .await
            .unwrap();
        let late = coordinator
            .admit_legacy(&proof, "late", &execution.endpoint)
            .await
            .unwrap();
        assert!(coordinator
            .prepare_legacy_retirement_from_witness(&late, &prepared)
            .await
            .is_err());
        let plan = BTreeMap::from([
            ("base".into(), 12),
            (format!("shard:{}", uuid::Uuid::nil()), 15),
        ]);
        let planned = coordinator
            .plan_legacy_retirement(&prepared, plan)
            .await
            .unwrap();
        assert!(coordinator
            .finish_legacy_retirement(&planned)
            .await
            .unwrap());
        let witness = coordinator
            .legacy_retirement(&execution.id)
            .await
            .unwrap()
            .unwrap();
        let mut invented = witness.clone();
        invented.process.container_id = "containerd://unverified".into();
        assert!(coordinator
            .prepare_legacy_retirement_from_witness(&late, &invented)
            .await
            .is_err());
        let mut other_endpoint = late.clone();
        other_endpoint.endpoint = "http://different-worker:3000".into();
        assert!(coordinator
            .prepare_legacy_retirement_from_witness(&other_endpoint, &witness)
            .await
            .is_err());
        let journal = coordinator
            .prepare_legacy_retirement_from_witness(&late, &witness)
            .await
            .unwrap();
        assert_eq!(journal.process, process);
        assert_eq!(journal.execution, late);
        assert!(journal.versions.is_none());
        assert!(!journal.completed);
        assert_eq!(coordinator.get("late").await.unwrap(), Some(late.clone()));
        assert!(coordinator
            .finish_legacy_retirement(&journal)
            .await
            .is_err());
        let new_plan = BTreeMap::from([
            ("base".into(), 90),
            (format!("shard:{}", uuid::Uuid::nil()), 91),
        ]);
        let planned_late = coordinator
            .plan_legacy_retirement(&journal, new_plan)
            .await
            .unwrap();
        assert!(coordinator
            .finish_legacy_retirement(&planned_late)
            .await
            .unwrap());
        assert_eq!(
            coordinator.get("late").await.unwrap().unwrap().phase,
            Phase::Finished
        );
        let lock = coordinator
            .client
            .clone()
            .get(target_lock_key(&coordinator.prefix, "late"), None)
            .await
            .unwrap();
        assert_eq!(lock.kvs()[0].value(), execution_owner(&late).as_bytes());
        assert_eq!(
            coordinator.legacy_retirement(&execution.id).await.unwrap(),
            Some(witness)
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Retirement {
    pub execution: Execution,
    pub process: ProcessIdentity,
    pub versions: Option<BTreeMap<String, u64>>,
    pub completed: bool,
}

fn bytes(record: &Retirement) -> Vec<u8> {
    serde_json::to_vec(record).expect("retirement is JSON serializable")
}

fn encode_intent(record: &WorkerRetirementIntent) -> Vec<u8> {
    serde_json::to_vec(record).expect("worker retirement intent is JSON serializable")
}

fn valid_process(process: &ProcessIdentity) -> bool {
    [
        &process.namespace,
        &process.pod,
        &process.pod_uid,
        &process.container,
        &process.container_id,
        &process.instance,
        &process.writer_id,
    ]
    .iter()
    .all(|s| !s.is_empty())
}

impl Coordinator {
    fn worker_retirement_endpoint_key(&self, endpoint: &str) -> String {
        let encoded: String = endpoint
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!(
            "{}/legacy-worker-retirement-endpoints/{encoded}",
            self.prefix.trim_end_matches('/')
        )
    }

    /// Exact endpoint lookup lets ordinary retries discover late admissions
    /// without scanning journals or requiring an agent to bind each target.
    pub async fn worker_retirement_for_endpoint(
        &self,
        endpoint: &str,
    ) -> Result<Option<WorkerRetirementIntent>> {
        let result = self
            .client
            .clone()
            .get(self.worker_retirement_endpoint_key(endpoint), None)
            .await
            .map_err(|e| e.to_string())?;
        let intent: Option<WorkerRetirementIntent> = result
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()?;
        if intent.as_ref().is_some_and(|i| i.endpoint != endpoint) {
            return Err("worker retirement endpoint index is inconsistent".into());
        }
        Ok(intent)
    }

    fn worker_retirement_key(&self, id: &str) -> Result<String> {
        uuid::Uuid::parse_str(id).map_err(|_| "invalid worker retirement ID".to_string())?;
        Ok(format!(
            "{}/legacy-worker-retirements/{id}",
            self.prefix.trim_end_matches('/')
        ))
    }

    pub async fn worker_retirement_intent(
        &self,
        id: &str,
    ) -> Result<Option<WorkerRetirementIntent>> {
        let result = self
            .client
            .clone()
            .get(self.worker_retirement_key(id)?, None)
            .await
            .map_err(|e| e.to_string())?;
        result
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()
    }

    /// Caller must capture the live worker through bracketed identity probes.
    /// Retrying an intent ID can never replace its original process identity.
    pub async fn prepare_worker_retirement(
        &self,
        intent: WorkerRetirementIntent,
    ) -> Result<WorkerRetirementIntent> {
        if intent.endpoint.is_empty()
            || intent.endpoint.len() > 2048
            || intent.base_uri.is_empty()
            || !valid_process(&intent.process)
        {
            return Err("invalid worker retirement intent".into());
        }
        let key = self.worker_retirement_key(&intent.id)?;
        let endpoint_key = self.worker_retirement_endpoint_key(&intent.endpoint);
        if self
            .transact(
                vec![
                    Compare::version(key.as_str(), CompareOp::Equal, 0),
                    Compare::version(endpoint_key.as_str(), CompareOp::Equal, 0),
                ],
                vec![
                    TxnOp::put(key, encode_intent(&intent), None),
                    TxnOp::put(endpoint_key, encode_intent(&intent), None),
                ],
            )
            .await?
        {
            return Ok(intent);
        }
        match self.worker_retirement_intent(&intent.id).await? {
            Some(prior)
                if prior == intent
                    && self
                        .worker_retirement_for_endpoint(&intent.endpoint)
                        .await?
                        .as_ref()
                        == Some(&intent) =>
            {
                Ok(prior)
            }
            _ => Err("worker retirement intent already identifies a different process".into()),
        }
    }

    /// After exact old-process death and fenced-successor checks, bind one
    /// target to the immutable intent. Every target still needs its own storage
    /// barrier plan and canonical completion; this does not release ownership.
    pub async fn prepare_legacy_retirement_from_worker(
        &self,
        old: &Execution,
        intent: &WorkerRetirementIntent,
    ) -> Result<Retirement> {
        if intent.endpoint != old.endpoint {
            return Err("worker intent belongs to a different endpoint".into());
        }
        self.prepare_retirement(
            old,
            intent.process.clone(),
            Some((
                self.worker_retirement_key(&intent.id)?,
                encode_intent(intent),
            )),
        )
        .await
    }

    fn retirement_key(&self, execution_id: &str) -> Result<String> {
        uuid::Uuid::parse_str(execution_id).map_err(|_| "invalid execution ID".to_string())?;
        Ok(format!(
            "{}/legacy-retirements/{execution_id}",
            self.prefix.trim_end_matches('/')
        ))
    }

    pub async fn legacy_retirement(&self, execution_id: &str) -> Result<Option<Retirement>> {
        let result = self
            .client
            .clone()
            .get(self.retirement_key(execution_id)?, None)
            .await
            .map_err(|e| e.to_string())?;
        result
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()
    }

    /// Called only after a live identity probe bracketed by Kubernetes identity
    /// checks. It records intent, NOT proof of death or permission to take over.
    pub async fn prepare_legacy_retirement(
        &self,
        old: &Execution,
        process: ProcessIdentity,
    ) -> Result<Retirement> {
        self.prepare_retirement(old, process, None).await
    }

    /// A target discovered during shutdown can use the already verified death
    /// of the same worker. The caller must recheck Kubernetes and the fenced
    /// successor. Only process identity is inherited, never storage versions.
    pub async fn prepare_legacy_retirement_from_witness(
        &self,
        old: &Execution,
        witness: &Retirement,
    ) -> Result<Retirement> {
        if !witness.completed
            || witness.versions.is_none()
            || witness.execution.id == old.id
            || witness.execution.endpoint != old.endpoint
        {
            return Err("completed retirement of the same worker is required".into());
        }
        self.prepare_retirement(
            old,
            witness.process.clone(),
            Some((self.retirement_key(&witness.execution.id)?, bytes(witness))),
        )
        .await
    }

    async fn prepare_retirement(
        &self,
        old: &Execution,
        process: ProcessIdentity,
        witness: Option<(String, Vec<u8>)>,
    ) -> Result<Retirement> {
        if old.protocol != 0
            || old.background
            || old.maintenance.is_some()
            || !matches!(old.phase, Phase::Running | Phase::Uncertain)
            || !valid_process(&process)
        {
            return Err("invalid legacy retirement preparation".into());
        }
        let next = Retirement {
            execution: old.clone(),
            process,
            versions: None,
            completed: false,
        };
        let key = self.retirement_key(&old.id)?;
        let mut comparisons = vec![
            Compare::value(
                execution_key(&self.prefix, &old.target),
                CompareOp::Equal,
                encode(old),
            ),
            Compare::value(
                target_lock_key(&self.prefix, &old.target),
                CompareOp::Equal,
                execution_owner(old),
            ),
            Compare::version(key.as_str(), CompareOp::Equal, 0),
        ];
        if let Some((key, value)) = witness {
            comparisons.push(Compare::value(key, CompareOp::Equal, value));
        }
        if self
            .transact(comparisons, vec![TxnOp::put(key, bytes(&next), None)])
            .await?
        {
            return Ok(next);
        }
        let prior = self
            .legacy_retirement(&old.id)
            .await?
            .ok_or("execution changed before retirement preparation")?;
        if prior.execution == *old && prior.process == next.process {
            Ok(prior)
        } else {
            Err("retirement identity changed; refusing replacement evidence".into())
        }
    }

    /// Persist the first post-termination snapshot before issuing any barriers.
    /// A competing recovery can adopt only the already recorded immutable plan.
    pub async fn plan_legacy_retirement(
        &self,
        old: &Retirement,
        versions: BTreeMap<String, u64>,
    ) -> Result<Retirement> {
        if old.completed
            || old.versions.is_some()
            || !versions.contains_key("base")
            || versions.len() != 2
            || versions.values().any(|v| *v == 0)
        {
            return Err("invalid legacy retirement plan".into());
        }
        let mut next = old.clone();
        next.versions = Some(versions);
        let key = self.retirement_key(&old.execution.id)?;
        if self
            .transact(
                vec![
                    Compare::value(key.as_str(), CompareOp::Equal, bytes(old)),
                    Compare::value(
                        execution_key(&self.prefix, &old.execution.target),
                        CompareOp::Equal,
                        encode(&old.execution),
                    ),
                ],
                vec![TxnOp::put(key, bytes(&next), None)],
            )
            .await?
        {
            return Ok(next);
        }
        let prior = self
            .legacy_retirement(&old.execution.id)
            .await?
            .ok_or("retirement plan disappeared")?;
        if prior.execution == old.execution
            && prior.process == old.process
            && prior.versions.is_some()
        {
            Ok(prior)
        } else {
            Err("retirement changed before barrier plan was persisted".into())
        }
    }

    /// Called only AFTER all persisted manifest versions have been fenced.
    /// Existing claim and lock remain; ordinary scheduler reconciliation owns
    /// release, retry accounting and the next attempt.
    pub async fn finish_legacy_retirement(&self, old: &Retirement) -> Result<bool> {
        if old.completed || old.versions.is_none() {
            return Err("retirement requires a persisted storage barrier plan".into());
        }
        let execution = &old.execution;
        let finished = execution.finished(Err("legacy process terminated and storage manifests fenced; retry with replacement executor".into()));
        let mut done = old.clone();
        done.completed = true;
        let key = self.retirement_key(&execution.id)?;
        self.transact(
            vec![
                Compare::value(key.as_str(), CompareOp::Equal, bytes(old)),
                Compare::value(
                    execution_key(&self.prefix, &execution.target),
                    CompareOp::Equal,
                    encode(execution),
                ),
                Compare::value(
                    target_lock_key(&self.prefix, &execution.target),
                    CompareOp::Equal,
                    execution_owner(execution),
                ),
            ],
            vec![
                TxnOp::put(key, bytes(&done), None),
                TxnOp::put(
                    execution_key(&self.prefix, &execution.target),
                    encode(&finished),
                    None,
                ),
            ],
        )
        .await
    }
}
