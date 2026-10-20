//! Durable admission for worker count/timer triggers. A native handoff may
//! close this gate, but closure alone never proves old writers are quiescent.
use crate::{
    encode, execution_key, execution_owner, target_lock_key, ClaimProof, Coordinator, Execution,
    Phase, Result,
};
use etcd_client::{Compare, CompareOp, TxnOp};

/// No lease: losing a controller cannot reopen background admission.
pub fn admission_key(prefix: &str, target: &str) -> String {
    execution_key(prefix, target).replace("/merge-executions/", "/background-merge-closed/")
}

impl Coordinator {
    /// Reserve before polling any storage work. The persistent request is in
    /// the same transaction so a crash cannot hide an admitted execution from
    /// the master. False means contention, a closed gate, or a retry deadline;
    /// it never authorizes an untracked fallback.
    pub async fn reserve_background(&self, execution: &Execution) -> Result<bool> {
        if !execution.background
            || execution.maintenance.is_some()
            || execution.phase != Phase::Reserved
            || execution.protocol != 2
            || execution.timeout_secs == 0
            || execution.queue_timeout_secs == 0
            || execution.idle_timeout_secs == 0
            || execution.target.is_empty()
            || execution.target.starts_with("datagen:")
            || execution.instance.is_empty()
            || uuid::Uuid::parse_str(&execution.id).is_err()
        {
            return Err("invalid background merge reservation".into());
        }
        let key = execution_key(&self.prefix, &execution.target);
        let failure_key = self.failure_key(&execution.target, &execution.endpoint);
        let prior = self
            .client
            .clone()
            .get(failure_key.clone(), None)
            .await
            .map_err(|e| e.to_string())?;
        let failure = prior.kvs().first();
        if let Some(row) = failure {
            let previous: crate::failure::ShardFailure =
                serde_json::from_slice(row.value()).map_err(|e| e.to_string())?;
            if previous.corrected().next_retry_ms > crate::failure::now_ms() {
                return Ok(false);
            }
        }
        self.transact(
            vec![
                Compare::version(key.as_str(), CompareOp::Equal, 0),
                Compare::version(
                    target_lock_key(&self.prefix, &execution.target),
                    CompareOp::Equal,
                    0,
                ),
                Compare::version(
                    admission_key(&self.prefix, &execution.target),
                    CompareOp::Equal,
                    0,
                ),
                Compare::version(
                    key.replace("/merge-executions/", "/catchup-active/"),
                    CompareOp::Equal,
                    0,
                ),
                Compare::version(
                    key.replace("/merge-executions/", "/merge-claims/"),
                    CompareOp::Equal,
                    0,
                ),
                Compare::version(
                    key.replace("/merge-executions/", "/compact-commit-ready/"),
                    CompareOp::Equal,
                    0,
                ),
                Compare::mod_revision(
                    failure_key,
                    CompareOp::Equal,
                    failure.map_or(0, |kv| kv.mod_revision()),
                ),
            ],
            vec![
                TxnOp::put(key, encode(execution), None),
                TxnOp::put(
                    target_lock_key(&self.prefix, &execution.target),
                    execution_owner(execution),
                    None,
                ),
                TxnOp::put(
                    self.request_key(&execution.target),
                    execution.target.as_str(),
                    None,
                ),
            ],
        )
        .await
    }

    /// Release only after local work and all commit leaves have drained, or a
    /// storage barrier has completed. A stale worker cannot remove a successor.
    /// The ordinary master release method can also reconcile these records.
    pub async fn release_background(&self, terminal: &Execution) -> Result<bool> {
        if !terminal.background
            || terminal.protocol != 2
            || !matches!(terminal.phase, Phase::Finished | Phase::Recovered)
        {
            return Err("background merge is not durably terminal".into());
        }
        let (mut compares, mut operations) = self.completion_changes(terminal).await?;
        compares.extend([
            // A scheduler may already be reconciling this execution. Leave
            // its terminal record intact so ordinary release can transfer the
            // write lock to that claim, rather than deleting its predecessor.
            Compare::version(
                execution_key(&self.prefix, &terminal.target)
                    .replace("/merge-executions/", "/merge-claims/"),
                CompareOp::Equal,
                0,
            ),
            Compare::value(
                execution_key(&self.prefix, &terminal.target),
                CompareOp::Equal,
                encode(terminal),
            ),
            Compare::value(
                target_lock_key(&self.prefix, &terminal.target),
                CompareOp::Equal,
                execution_owner(terminal),
            ),
        ]);
        operations.extend([
            self.remove_permits(terminal),
            self.remove_progress(terminal),
            self.remove_work_progress(terminal),
            TxnOp::delete(execution_key(&self.prefix, &terminal.target), None),
            TxnOp::delete(target_lock_key(&self.prefix, &terminal.target), None),
        ]);
        self.transact(compares, operations).await
    }

    /// The canonical scheduler claim serializes this with background reserve.
    /// This prevents NEW background work; it does not authorize native writes
    /// or certify coverage of historical untracked writers.
    pub async fn close_background_admission(
        &self,
        proof: &ClaimProof,
        target: &str,
    ) -> Result<bool> {
        self.transact(
            vec![
                Compare::value(proof.key.as_str(), CompareOp::Equal, proof.token.as_bytes()),
                Compare::value(
                    target_lock_key(&self.prefix, target),
                    CompareOp::Equal,
                    proof.token.as_bytes(),
                ),
                Compare::version(execution_key(&self.prefix, target), CompareOp::Equal, 0),
            ],
            vec![TxnOp::put(
                admission_key(&self.prefix, target),
                "closed",
                None,
            )],
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use etcd_client::{Client, Txn};

    async fn fixture() -> (Coordinator, Client, String) {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").expect("isolated test etcd required");
        let client = Client::connect([endpoint], None).await.unwrap();
        let prefix = format!("/background-admission-test/{}", uuid::Uuid::new_v4());
        (Coordinator::new(client.clone(), &prefix), client, prefix)
    }

    fn execution() -> Execution {
        let mut execution = Execution::new("hot", "http://worker", "worker-boot", 60);
        execution.background = true;
        execution
    }

    #[test]
    fn old_readers_cannot_cas_background_records() {
        let background = execution();
        let bytes = encode(&background);
        assert_eq!(
            serde_json::from_slice::<Execution>(&bytes).unwrap(),
            background
        );
        let mut old_reader = background.clone();
        old_reader.background = false;
        assert_ne!(encode(&old_reader), bytes);
        assert!(!String::from_utf8(encode(&old_reader))
            .unwrap()
            .contains("background"));
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn concurrent_triggers_admit_one_durable_writer_and_request() {
        let (coordinator, _, _) = fixture().await;
        let mut tasks = Vec::new();
        for _ in 0..20 {
            let c = coordinator.clone();
            tasks.push(tokio::spawn(async move {
                let execution = execution();
                c.reserve_background(&execution)
                    .await
                    .unwrap()
                    .then_some(execution)
            }));
        }
        let mut admitted = Vec::new();
        for task in tasks {
            admitted.extend(task.await.unwrap());
        }
        assert_eq!(admitted.len(), 1);
        assert_eq!(
            coordinator.get("hot").await.unwrap(),
            Some(admitted[0].clone())
        );
        let (requests, next) = coordinator.request_page(None).await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].target, "hot");
        assert!(next.is_none());
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn closing_race_never_forgets_a_writer_and_survives_controller_loss() {
        let (coordinator, mut client, prefix) = fixture().await;
        let proof = ClaimProof {
            key: format!("{prefix}/claim"),
            token: "master".into(),
            lease_id: 0,
        };
        client
            .put(proof.key.as_str(), proof.token.as_str(), None)
            .await
            .unwrap();
        let background = execution();
        let master_claim = client.txn(
            Txn::new()
                .when([Compare::version(
                    target_lock_key(&prefix, "hot"),
                    CompareOp::Equal,
                    0,
                )])
                .and_then([TxnOp::put(
                    target_lock_key(&prefix, "hot"),
                    proof.token.as_str(),
                    None,
                )]),
        );
        let (worker, master) =
            tokio::join!(coordinator.reserve_background(&background), master_claim);
        let worker = worker.unwrap();
        assert_ne!(worker, master.unwrap().succeeded());
        if worker {
            assert!(!coordinator
                .close_background_admission(&proof, "hot")
                .await
                .unwrap());
            assert!(coordinator.release_background(&background).await.is_err());
            let running = coordinator.start(&background).await.unwrap().unwrap();
            assert!(coordinator.finish(&running, Ok(3)).await.unwrap());
            assert!(coordinator
                .release_background(&running.finished(Ok(3)))
                .await
                .unwrap());
            client
                .put(target_lock_key(&prefix, "hot"), proof.token.as_str(), None)
                .await
                .unwrap();
        }
        assert!(coordinator
            .close_background_admission(&proof, "hot")
            .await
            .unwrap());
        // Model expired controller/task ownership; only its ephemeral state
        // disappears. The separate gate remains closed across a fresh client.
        client.delete(proof.key, None).await.unwrap();
        client
            .delete(target_lock_key(&prefix, "hot"), None)
            .await
            .unwrap();
        let restarted = Coordinator::new(client, prefix);
        assert!(!restarted.reserve_background(&execution()).await.unwrap());
        assert!(restarted.get("hot").await.unwrap().is_none());
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn uncertain_writer_requires_fencing_and_cannot_release_a_successor() {
        let (coordinator, mut client, prefix) = fixture().await;
        let background = execution();
        assert!(coordinator.reserve_background(&background).await.unwrap());
        let running = coordinator.start(&background).await.unwrap().unwrap();
        coordinator
            .authorize_commit(&running, "memory://dataset", "base", 7)
            .await
            .unwrap();
        assert!(coordinator
            .report_uncertain(&running, "lost PUT response".into())
            .await
            .unwrap());
        let uncertain = coordinator.get("hot").await.unwrap().unwrap();
        assert!(coordinator.release_background(&uncertain).await.is_err());
        assert!(!coordinator.reserve_background(&execution()).await.unwrap());
        let proof = ClaimProof {
            key: format!("{prefix}/claim"),
            token: "master".into(),
            lease_id: 0,
        };
        client
            .put(proof.key.as_str(), proof.token.as_str(), None)
            .await
            .unwrap();
        let frozen = coordinator
            .freeze(&proof, &uncertain)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            coordinator.watermarks(&frozen).await.unwrap().versions["base"],
            7
        );
        assert!(coordinator
            .authorize_commit(&running, "memory://dataset", "base", 8)
            .await
            .is_err());
        // Only exercise metadata acknowledgement here; real object-store
        // barrier behavior is covered by the core/master storage tests.
        assert!(coordinator.finish_recovery(&proof, &frozen).await.unwrap());
        let recovered = coordinator.get("hot").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &recovered).await.unwrap());
        let successor = Execution::new("hot", "http://successor", "next-boot", 60);
        assert!(coordinator.reserve(&proof, &successor).await.unwrap());
        assert!(!coordinator.release_background(&recovered).await.unwrap());
        assert_eq!(coordinator.get("hot").await.unwrap(), Some(successor));
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn terminal_background_writer_transfers_to_a_waiting_master_claim() {
        let (coordinator, mut client, prefix) = fixture().await;
        let background = execution();
        assert!(coordinator.reserve_background(&background).await.unwrap());
        let running = coordinator.start(&background).await.unwrap().unwrap();
        let proof = ClaimProof {
            key: format!("{prefix}/claim"),
            token: "master".into(),
            lease_id: 0,
        };
        client
            .put(proof.key.as_str(), proof.token.as_str(), None)
            .await
            .unwrap();
        client
            .put(
                execution_key(&prefix, "hot").replace("/merge-executions/", "/merge-claims/"),
                proof.token.as_str(),
                None,
            )
            .await
            .unwrap();
        assert!(coordinator.finish(&running, Ok(4)).await.unwrap());
        let terminal = running.finished(Ok(4));
        assert!(!coordinator.release_background(&terminal).await.unwrap());
        assert_eq!(
            coordinator.get("hot").await.unwrap(),
            Some(terminal.clone())
        );
        assert!(coordinator.release(&proof, &terminal).await.unwrap());
        let next = Execution::new("hot", "http://next", "next-boot", 60);
        assert!(coordinator.reserve(&proof, &next).await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn background_failures_preserve_retry_budget_and_external_admission() {
        let (coordinator, mut client, prefix) = fixture().await;
        let key = execution_key(&prefix, "hot");
        for reserved in [
            target_lock_key(&prefix, "hot"),
            key.replace("/merge-executions/", "/catchup-active/"),
            key.replace("/merge-executions/", "/merge-claims/"),
            key.replace("/merge-executions/", "/compact-commit-ready/"),
        ] {
            client.put(reserved.as_str(), "other", None).await.unwrap();
            assert!(!coordinator.reserve_background(&execution()).await.unwrap());
            client.delete(reserved, None).await.unwrap();
        }
        let background = execution();
        assert!(coordinator.reserve_background(&background).await.unwrap());
        let running = coordinator.start(&background).await.unwrap().unwrap();
        let error = Err("storage timeout".into());
        assert!(coordinator.finish(&running, error.clone()).await.unwrap());
        assert!(coordinator
            .release_background(&running.finished(error))
            .await
            .unwrap());
        assert!(!coordinator.reserve_background(&execution()).await.unwrap());
        assert_eq!(
            coordinator
                .failure("hot", "http://worker")
                .await
                .unwrap()
                .unwrap()
                .consecutive_attempts,
            1
        );
    }
}
