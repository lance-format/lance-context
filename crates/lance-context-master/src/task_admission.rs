//! On-demand diagnosis only: bounded exact-key reads, no claim/recovery or writes.
use super::*;
use serde::Serialize;

const MAX_DEPENDENCIES: usize = 64;

#[derive(Debug, Serialize)]
pub(crate) struct AdmissionExplanation {
    pub task_id: String,
    pub target: String,
    pub kind: TaskKind,
    pub state: TaskState,
    pub observed_at_ms: i64,
    pub snapshot_revision: Option<i64>,
    /// This master's configured admission predicates, not slot availability,
    /// queue fairness, a guarantee the final claim CAS will win, or a fence.
    pub reason: &'static str,
    pub dependency_task_id: Option<String>,
    pub skip: Option<eligibility::Skip>,
    pub claim_shape: Option<eligibility::ClaimShape>,
    pub ownership: Option<OwnershipPresence>,
    pub orphaned_catchup_recovery: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct OwnershipPresence {
    merge_execution: bool,
    target_lock: bool,
    target_lock_matches_execution: bool,
    catchup_active: bool,
    merge_claim: bool,
    task_claim: bool,
    preparation: bool,
    prepared_commit: bool,
}

impl AdmissionExplanation {
    fn new(task: &TaskRecord, reason: &'static str) -> Self {
        Self {
            task_id: task.id.clone(),
            target: task.target.clone(),
            kind: task.kind,
            state: task.state,
            observed_at_ms: Utc::now().timestamp_millis(),
            snapshot_revision: None,
            reason,
            dependency_task_id: None,
            skip: None,
            claim_shape: None,
            ownership: None,
            orphaned_catchup_recovery: false,
        }
    }
}

impl EtcdTaskStore {
    pub(super) async fn explain_admission(
        &self,
        id: &str,
    ) -> lance::Result<Option<AdmissionExplanation>> {
        let Some(task) = self.get(id).await? else {
            return Ok(None);
        };
        let mut result = AdmissionExplanation::new(&task, "not_queued");
        if task.state != TaskState::Queued {
            return Ok(Some(result));
        }
        // Stay below etcd's transaction operation limit even for malformed or
        // historical tasks with huge dependency lists. Never report them ready.
        if task.depends_on.len() > MAX_DEPENDENCIES {
            result.reason = "dependency_limit";
            return Ok(Some(result));
        }
        let execution_key = lance_context_merge::execution_key(&self.prefix, &task.target);
        let mut keys = vec![
            execution_key.clone(),
            self.target_lock_key(&task.target),
            crate::catchup::store::active_key(&self.prefix, &task.target),
            execution_key.replace("/merge-executions/", "/merge-claims/"),
            self.claim_key(id),
            format!(
                "{}/compact-preparations/{}",
                self.prefix,
                encode_segment(&task.target)
            ),
            self.compaction_commit_key(&task.target),
            self.queue_key(id),
            self.task_key(id),
            execution_key.replace("/merge-executions/", "/catchup-records/"),
        ];
        keys.extend(task.depends_on.iter().map(|id| self.task_key(id)));
        let response = self
            .client
            .clone()
            .txn(
                Txn::new().and_then(
                    keys.iter()
                        .map(|key| TxnOp::get(key.as_str(), None))
                        .collect::<Vec<_>>(),
                ),
            )
            .await
            .map_err(etcd_error("read task diagnostic snapshot"))?;
        result.snapshot_revision = response.header().map(|h| h.revision());
        result.observed_at_ms = Utc::now().timestamp_millis();
        let values = response
            .op_responses()
            .into_iter()
            .map(|op| match op {
                TxnOpResponse::Get(value) => Ok(value.kvs().first().map(|kv| kv.value().to_vec())),
                _ => Err(lance::Error::io("invalid task diagnostic response")),
            })
            .collect::<lance::Result<Vec<_>>>()?;
        if values.len() != keys.len() {
            return Err(lance::Error::io("incomplete task diagnostic snapshot"));
        }
        // The initial read only located the keys. Every fact used below must
        // refer to the same task and the same etcd revision, including deps.
        let current = values[8]
            .as_deref()
            .map(|v| decode_task(v, &keys[8]))
            .transpose()?;
        if current.as_ref() != Some(&task) {
            result.reason = "task_changed_retry";
            return Ok(Some(result));
        }
        let queued = values[7]
            .as_deref()
            .map(|v| decode_task(v, &keys[7]))
            .transpose()?;
        if queued.as_ref() != Some(&task) {
            result.reason = "queue_entry_missing_or_changed";
            return Ok(Some(result));
        }
        if task.kind == TaskKind::MergeWal
            && !self.wal_claim_targets.is_empty()
            && !self.wal_claim_targets.contains(&task.target)
        {
            result.reason = "wal_target_not_selected";
            return Ok(Some(result));
        }
        let mut waiting = None;
        for (index, dependency) in task.depends_on.iter().enumerate() {
            let record = values[10 + index]
                .as_deref()
                .map(|v| decode_task(v, &keys[10 + index]))
                .transpose()?;
            match record {
                Some(record) if record.state == TaskState::Done => {}
                Some(record) if record.state != TaskState::Failed => {
                    waiting.get_or_insert_with(|| dependency.clone());
                }
                failed => {
                    result.reason = if failed.is_none() {
                        "dependency_missing"
                    } else {
                        "dependency_failed"
                    };
                    result.dependency_task_id = Some(dependency.clone());
                    return Ok(Some(result));
                }
            }
        }
        if let Some(dependency) = waiting {
            result.reason = "dependency_waiting";
            result.dependency_task_id = Some(dependency);
            return Ok(Some(result));
        }
        let execution: Option<lance_context_merge::Execution> = values[0]
            .as_deref()
            .map(serde_json::from_slice)
            .transpose()
            .map_err(|_| lance::Error::io("invalid merge execution metadata"))?;
        let owner = execution.as_ref().map(lance_context_merge::execution_owner);
        let shape = eligibility::claim_shape(
            task.kind,
            &task.target,
            &self.eligibility_policy(),
            values[2].is_some(),
        );
        let snapshot = eligibility::Snapshot {
            merge_execution: values[0].as_deref(),
            target_lock: values[1].as_deref(),
            catchup_active: values[2].as_deref(),
            merge_claim: values[3].as_deref(),
            task_claim: values[4].as_deref(),
            compact_preparation: values[5].as_deref(),
            compact_commit_ready: values[6].as_deref(),
        };
        result.orphaned_catchup_recovery = values[9].is_none()
            && eligibility::orphaned_catchup_candidate(
                task.kind,
                &task.target,
                &self.eligibility_policy(),
                execution.as_ref(),
                values[2].as_deref(),
            );
        result.skip = eligibility::admission_filter(
            task.kind,
            shape,
            &snapshot,
            owner.as_deref(),
            execution.as_ref().is_some_and(|e| e.maintenance.is_none()),
            if result.orphaned_catchup_recovery {
                values[2].as_deref()
            } else {
                None
            },
        )
        .err();
        result.reason = if result.skip.is_some() {
            "admission_blocked"
        } else {
            "admission_preconditions_satisfied"
        };
        result.claim_shape = Some(shape);
        result.ownership = Some(OwnershipPresence {
            merge_execution: values[0].is_some(),
            target_lock: values[1].is_some(),
            target_lock_matches_execution: owner
                .as_deref()
                .is_some_and(|owner| values[1].as_deref() == Some(owner.as_bytes())),
            catchup_active: values[2].is_some(),
            merge_claim: values[3].is_some(),
            task_claim: values[4].is_some(),
            preparation: values[5].is_some(),
            prepared_commit: values[5].is_some() && values[5] == values[6],
        });
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    async fn fixture() -> TaskStore {
        let mut cfg = MasterConfig::parse_from(["test"]);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/admission-diagnostic/{}", generate_id());
        cfg.maintenance.index_prepare_targets = vec!["hot".into()];
        cfg.maintenance.compaction_prepare_targets = vec!["hot".into()];
        cfg.maintenance.wal_claim_targets = vec!["allowed".into()];
        TaskStore::open(&cfg).await.unwrap()
    }

    async fn contents(store: &TaskStore) -> Vec<(Vec<u8>, Vec<u8>, i64, i64)> {
        store
            .inner
            .client
            .clone()
            .get(
                format!("{}/", store.inner.prefix),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .unwrap()
            .kvs()
            .iter()
            .map(|kv| {
                (
                    kv.key().to_vec(),
                    kv.value().to_vec(),
                    kv.mod_revision(),
                    kv.lease(),
                )
            })
            .collect()
    }

    async fn explain_read_only(store: &TaskStore, id: &str) -> AdmissionExplanation {
        let before = contents(store).await;
        let diagnostic = store.explain_admission(id).await.unwrap().unwrap();
        assert_eq!(
            contents(store).await,
            before,
            "diagnostic must not write, grant claims, recover or prune"
        );
        diagnostic
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn admission_diagnostic_explains_done_parent_and_live_preparation() {
        let store = fixture().await;
        let parent = store
            .enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store.finish(claim, Ok("done".into())).await.unwrap();
        store
            .enqueue(TaskKind::IndexId, "hot", vec![])
            .await
            .unwrap();
        let blocker = store
            .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
            .await
            .unwrap()
            .unwrap();
        assert!(blocker.preparing_maintenance());
        let child = store
            .enqueue(TaskKind::IndexId, "hot", vec![parent.id])
            .await
            .unwrap();
        let result = explain_read_only(&store, &child.id).await;
        assert_eq!(result.reason, "admission_blocked");
        assert_eq!(result.skip, Some(eligibility::Skip::OwnershipBlocked));
        assert!(result.snapshot_revision.unwrap() > 0);
        assert!(result.claim_shape.unwrap().preparing);
        assert!(result.ownership.as_ref().unwrap().preparation);
        assert!(!result.ownership.as_ref().unwrap().task_claim);
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains(&blocker.backend.token));
        assert!(!json.contains(&store.inner.prefix));
        assert!(store
            .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
            .await
            .unwrap()
            .is_none());
        store.finish(blocker, Ok("done".into())).await.unwrap();
        let ready = explain_read_only(&store, &child.id).await;
        assert_eq!(ready.reason, "admission_preconditions_satisfied");
        assert!(ready.skip.is_none());
        let claim = store
            .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, child.id);
        assert_eq!(
            explain_read_only(&store, &child.id).await.reason,
            "not_queued"
        );
        store.finish(claim, Ok("done".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn admission_diagnostic_dependencies_and_bounds_do_not_mutate_tasks() {
        let store = fixture().await;
        let parent = store
            .enqueue(TaskKind::Repair, "parent", vec![])
            .await
            .unwrap();
        let child = store
            .enqueue(TaskKind::IndexId, "hot", vec![parent.id.clone()])
            .await
            .unwrap();
        let waiting = explain_read_only(&store, &child.id).await;
        assert_eq!(waiting.reason, "dependency_waiting");
        assert_eq!(
            waiting.dependency_task_id.as_deref(),
            Some(parent.id.as_str())
        );
        let claim = store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .unwrap();
        store
            .finish(claim, Err("synthetic failure".into()))
            .await
            .unwrap();
        assert_eq!(
            explain_read_only(&store, &child.id).await.reason,
            "dependency_failed"
        );
        assert_eq!(
            store.get(&child.id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        let missing = store
            .enqueue(TaskKind::IndexId, "hot", vec![generate_id()])
            .await
            .unwrap();
        assert_eq!(
            explain_read_only(&store, &missing.id).await.reason,
            "dependency_missing"
        );
        let large = store
            .enqueue(
                TaskKind::IndexId,
                "hot",
                (0..65).map(|_| generate_id()).collect(),
            )
            .await
            .unwrap();
        let bounded = explain_read_only(&store, &large.id).await;
        assert_eq!(bounded.reason, "dependency_limit");
        assert!(bounded.snapshot_revision.is_none());
        let wal = store
            .enqueue(TaskKind::MergeWal, "excluded", vec![])
            .await
            .unwrap();
        assert_eq!(
            explain_read_only(&store, &wal.id).await.reason,
            "wal_target_not_selected"
        );
        store
            .inner
            .client
            .clone()
            .delete(store.inner.queue_key(&wal.id), None)
            .await
            .unwrap();
        assert_eq!(
            explain_read_only(&store, &wal.id).await.reason,
            "queue_entry_missing_or_changed"
        );
        assert!(store
            .explain_admission(&generate_id())
            .await
            .unwrap()
            .is_none());
    }
}
