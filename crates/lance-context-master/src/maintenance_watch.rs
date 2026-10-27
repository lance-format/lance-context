//! Exact-key notifications for one prepared maintenance operation. These are
//! hints and progress observations; only `promote_maintenance` grants a writer.
use super::{etcd_error, TaskClaim, TaskStore};
use etcd_client::{EventType, GetOptions, Txn, TxnOp, TxnOpResponse, WatchOptions, WatchStream};
use lance_context_merge::{progress::ExecutionProgress, Execution};
use lance_context_metrics::etcd::{ObserveEtcd, Operation};
use std::collections::HashMap;

pub(crate) enum MaintenanceChange {
    Ownership,
    Progress,
    Resnapshot,
}

pub(crate) struct MaintenanceWatch {
    stream: WatchStream,
    snapshot: MaintenanceSnapshot,
}

/// A bounded observation, deliberately separate from subscription so the gap
/// between the two is covered by replay from its original revision.
pub(crate) struct MaintenanceSnapshot {
    revision: i64,
    values: HashMap<Vec<u8>, Option<Vec<u8>>>,
    claim_key: Vec<u8>,
    preparation_key: Vec<u8>,
    token: Vec<u8>,
    execution_key: Vec<u8>,
    progress_key: Option<Vec<u8>>,
    execution: Option<Execution>,
    progress: Option<ExecutionProgress>,
}

impl MaintenanceWatch {
    pub(crate) fn execution(&self) -> Option<&Execution> {
        self.snapshot.execution.as_ref()
    }

    pub(crate) fn progress(&self) -> Option<&ExecutionProgress> {
        self.snapshot.progress.as_ref()
    }

    pub(crate) fn preparation_owned(&self) -> bool {
        [&self.snapshot.claim_key, &self.snapshot.preparation_key]
            .iter()
            .all(|key| {
                self.snapshot.values.get(*key).and_then(Option::as_ref)
                    == Some(&self.snapshot.token)
            })
    }

    pub(crate) async fn changed(&mut self) -> lance::Result<MaintenanceChange> {
        while let Some(response) = self
            .stream
            .message()
            .await
            .map_err(etcd_error("watch maintenance commit turn"))?
        {
            if response.canceled() {
                return Err(lance::Error::io("maintenance watch canceled; resnapshot"));
            }
            let mut ownership = false;
            let mut progress = false;
            for event in response.events() {
                let kv = event
                    .kv()
                    .ok_or_else(|| lance::Error::io("maintenance watch event missing key"))?;
                let value = (event.event_type() == EventType::Put).then(|| kv.value().to_vec());
                if kv.key() == self.snapshot.execution_key {
                    let next: Option<Execution> = value
                        .as_deref()
                        .map(serde_json::from_slice)
                        .transpose()
                        .map_err(|e| lance::Error::io(e.to_string()))?;
                    if next.as_ref().map(|e| &e.id)
                        != self.snapshot.execution.as_ref().map(|e| &e.id)
                    {
                        // Progress keys are execution-scoped. Close this stream
                        // and capture the successor and its keys at one revision.
                        return Ok(MaintenanceChange::Resnapshot);
                    }
                    // Promotion compares the complete execution value. Even
                    // a same-ID phase change may have rejected its last CAS.
                    ownership |= next != self.snapshot.execution;
                    self.snapshot.execution = next;
                } else if self.snapshot.progress_key.as_deref() == Some(kv.key()) {
                    let next = value
                        .as_deref()
                        .map(serde_json::from_slice::<ExecutionProgress>)
                        .transpose()
                        .map_err(|e| lance::Error::io(e.to_string()))?;
                    progress |= next != self.snapshot.progress;
                    self.snapshot.progress = next;
                } else if let Some(old) = self.snapshot.values.get_mut(kv.key()) {
                    ownership |= *old != value;
                    *old = value;
                }
            }
            if ownership {
                return Ok(MaintenanceChange::Ownership);
            }
            if progress {
                return Ok(MaintenanceChange::Progress);
            }
            // Creation acknowledgements and unchanged heartbeats are quiet.
        }
        Err(lance::Error::io("maintenance watch ended; resnapshot"))
    }
}

impl TaskStore {
    pub(crate) async fn watch_maintenance_commit(
        &self,
        claim: &TaskClaim,
    ) -> lance::Result<MaintenanceWatch> {
        self.capture_maintenance_commit(claim)
            .await?
            .subscribe(&self.inner.client)
            .await
    }

    pub(crate) async fn capture_maintenance_commit(
        &self,
        claim: &TaskClaim,
    ) -> lance::Result<MaintenanceSnapshot> {
        let b = &claim.backend;
        let preparation = b
            .preparation_key
            .as_ref()
            .ok_or_else(|| lance::Error::io("maintenance preparation key missing"))?;
        let execution_key =
            lance_context_merge::execution_key(&self.inner.prefix, &claim.task.target);
        let keys = [
            b.claim_key.clone(),
            preparation.clone(),
            execution_key.clone(),
            self.inner.target_lock_key(&claim.task.target),
            execution_key.replace("/merge-executions/", "/merge-claims/"),
            crate::catchup::store::active_key(&self.inner.prefix, &claim.task.target),
        ];
        let response = self
            .inner
            .client
            .clone()
            .txn(
                Txn::new().and_then(
                    keys.iter()
                        .map(|k| TxnOp::get(k.as_str(), None))
                        .collect::<Vec<_>>(),
                ),
            )
            .observed(Operation::TaskReadTxn)
            .await
            .map_err(etcd_error("capture maintenance commit wait"))?;
        let revision = response
            .header()
            .map(|h| h.revision())
            .filter(|r| *r > 0 && *r < i64::MAX)
            .ok_or_else(|| lance::Error::io("maintenance snapshot revision missing"))?;
        let responses = response.op_responses();
        if responses.len() != keys.len() {
            return Err(lance::Error::io("maintenance snapshot incomplete"));
        }
        let mut values = HashMap::new();
        for (key, response) in keys.iter().zip(responses) {
            let TxnOpResponse::Get(get) = response else {
                return Err(lance::Error::io("invalid maintenance snapshot response"));
            };
            values.insert(
                key.as_bytes().to_vec(),
                get.kvs().first().map(|kv| kv.value().to_vec()),
            );
        }
        let execution: Option<Execution> = values[execution_key.as_bytes()]
            .as_deref()
            .map(serde_json::from_slice)
            .transpose()
            .map_err(|e| lance::Error::io(e.to_string()))?;
        let progress_key = execution.as_ref().map(|e| {
            format!(
                "{}/merge-progress/{}",
                self.inner.prefix.trim_end_matches('/'),
                e.id
            )
        });
        let progress = if let Some(key) = &progress_key {
            let response = self
                .inner
                .client
                .clone()
                .get(
                    key.as_str(),
                    Some(GetOptions::new().with_revision(revision)),
                )
                .observed(Operation::TaskRead)
                .await
                .map_err(etcd_error("capture maintenance wait progress"))?;
            response
                .kvs()
                .first()
                .map(|kv| serde_json::from_slice(kv.value()))
                .transpose()
                .map_err(|e| lance::Error::io(e.to_string()))?
        } else {
            None
        };
        Ok(MaintenanceSnapshot {
            revision,
            values,
            claim_key: b.claim_key.as_bytes().to_vec(),
            preparation_key: preparation.as_bytes().to_vec(),
            token: b.token.as_bytes().to_vec(),
            execution_key: execution_key.into_bytes(),
            progress_key: progress_key.map(String::into_bytes),
            execution,
            progress,
        })
    }
}

impl MaintenanceSnapshot {
    pub(crate) async fn subscribe(
        self,
        client: &etcd_client::Client,
    ) -> lance::Result<MaintenanceWatch> {
        // Replay even changes made while the snapshot or subscriptions were
        // assembled. A compacted revision requires a new complete capture.
        let options = WatchOptions::new().with_start_revision(self.revision + 1);
        let mut stream = client
            .clone()
            .watch(self.claim_key.clone(), Some(options.clone()))
            .observed(Operation::DispatchWatch)
            .await
            .map_err(etcd_error("open maintenance commit watch"))?;
        for key in self
            .values
            .keys()
            .filter(|key| **key != self.claim_key)
            .chain(self.progress_key.iter())
        {
            stream
                .watch(key.clone(), Some(options.clone()))
                .await
                .map_err(etcd_error("add maintenance commit watch key"))?;
        }
        Ok(MaintenanceWatch {
            stream,
            snapshot: self,
        })
    }
}
