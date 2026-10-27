//! Execution progress is separate from commit ownership. A heartbeat with an
//! unchanged sequence must never extend a no-progress deadline.
use crate::{encode, execution_key, Coordinator, Execution, Phase, Result};
use etcd_client::{Compare, CompareOp, Txn, TxnOp, TxnOpResponse, WatchOptions, WatchStream};
use lance_context_metrics::etcd::{ObserveEtcd, Operation};

/// Exact keys on one stream, owned by the waiting execution. Dropping this
/// value closes the stream; there is no detached task or fleet-wide subscription.
pub struct ExecutionWatch {
    stream: WatchStream,
    progress_key: Option<Vec<u8>>,
}

/// A progress event is evidence of work by this execution, never authorization
/// to commit, release ownership, or replace an executor. Ownership changes and
/// deleted progress require a new authoritative snapshot.
#[derive(Debug, PartialEq, Eq)]
pub enum ExecutionUpdate {
    Progress(u64),
    Refresh,
}

impl ExecutionWatch {
    pub async fn changed(&mut self) -> Result<()> {
        self.next_update().await.map(|_| ())
    }

    /// Retain the progress payload instead of turning every notification into
    /// another pair of execution/progress GETs. The exact execution ID is part
    /// of the subscribed progress key; callers must ignore non-increasing
    /// sequences and still bound their ownership/freshness checks independently.
    pub async fn next_update(&mut self) -> Result<ExecutionUpdate> {
        while let Some(response) = self.stream.message().await.map_err(|e| e.to_string())? {
            if response.canceled() {
                return Err(
                    "execution watch canceled; refresh snapshot before resubscribing".into(),
                );
            }
            let mut progress = None;
            for event in response.events() {
                let kv = event.kv().ok_or("execution watch event missing value")?;
                if self.progress_key.as_deref() != Some(kv.key())
                    || event.event_type() != etcd_client::EventType::Put
                {
                    return Ok(ExecutionUpdate::Refresh);
                }
                let current: ExecutionProgress =
                    serde_json::from_slice(kv.value()).map_err(|e| e.to_string())?;
                progress = Some(
                    progress.map_or(current.sequence, |prior: u64| prior.max(current.sequence)),
                );
            }
            if let Some(sequence) = progress {
                return Ok(ExecutionUpdate::Progress(sequence));
            }
        }
        Err("execution watch ended".into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionProgress {
    pub sequence: u64,
}

/// Execution, progress sequence and work report read together at one etcd
/// revision. `work_report` is only present when it was published for the
/// same sequence as `progress`; a report from an older sequence is dropped
/// rather than shown beside a newer heartbeat.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProgressSnapshot {
    pub execution: Execution,
    pub progress: Option<ExecutionProgress>,
    pub work_report: Option<serde_json::Value>,
    /// etcd revision the snapshot was taken at.
    pub revision: i64,
}

/// Cache the last acknowledged progress revision. Every publication compares
/// that revision as well as execution ownership, so an ambiguously completed
/// RPC cannot overwrite a newer report when it eventually reaches etcd.
pub struct WorkProgressPublisher {
    coordinator: Coordinator,
    execution: Execution,
    revision: i64,
    sequence: u64,
}

impl WorkProgressPublisher {
    pub async fn publish(&mut self, sequence: u64, report: serde_json::Value) -> Result<bool> {
        let execution_key = execution_key(&self.coordinator.prefix, &self.execution.target);
        let progress_key = self.coordinator.progress_key(&self.execution);
        let encoded = encode(&self.execution);
        if sequence < self.sequence {
            // A superseded retry is not new progress. Still check ownership.
            return Ok(self.coordinator.get(&self.execution.target).await?.as_ref()
                == Some(&self.execution));
        }
        for _ in 0..3 {
            let response = self
                .coordinator
                .client
                .clone()
                .txn(
                    Txn::new()
                        .when(vec![
                            Compare::value(
                                execution_key.clone(),
                                CompareOp::Equal,
                                encoded.clone(),
                            ),
                            Compare::mod_revision(
                                progress_key.clone(),
                                CompareOp::Equal,
                                self.revision,
                            ),
                        ])
                        .and_then(vec![
                            TxnOp::put(
                                progress_key.clone(),
                                serde_json::to_vec(&ExecutionProgress { sequence }).unwrap(),
                                None,
                            ),
                            TxnOp::put(
                                self.coordinator.work_key(&self.execution),
                                serde_json::to_vec(&report).map_err(|e| e.to_string())?,
                                None,
                            ),
                        ])
                        .or_else(vec![
                            TxnOp::get(execution_key.clone(), None),
                            TxnOp::get(progress_key.clone(), None),
                        ]),
                )
                .observed(Operation::MergeWorkProgress)
                .await
                .map_err(|e| e.to_string())?;
            if response.succeeded() {
                self.revision = response
                    .header()
                    .ok_or("progress publication missing revision")?
                    .revision();
                self.sequence = sequence;
                return Ok(true);
            }
            // The failed CAS reads ownership and progress at the same revision.
            // It also resolves a lost acknowledgement without another read RPC.
            let responses = response.op_responses();
            let [TxnOpResponse::Get(owner), TxnOpResponse::Get(progress)] = responses.as_slice()
            else {
                return Err("invalid progress publication conflict response".into());
            };
            if owner.kvs().first().map(|kv| kv.value()) != Some(encoded.as_slice()) {
                return Ok(false);
            }
            let previous = progress.kvs().first();
            self.revision = previous.map_or(0, |kv| kv.mod_revision());
            self.sequence = previous
                .map(|kv| serde_json::from_slice::<ExecutionProgress>(kv.value()))
                .transpose()
                .map_err(|e| e.to_string())?
                .map_or(0, |p| p.sequence);
            if previous.is_some() && self.sequence > sequence {
                return Ok(true);
            }
        }
        Err("progress publication contention exceeded limit".into())
    }
}

impl Coordinator {
    /// Watch only durable ownership, not this execution's own progress writes.
    /// Start after an authoritative read so revocation during setup is replayed.
    /// Stream errors require a fresh read; they are never evidence of fencing.
    pub async fn watch_execution_owner(
        &self,
        target: &str,
        revision: i64,
    ) -> Result<ExecutionWatch> {
        if revision <= 0 || revision == i64::MAX {
            return Err("invalid execution watch revision".into());
        }
        let stream = self
            .client
            .clone()
            .watch(
                execution_key(&self.prefix, target),
                Some(WatchOptions::new().with_start_revision(revision + 1)),
            )
            .observed(Operation::MergeOwnerWatch)
            .await
            .map_err(|e| e.to_string())?;
        Ok(ExecutionWatch {
            stream,
            progress_key: None,
        })
    }

    /// Replay changes after the authoritative execution read, including changes
    /// that occur before either watch is registered. After disconnect or
    /// compaction, callers must take a new snapshot before calling this again.
    pub async fn watch_execution(
        &self,
        execution: &Execution,
        revision: i64,
    ) -> Result<ExecutionWatch> {
        if revision <= 0 || revision == i64::MAX {
            return Err("invalid execution watch revision".into());
        }
        let options = WatchOptions::new().with_start_revision(revision + 1);
        let mut stream = self
            .client
            .clone()
            .watch(
                execution_key(&self.prefix, &execution.target),
                Some(options.clone()),
            )
            .observed(Operation::MergeProgressWatch)
            .await
            .map_err(|e| e.to_string())?;
        stream
            .watch(self.progress_key(execution), Some(options))
            .await
            .map_err(|e| e.to_string())?;
        Ok(ExecutionWatch {
            stream,
            progress_key: Some(self.progress_key(execution).into_bytes()),
        })
    }

    pub fn work_progress_publisher(&self, execution: &Execution) -> WorkProgressPublisher {
        WorkProgressPublisher {
            coordinator: self.clone(),
            execution: execution.clone(),
            revision: 0,
            sequence: 0,
        }
    }
    fn work_key(&self, execution: &Execution) -> String {
        format!(
            "{}/merge-work/{}",
            self.prefix.trim_end_matches('/'),
            execution.id
        )
    }

    /// Publish work details atomically with the existing progress sequence.
    /// Keep the old sequence wire format intact: mixed-version executors use
    /// its exact bytes in the no-progress revocation compare-and-swap.
    pub async fn publish_work_progress(
        &self,
        execution: &Execution,
        sequence: u64,
        report: serde_json::Value,
    ) -> Result<bool> {
        self.work_progress_publisher(execution)
            .publish(sequence, report)
            .await
    }

    pub async fn work_progress(&self, execution: &Execution) -> Result<Option<serde_json::Value>> {
        let response = self
            .client
            .clone()
            .get(self.work_key(execution), None)
            .observed(Operation::MergeRead)
            .await
            .map_err(|e| e.to_string())?;
        response
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()
    }

    /// Read execution, progress and work report for `target` atomically.
    ///
    /// Two round trips: the first learns the execution id (the progress and
    /// work keys are per execution, not per target), the second reads all
    /// three keys in one transaction so they come from a single revision.
    /// If the execution changed between the two reads the transaction's
    /// compare fails and we loop; a target that flips executions faster than
    /// two etcd reads is not something a status endpoint needs to render.
    pub async fn progress_snapshot(&self, target: &str) -> Result<Option<ProgressSnapshot>> {
        let execution_key = execution_key(&self.prefix, target);
        for _ in 0..3 {
            let Some(execution) = self.get(target).await? else {
                return Ok(None);
            };
            let encoded = encode(&execution);
            let response = self
                .client
                .clone()
                .txn(
                    Txn::new()
                        .when(vec![Compare::value(
                            execution_key.clone(),
                            CompareOp::Equal,
                            encoded,
                        )])
                        .and_then(vec![
                            TxnOp::get(self.progress_key(&execution), None),
                            TxnOp::get(self.work_key(&execution), None),
                        ]),
                )
                .observed(Operation::MergeRead)
                .await
                .map_err(|e| e.to_string())?;
            if !response.succeeded() {
                continue;
            }
            let revision = response.header().map_or(0, |h| h.revision());
            let mut values = response.op_responses().into_iter().map(|op| match op {
                TxnOpResponse::Get(get) => get.kvs().first().map(|kv| kv.value().to_vec()),
                _ => None,
            });
            let progress: Option<ExecutionProgress> = values
                .next()
                .flatten()
                .map(|v| serde_json::from_slice(&v).map_err(|e| e.to_string()))
                .transpose()?;
            let work_report: Option<serde_json::Value> = values
                .next()
                .flatten()
                .map(|v| serde_json::from_slice(&v).map_err(|e| e.to_string()))
                .transpose()?;
            let work_report = work_report.filter(|report| {
                report["sequence"].as_u64() == progress.as_ref().map(|p| p.sequence)
            });
            return Ok(Some(ProgressSnapshot {
                execution,
                progress,
                work_report,
                revision,
            }));
        }
        Err(format!(
            "merge execution for {target} changed on every read; retry later"
        ))
    }

    pub(crate) fn remove_work_progress(&self, execution: &Execution) -> TxnOp {
        TxnOp::delete(self.work_key(execution), None)
    }

    fn progress_key(&self, execution: &Execution) -> String {
        format!(
            "{}/merge-progress/{}",
            self.prefix.trim_end_matches('/'),
            execution.id
        )
    }

    pub async fn publish_progress(&self, execution: &Execution, sequence: u64) -> Result<bool> {
        self.transact_observed(
            if sequence == 0 {
                Operation::MergeProgressInitial
            } else {
                Operation::MergeProgressChanged
            },
            vec![Compare::value(
                execution_key(&self.prefix, &execution.target),
                CompareOp::Equal,
                encode(execution),
            )],
            vec![TxnOp::put(
                self.progress_key(execution),
                serde_json::to_vec(&ExecutionProgress { sequence }).unwrap(),
                None,
            )],
        )
        .await
    }

    pub async fn progress(&self, execution: &Execution) -> Result<Option<ExecutionProgress>> {
        let response = self
            .client
            .clone()
            .get(self.progress_key(execution), None)
            .observed(Operation::MergeRead)
            .await
            .map_err(|e| e.to_string())?;
        response
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()
    }

    /// Close commit admission only if both ownership and the observed completed
    /// work still match. A concurrent progress publication defeats this CAS.
    /// This is not a storage fence: recovery must still cover admitted writes.
    pub async fn revoke_stalled(
        &self,
        execution: &Execution,
        sequence: Option<u64>,
    ) -> Result<bool> {
        if execution.phase != Phase::Running {
            return Ok(false);
        }
        let mut next = execution.finished(Err(
            "confirmed no-progress execution; storage recovery required".into(),
        ));
        next.phase = Phase::Uncertain;
        let progress = match sequence {
            Some(sequence) => Compare::value(
                self.progress_key(execution),
                CompareOp::Equal,
                serde_json::to_vec(&ExecutionProgress { sequence }).unwrap(),
            ),
            None => Compare::version(self.progress_key(execution), CompareOp::Equal, 0),
        };
        self.transact(
            vec![
                Compare::value(
                    execution_key(&self.prefix, &execution.target),
                    CompareOp::Equal,
                    encode(execution),
                ),
                progress,
            ],
            vec![TxnOp::put(
                execution_key(&self.prefix, &execution.target),
                encode(&next),
                None,
            )],
        )
        .await
    }

    pub(crate) fn remove_progress(&self, execution: &Execution) -> TxnOp {
        TxnOp::delete(self.progress_key(execution), None)
    }
}
