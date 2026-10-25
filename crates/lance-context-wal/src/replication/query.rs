//! Bounded read-index queries with an application view captured at completion.
//! A result describes committed state during the request; it is not a lease,
//! authorization token or permission to write an external Lance manifest.
use super::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Query {
    ClosePermit {
        table: Uuid,
        transaction: Uuid,
        plan_index: u64,
        source_group: u64,
    },
    SourceClosure {
        table: Uuid,
    },
    Manifest {
        table: Uuid,
        version: Option<u64>,
    },
    Publisher {
        table: Uuid,
    },
    Accepted {
        after: u64,
        limit: usize,
    },
    Producer {
        table: Uuid,
        bucket: u32,
        producer: Uuid,
    },
    Pin {
        table: Uuid,
        transaction: Uuid,
        plan_index: u64,
        source_group: u64,
    },
    SourcePin {
        table: Uuid,
    },
    ClosedProof {
        table: Uuid,
    },
    ProducerSlot {
        slot: u32,
    },
    /// Compact table binding and publication cursors, without plans/descriptors.
    Visibility {
        table: Uuid,
    },
    /// A bounded scan of committed metadata, returning only this table/bucket.
    /// Empty pages may still advance over unrelated records.
    AcceptedBucket {
        table: Uuid,
        bucket: u32,
        after: u64,
        limit: usize,
    },
    /// Fencing identity without plans, file descriptors or manifest history.
    PublisherBinding {
        table: Uuid,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Value {
    ClosePermit(Option<Box<source::ClosurePermit>>),
    SourceClosure(Option<Arc<source::Closed>>),
    Manifest(Option<Arc<publication::ManifestRecord>>),
    Publisher(Option<Box<publication::Table>>),
    Accepted(Page),
    Producer(Option<(BatchMeta, u64)>),
    CursorAhead {
        applied: u64,
    },
    Pin(Option<Box<source::Permit>>),
    SourcePin(Option<Arc<source::Pin>>),
    ClosedProof(Option<Box<source::ClosedProof>>),
    /// A historical position is outside catalog retention. This is neither a
    /// missing-object result nor proof that a supplied staging object lost.
    ManifestRetired(publication::HistoryBoundary),
    ProducerSlot(Option<producer::View>),
    Visibility(Option<Box<Visibility>>),
    PublisherBinding(Option<Box<PublisherBinding>>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublisherBinding {
    pub authority: publication::Authority,
    pub destination: publication::Destination,
    pub ingest: crate::ingest_contract::IngestContract,
    pub incarnation: Option<Uuid>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Visibility {
    pub destination: publication::Destination,
    pub ingest: crate::ingest_contract::IngestContract,
    pub incarnation: Option<Uuid>,
    pub checkpoint: BTreeMap<u64, u64>,
    pub minimum_read_version: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cut {
    pub cluster: Uuid,
    pub layout: [u8; 32],
    pub group: u64,
    pub term: u64,
    pub quorum_index: u64,
    /// The query and its cursor are captured together after this position applies.
    pub applied: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Observed {
        cut: Cut,
        value: Value,
    },
    /// No read value was released before leadership changed. Retry at a leader.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Completion {
    pub group: u64,
    pub request: u64,
    pub outcome: Outcome,
}

impl Query {
    pub(super) fn validate(&self, layout: &Layout, group: u64) -> Result<()> {
        match self {
            Self::ProducerSlot { slot } => ensure!(
                *slot < producer::MAX_SLOTS,
                "producer slot query out of bounds"
            ),
            Self::Manifest { table, version } => ensure!(
                !table.is_nil()
                    && group == publication::control_group(layout)
                    && version.is_none_or(|v| v > 0 && v < u64::MAX),
                "invalid manifest query scope"
            ),
            Self::ClosePermit {
                table,
                transaction,
                plan_index,
                source_group,
            }
            | Self::Pin {
                table,
                transaction,
                plan_index,
                source_group,
            } => ensure!(
                !table.is_nil()
                    && !transaction.is_nil()
                    && *plan_index > 0
                    && *plan_index < u64::MAX
                    && layout.groups.contains_key(source_group)
                    && group == publication::control_group(layout),
                "invalid catalog dependency query"
            ),
            Self::SourceClosure { table }
            | Self::SourcePin { table }
            | Self::ClosedProof { table } => {
                ensure!(!table.is_nil(), "nil source table")
            }
            Self::Publisher { table }
            | Self::Visibility { table }
            | Self::PublisherBinding { table } => ensure!(
                !table.is_nil() && group == publication::control_group(layout),
                "invalid publisher query scope"
            ),
            Self::Accepted { after, limit } => ensure!(
                *after < u64::MAX && (1..=128).contains(limit),
                "invalid accepted query bounds"
            ),
            Self::AcceptedBucket {
                table,
                bucket,
                after,
                limit,
            } => ensure!(
                !table.is_nil()
                    && layout.group_for_bucket(*bucket)? == group
                    && *after < u64::MAX
                    && (1..=128).contains(limit),
                "invalid accepted bucket query scope"
            ),
            Self::Producer {
                table,
                bucket,
                producer,
            } => ensure!(
                !table.is_nil() && !producer.is_nil() && layout.group_for_bucket(*bucket)? == group,
                "invalid producer query scope"
            ),
        }
        Ok(())
    }

    pub(super) fn evaluate(
        &self,
        state: &State,
        cut: &Cut,
        layout: &Layout,
        limits: &Limits,
        signers: source::QuerySigners<'_>,
    ) -> Result<Value> {
        Ok(match self {
            Self::ProducerSlot { slot } => Value::ProducerSlot(state.producer_view(*slot)),
            Self::ClosedProof { table } => Value::ClosedProof(
                state
                    .closed
                    .get(table)
                    .map(|closed| {
                        source::ClosedProof::issue(
                            signers.source.context("source signer unavailable")?,
                            cut.clone(),
                            layout,
                            limits,
                            closed.clone(),
                        )
                        .map(Box::new)
                    })
                    .transpose()?,
            ),
            Self::SourceClosure { table } => Value::SourceClosure(state.closed.get(table).cloned()),
            Self::ClosePermit {
                table,
                transaction,
                plan_index,
                source_group,
            } => {
                let selected = state.catalog.tables.get(table).filter(|table| {
                    table.progress.published.is_some()
                        && table.progress.pending.as_ref().is_some_and(|p| {
                            p.index == *plan_index
                                && p.plan.transaction == *transaction
                                && p.plan.sources.contains_key(source_group)
                        })
                });
                Value::ClosePermit(
                    selected
                        .map(|table| {
                            source::ClosurePermit::issue(
                                signers
                                    .catalog
                                    .context("catalog closure signer unavailable")?,
                                cut.clone(),
                                layout,
                                limits,
                                table,
                                *source_group,
                            )
                            .map(Box::new)
                        })
                        .transpose()?,
                )
            }
            Self::Manifest { table, version } => {
                let manifests = state
                    .catalog
                    .tables
                    .get(table)
                    .and_then(|t| t.manifests.as_ref());
                match manifests {
                    Some(m) if version.is_some_and(|v| m.history_boundary().is_retired(v)) => {
                        Value::ManifestRetired(m.history_boundary())
                    }
                    _ => Value::Manifest(
                        manifests
                            .and_then(|m| m.versions.get(&version.unwrap_or(m.head)))
                            .cloned(),
                    ),
                }
            }
            Self::Pin {
                table,
                transaction,
                plan_index,
                source_group,
            } => {
                let selected = state.catalog.tables.get(table).filter(|table| {
                    table.progress.pending.as_ref().is_some_and(|p| {
                        p.index == *plan_index
                            && p.plan.transaction == *transaction
                            && p.plan.sources.contains_key(source_group)
                    })
                });
                Value::Pin(
                    selected
                        .map(|table| {
                            source::Permit::issue(
                                signers
                                    .catalog
                                    .context("catalog dependency signer unavailable")?,
                                cut.clone(),
                                layout,
                                limits,
                                table,
                                *source_group,
                            )
                            .map(Box::new)
                        })
                        .transpose()?,
                )
            }
            Self::SourcePin { table } => Value::SourcePin(state.pins.get(table).cloned()),
            Self::Publisher { table } => {
                Value::Publisher(state.catalog.tables.get(table).cloned().map(Box::new))
            }
            Self::PublisherBinding { table } => {
                Value::PublisherBinding(state.catalog.tables.get(table).map(|table| {
                    Box::new(PublisherBinding {
                        authority: table.authority.clone(),
                        destination: table.destination.clone(),
                        ingest: table.ingest.clone(),
                        incarnation: table.manifests.as_ref().map(|m| m.incarnation),
                    })
                }))
            }
            Self::Visibility { table } => {
                Value::Visibility(state.catalog.tables.get(table).map(|table| {
                    Box::new(Visibility {
                        destination: table.destination.clone(),
                        ingest: table.ingest.clone(),
                        incarnation: table.manifests.as_ref().map(|m| m.incarnation),
                        checkpoint: table.progress.checkpoint.clone(),
                        minimum_read_version: table.progress.minimum_read_version,
                    })
                }))
            }
            Self::Accepted { after, .. } | Self::AcceptedBucket { after, .. }
                if *after > state.applied =>
            {
                Value::CursorAhead {
                    applied: state.applied,
                }
            }
            Self::Accepted { after, limit } => Value::Accepted(state.page(*after, *limit)?),
            Self::AcceptedBucket {
                table,
                bucket,
                after,
                limit,
            } => Value::Accepted(state.bucket_page(*table, *bucket, *after, *limit)?),
            Self::Producer {
                table,
                bucket,
                producer,
            } => Value::Producer(state.producers.get(&(*table, *bucket, *producer)).cloned()),
        })
    }
}

struct Pending {
    request: u64,
    term: u64,
    query: Query,
    confirmed: Option<u64>,
}

pub(super) struct Tracker {
    incarnation: Uuid,
    next: u64,
    pending: BTreeMap<[u8; 32], Pending>,
}

impl Default for Tracker {
    fn default() -> Self {
        Self {
            incarnation: Uuid::new_v4(),
            next: 0,
            pending: BTreeMap::new(),
        }
    }
}

impl Tracker {
    pub fn len(&self) -> usize {
        self.pending.len()
    }
    pub fn contains(&self, request: u64) -> bool {
        self.pending.values().any(|p| p.request == request)
    }
    pub fn enqueue(&mut self, request: u64, term: u64, query: Query) -> Result<Vec<u8>> {
        self.next = self.next.checked_add(1).context("read context exhausted")?;
        let mut context = [0; 32];
        context[..8].copy_from_slice(b"LCREAD01");
        context[8..24].copy_from_slice(self.incarnation.as_bytes());
        context[24..].copy_from_slice(&self.next.to_le_bytes());
        ensure!(
            self.pending
                .insert(
                    context,
                    Pending {
                        request,
                        term,
                        query,
                        confirmed: None
                    }
                )
                .is_none(),
            "reused read context"
        );
        Ok(context.to_vec())
    }
    pub fn confirm(&mut self, context: &[u8], index: u64) -> Result<()> {
        let Ok(key) = <[u8; 32]>::try_from(context) else {
            return Ok(());
        };
        let Some(pending) = self.pending.get_mut(&key) else {
            return Ok(());
        };
        ensure!(
            index > 0 && index < u64::MAX,
            "invalid confirmed read index"
        );
        ensure!(
            pending.confirmed.is_none_or(|old| old == index),
            "read index changed"
        );
        pending.confirmed = Some(index);
        Ok(())
    }
    pub fn finish(
        &mut self,
        cut: Cut,
        state: &State,
        leader: bool,
        evaluate: impl Fn(&Query, &Cut, &State) -> Result<Value>,
    ) -> Result<Vec<Completion>> {
        let done: Vec<_> = self
            .pending
            .iter()
            .filter_map(|(key, p)| {
                (!leader
                    || p.term != cut.term
                    || p.confirmed.is_some_and(|index| index <= state.applied))
                .then_some(*key)
            })
            .collect();
        let mut results = Vec::with_capacity(done.len());
        for key in done {
            let p = self.pending.remove(&key).expect("selected pending read");
            let outcome = if !leader || p.term != cut.term {
                Outcome::Unknown
            } else {
                let observed_cut = Cut {
                    quorum_index: p.confirmed.expect("confirmed completed read"),
                    ..cut.clone()
                };
                let value = evaluate(&p.query, &observed_cut, state)?;
                Outcome::Observed {
                    cut: observed_cut,
                    value,
                }
            };
            results.push(Completion {
                group: cut.group,
                request: p.request,
                outcome,
            });
        }
        Ok(results)
    }
}
