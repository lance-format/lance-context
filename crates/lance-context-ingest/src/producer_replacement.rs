//! Replace an abandoned producer using existing replicated generation fences.
//! This does not resolve its application/source obligations or discard its log.
use super::*;

#[derive(Debug)]
pub enum ProducerReplacement {
    /// The predecessor is fenced, but its source tail is not retired yet.
    /// Retry this same replacement operation after publication/retirement.
    WaitingForRetirement {
        last: Option<producer::Last>,
    },
    Opened(Producer),
}

/// Another generation/operation owns the slot. Never choose a different
/// predecessor automatically in response to this error.
#[derive(Debug)]
pub struct ProducerReplacementFenced;
impl std::fmt::Display for ProducerReplacementFenced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("producer replacement fenced")
    }
}
impl std::error::Error for ProducerReplacementFenced {}

impl Ingress {
    async fn producer_slot(&self, previous: &Producer) -> Result<producer::View> {
        let group = self.group(previous.bucket())?;
        let response = self
            .client
            .request_routed(Request::Query {
                group,
                query: query::Query::ProducerSlot {
                    slot: previous.open.slot,
                },
            })
            .await?;
        let Reply::Query(query::Outcome::Observed {
            cut,
            value: query::Value::ProducerSlot(Some(view)),
        }) = response.reply
        else {
            bail!("producer slot quorum observation unavailable");
        };
        ensure!(
            cut.cluster == self.scope.cluster
                && cut.layout == self.scope.layout
                && cut.group == group
                && cut.term > 0
                && cut.applied >= cut.quorum_index
                && view.slot == previous.open.slot,
            "foreign producer slot observation"
        );
        Ok(view)
    }

    fn successor(&self, view: &producer::View, open: &ProducerOpen) -> Result<Option<Producer>> {
        let generation = open
            .expected_generation
            .checked_add(1)
            .context("producer generation exhausted")?;
        if view.generation != generation {
            return Ok(None);
        }
        ensure!(
            view.lease
                .as_ref()
                .is_some_and(|lease| lease.table == self.scope.table
                    && lease.bucket == open.bucket
                    && lease.operation == open.operation
                    && !lease.closed),
            ProducerReplacementFenced
        );
        Ok(Some(Producer {
            scope: self.scope.clone(),
            open: open.clone(),
            id: producer::producer_id(open.slot, generation),
            generation,
        }))
    }

    /// Fence this exact predecessor, then reuse its slot only after the existing
    /// signed retirement protocol has covered its latest source record.
    ///
    /// The caller authorizes the table and predecessor, persists `operation`
    /// before the first call, and retries it unchanged on an unknown response.
    /// A successful replacement starts a new producer sequence; replay from an
    /// independently confirmed source checkpoint is safe only for idempotent
    /// business operations such as first-create. It does not resolve the old
    /// request results, advance a source cursor, or make arbitrary patches safe
    /// to replay. A lost replacement executor needs another new attempt, not
    /// an empty journal reopened with the same successful replacement tuple.
    pub async fn replace_producer(
        &self,
        previous: &Producer,
        operation: Uuid,
    ) -> Result<ProducerReplacement> {
        ensure!(
            *previous.scope == *self.scope
                && self.scope.contract.policy == WritePolicy::TurnIngestion,
            "replacement requires the original managed turn producer scope"
        );
        ensure!(
            !operation.is_nil()
                && operation != previous.open.operation
                && previous.generation < u64::MAX,
            "invalid replacement operation or exhausted generation"
        );
        let open = ProducerOpen {
            bucket: previous.bucket(),
            slot: previous.open.slot,
            expected_generation: previous.generation,
            operation,
        };
        let view = self.producer_slot(previous).await?;
        if let Some(next) = self.successor(&view, &open)? {
            return Ok(ProducerReplacement::Opened(next));
        }
        ensure!(
            view.generation == previous.generation
                && view
                    .lease
                    .as_ref()
                    .is_some_and(|lease| lease.table == self.scope.table
                        && lease.bucket == previous.bucket()
                        && lease.operation == previous.open.operation),
            ProducerReplacementFenced
        );

        // A pending poll is a quorum read, not another replicated Open command.
        if view.lease.as_ref().unwrap().closed && !view.reusable {
            return Ok(ProducerReplacement::WaitingForRetirement { last: view.last });
        }

        if !view.lease.as_ref().unwrap().closed {
            let response = self
                .client
                .request_routed(Request::Commit(Box::new(Command::Producer(
                    producer::Command {
                        cluster: self.scope.cluster,
                        layout: self.scope.layout,
                        group: self.group(previous.bucket())?,
                        table: self.scope.table,
                        slot: previous.open.slot,
                        action: producer::Action::Revoke {
                            generation: previous.generation,
                        },
                    },
                ))))
                .await?;
            match response.reply {
                Reply::Commit(Outcome::Committed(Applied::Producer(producer::Reply::Closed {
                    ..
                }))) => {}
                Reply::Commit(Outcome::Committed(Applied::Producer(producer::Reply::Fenced))) => {
                    // A concurrent retry of this same operation may have opened
                    // its successor. Observe that identity without revoking it.
                    let view = self.producer_slot(previous).await?;
                    return self
                        .successor(&view, &open)?
                        .map(ProducerReplacement::Opened)
                        .ok_or_else(|| ProducerReplacementFenced.into());
                }
                _ => bail!("producer revocation outcome unknown; retain replacement identity"),
            }
        }
        let response = self
            .client
            .request_routed(Request::Commit(Box::new(Command::Producer(
                producer::Command {
                    cluster: self.scope.cluster,
                    layout: self.scope.layout,
                    group: self.group(open.bucket)?,
                    table: self.scope.table,
                    slot: open.slot,
                    action: producer::Action::Open {
                        expected_generation: open.expected_generation,
                        operation: open.operation,
                        bucket: open.bucket,
                    },
                },
            ))))
            .await?;
        match response.reply {
            Reply::Commit(Outcome::Committed(Applied::Producer(producer::Reply::Opened {
                producer: id,
                generation,
                closed: false,
            }))) => {
                ensure!(
                    generation == previous.generation + 1
                        && id == producer::producer_id(open.slot, generation),
                    "foreign replacement allocation"
                );
                Ok(ProducerReplacement::Opened(Producer {
                    scope: self.scope.clone(),
                    open,
                    id,
                    generation,
                }))
            }
            Reply::Commit(Outcome::Committed(Applied::Producer(producer::Reply::Busy))) => {
                let view = self.producer_slot(previous).await?;
                if let Some(next) = self.successor(&view, &open)? {
                    return Ok(ProducerReplacement::Opened(next));
                }
                ensure!(
                    view.generation == previous.generation,
                    ProducerReplacementFenced
                );
                Ok(ProducerReplacement::WaitingForRetirement { last: view.last })
            }
            Reply::Commit(Outcome::Committed(Applied::Producer(
                producer::Reply::Fenced | producer::Reply::Opened { closed: true, .. },
            ))) => Err(ProducerReplacementFenced.into()),
            _ => bail!("replacement allocation outcome unknown; retain replacement identity"),
        }
    }
}
