//! Resolve a source closure, with one exact resubmission after an unknown result.
use super::*;

impl Publisher {
    pub(crate) async fn close_source(
        &self,
        permit: source::ClosurePermit,
    ) -> Result<source::ClosedProof> {
        let expected = permit.closure().clone();
        for attempt in 0..2 {
            let result = commit(&self.ingress.client, Command::Close(permit.clone())).await;
            if let Ok(reply) = &result {
                ensure!(
                    matches!(reply, Applied::Source(source::Reply::Closed { .. })),
                    "source closure rejected: {reply:?}"
                );
            }
            // A timeout or lost response says nothing about whether Close committed.
            // A missing first closure or the exact preceding source boundary may
            // mean the command never committed. Neither is a successful closure.
            // Only a freshly confirmed, unchanged catalog permit allows one more
            // submission of the original idempotent command, under the same owner.
            let resume = attempt == 0 && result.is_err();
            let observed: Result<Option<source::ClosedProof>> = async {
                if result.is_err() {
                    // A source proof is not publisher authority. Preserve the same
                    // ownership and incarnation checks used for unknown Advance.
                    self.state().await?;
                }
                let query::Value::ClosedProof(proof) = query(
                    &self.ingress,
                    expected.group,
                    query::Query::ClosedProof {
                        table: self.authority.table,
                    },
                )
                .await?
                else {
                    bail!("invalid source closure observation");
                };
                if let Some(proof) = proof {
                    let closed = proof.closed().permit.closure();
                    ensure!(
                        proof.source_cut().group == expected.group,
                        "foreign source closure proof group"
                    );
                    if closed == &expected {
                        return Ok(Some(*proof));
                    }
                    ensure!(
                        resume
                            && closed.table == expected.table
                            && closed.incarnation == expected.incarnation
                            && closed.group == expected.group
                            && closed.through == expected.after
                            && closed.plan_index < expected.plan_index
                            && closed.version < expected.version
                            && closed.published_index < expected.published_index,
                        "foreign source closure proof: expected {expected:?}, observed {closed:?}"
                    );
                } else {
                    ensure!(
                        resume && expected.after == 0,
                        "source closure proof unavailable"
                    );
                }
                let query::Value::ClosePermit(Some(current)) = query(
                    &self.ingress,
                    self.authority.group,
                    query::Query::ClosePermit {
                        table: expected.table,
                        transaction: expected.transaction,
                        plan_index: expected.plan_index,
                        source_group: expected.group,
                    },
                )
                .await?
                else {
                    bail!("original source closure authorization unavailable");
                };
                ensure!(
                    current.closure() == &expected,
                    "original source closure authorization changed"
                );
                self.state().await?;
                Ok(None)
            }
            .await;
            match (result, observed) {
                (_, Ok(Some(proof))) => return Ok(proof),
                (_, Ok(None)) => continue,
                (Err(error), Err(observation)) => {
                    return Err(error).context(format!(
                        "original source closure not confirmed: {observation:#}"
                    ))
                }
                (_, Err(error)) => return Err(error),
            }
        }
        unreachable!("only the first unknown closure can be resubmitted")
    }
}
