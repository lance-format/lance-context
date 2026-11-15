//! Resolve an uncertain source pin without selecting a new publication prefix.
use super::*;

impl Publisher {
    pub(crate) async fn pin_source(&self, permit: source::Permit) -> Result<()> {
        let expected = permit.selection();
        loop {
            let response = self
                .ingress
                .client
                .request_routed(Request::Commit(Box::new(Command::Pin(permit.clone()))))
                .await;
            let (error, uncertain) = match response.map(|r| r.reply) {
                Ok(Reply::Commit(Outcome::Committed(Applied::Source(source::Reply::Pinned {
                    ..
                })))) => return Ok(()),
                Ok(Reply::Commit(Outcome::Unknown)) => (
                    anyhow::anyhow!(
                        "durable command outcome unknown; reconcile the original operation"
                    ),
                    true,
                ),
                Ok(reply) => bail!("source pin rejected: {reply:?}"),
                Err(error) => (error, false),
            };
            let observed: Result<bool> = async {
                // A source pin is not publisher authority. A fenced owner may
                // neither confirm this operation nor resubmit its old permit.
                self.state().await?;
                let query::Value::SourcePin(pin) = query(
                    &self.ingress,
                    expected.group,
                    query::Query::SourcePin {
                        table: expected.table,
                    },
                )
                .await?
                else {
                    bail!("invalid source pin observation");
                };
                if let Some(pin) = pin {
                    ensure!(
                        pin.permit.selection() == expected,
                        "source pin differs from original selection"
                    );
                    return Ok(true);
                }
                ensure!(uncertain, "original source pin remains unconfirmed");
                let query::Value::Pin(Some(current)) = query(
                    &self.ingress,
                    self.authority.group,
                    query::Query::Pin {
                        table: expected.table,
                        transaction: expected.transaction,
                        plan_index: expected.plan_index,
                        source_group: expected.group,
                    },
                )
                .await?
                else {
                    bail!("original source pin authorization unavailable");
                };
                ensure!(
                    current.selection() == expected,
                    "original source pin authorization changed"
                );
                self.state().await?;
                // Absence does not prove that the original command failed. The
                // source deduplicates this exact selection if both copies commit.
                Ok(false)
            }
            .await;
            match observed {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    // Structured Unknown plus fresh authority and the identical
                    // selected permit authorize this idempotent control request.
                    // Keep the caller's frozen generation while bounded apply
                    // and point reads remain live. No business write or file
                    // preparation is repeated; cancellation retains the plan.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(observation) => {
                    return Err(error).context(format!(
                        "original source pin not confirmed: {observation:#}"
                    ));
                }
            }
        }
    }
}
