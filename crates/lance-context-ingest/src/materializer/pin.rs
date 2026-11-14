//! Resolve an uncertain source pin without selecting a new publication prefix.
use super::*;

impl Publisher {
    pub(crate) async fn pin_source(&self, permit: source::Permit) -> Result<()> {
        let expected = permit.selection();
        for attempt in 0..2 {
            let error = match commit(&self.ingress.client, Command::Pin(permit.clone())).await {
                Ok(Applied::Source(source::Reply::Pinned { .. })) => return Ok(()),
                Ok(reply) => bail!("source pin rejected: {reply:?}"),
                Err(error) => error,
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
                ensure!(attempt == 0, "original source pin remains unconfirmed");
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
                Ok(false) => continue,
                Err(observation) => {
                    return Err(error).context(format!(
                        "original source pin not confirmed: {observation:#}"
                    ));
                }
            }
        }
        unreachable!("only the first uncertain pin may be resubmitted")
    }
}
