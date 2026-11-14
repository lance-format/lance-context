//! Resolve an uncertain advancement without selecting a new publication.
use super::*;

impl Publisher {
    pub(crate) async fn record_advance(
        &self,
        request: p::AdvancePublication,
    ) -> Result<Arc<p::Advanced>> {
        ensure!(
            request.authority == self.authority,
            "foreign advancement authority"
        );
        for attempt in 0..2 {
            let error = match commit(&self.ingress.client, Command::Advance(request.clone())).await
            {
                Ok(Applied::Publisher(p::Reply::Advanced(advanced))) => return Ok(advanced),
                Ok(reply) => bail!("source cursor advancement rejected: {reply:?}"),
                Err(error) => error,
            };
            let observed: Result<Option<Arc<p::Advanced>>> = async {
                // state() verifies current authority, table binding and incarnation.
                let view = self.state().await?;
                if let Some(advanced) = &view.progress.advanced {
                    if request.matches_receipt(advanced)? {
                        return Ok(Some(advanced.clone()));
                    }
                }
                ensure!(attempt == 0, "original advancement remains unconfirmed");
                let pending = view
                    .progress
                    .pending
                    .context("original publication no longer pending")?;
                let published = view
                    .progress
                    .published
                    .context("original publication unavailable")?;
                ensure!(
                    published.index == request.published_index
                        && request.proofs.keys().eq(pending.plan.sources.keys()),
                    "original publication or closure set changed"
                );
                for (group, proof) in &request.proofs {
                    let query::Value::ClosePermit(Some(permit)) = query(
                        &self.ingress,
                        self.authority.group,
                        query::Query::ClosePermit {
                            table: self.authority.table,
                            transaction: pending.plan.transaction,
                            plan_index: pending.index,
                            source_group: *group,
                        },
                    )
                    .await?
                    else {
                        bail!("original closure authorization unavailable");
                    };
                    ensure!(
                        permit.closure() == proof.closed().permit.closure(),
                        "original closure authorization changed"
                    );
                }
                self.state().await?;
                // Absence is not proof of non-commit. Both copies may commit;
                // the catalog deduplicates the exact publication and closures.
                // Reuse the original proofs, never close a new source range.
                Ok(None)
            }
            .await;
            match observed {
                Ok(Some(advanced)) => return Ok(advanced),
                Ok(None) => continue,
                Err(observation) => {
                    return Err(error).context(format!(
                        "original publication advancement not confirmed: {observation:#}"
                    ))
                }
            }
        }
        unreachable!("only the first uncertain advancement may be resubmitted")
    }
}
