//! Resolve a source closure from its quorum proof without resubmitting Close.
use super::*;

impl Publisher {
    pub(crate) async fn close_source(
        &self,
        permit: source::ClosurePermit,
    ) -> Result<source::ClosedProof> {
        let expected = permit.closure().clone();
        let result = commit(&self.ingress.client, Command::Close(permit)).await;
        if let Ok(reply) = &result {
            ensure!(
                matches!(reply, Applied::Source(source::Reply::Closed { .. })),
                "source closure rejected: {reply:?}"
            );
        }
        // A timeout or lost response says nothing about whether Close committed.
        // Keep the original error unless one bounded quorum observation proves
        // this exact closure. A missing or newer closure cannot stand in for it.
        let observed: Result<source::ClosedProof> = async {
            if result.is_err() {
                // A source proof is not publisher authority. Preserve the same
                // ownership and incarnation checks used for unknown Advance.
                self.state().await?;
            }
            let query::Value::ClosedProof(Some(proof)) = query(
                &self.ingress,
                expected.group,
                query::Query::ClosedProof {
                    table: self.authority.table,
                },
            )
            .await?
            else {
                bail!("source closure proof unavailable");
            };
            ensure!(
                proof.closed().permit.closure() == &expected
                    && proof.source_cut().group == expected.group,
                "foreign source closure proof"
            );
            Ok(*proof)
        }
        .await;
        match (result, observed) {
            (_, Ok(proof)) => Ok(proof),
            (Err(error), Err(observation)) => Err(error).context(format!(
                "original source closure not confirmed: {observation:#}"
            )),
            (_, Err(error)) => Err(error),
        }
    }
}
