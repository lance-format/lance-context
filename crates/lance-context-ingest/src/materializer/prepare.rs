//! Confirm an uncertain Prepare against its exact retained catalog result.
use super::*;

impl Publisher {
    pub(crate) async fn record_preparation(&self, request: p::PrepareRequest) -> Result<()> {
        ensure!(
            request.authority == self.authority,
            "foreign preparation authority"
        );
        let submitted = commit(&self.ingress.client, Command::Prepare(request.clone())).await;
        match submitted {
            Ok(Applied::Publisher(p::Reply::Prepared { .. })) => Ok(()),
            Ok(reply) => bail!("materialization result rejected: {reply:?}"),
            Err(error) => {
                // state() checks authority, binding and incarnation. Only the
                // complete original preparation can resolve the unknown result;
                // a matching UUID alone cannot authorize different files.
                // Observe once, without resubmitting Prepare or writing files.
                match self.state().await {
                    Ok(view)
                        if view.progress.prepared.as_ref().is_some_and(|prepared| {
                            prepared.transaction == request.transaction
                                && prepared.plan_index == request.plan_index
                                && prepared.descriptors == request.descriptors
                        }) =>
                    {
                        Ok(())
                    }
                    Ok(_) => Err(error).context("original preparation not confirmed"),
                    Err(observation) => Err(error).context(format!(
                        "original preparation observation failed: {observation:#}"
                    )),
                }
            }
        }
    }
}
