//! Resolve an uncertain Prepare using its retained result or exact active plan.
use super::*;

impl Publisher {
    pub(crate) async fn record_preparation(&self, request: p::PrepareRequest) -> Result<()> {
        ensure!(
            request.authority == self.authority,
            "foreign preparation authority"
        );
        for attempt in 0..2 {
            match commit(&self.ingress.client, Command::Prepare(request.clone())).await {
                Ok(Applied::Publisher(p::Reply::Prepared { .. })) => return Ok(()),
                Ok(reply) => bail!("materialization result rejected: {reply:?}"),
                Err(error) => {
                    // state() validates owner, binding and incarnation. A missing
                    // result is not proof of failure: the original proposal can
                    // still commit. Only an unchanged active plan permits one
                    // resubmission of the exact idempotent descriptor command.
                    // Files and business mutations are never regenerated here.
                    match self.state().await {
                        Ok(view) => {
                            if let Some(prepared) = &view.progress.prepared {
                                if prepared.transaction == request.transaction
                                    && prepared.plan_index == request.plan_index
                                    && prepared.descriptors == request.descriptors
                                {
                                    return Ok(());
                                }
                            } else if attempt == 0
                                && view.progress.pending.as_ref().is_some_and(|pending| {
                                    pending.index == request.plan_index
                                        && pending.plan.transaction == request.transaction
                                })
                            {
                                continue;
                            }
                            return Err(error).context("original preparation not confirmed");
                        }
                        Err(observation) => {
                            return Err(error).context(format!(
                                "original preparation observation failed: {observation:#}"
                            ));
                        }
                    }
                }
            }
        }
        unreachable!("only the first uncertain submission may be repeated")
    }
}
