//! Resolve an uncertain Confirm using the exact retained quorum outcome.
use super::*;

impl Publisher {
    pub(crate) async fn confirm_publication(
        &self,
        request: p::ConfirmPublication,
    ) -> Result<Arc<p::Published>> {
        ensure!(
            request.authority == self.authority
                && request.incarnation == self.ingress.scope.incarnation,
            "foreign publication confirmation authority or incarnation"
        );
        match commit(&self.ingress.client, Command::Confirm(request.clone())).await {
            Ok(Applied::Publisher(p::Reply::Published(published))) => {
                ensure!(matches(&request, &published), "foreign publication outcome");
                Ok(published)
            }
            Ok(reply) => bail!("publication confirmation rejected: {reply:?}"),
            Err(error) => {
                // One bounded quorum observation, never a second submission.
                // state() checks authority, binding and incarnation. Selected
                // files alone are insufficient: Confirm must have committed.
                match self.state().await {
                    Ok(view) => {
                        if let Some(published) = view
                            .progress
                            .published
                            .iter()
                            .chain(view.progress.advanced.iter().map(|a| &a.publication))
                            .find(|published| matches(&request, published))
                        {
                            return Ok(published.clone());
                        }
                        Err(error).context("original publication confirmation not observed")
                    }
                    Err(observation) => Err(error).context(format!(
                        "original publication confirmation observation failed: {observation:#}"
                    )),
                }
            }
        }
    }
}

fn matches(request: &p::ConfirmPublication, published: &p::Published) -> bool {
    let manifest = &published.manifest;
    request.incarnation == published.incarnation
        && request.transaction == manifest.transaction
        && request.plan_index == manifest.plan_index
        && request.prepared_index == manifest.prepared_index
        && request.descriptor_digest == manifest.descriptor_digest
        && request.target == manifest.version
        && request.manifest_identity == manifest.identity
}
