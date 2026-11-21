//! Resolve an uncertain Confirm without replacing its selected publication.
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
        loop {
            let response = self
                .ingress
                .client
                .request_routed(Request::Commit(Box::new(Command::Confirm(request.clone()))))
                .await;
            let (error, uncertain) = match response.map(|r| r.reply) {
                Ok(Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Published(
                    published,
                ))))) => {
                    ensure!(matches(&request, &published), "foreign publication outcome");
                    return Ok(published);
                }
                Ok(Reply::Unknown | Reply::Commit(Outcome::Unknown)) => (
                    anyhow::anyhow!(
                        "durable command outcome unknown; reconcile the original operation"
                    ),
                    true,
                ),
                Ok(reply) => bail!("publication confirmation rejected: {reply:?}"),
                Err(error) => (error, false),
            };
            let observed: Result<Option<Arc<p::Published>>> = async {
                // state() checks the quorum observation, current owner, binding
                // and incarnation. Files alone never prove Confirm committed.
                let view = self.state().await?;
                if let Some(published) = view
                    .progress
                    .published
                    .iter()
                    .chain(view.progress.advanced.iter().map(|a| &a.publication))
                    .find(|published| matches(&request, published))
                {
                    return Ok(Some(published.clone()));
                }
                ensure!(uncertain, "original publication confirmation not observed");
                validate_pending(&request, &view)?;
                Ok(None)
            }
            .await;
            match observed {
                Ok(Some(published)) => return Ok(published),
                Ok(None) => {
                    // Only an explicit Unknown plus the identical selected plan
                    // permits reissuing this idempotent control command. Keep
                    // the caller's frozen generation/files and memory permits.
                    // Apply/read stay live under their existing bounds, and
                    // cancellation leaves the same durable recovery evidence.
                    // No uploads or business mutations are repeated here.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(observation) => {
                    return Err(error).context(format!(
                        "original publication confirmation observation failed: {observation:#}"
                    ));
                }
            }
        }
    }
}

fn validate_pending(request: &p::ConfirmPublication, view: &p::Table) -> Result<()> {
    ensure!(
        view.progress.published.is_none(),
        "different publication confirmed"
    );
    let pending = view
        .progress
        .pending
        .as_ref()
        .context("original plan unavailable")?;
    let prepared = view
        .progress
        .prepared
        .as_ref()
        .context("original preparation unavailable")?;
    let manifests = view
        .manifests
        .as_ref()
        .context("original manifests unavailable")?;
    let selected = manifests
        .versions
        .get(&request.target)
        .context("original manifest unavailable")?;
    ensure!(
        pending.index == request.plan_index
            && pending.plan.transaction == request.transaction
            && pending.plan.target_version()? == request.target
            && prepared.index == request.prepared_index
            && prepared.plan_index == request.plan_index
            && prepared.transaction == request.transaction
            && prepared.descriptors.digest == request.descriptor_digest
            && manifests.head == request.target
            && manifests.incarnation == request.incarnation
            && matches_manifest(request, selected),
        "original publication confirmation identity changed"
    );
    Ok(())
}

fn matches_manifest(request: &p::ConfirmPublication, manifest: &p::ManifestRecord) -> bool {
    request.transaction == manifest.transaction
        && request.plan_index == manifest.plan_index
        && request.prepared_index == manifest.prepared_index
        && request.descriptor_digest == manifest.descriptor_digest
        && request.target == manifest.version
        && request.manifest_identity == manifest.identity
}

fn matches(request: &p::ConfirmPublication, published: &p::Published) -> bool {
    request.incarnation == published.incarnation && matches_manifest(request, &published.manifest)
}
