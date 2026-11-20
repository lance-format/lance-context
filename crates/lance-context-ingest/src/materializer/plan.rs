//! Resolve an uncertain plan without replacing its selection or authority.
use super::*;

impl Publisher {
    pub(crate) async fn record_plan(&self, plan: p::Plan) -> Result<()> {
        let request = p::PlanRequest {
            authority: self.authority.clone(),
            plan,
        };
        loop {
            let response = self
                .ingress
                .client
                .request_routed(Request::Commit(Box::new(Command::Plan(request.clone()))))
                .await;
            let (error, uncertain) = match response.map(|r| r.reply) {
                Ok(Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Planned {
                    ..
                })))) => return Ok(()),
                Ok(reply @ (Reply::Unknown | Reply::Commit(Outcome::Unknown))) => (
                    anyhow::anyhow!("publication plan outcome unknown: {reply:?}"),
                    true,
                ),
                Ok(reply) => bail!("publication selection rejected: {reply:?}"),
                Err(error) => (error, false),
            };
            let observed: Result<bool> = async {
                // This quorum observation also checks owner, table binding and
                // incarnation. A matching UUID alone does not confirm the plan.
                let view = self.state().await?;
                if let Some(pending) = &view.progress.pending {
                    ensure!(
                        pending.plan == request.plan,
                        "different publication plan selected"
                    );
                    return Ok(true);
                }
                ensure!(uncertain, "original publication plan not observed");
                validate_predecessor(&request.plan, &view)?;
                Ok(false)
            }
            .await;
            match observed {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    // Catalog::plan checks authority before accepting an exact
                    // duplicate and returns its original index. Retain the same
                    // request even if the first attempt commits after our read.
                    // No new selection, upload or business mutation occurs here.
                    // Dropping this future cancels retries; staging keeps its
                    // existing bounds while this one control request is pending.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(observation) => {
                    return Err(error).context(format!(
                        "original publication plan observation failed: {observation:#}"
                    ));
                }
            }
        }
    }
}

fn validate_predecessor(plan: &p::Plan, view: &p::Table) -> Result<()> {
    ensure!(
        view.progress.prepared.is_none()
            && view.progress.published.is_none()
            && view.progress.minimum_read_version <= plan.read_version
            && view
                .manifests
                .as_ref()
                .is_some_and(|m| m.head == plan.read_version)
            && plan.sources.iter().all(|(group, range)| {
                range.after == view.progress.checkpoint.get(group).copied().unwrap_or(0)
            }),
        "original publication plan predecessor changed"
    );
    Ok(())
}
