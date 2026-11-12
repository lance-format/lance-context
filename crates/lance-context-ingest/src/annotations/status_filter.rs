use super::*;
use lance_context_core::annotation_staging::{StatusMatches, MAX_STATUS_MATCHES};

impl Service {
    pub(crate) fn validate_status_filter(
        &self,
        bucket: u32,
        status: Option<&str>,
        limit: usize,
    ) -> Result<()> {
        ensure!(
            bucket == self.bucket()
                && (1..=MAX_STATUS_MATCHES).contains(&limit)
                && status.is_none_or(|s| s.len() <= self.limits.value_bytes),
            "invalid annotation status filter"
        );
        Ok(())
    }

    /// The HTTP owner fences the result after this query. The acquired epoch's
    /// recovery cut is required, but a later unacknowledged proposal cannot
    /// prevent a query of the already selected materialized snapshot.
    /// None means shared memory refused admission before native work started.
    pub(crate) async fn filter_status_owned(
        &self,
        status: Option<String>,
        limit: usize,
        response_bytes: usize,
        recovery_cut: u64,
        deadline: Duration,
    ) -> Result<Option<(StatusMatches, MergeReservation)>> {
        tokio::time::timeout(deadline, async {
            self.wait_through(recovery_cut).await?;
            // HTTP admission bounds these waiters. Acquire the shared reader
            // before capturing a base, within the original request deadline.
            let read = self
                .shared
                .staging
                .capture_status_when_available(status, limit, response_bytes)
                .await?;
            let workspace = read.workspace_bytes()?;
            ensure!(
                workspace <= self.ingress.budget.limit(),
                "status filter workspace exceeds budget"
            );
            // Optional filtering must not queue a large reservation ahead of
            // ingestion. Refuse promptly when shared capacity is occupied.
            let Some(reservation) = self.ingress.budget.try_reserve(workspace) else {
                return Ok(None);
            };
            // Native query tasks may outlive cancellation of an HTTP request.
            // Keep their base pin and reservation until the actual work ends.
            tokio::spawn(async move {
                let result = read.resolve().await?;
                Ok::<_, anyhow::Error>(Some((result, reservation)))
            })
            .await?
        })
        .await
        .context("annotation status query deadline expired")?
    }
}
