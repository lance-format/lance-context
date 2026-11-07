//! Body maintenance shares the owner's plan, budgets and selected narrow head.
use super::*;
use lance::dataset::{
    transaction::{Operation, RewriteGroup, Transaction},
    CommitBuilder,
};
use lance_context_core::annotation_staging::BodySnapshot;

impl Publisher {
    pub(super) async fn body_compaction_candidate(
        &mut self,
        version: u64,
    ) -> Result<Option<Dataset>> {
        let (Some(body), Some(options)) = (&self.service.shared.body, &self.options.compaction)
        else {
            return Ok(None);
        };
        let base = checkpoint::load(&self.service.ingress, &self.storage, version).await?;
        let dataset = mixed::open_body(
            body,
            &base,
            &self.service.ingress.scope.incarnation.to_string(),
        )
        .await?;
        if !materializer::compaction::has_work_on(&dataset, options).await? {
            return Ok(None);
        }
        // Prefer the side with more fragments when both need maintenance.
        // This is derived from the fixed base, including after a plan-only
        // restart; continuous imports must not starve either table's rewrite.
        if dataset.manifest().fragments.len() < base.manifest().fragments.len()
            && materializer::compaction::has_work(&self.catalog, version).await?
        {
            return Ok(None);
        }
        Ok(Some(dataset))
    }

    pub(super) async fn prepare_body_compaction(
        &mut self,
        pending: &p::PendingPlan,
    ) -> Result<bool> {
        let Some(body_base) = self
            .body_compaction_candidate(pending.plan.read_version)
            .await?
        else {
            return Ok(false);
        };
        let base = checkpoint::load(
            &self.service.ingress,
            &self.storage,
            pending.plan.read_version,
        )
        .await?;
        let native = base
            .read_transaction()
            .await?
            .context("body maintenance base has no native transaction")?;
        let cut = native
            .transaction_properties
            .as_ref()
            .and_then(|p| p.get("lance-context.annotation-cut"))
            .context("body maintenance base has no cut")?
            .clone();
        let (result, _reservation) = materializer::compaction::prepare_body_dataset(
            body_base.clone(),
            self.options.compaction.clone().unwrap(),
            self.service.ingress.scope.contract.max_record_bytes as usize,
            self.budget.clone(),
            self.service.shared.phases.clone(),
        )
        .await?;
        let mut transaction = Transaction::new_from_version(
            body_base.version_id(),
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: result.original_fragments,
                    new_fragments: result.new_fragments,
                }],
                rewritten_indices: Vec::new(),
                frag_reuse_index: None,
            },
        );
        transaction.uuid = pending.plan.transaction.to_string();
        transaction.transaction_properties = Some(Arc::new(std::collections::HashMap::from([(
            "lance-context.annotation-cut".into(),
            cut,
        )])));
        let selected = CommitBuilder::new(Arc::new(body_base))
            .with_detached(true)
            .with_max_retries(1)
            .with_skip_auto_cleanup(true)
            .execute(transaction)
            .await?;
        let reference = BodySnapshot::capture(
            self.service.ingress.scope.incarnation.to_string(),
            &selected,
        )
        .await?;
        let descriptors = publication::mixed::maintenance::descriptors(
            &base,
            reference,
            &self.selection(pending),
            &self.storage,
            self.service.shared.body.as_ref().unwrap(),
            self.service.ingress.client.max_descriptor_bytes(),
        )
        .await?;
        self.catalog
            .record_preparation(p::PrepareRequest {
                authority: self.catalog.authority().clone(),
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                descriptors,
            })
            .await?;
        Ok(true)
    }
}
