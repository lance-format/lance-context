//! Install a confirmed physical rewrite without retiring annotation updates.
use super::*;
use lance_table::transaction::{Operation, Transaction};

impl AnnotationStaging {
    /// Switch the base after a confirmed, row-preserving native compaction.
    /// The caller must validate the rewrite against its pinned source snapshot
    /// and resolve the exact fenced publication before calling this method.
    /// Matching a native transaction is not itself a quorum or reclamation proof.
    ///
    /// Compaction must preserve the base's annotation cut and request results.
    /// It cannot overlap a frozen publication. Active updates and their applied
    /// position remain intact; captured cold readers continue pinning the old
    /// base through the same registry used by ordinary annotation publication.
    pub async fn install_compacted(
        &self,
        transaction: &Transaction,
        selected: Arc<Dataset>,
    ) -> Result<()> {
        let (base, materialized) = {
            let state = self.state.read().await;
            if state.frozen.is_some() {
                return Err(invalid(
                    "compaction overlaps a frozen annotation generation",
                ));
            }
            (state.base.clone(), state.materialized)
        };
        if !matches!(&transaction.operation, Operation::Rewrite { groups, .. } if !groups.is_empty())
            || transaction.read_version != base.version_id()
            || selected.uri() != base.uri()
            || selected.version_id() != base.version_id() + 1
            || selected.schema() != base.schema()
            || selected.read_transaction().await?.as_ref() != Some(transaction)
        {
            return Err(invalid("selected snapshot is not this native compaction"));
        }
        let original = base
            .read_transaction()
            .await?
            .ok_or_else(|| invalid("compaction base has no annotation transaction"))?;
        let properties = original.transaction_properties.as_ref();
        let cut = properties
            .and_then(|p| p.get("lance-context.annotation-cut"))
            .ok_or_else(|| invalid("compaction base has no annotation cut"))?;
        let (scope, through): (Scope, u64) =
            serde_json::from_str(cut).map_err(|e| invalid(e.to_string()))?;
        if scope != self.scope || through != materialized {
            return Err(invalid("compaction base annotation cut differs"));
        }
        for key in [
            "lance-context.annotation-cut",
            RESULTS_PROPERTY,
            BODY_SNAPSHOT_PROPERTY,
        ] {
            if properties.and_then(|p| p.get(key))
                != transaction
                    .transaction_properties
                    .as_ref()
                    .and_then(|p| p.get(key))
            {
                return Err(invalid(
                    "compaction changed annotation cut, request results or body reference",
                ));
            }
        }
        let _mutation = self.mutation.lock().await;
        let mut state = self.state.write().await;
        if state.frozen.is_some() || !Arc::ptr_eq(&state.base, &base) {
            return Err(invalid(
                "annotation base changed during compaction installation",
            ));
        }
        state.base = state.views.install(selected);
        Ok(())
    }
}
