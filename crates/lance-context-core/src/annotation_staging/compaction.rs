//! Install confirmed native maintenance without retiring annotation updates.
use super::*;
use lance_table::transaction::{Operation, Transaction};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Maintenance {
    NarrowCompaction,
    BodyCompaction,
    Index,
}

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
        self.install_maintenance(transaction, selected, Maintenance::NarrowCompaction)
            .await
    }

    /// Install a confirmed body-only rewrite selected by a native metadata
    /// transaction. The caller verifies the exact row-preserving body rewrite
    /// and fenced publication. Narrow files, cut and results remain unchanged;
    /// active updates and already captured body readers keep their own views.
    pub async fn install_body_compacted(
        &self,
        transaction: &Transaction,
        selected: Arc<Dataset>,
    ) -> Result<()> {
        self.install_maintenance(transaction, selected, Maintenance::BodyCompaction)
            .await
    }

    /// Switch the base after a confirmed native index-only publication.
    /// The caller validates index fields, coverage and replacement identities
    /// against its pinned source and resolves the exact fenced publication.
    /// This method does not build, schedule or authorize an index publication.
    ///
    /// Data fragments, annotation cut, request results and body reference must
    /// stay unchanged. Active updates and captured readers retain their views;
    /// a frozen annotation publication must finish before this installation.
    pub async fn install_indexed(
        &self,
        transaction: &Transaction,
        selected: Arc<Dataset>,
    ) -> Result<()> {
        self.install_maintenance(transaction, selected, Maintenance::Index)
            .await
    }

    async fn install_maintenance(
        &self,
        transaction: &Transaction,
        selected: Arc<Dataset>,
        maintenance: Maintenance,
    ) -> Result<()> {
        let (base, materialized) = {
            let state = self.state.read().await;
            if state.frozen.is_some() {
                return Err(invalid(
                    "maintenance overlaps a frozen annotation generation",
                ));
            }
            (state.base.clone(), state.materialized)
        };
        let body_only = maintenance == Maintenance::BodyCompaction;
        let operation = match maintenance {
            Maintenance::BodyCompaction => {
                matches!(&transaction.operation, Operation::UpdateConfig {
                config_updates: None,
                table_metadata_updates: None,
                schema_metadata_updates: None,
                field_metadata_updates,
            } if field_metadata_updates.is_empty())
                    && selected.manifest().fragments == base.manifest().fragments
            }
            Maintenance::NarrowCompaction => {
                matches!(&transaction.operation, Operation::Rewrite { groups, .. } if !groups.is_empty())
            }
            Maintenance::Index => {
                matches!(&transaction.operation, Operation::CreateIndex { new_indices, .. } if !new_indices.is_empty())
                    && selected.manifest().fragments == base.manifest().fragments
            }
        };
        if !operation
            || transaction.read_version != base.version_id()
            || selected.uri() != base.uri()
            || selected.version_id() != base.version_id() + 1
            || selected.schema() != base.schema()
            || selected.read_transaction().await?.as_ref() != Some(transaction)
        {
            return Err(invalid("selected snapshot is not this native maintenance"));
        }
        let original = base
            .read_transaction()
            .await?
            .ok_or_else(|| invalid("maintenance base has no annotation transaction"))?;
        let properties = original.transaction_properties.as_ref();
        let cut = properties
            .and_then(|p| p.get("lance-context.annotation-cut"))
            .ok_or_else(|| invalid("maintenance base has no annotation cut"))?;
        let (scope, through): (Scope, u64) =
            serde_json::from_str(cut).map_err(|e| invalid(e.to_string()))?;
        if scope != self.scope || through != materialized {
            return Err(invalid("maintenance base annotation cut differs"));
        }
        for key in [
            "lance-context.annotation-cut",
            RESULTS_PROPERTY,
            BODY_SNAPSHOT_PROPERTY,
        ] {
            if body_only && key == BODY_SNAPSHOT_PROPERTY {
                continue;
            }
            if properties.and_then(|p| p.get(key))
                != transaction
                    .transaction_properties
                    .as_ref()
                    .and_then(|p| p.get(key))
            {
                return Err(invalid(
                    "maintenance changed annotation cut, request results or body reference",
                ));
            }
        }
        if body_only {
            let prior = BodySnapshot::read(&base)
                .await?
                .ok_or_else(|| invalid("body maintenance base has no body reference"))?;
            let next = BodySnapshot::read(&selected)
                .await?
                .ok_or_else(|| invalid("body maintenance dropped its body reference"))?;
            if next.uri != prior.uri
                || next.table_instance != prior.table_instance
                || next.version == prior.version
                || !lance_table::format::is_detached_version(next.version)
                || next.transaction != transaction.uuid
            {
                return Err(invalid(
                    "body maintenance changed its binding or native identity",
                ));
            }
        }
        let _mutation = self.mutation.lock().await;
        let mut state = self.state.write().await;
        if state.frozen.is_some() || !Arc::ptr_eq(&state.base, &base) {
            return Err(invalid(
                "annotation base changed during maintenance installation",
            ));
        }
        state.base = if maintenance == Maintenance::Index {
            // Fragments, schema, cut and request results were checked above.
            // Keep decoded patch values usable, while readers still retain
            // their original physical base for reclamation protection.
            state.views.install_indexed(selected, &base)
        } else {
            state.views.install(selected)
        };
        Ok(())
    }
}
