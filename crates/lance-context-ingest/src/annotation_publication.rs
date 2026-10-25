//! Preserve a frozen annotation transaction through the existing WAL catalog.
//!
//! This adapter handles preparation metadata and the fenced Lance commit. Its
//! caller still orders committed patches, pins the complete source prefix,
//! records preparation, confirms the selected manifest, and authorizes closure.
//! It never acknowledges a mutation or advances a source cursor itself.
use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use lance::{dataset::CommitBuilder, Dataset};
use lance_context_core::annotation_staging::{PreparedAnnotations, Scope};
use lance_context_wal::{placement::Layout, record::RecordKind, replication::publication as p};
use lance_table::{
    format::pb,
    transaction::{Operation, Transaction, UpdateMode},
};
use prost::Message;
use std::{collections::BTreeSet, sync::Arc};
use uuid::Uuid;

use crate::materializer::Storage;

mod metadata;

/// The first annotation publisher covers one shard in one source group. A
/// source pin must prove this plan contains every relevant record in its range;
/// a per-shard cut alone cannot authorize closing a shared group's prefix.
pub struct Selection {
    pub table: Uuid,
    pub incarnation: Uuid,
    pub pending: p::PendingPlan,
}
impl Selection {
    pub(crate) fn scope(&self) -> Result<(Scope, u64)> {
        let plan = &self.pending.plan;
        ensure!(
            !self.table.is_nil()
                && !self.incarnation.is_nil()
                && self.pending.index > 0
                && self.pending.index < u64::MAX
                && !plan.transaction.is_nil()
                && plan.sources.len() == 1,
            "annotation publication requires one scoped source selection"
        );
        plan.target_version()?;
        let (&group, range) = plan.sources.first_key_value().unwrap();
        let (_, first) = range.records.first().context("empty annotation prefix")?;
        ensure!(
            group > 0
                && range.after < range.through
                && range.through < u64::MAX
                && range.records.len() <= p::MAX_INPUTS
                && range.records.windows(2).all(|w| w[0].0 < w[1].0)
                && range.records.iter().all(|(index, meta)| {
                    *index > range.after
                        && *index <= range.through
                        && meta.kind == RecordKind::Data
                        && meta.stream == self.table
                        && meta.bucket == first.bucket
                }),
            "annotation selection mixes shards or contains an invalid prefix"
        );
        Ok((
            Scope {
                table_instance: self.incarnation.to_string(),
                source_group: group,
                shard: first.bucket,
            },
            range.through,
        ))
    }
}

/// Encode the original native transaction, including its cut, file selection,
/// modified fields and row-version offsets. Only metadata is encoded here.
/// Retain these exact bytes through an ambiguous Prepare RPC; protobuf maps
/// need not encode in the same order on a subsequent call.
pub fn descriptors(
    prepared: &PreparedAnnotations,
    selection: &Selection,
) -> Result<p::Descriptors> {
    let transaction = &prepared.merge().transaction;
    validate(prepared.base(), transaction, selection)?;
    ensure!(
        prepared.through() == selection.scope()?.1,
        "frozen annotation cut differs from source selection"
    );
    p::Descriptors::lance_annotations(Bytes::from(
        pb::Transaction::from(transaction).encode_to_vec(),
    ))
}

/// Persist large native preparation metadata with its immutable Lance files.
/// Small preparations retain the existing inline encoding. An upload error
/// stops this attempt before Prepare; recovery still follows the original plan.
pub async fn persist_descriptors(
    prepared: &PreparedAnnotations,
    selection: &Selection,
    storage: &Storage,
    maximum: u64,
) -> Result<p::Descriptors> {
    ensure!(
        (1..=p::MAX_DESCRIPTOR_BYTES as u64).contains(&maximum),
        "invalid annotation descriptor budget"
    );
    ensure!(
        prepared.base().uri() == storage.uri,
        "foreign metadata storage"
    );
    let transaction = &prepared.merge().transaction;
    validate(prepared.base(), transaction, selection)?;
    ensure!(
        prepared.through() == selection.scope()?.1,
        "foreign metadata cut"
    );
    let native = pb::Transaction::from(transaction);
    ensure!(
        native.encoded_len() <= metadata::MAX_NATIVE_BYTES,
        "native annotation preparation exceeds metadata limit"
    );
    let bytes = Bytes::from(native.encode_to_vec());
    if bytes.len() as u64 <= maximum {
        p::Descriptors::lance_annotations(bytes)
    } else {
        metadata::persist(prepared.base(), storage, selection, bytes, maximum).await
    }
}

/// Recover the exact preparation against its original base. Do not run merge
/// insert again when the catalog already retains these descriptors.
pub fn transaction(
    base: &Dataset,
    descriptors: &p::Descriptors,
    selection: &Selection,
) -> Result<Transaction> {
    descriptors.validate(p::MAX_DESCRIPTOR_BYTES as u64)?;
    ensure!(descriptors.format == 4, "not annotation descriptors");
    decode(base, descriptors.bytes.clone(), selection)
}

fn decode(base: &Dataset, bytes: Bytes, selection: &Selection) -> Result<Transaction> {
    ensure!(
        bytes.len() <= metadata::MAX_NATIVE_BYTES,
        "native annotation metadata too large"
    );
    let transaction = Transaction::try_from(pb::Transaction::decode(bytes)?)?;
    validate(base, &transaction, selection)?;
    Ok(transaction)
}

async fn resolve_transaction(
    base: &Dataset,
    descriptors: &p::Descriptors,
    selection: &Selection,
    storage: &Storage,
) -> Result<Transaction> {
    if descriptors.format == 4 {
        transaction(base, descriptors, selection)
    } else {
        ensure!(descriptors.format == 5, "not annotation descriptors");
        descriptors.validate(p::MAX_DESCRIPTOR_BYTES as u64)?;
        decode(
            base,
            metadata::load(base, storage, selection, descriptors).await?,
            selection,
        )
    }
}

/// Resolve a pending publication that already selected its manifest, including
/// a crash before confirmation. Obtain `record` from the scoped quorum catalog;
/// do not infer selection from a blob listing or a locally uploaded manifest.
pub async fn selected_cut(
    selected: &Dataset,
    original_base: &Dataset,
    descriptors: &p::Descriptors,
    selection: &Selection,
    record: &p::ManifestRecord,
    storage: &Storage,
) -> Result<u64> {
    let expected = resolve_transaction(original_base, descriptors, selection, storage).await?;
    ensure!(
        selected.uri() == original_base.uri()
            && selected.version_id() == selection.pending.plan.target_version()?
            && record.version == selected.version_id()
            && record.transaction == selection.pending.plan.transaction
            && record.plan_index == selection.pending.index
            && record.prepared_index > selection.pending.index
            && record.descriptor_digest == descriptors.digest
            && selected.read_transaction().await?.as_ref() == Some(&expected),
        "selected annotation snapshot differs from durable preparation"
    );
    Ok(selection.scope()?.1)
}

fn validate(base: &Dataset, transaction: &Transaction, selection: &Selection) -> Result<()> {
    ensure!(
        transaction.read_version == base.version_id()
            && transaction.read_version == selection.pending.plan.read_version
            && transaction.uuid == selection.pending.plan.transaction.to_string()
            && transaction.tag.is_none(),
        "annotation transaction differs from its fixed publication target"
    );
    let cut = transaction
        .transaction_properties
        .as_ref()
        .and_then(|p| p.get("lance-context.annotation-cut"))
        .context("annotation transaction has no source cut")?;
    ensure!(
        serde_json::from_str::<(Scope, u64)>(cut)? == selection.scope()?,
        "annotation transaction has a foreign source cut"
    );
    let results = transaction
        .transaction_properties
        .as_ref()
        .and_then(|p| p.get(lance_context_core::annotation_staging::RESULTS_PROPERTY));
    if let Some(results) = results {
        crate::annotations::checkpoint::Snapshot::decode(results, selection.scope()?.1)?;
    }
    if let Operation::UpdateConfig {
        config_updates,
        table_metadata_updates,
        schema_metadata_updates,
        field_metadata_updates,
    } = &transaction.operation
    {
        ensure!(
            results.is_some()
                && config_updates.is_none()
                && table_metadata_updates.is_none()
                && schema_metadata_updates.is_none()
                && field_metadata_updates.is_empty(),
            "result-only publication may only record the application cut and results"
        );
        return Ok(());
    }
    let Operation::Update {
        removed_fragment_ids,
        updated_fragments,
        new_fragments,
        fields_modified,
        compacted_sstables,
        fields_for_preserving_frag_bitmap,
        update_mode,
        inserted_rows_filter,
        updated_fragment_offsets,
    } = &transaction.operation
    else {
        anyhow::bail!("annotation preparation is not a native update");
    };
    ensure!(
        removed_fragment_ids.is_empty()
            && new_fragments.is_empty()
            && !updated_fragments.is_empty()
            && compacted_sstables.is_empty()
            && fields_for_preserving_frag_bitmap.is_empty()
            && *update_mode == Some(UpdateMode::RewriteColumns)
            && inserted_rows_filter.is_none()
            && updated_fragment_offsets.is_some(),
        "annotation preparation must preserve existing rows and rewrite columns"
    );
    // Join keys may also be present in the native partial update source. All
    // other fields (especially the body) must keep their original files.
    let allowed: BTreeSet<i32> = ["id", "labels", "status", "score"]
        .into_iter()
        .map(|name| base.schema().project(&[name]))
        .collect::<lance::Result<Vec<_>>>()?
        .iter()
        .flat_map(|schema| schema.field_ids())
        .collect();
    ensure!(
        !fields_modified.is_empty()
            && fields_modified
                .iter()
                .all(|id| allowed.contains(&(*id as i32))),
        "annotation update modifies a field outside the narrow projection"
    );
    let mut ids = BTreeSet::new();
    for updated in updated_fragments {
        let original = base
            .manifest()
            .fragments
            .iter()
            .find(|f| f.id == updated.id)
            .context("annotation update references a foreign fragment")?;
        ensure!(
            ids.insert(updated.id)
                && updated.physical_rows == original.physical_rows
                && updated.deletion_file == original.deletion_file
                && updated.row_id_meta == original.row_id_meta
                && updated.created_at_version_meta == original.created_at_version_meta
                && updated.overlays == original.overlays,
            "annotation update changes fragment membership"
        );
        for old in &original.files {
            // Native column updates tombstone replaced fields (-2) in old
            // mixed-column files. Preserve every other field/column mapping.
            let mut expected = old.clone();
            expected.fields = old
                .fields
                .iter()
                .map(|id| {
                    if fields_modified.contains(&(*id as u32)) {
                        -2
                    } else {
                        *id
                    }
                })
                .collect::<Vec<_>>()
                .into();
            ensure!(
                updated.files.contains(&expected),
                "annotation update replaced a base file"
            );
        }
        let mut file_ids = BTreeSet::new();
        for file in &updated.files {
            ensure!(
                file_ids.insert((file.base_id, file.path.as_str())),
                "annotation update repeats a file reference"
            );
            if original
                .files
                .iter()
                .any(|old| old.path == file.path && old.base_id == file.base_id)
            {
                continue;
            }
            // An ID slot can be tombstoned when preparation keeps the original
            // physical key column. Require a live narrow field as well.
            ensure!(
                file.fields.iter().any(|id| allowed.contains(id))
                    && file
                        .fields
                        .iter()
                        .all(|id| *id == -2 || allowed.contains(id)),
                "annotation update writes a field outside the narrow projection"
            );
        }
    }
    Ok(())
}

/// Write the fixed manifest through the existing quorum catalog. Success here
/// is manifest selection, not publication confirmation or source GC. An error
/// may be an unknown outcome: reconcile the same target/identity at quorum.
pub async fn publish(
    catalog: Arc<dyn p::lance::ManifestCatalog>,
    layout: &Layout,
    storage: &Storage,
    commit: p::lance::ManifestCommit,
    selection: &Selection,
    descriptors: &p::Descriptors,
    base: Arc<Dataset>,
) -> Result<Dataset> {
    ensure!(
        commit.authority.table == selection.table
            && commit.incarnation == selection.incarnation
            && commit.transaction == selection.pending.plan.transaction
            && commit.plan_index == selection.pending.index
            && commit.descriptor_digest == descriptors.digest
            && commit.target == selection.pending.plan.target_version()?
            && base.uri() == storage.uri,
        "annotation manifest scope differs from preparation"
    );
    let transaction = resolve_transaction(&base, descriptors, selection, storage).await?;
    let handler = p::lance::ManagedManifest::new(catalog, storage.base.clone(), layout, commit)?;
    Ok(CommitBuilder::new(base)
        .with_commit_handler(Arc::new(handler))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?)
}
