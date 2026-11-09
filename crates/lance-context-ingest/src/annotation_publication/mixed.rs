//! Explicit paired publication; the patch-only adapter keeps rejecting inserts.
use super::*;
use lance_context_core::annotation_staging::{BodySnapshot, BODY_SNAPSHOT_PROPERTY};
use lance_table::format::Fragment;

pub mod maintenance;
mod reader;

fn inserts(transaction: &Transaction) -> &[Fragment] {
    match &transaction.operation {
        Operation::Append { fragments } => fragments,
        Operation::Update { new_fragments, .. } => new_fragments,
        _ => &[],
    }
}

pub(super) fn validate_narrow(base: &Dataset, transaction: &Transaction) -> Result<()> {
    let names = base
        .schema()
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        names == BTreeSet::from(["id", "rollout_id", "labels", "status", "score"]),
        "mixed annotations require the exact narrow table"
    );
    let fields = base
        .schema()
        .field_ids()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut files = base
        .manifest()
        .fragments
        .iter()
        .flat_map(|f| f.files.iter())
        .map(|f| (f.base_id, f.path.as_str()))
        .collect::<BTreeSet<_>>();
    for fragment in inserts(transaction) {
        ensure!(
            fragment.physical_rows.is_some_and(|n| n > 0)
                && fragment.deletion_file.is_none()
                && fragment.overlays.is_empty(),
            "new annotation fragment changes membership"
        );
        let mut written = BTreeSet::new();
        let mut recorded = BTreeSet::new();
        for file in &fragment.files {
            ensure!(
                files.insert((file.base_id, file.path.as_str()))
                    && file
                        .fields
                        .iter()
                        .all(|id| fields.contains(id) && recorded.insert(*id)),
                "new annotation fragment aliases files or writes foreign/duplicate fields"
            );
            // V2.1+ maps physical leaf columns; a list's structural parent
            // need not have its own entry. Use Lance's version-aware schema
            // projection to prove complete logical coverage (including labels).
            written.extend(file.schema(base.schema()).field_ids());
        }
        ensure!(
            written == fields,
            "new annotation fragment omits narrow fields"
        );
    }
    if matches!(transaction.operation, Operation::Append { .. }) {
        ensure!(!inserts(transaction).is_empty(), "empty mixed append");
    }
    Ok(())
}

fn rows(fragments: &[Fragment]) -> Result<u64> {
    fragments.iter().try_fold(0u64, |total, f| {
        total
            .checked_add(f.physical_rows.context("body fragment lacks row count")? as u64)
            .context("body row count overflow")
    })
}

async fn validate(
    base: &Dataset,
    transaction: &Transaction,
    selection: &Selection,
    body: &Storage,
) -> Result<()> {
    if selection.pending.plan.is_maintenance() {
        if matches!(transaction.operation, Operation::Rewrite { .. }) {
            super::compaction::validate(base, transaction, selection).await?;
            validate_narrow(base, transaction)?;
            let reference = BodySnapshot::read(base)
                .await?
                .context("missing body reference")?;
            ensure!(
                reference.uri == body.uri
                    && reference.table_instance == selection.incarnation.to_string(),
                "narrow maintenance has foreign body destination"
            );
            return Ok(());
        }
        return maintenance::validate(base, transaction, selection, body).await;
    }
    validate_native(base, transaction, selection, true)?;
    let encoded = transaction
        .transaction_properties
        .as_ref()
        .and_then(|p| p.get(BODY_SNAPSHOT_PROPERTY))
        .context("mixed preparation has no body identity")?;
    ensure!(
        encoded.len() <= 32 << 10,
        "mixed body identity is oversized"
    );
    let reference: BodySnapshot = serde_json::from_str(encoded)?;
    let prior = BodySnapshot::read(base)
        .await?
        .context("mixed base has no body identity")?;
    ensure!(
        reference.uri == body.uri
            && prior.uri == body.uri
            && reference.table_instance == selection.incarnation.to_string()
            && prior.table_instance == reference.table_instance,
        "mixed body identity differs from authorized destination/incarnation"
    );
    let inserted = rows(inserts(transaction))?;
    if reference == prior {
        ensure!(inserted == 0, "new annotations have no new body snapshot");
        return Ok(());
    }
    ensure!(
        inserted > 0
            && reference.version != prior.version
            && lance_table::format::is_detached_version(reference.version),
        "mixed ingestion may only advance the body for new turns"
    );
    let previous = load_body(body, &prior, &reference.table_instance).await?;
    let next = load_body(body, &reference, &reference.table_instance).await?;
    let native = next
        .read_transaction()
        .await?
        .context("body has no transaction")?;
    ensure!(
        native.read_version == prior.version
            && native.uuid == transaction.uuid
            && native.tag.is_none()
            && native
                .transaction_properties
                .as_ref()
                .and_then(|p| p.get("lance-context.annotation-cut"))
                == transaction
                    .transaction_properties
                    .as_ref()
                    .and_then(|p| p.get("lance-context.annotation-cut")),
        "body snapshot is not the original paired preparation"
    );
    let Operation::Append { fragments } = &native.operation else {
        anyhow::bail!("mixed body preparation must append first-create rows");
    };
    ensure!(
        rows(fragments)? == inserted
            && next.manifest().fragments.len()
                == previous.manifest().fragments.len() + fragments.len()
            && previous
                .manifest()
                .fragments
                .iter()
                .all(|old| next.manifest().fragments.contains(old)),
        "body snapshot drops prior rows or differs from narrow insert coverage"
    );
    Ok(())
}

pub async fn persist_descriptors(
    prepared: &PreparedAnnotations,
    selection: &Selection,
    storage: &Storage,
    body: &Storage,
    maximum: u64,
) -> Result<p::Descriptors> {
    validate(
        prepared.base(),
        &prepared.merge().transaction,
        selection,
        body,
    )
    .await?;
    persist_native(prepared, selection, storage, maximum).await
}

pub(crate) async fn resolve(
    base: &Dataset,
    descriptors: &p::Descriptors,
    selection: &Selection,
    storage: &Storage,
    body: &Storage,
) -> Result<Transaction> {
    descriptors.validate(p::MAX_DESCRIPTOR_BYTES as u64)?;
    let bytes = match descriptors.format {
        4 => descriptors.bytes.clone(),
        5 => metadata::load(base, storage, selection, descriptors).await?,
        _ => anyhow::bail!("not mixed native descriptors"),
    };
    ensure!(
        bytes.len() <= metadata::MAX_NATIVE_BYTES,
        "mixed native metadata is oversized"
    );
    let transaction = Transaction::try_from(pb::Transaction::decode(bytes)?)?;
    validate(base, &transaction, selection, body).await?;
    Ok(transaction)
}

pub async fn selected_cut(
    selected: &Dataset,
    base: &Dataset,
    descriptors: &p::Descriptors,
    selection: &Selection,
    record: &p::ManifestRecord,
    storage: &Storage,
    body: &Storage,
) -> Result<u64> {
    let expected = resolve(base, descriptors, selection, storage, body).await?;
    selected_identity(selected, base, descriptors, selection, record, &expected).await
}

#[allow(clippy::too_many_arguments)] // Same commit authority, plus the paired destination.
pub async fn publish(
    catalog: Arc<dyn p::lance::ManifestCatalog>,
    layout: &Layout,
    storage: &Storage,
    body: &Storage,
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
        "mixed manifest scope differs from preparation"
    );
    let transaction = resolve(&base, descriptors, selection, storage, body).await?;
    let handler = p::lance::ManagedManifest::new(catalog, storage.base.clone(), layout, commit)?;
    Ok(CommitBuilder::new(base)
        .with_commit_handler(Arc::new(handler))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?)
}

async fn load_body(
    storage: &Storage,
    reference: &lance_context_core::annotation_staging::BodySnapshot,
    incarnation: &str,
) -> Result<lance::Dataset> {
    ensure!(
        reference.uri == storage.uri && reference.table_instance == incarnation,
        "body snapshot differs from the authorized destination"
    );
    let body = lance::dataset::builder::DatasetBuilder::from_uri(&storage.uri)
        .with_storage_options(storage.options.clone())
        .with_session(storage.session.clone())
        .with_commit_handler(Arc::new(reader::BodyManifestReader))
        .with_version(reference.version)
        .load()
        .await?;
    ensure!(
        body.read_transaction()
            .await?
            .is_some_and(|tx| tx.uuid == reference.transaction),
        "body native identity differs from selected reference"
    );
    let schema: arrow_schema::Schema = body.schema().into();
    ensure!(
        schema == crate::rollout_schema(),
        "body snapshot has a foreign schema"
    );
    Ok(body)
}
