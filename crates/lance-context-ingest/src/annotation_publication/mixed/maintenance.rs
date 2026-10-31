//! Select a row-preserving detached body rewrite without changing annotations.
use super::*;
use lance::dataset::optimize::RewriteResult;
use lance_context_core::annotation_staging::RESULTS_PROPERTY;

pub async fn descriptors(
    base: &Dataset,
    reference: BodySnapshot,
    selection: &Selection,
    storage: &Storage,
    body: &Storage,
    maximum: u64,
) -> Result<p::Descriptors> {
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: Default::default(),
        },
    );
    transaction.uuid = selection.pending.plan.transaction.to_string();
    let mut properties = base
        .read_transaction()
        .await?
        .and_then(|t| t.transaction_properties)
        .context("body maintenance base has no checkpoint")?;
    Arc::make_mut(&mut properties).insert(
        BODY_SNAPSHOT_PROPERTY.into(),
        serde_json::to_string(&reference)?,
    );
    transaction.transaction_properties = Some(properties);
    validate(base, &transaction, selection, body).await?;
    persist_transaction(base, &transaction, selection, storage, maximum).await
}

pub(super) async fn validate(
    base: &Dataset,
    transaction: &Transaction,
    selection: &Selection,
    body: &Storage,
) -> Result<()> {
    selection.identity()?;
    ensure!(
        selection.pending.plan.is_maintenance()
            && transaction.read_version == base.version_id()
            && transaction.read_version == selection.pending.plan.read_version
            && transaction.uuid == selection.pending.plan.transaction.to_string()
            && transaction.tag.is_none(),
        "foreign body maintenance identity"
    );
    ensure!(
        matches!(&transaction.operation, Operation::UpdateConfig {
        config_updates: None, table_metadata_updates: None,
        schema_metadata_updates: None, field_metadata_updates,
    } if field_metadata_updates.is_empty()),
        "body maintenance modifies narrow data or configuration"
    );
    let old = base
        .read_transaction()
        .await?
        .and_then(|t| t.transaction_properties)
        .context("body maintenance base has no checkpoint")?;
    let properties = transaction
        .transaction_properties
        .as_ref()
        .context("body maintenance dropped the checkpoint")?;
    let encoded = properties
        .get(BODY_SNAPSHOT_PROPERTY)
        .context("body maintenance has no body reference")?;
    ensure!(encoded.len() <= 32 << 10, "oversized body reference");
    let reference: BodySnapshot = serde_json::from_str(encoded)?;
    let mut expected = (*old).clone();
    expected.insert(BODY_SNAPSHOT_PROPERTY.into(), encoded.clone());
    ensure!(
        properties.as_ref() == &expected,
        "body maintenance changed annotation checkpoint properties"
    );
    let cut = properties
        .get("lance-context.annotation-cut")
        .context("body maintenance has no cut")?;
    let (scope, through): (Scope, u64) = serde_json::from_str(cut)?;
    ensure!(
        scope.table_instance == selection.incarnation.to_string()
            && scope.source_group > 0
            && scope.shard < 65_536
            && through > 0
            && through < u64::MAX,
        "foreign body maintenance cut"
    );
    // Only the body reference may change. The selected base's source group,
    // bucket, cut and request results remain identical. next_group is a plan
    // scheduling cursor and may differ in an already retained maintenance plan.
    crate::annotations::checkpoint::Snapshot::decode(
        properties
            .get(RESULTS_PROPERTY)
            .context("body maintenance dropped request results")?,
        through,
    )?;
    let prior = BodySnapshot::read(base)
        .await?
        .context("base has no body reference")?;
    ensure!(
        prior.uri == body.uri
            && reference.uri == body.uri
            && prior.table_instance == scope.table_instance
            && reference.table_instance == scope.table_instance
            && reference.version != prior.version
            && lance_table::format::is_detached_version(reference.version)
            && reference.transaction == transaction.uuid,
        "body maintenance changed destination or native identity"
    );
    let previous = super::load_body(body, &prior, &scope.table_instance).await?;
    let next = super::load_body(body, &reference, &scope.table_instance).await?;
    let native = next
        .read_transaction()
        .await?
        .context("body rewrite has no transaction")?;
    ensure!(
        native.read_version == prior.version
            && native.uuid == transaction.uuid
            && native.tag.is_none()
            && next.schema() == previous.schema()
            && native
                .transaction_properties
                .as_ref()
                .and_then(|p| p.get("lance-context.annotation-cut"))
                == Some(cut),
        "body rewrite differs from original selected pair"
    );
    let Operation::Rewrite {
        groups,
        rewritten_indices,
        frag_reuse_index,
    } = &native.operation
    else {
        anyhow::bail!("body maintenance is not a native rewrite");
    };
    ensure!(
        groups.len() == 1 && rewritten_indices.is_empty() && frag_reuse_index.is_none(),
        "unsupported body rewrite operation"
    );
    let group = &groups[0];
    let result = RewriteResult {
        metrics: Default::default(),
        new_fragments: group.new_fragments.clone(),
        read_version: native.read_version,
        original_fragments: group.old_fragments.clone(),
        row_addrs: None,
    };
    crate::materializer::compaction::validate_result(&previous, &result).await?;
    ensure!(
        next.manifest().fragments.len()
            == previous.manifest().fragments.len() - result.original_fragments.len()
                + result.new_fragments.len()
            && previous
                .manifest()
                .fragments
                .iter()
                .filter(|f| !result.original_fragments.contains(f))
                .all(|f| next.manifest().fragments.contains(f))
            && result.new_fragments.iter().all(|expected| next
                .manifest()
                .fragments
                .iter()
                .find(|actual| actual.id == expected.id)
                .is_some_and(|actual| finalized_fragment_matches(expected, actual))),
        "body maintenance changed unrelated fragments"
    );
    Ok(())
}

fn finalized_fragment_matches(expected: &Fragment, actual: &Fragment) -> bool {
    if expected.files.len() != actual.files.len() {
        return false;
    }
    let mut finalized = expected.clone();
    for (file, observed) in finalized.files.iter_mut().zip(&actual.files) {
        if file.file_size_bytes.get().is_none() {
            // Lance's binary-copy rewrite may leave this cache unknown in the
            // native transaction. migrate_manifest fills it before selection.
            // Permit only that completion; known sizes and every other field
            // (including paths, mappings, row IDs and row lineage) stay exact.
            if observed.file_size_bytes.get().is_none() {
                return false;
            }
            file.file_size_bytes = observed.file_size_bytes.clone();
        }
    }
    finalized == *actual
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_io::utils::CachedFileSize;
    use lance_table::format::{DataFile, RowIdMeta};

    #[test]
    fn body_compaction_accepts_only_unknown_size_completion() {
        let mut expected = Fragment::new(3);
        expected.physical_rows = Some(3);
        expected.files.push(DataFile {
            path: "body.lance".into(),
            fields: vec![0].into(),
            column_indices: vec![0].into(),
            file_major_version: 2,
            file_minor_version: 2,
            file_size_bytes: CachedFileSize::new(0),
            base_id: None,
        });
        let mut actual = expected.clone();
        actual.files[0].file_size_bytes = CachedFileSize::new(64943);
        assert!(finalized_fragment_matches(&expected, &actual));
        let mut foreign = actual.clone();
        foreign.files[0].path = "other.lance".into();
        assert!(!finalized_fragment_matches(&expected, &foreign));
        let mut foreign = actual.clone();
        foreign.row_id_meta = Some(RowIdMeta::Column);
        assert!(!finalized_fragment_matches(&expected, &foreign));
        expected.files[0].file_size_bytes = CachedFileSize::new(7);
        assert!(!finalized_fragment_matches(&expected, &actual));
    }
}
