//! Large narrow rewrites use the existing native preparation/reference encoding.
use super::*;
use lance::dataset::optimize::RewriteResult;
use lance_context_core::annotation_staging::RESULTS_PROPERTY;
use lance_table::transaction::RewriteGroup;

pub(crate) async fn descriptors(
    base: &Dataset,
    result: RewriteResult,
    selection: &Selection,
    storage: &Storage,
    maximum: u64,
) -> Result<p::Descriptors> {
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: result.original_fragments,
                new_fragments: result.new_fragments,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    );
    ensure!(
        result.read_version == base.version_id() && result.row_addrs.is_none(),
        "foreign narrow compaction result"
    );
    transaction.uuid = selection.pending.plan.transaction.to_string();
    transaction.transaction_properties = base
        .read_transaction()
        .await?
        .and_then(|t| t.transaction_properties);
    validate(base, &transaction, selection).await?;
    persist_transaction(base, &transaction, selection, storage, maximum).await
}

pub(super) async fn validate(
    base: &Dataset,
    transaction: &Transaction,
    selection: &Selection,
) -> Result<()> {
    selection.identity()?;
    ensure!(
        selection.pending.plan.is_maintenance()
            && transaction.read_version == base.version_id()
            && transaction.read_version == selection.pending.plan.read_version
            && transaction.uuid == selection.pending.plan.transaction.to_string()
            && transaction.tag.is_none(),
        "foreign narrow maintenance identity"
    );
    let old = base
        .read_transaction()
        .await?
        .and_then(|t| t.transaction_properties)
        .context("narrow maintenance base has no checkpoint")?;
    ensure!(
        transaction.transaction_properties.as_ref() == Some(&old),
        "narrow maintenance changed checkpoint properties"
    );
    let (scope, through): (Scope, u64) = serde_json::from_str(
        old.get("lance-context.annotation-cut")
            .context("narrow maintenance has no cut")?,
    )?;
    ensure!(
        scope.table_instance == selection.incarnation.to_string()
            && scope.source_group > 0
            && scope.shard < 65_536
            && through > 0
            && through < u64::MAX,
        "foreign narrow maintenance cut"
    );
    // The exact selected base properties above bind the source group and bucket.
    // An empty-source plan's next_group is a scheduling cursor, not the cut's
    // identity; retained plans may have the layout's first group as that cursor.
    crate::annotations::checkpoint::Snapshot::decode(
        old.get(RESULTS_PROPERTY)
            .context("narrow maintenance dropped request results")?,
        through,
    )?;
    let Operation::Rewrite {
        groups,
        rewritten_indices,
        frag_reuse_index,
    } = &transaction.operation
    else {
        anyhow::bail!("narrow maintenance is not a native rewrite");
    };
    ensure!(
        groups.len() == 1 && rewritten_indices.is_empty() && frag_reuse_index.is_none(),
        "unsupported narrow rewrite operation"
    );
    crate::materializer::compaction::validate_result(
        base,
        &RewriteResult {
            metrics: Default::default(),
            new_fragments: groups[0].new_fragments.clone(),
            read_version: transaction.read_version,
            original_fragments: groups[0].old_fragments.clone(),
            row_addrs: None,
        },
    )
    .await
}
