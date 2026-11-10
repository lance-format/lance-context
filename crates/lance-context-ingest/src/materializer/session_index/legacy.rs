//! Extend an existing Lance 13 BTree with real stable IDs, without rebuilding
//! its existing segments or mixing them with Lance 14 physical addresses.
use super::*;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::DataFusionError;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::TryStreamExt;
use lance::dataset::scanner::ColumnOrdering;
use lance_index::scalar::{
    btree::{train_btree_index, DEFAULT_BTREE_BATCH_SIZE},
    lance_format::LanceIndexStore,
};

#[cfg(test)]
mod tests;

pub(super) async fn build(
    dataset: &Dataset,
    key: Key,
    ids: Vec<u32>,
    template: &IndexMetadata,
) -> Result<IndexMetadata> {
    validate_metadata(dataset, template, key)?;
    ensure!(
        template.index_version == 0 && dataset.manifest().uses_stable_row_ids(),
        "legacy BTree extension requires a stable-ID version-0 source"
    );
    ensure!(
        !ids.is_empty() && ids.len() <= 512,
        "legacy index fragment bound exceeded"
    );
    let coverage: BTreeSet<_> = ids.iter().copied().collect();
    ensure!(
        coverage.len() == ids.len(),
        "duplicate legacy index source fragment"
    );
    let fragments: Vec<_> = ids
        .iter()
        .map(|&id| {
            dataset
                .get_fragment(id as usize)
                .map(|f| f.metadata().clone())
                .context("legacy index source fragment missing")
        })
        .collect::<Result<_>>()?;
    let mut rows = 0usize;
    for fragment in &fragments {
        let count = fragment
            .physical_rows
            .context("legacy index source row count unknown")?;
        rows = rows
            .checked_add(count)
            .context("legacy index row count overflow")?;
        ensure!(
            fragment.deletion_file.is_none() && fragment.overlays.is_empty() && rows <= MAX_ROWS,
            "legacy index source exceeds append-only build bounds"
        );
    }
    let mut scan = dataset.scan();
    scan.project(&[key.column()])?
        .with_row_id()
        .with_fragments(fragments)
        .batch_size(DEFAULT_BTREE_BATCH_SIZE as usize)
        .io_buffer_size(32 << 20);
    scan.order_by(Some(vec![ColumnOrdering::asc_nulls_first(
        key.column().to_owned(),
    )]))?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8, true),
        Field::new("_rowaddr", DataType::UInt64, false),
    ]));
    // The low-level native page writer names this input `_rowaddr`. Supply
    // actual `_rowid` integers and persist version 0, whose loader interprets
    // these integers as stable IDs. Never relabel address-valued native pages.
    // Both Arrow arrays are shared; this adapter copies no value buffers.
    let batch_schema = schema.clone();
    let column = key.column();
    let stream = scan
        .try_into_stream()
        .await?
        .map_err(|error| DataFusionError::External(Box::new(error)))
        .and_then(move |batch| {
            let result = (|| {
                let values = batch
                    .column_by_name(column)
                    .context("legacy index value column missing")?;
                let ids = batch
                    .column_by_name("_rowid")
                    .context("legacy index stable IDs missing")?;
                Ok::<_, anyhow::Error>(RecordBatch::try_new(
                    batch_schema.clone(),
                    vec![values.clone(), ids.clone()],
                )?)
            })()
            .map_err(|error| DataFusionError::External(error.into()));
            std::future::ready(result)
        });
    let mut index = template.clone();
    index.uuid = Uuid::new_v4();
    index.dataset_version = dataset.version_id();
    index.fragment_bitmap = Some(coverage.into_iter().collect());
    index.created_at = None;
    let store = LanceIndexStore::new(
        dataset.object_store(None).await?,
        dataset.indices_dir().child(index.uuid.to_string()),
        Arc::new(dataset.session().file_metadata_cache().clone()),
    );
    index.files = Some(
        train_btree_index(
            Box::pin(RecordBatchStreamAdapter::new(schema, stream)),
            &store,
            DEFAULT_BTREE_BATCH_SIZE,
            Some(ids),
            None,
        )
        .await?,
    );
    Ok(index)
}
