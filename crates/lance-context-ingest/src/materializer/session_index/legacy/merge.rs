//! Consolidate homogeneous version-0 segments without scanning turn bodies.
use super::*;
use lance::{dataset::rowids::load_row_id_sequence, index::DatasetIndexInternalExt};
use lance_index::{
    metrics::NoOpMetricsCollector,
    scalar::{btree::BTreeIndex, OldIndexDataFilter},
};

pub(crate) async fn merge(
    dataset: &Dataset,
    key: Key,
    segments: Vec<IndexMetadata>,
) -> Result<IndexMetadata> {
    ensure!(
        dataset.manifest().uses_stable_row_ids() && (2..=64).contains(&segments.len()),
        "legacy index merge requires bounded stable-ID sources"
    );
    let existing = dataset.load_indices().await?;
    ensure!(
        !existing.iter().any(lance_index::is_system_index),
        "legacy index merge cannot bypass a system index"
    );
    let mut coverage = BTreeSet::new();
    let mut sources = Vec::new();
    let mut filters = Vec::new();
    for segment in &segments {
        validate_metadata(dataset, segment, key)?;
        ensure!(
            segment.index_version == 0 && existing.contains(segment),
            "legacy index merge changed its source or row domain"
        );
        let mut filter = OldIndexDataFilter::RowIds(Default::default());
        let OldIndexDataFilter::RowIds(allowed) = &mut filter else {
            unreachable!()
        };
        for id in segment.fragment_bitmap.as_ref().unwrap().iter() {
            ensure!(
                coverage.insert(id),
                "legacy index merge has overlapping coverage"
            );
            let fragment = dataset
                .get_fragment(id as usize)
                .context("legacy merge fragment missing")?;
            let metadata = fragment.metadata();
            ensure!(
                metadata.deletion_file.is_none() && metadata.overlays.is_empty(),
                "legacy index merge requires append-only fragments"
            );
            // Metadata row-ID sequences retain compressed ranges. Only IDs in
            // this segment's CURRENT coverage may survive. A mutable column
            // rewrite can leave stale values in pages outside that coverage.
            let sequence = load_row_id_sequence(dataset, metadata).await?;
            let OldIndexDataFilter::RowIds(rows) =
                OldIndexDataFilter::RowIds(sequence.as_ref().into())
            else {
                unreachable!()
            };
            *allowed |= rows;
        }
        let opened = dataset
            .open_scalar_index_for_maintenance(key.column(), &segment.uuid, &NoOpMetricsCollector)
            .await?;
        ensure!(
            !opened.results_are_row_addresses(),
            "legacy merge opened an address index"
        );
        let btree = opened
            .as_any()
            .downcast_ref::<BTreeIndex>()
            .context("legacy merge source is not a BTree")?;
        sources.push(Arc::new(btree.clone()));
        filters.push(Some(filter));
    }
    ensure!(!coverage.is_empty(), "legacy merge has no live coverage");
    let mut index = segments[0].clone();
    index.uuid = Uuid::new_v4();
    index.dataset_version = dataset.version_id();
    index.fragment_bitmap = Some(coverage.into_iter().collect());
    index.created_at = None;
    let store = LanceIndexStore::new(
        dataset.object_store(None).await?,
        dataset.indices_dir().join(index.uuid.to_string()),
        Arc::new(dataset.session().file_metadata_cache().clone()),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "value",
            dataset
                .schema()
                .field(key.column())
                .context("legacy index column missing")?
                .data_type(),
            true,
        ),
        Field::new("_rowaddr", DataType::UInt64, false),
    ]));
    // The high-level Lance14 merge refuses all legacy stable-ID segments.
    // The native page merger supports a homogeneous ID domain and preserves it;
    // the exact ID filters above replace its address-oriented wrapper filters.
    let output = BTreeIndex::merge_segments(
        &sources,
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::empty(),
        )),
        &store,
        &filters,
    )
    .await?;
    ensure!(
        output.index_version == 0,
        "legacy merge changed its result domain"
    );
    index.index_details = Some(Arc::new(output.index_details));
    index.files = Some(index_files_to_table(output.files));
    Ok(index)
}
