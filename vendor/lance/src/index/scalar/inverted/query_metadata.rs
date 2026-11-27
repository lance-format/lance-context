//! Prepare existing query metadata for caller-owned publication.
use super::*;
use lance_index::progress::IndexBuildProgress;
use lance_index::scalar::inverted::rewrite::{
    QueryMetadataRewriteOptions, query_metadata_ready, rewrite_query_metadata,
};

/// A converted segment that is not yet published in any dataset manifest.
#[derive(Debug)]
pub struct PreparedQueryIndex {
    pub index: IndexMetadata,
    pub partitions: usize,
    pub document_rows: u64,
}

/// Inspect query-metadata availability on one committed local FTS segment.
/// Reads file schemas only; it does not validate posting contents or ranking.
pub async fn query_index_metadata_ready(
    dataset: &Dataset,
    column: &str,
    index_uuid: &Uuid,
) -> Result<bool> {
    let indices = dataset.load_indices().await?;
    let source = indices
        .iter()
        .find(|index| &index.uuid == index_uuid)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "index {index_uuid} is not committed in this snapshot"
            ))
        })?;
    let field = dataset
        .schema()
        .field(column)
        .ok_or_else(|| Error::invalid_input(format!("column {column} is absent")))?;
    if source.fields != [field.id] || source.base_id.is_some() {
        return Err(Error::not_supported(
            "query metadata inspection requires one local indexed field",
        ));
    }
    load_segment_details(dataset, column, std::slice::from_ref(source)).await?;
    let store = LanceIndexStore::from_dataset_for_existing(dataset, source).await?;
    query_metadata_ready(&store).await
}

/// Convert one committed FTS segment to paged dictionary and corpus metadata.
/// The caller owns the deadline and may publish `index` only after success,
/// replacing the original UUID with the same fragment coverage. The old files
/// and dataset manifest are unchanged. The caller exclusively owns the fresh
/// destination UUID and may bind object-copy routing before calling this API.
/// External index bases are unsupported.
pub async fn prepare_query_index(
    dataset: &Dataset,
    column: &str,
    index_uuid: &Uuid,
    destination_uuid: &Uuid,
    options: &QueryMetadataRewriteOptions,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<PreparedQueryIndex> {
    let indices = dataset.load_indices().await?;
    if indices.iter().any(|index| &index.uuid == destination_uuid) {
        return Err(Error::invalid_input(
            "query index destination UUID is already committed",
        ));
    }
    let source = indices
        .iter()
        .find(|index| &index.uuid == index_uuid)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "index {index_uuid} is not committed in this snapshot"
            ))
        })?;
    let field = dataset
        .schema()
        .field(column)
        .ok_or_else(|| Error::invalid_input(format!("column {column} is absent")))?;
    if source.fields != [field.id] || source.base_id.is_some() {
        return Err(Error::not_supported(
            "query index conversion requires one local indexed field",
        ));
    }
    load_segment_details(dataset, column, std::slice::from_ref(source)).await?;
    let source_store = LanceIndexStore::from_dataset_for_existing(dataset, source).await?;
    let destination = LanceIndexStore::from_dataset_for_new(dataset, destination_uuid)?;
    let rewritten = rewrite_query_metadata(&source_store, &destination, options, progress).await?;
    Ok(PreparedQueryIndex {
        index: IndexMetadata {
            uuid: *destination_uuid,
            files: Some(index_files_to_table(rewritten.files)),
            created_at: chrono::DateTime::from_timestamp_millis(
                chrono::Utc::now().timestamp_millis(),
            ),
            ..source.clone()
        },
        partitions: rewritten.partitions,
        document_rows: rewritten.document_rows,
    })
}
