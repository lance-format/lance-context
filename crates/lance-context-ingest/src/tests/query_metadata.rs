//! Exercise the public metadata conversion API through detached publication.
use anyhow::Result;
use arrow_array::{
    ArrayRef, Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
};
use futures::TryStreamExt;
use lance::{
    dataset::{
        transaction::{Operation, TransactionBuilder},
        CommitBuilder, WriteParams,
    },
    index::{
        scalar::{
            load_segments, open_query_matches, prepare_query_index, query_index_metadata_ready,
        },
        DatasetIndexExt,
    },
    Dataset,
};
use lance_index::{
    metrics::NoOpMetricsCollector,
    progress::noop_progress,
    scalar::{
        inverted::{
            rewrite::QueryMetadataRewriteOptions, DocumentGranularity, InvertedIndexParams,
        },
        FullTextSearchQuery,
    },
    IndexType,
};
use std::sync::Arc;

async fn scores(dataset: &Dataset) -> Result<Vec<(u64, f32)>> {
    let mut scan = dataset.scan();
    scan.full_text_search(FullTextSearchQuery::new("needle ending".into()))?;
    let batches: Vec<RecordBatch> = scan
        .with_row_id()
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    let mut result = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("_rowid")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let scores = batch
            .column_by_name("_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        result.extend(
            ids.values()
                .iter()
                .copied()
                .zip(scores.values().iter().copied()),
        );
    }
    result.sort_by_key(|(id, _)| *id);
    Ok(result)
}

async fn unranked(dataset: &Dataset) -> Result<Vec<u64>> {
    let mut rows = Vec::new();
    for segment in load_segments(dataset, "text", DocumentGranularity::Row)
        .await?
        .unwrap()
    {
        let (mut stream, _) = open_query_matches(
            dataset,
            "text",
            &segment,
            "needle",
            1,
            &NoOpMetricsCollector,
        )
        .await?;
        while let Some(batch) = stream.try_next().await? {
            assert!(batch.len() <= 1);
            rows.extend(batch);
        }
    }
    rows.sort_unstable();
    Ok(rows)
}

#[tokio::test]
async fn query_metadata_rewrite_detached_publication_preserves_scores_and_unranked_rows(
) -> Result<()> {
    for stable in [false, true] {
        let root = tempfile::tempdir()?;
        let uri = root.path().to_str().unwrap();
        let batch = RecordBatch::try_from_iter(vec![(
            "text",
            Arc::new(StringArray::from(vec![
                Some("needle retained"),
                Some("needle needle ending"),
                Some("ending"),
                None,
            ])) as ArrayRef,
        )])?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            uri,
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await?;
        let params = InvertedIndexParams::default()
            .with_position(true)
            .stem(false)
            .remove_stop_words(false);
        let mut parts = Vec::new();
        for fragment in dataset.get_fragments() {
            parts.push(
                dataset
                    .create_index_builder(&["text"], IndexType::Inverted, &params)
                    .name("fts".into())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await?,
            );
        }
        dataset
            .commit_existing_index_segments("fts", "text", parts)
            .await?;
        dataset = Dataset::open(uri).await?;
        let main_version = dataset.version_id();
        let originals = dataset.load_indices().await?;
        let expected_scores = scores(&dataset).await?;
        let expected_rows = unranked(&dataset).await?;
        assert!(!expected_scores.is_empty());
        assert_eq!(expected_rows.len(), 2);
        let options = QueryMetadataRewriteOptions {
            document_batch_rows: 1,
            ..Default::default()
        };
        for original in originals.iter() {
            let destination = uuid::Uuid::new_v4();
            assert!(query_index_metadata_ready(&dataset, "text", &original.uuid).await?);
            assert!(
                query_index_metadata_ready(&dataset, "missing", &original.uuid)
                    .await
                    .is_err()
            );
            assert!(prepare_query_index(
                &dataset,
                "text",
                &original.uuid,
                &original.uuid,
                &options,
                noop_progress()
            )
            .await
            .is_err());
            let before = dataset.load_indices().await?;
            let converted = prepare_query_index(
                &dataset,
                "text",
                &original.uuid,
                &destination,
                &options,
                noop_progress(),
            )
            .await?;
            assert_eq!(converted.index.uuid, destination);
            assert_eq!(converted.index.fields, original.fields);
            assert_eq!(converted.index.fragment_bitmap, original.fragment_bitmap);
            assert_eq!(converted.index.index_details, original.index_details);
            assert_eq!(converted.index.dataset_version, original.dataset_version);
            assert_eq!(converted.index.index_version, original.index_version);
            assert!(converted.partitions > 0);
            assert!(converted.document_rows > 0);
            assert_eq!(dataset.load_indices().await?.as_ref(), before.as_ref());
            let expected_metadata = converted.index.clone();
            let transaction = TransactionBuilder::new(
                dataset.version_id(),
                Operation::CreateIndex {
                    new_indices: vec![converted.index],
                    removed_indices: vec![original.clone()],
                },
            )
            .build();
            dataset = CommitBuilder::new(Arc::new(dataset))
                .with_detached(true)
                .with_skip_auto_cleanup(true)
                .execute(transaction)
                .await?;
            assert!(dataset.load_indices().await?.contains(&expected_metadata));
            assert!(query_index_metadata_ready(&dataset, "text", &destination).await?);
            assert_eq!(scores(&dataset).await?, expected_scores);
            assert_eq!(unranked(&dataset).await?, expected_rows);
            assert_eq!(dataset.load_indices().await?.len(), originals.len());
            let mainline = Dataset::open(uri).await?;
            assert_eq!(mainline.version_id(), main_version);
            assert_eq!(mainline.load_indices().await?.as_ref(), originals.as_ref());
        }
    }
    Ok(())
}
