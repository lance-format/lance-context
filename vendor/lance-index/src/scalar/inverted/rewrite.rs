//! Convert query metadata without reading source documents or retokenizing postings.

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch, types::UInt32Type};
use arrow_schema::Schema;
use lance_core::{Error, Result};

use super::builder::{doc_file_path, posting_file_path, token_file_path};
use super::documents::TOTAL_TOKENS_KEY;
use super::{METADATA_FILE, NUM_TOKEN_COL, TOKEN_SET_FORMAT_KEY, TokenSet, TokenSetFormat};
use crate::progress::IndexBuildProgress;
use crate::scalar::{IndexFile, IndexStore, lance_format::LanceIndexStore};

/// Bounds for one sequential partition conversion. The caller owns the process
/// memory limit and deadline; dictionary decoding can allocate more than its
/// on-disk size. No partition handles survive conversion of that partition.
#[derive(Clone, Debug)]
pub struct QueryMetadataRewriteOptions {
    pub document_batch_rows: usize,
    pub max_token_file_bytes: u64,
}

impl Default for QueryMetadataRewriteOptions {
    fn default() -> Self {
        Self {
            document_batch_rows: 8192,
            max_token_file_bytes: 512 << 20,
        }
    }
}

/// Files for a new, unpublished index UUID. Counts include physical document
/// rows, preserving the original index's corpus-statistics semantics.
#[derive(Debug)]
pub struct RewrittenQueryMetadata {
    pub files: Vec<IndexFile>,
    pub partitions: usize,
    pub document_rows: u64,
}

/// Inspect only file schemas for paged dictionaries and persisted corpus totals.
/// This checks query-metadata availability, not posting contents or ranking.
/// Unsupported or malformed formats fail rather than being classified as ready.
pub async fn query_metadata_ready(source: &LanceIndexStore) -> Result<bool> {
    let metadata = source.open_index_file(METADATA_FILE).await?;
    let format = metadata
        .schema()
        .metadata
        .get(TOKEN_SET_FORMAT_KEY)
        .map(|value| TokenSetFormat::from_str(value))
        .transpose()?;
    if format != Some(TokenSetFormat::Fst) {
        return Err(Error::not_supported(
            "query metadata inspection requires a partitioned FST index",
        ));
    }
    let partitions: Vec<u64> = serde_json::from_str(
        metadata
            .schema()
            .metadata
            .get("partitions")
            .ok_or_else(|| Error::index("index metadata has no partitions"))?,
    )?;
    if partitions.iter().copied().collect::<HashSet<_>>().len() != partitions.len() {
        return Err(Error::index("index metadata contains duplicate partitions"));
    }
    for id in partitions {
        let tokens = source.open_index_file(&token_file_path(id)).await?;
        match tokens
            .schema()
            .metadata
            .get(super::token_lookup::LOOKUP_VERSION_KEY)
            .map(String::as_str)
        {
            None => return Ok(false),
            Some("1") => {}
            Some(_) => return Err(Error::index("unsupported token lookup version")),
        }
        let docs = source.open_index_file(&doc_file_path(id)).await?;
        let Some(total) = docs.schema().metadata.get(TOTAL_TOKENS_KEY) else {
            return Ok(false);
        };
        total
            .parse::<u64>()
            .map_err(|error| Error::index(format!("invalid corpus token total: {error}")))?;
    }
    Ok(true)
}

/// Rewrite a partitioned FST index into a fresh, empty store on the same backing
/// object store. The source is immutable and the caller must exclusively own
/// the destination UUID. Posting files and metadata are copied through the
/// concrete Lance store's object-copy path, including any global buffers.
///
/// Dictionaries are converted one partition at a time; document rows are copied
/// in bounded batches while summing their exact lengths. Row IDs, row order,
/// token IDs, postings, deletions, and tokenizer settings are unchanged. Only
/// the caller may publish the returned files, after this function succeeds.
/// Errors can leave an unpublished destination, which must not be reused.
pub async fn rewrite_query_metadata(
    source: &LanceIndexStore,
    destination: &LanceIndexStore,
    options: &QueryMetadataRewriteOptions,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<RewrittenQueryMetadata> {
    if !(1..=65536).contains(&options.document_batch_rows) || options.max_token_file_bytes == 0 {
        return Err(Error::invalid_input(
            "invalid query metadata rewrite bounds",
        ));
    }
    if !destination.list_files_with_sizes().await?.is_empty() {
        return Err(Error::invalid_input(
            "query metadata destination must be empty",
        ));
    }
    let metadata = source.open_index_file(METADATA_FILE).await?;
    let format = metadata
        .schema()
        .metadata
        .get(TOKEN_SET_FORMAT_KEY)
        .map(|value| TokenSetFormat::from_str(value))
        .transpose()?;
    if format != Some(TokenSetFormat::Fst) {
        return Err(Error::not_supported(
            "query metadata rewrite requires a partitioned FST index",
        ));
    }
    let partitions: Vec<u64> = serde_json::from_str(
        metadata
            .schema()
            .metadata
            .get("partitions")
            .ok_or_else(|| Error::index("index metadata has no partitions"))?,
    )?;
    if partitions.iter().copied().collect::<HashSet<_>>().len() != partitions.len() {
        return Err(Error::index("index metadata contains duplicate partitions"));
    }
    let source_files: HashMap<_, _> = source
        .list_files_with_sizes()
        .await?
        .into_iter()
        .map(|file| (file.path, file.size_bytes))
        .collect();
    for id in &partitions {
        for path in [
            token_file_path(*id),
            doc_file_path(*id),
            posting_file_path(*id),
        ] {
            if !source_files.contains_key(&path) {
                return Err(Error::index(format!(
                    "index partition file {path} is absent"
                )));
            }
        }
        if source_files
            .get(&token_file_path(*id))
            .is_some_and(|size| *size > options.max_token_file_bytes)
        {
            return Err(Error::invalid_input(format!(
                "partition {id} token file exceeds rewrite byte bound"
            )));
        }
    }
    progress
        .stage_start(
            "query_metadata",
            Some(partitions.len() as u64),
            "partitions",
        )
        .await?;
    let mut files = Vec::with_capacity(partitions.len() * 3 + 1);
    let mut document_rows = 0u64;
    for (position, id) in partitions.iter().copied().enumerate() {
        let token_path = token_file_path(id);
        let reader = source.open_index_file(&token_path).await?;
        let mut token_metadata = reader.schema().metadata.clone();
        let tokens = TokenSet::load(reader, TokenSetFormat::Fst).await?;
        let batch = tokens.to_batch(TokenSetFormat::Fst)?;
        token_metadata.extend(batch.schema().metadata().clone());
        let schema =
            Arc::new(Schema::new(batch.schema().fields().clone()).with_metadata(token_metadata));
        let batch = RecordBatch::try_new(schema.clone(), batch.columns().to_vec())?;
        let mut writer = destination.new_index_file(&token_path, schema).await?;
        writer.write_record_batch(batch).await?;
        files.push(writer.finish().await?);

        let docs_path = doc_file_path(id);
        let reader = source.open_index_file(&docs_path).await?;
        let mut docs_metadata = reader.schema().metadata.clone();
        let declared_total = docs_metadata
            .get(TOTAL_TOKENS_KEY)
            .map(|value| {
                value.parse::<u64>().map_err(|error| {
                    Error::index(format!("invalid corpus total for partition {id}: {error}"))
                })
            })
            .transpose()?;
        let mut writer = destination
            .new_index_file(&docs_path, Arc::new(reader.schema().into()))
            .await?;
        let mut total_tokens = 0u64;
        for start in (0..reader.num_rows()).step_by(options.document_batch_rows) {
            let end = start
                .saturating_add(options.document_batch_rows)
                .min(reader.num_rows());
            let batch = reader.read_range(start..end, None).await?;
            if batch.num_rows() != end - start {
                return Err(Error::index(format!(
                    "document row count changed for partition {id}"
                )));
            }
            let lengths = batch
                .column_by_name(NUM_TOKEN_COL)
                .and_then(|array| array.as_primitive_opt::<UInt32Type>())
                .filter(|array| array.null_count() == 0)
                .ok_or_else(|| {
                    Error::index(format!("invalid document lengths for partition {id}"))
                })?;
            for length in lengths.values() {
                total_tokens = total_tokens
                    .checked_add(u64::from(*length))
                    .ok_or_else(|| Error::index("corpus token total overflow"))?;
            }
            writer.write_record_batch(batch).await?;
        }
        if declared_total.is_some_and(|total| total != total_tokens) {
            return Err(Error::index(format!(
                "stored corpus total disagrees with partition {id} rows"
            )));
        }
        docs_metadata.insert(TOTAL_TOKENS_KEY.to_owned(), total_tokens.to_string());
        files.push(writer.finish_with_metadata(docs_metadata).await?);
        document_rows = document_rows
            .checked_add(reader.num_rows() as u64)
            .ok_or_else(|| Error::index("document row total overflow"))?;
        files.push(
            source
                .copy_index_file(&posting_file_path(id), destination)
                .await?,
        );
        progress
            .stage_progress("query_metadata", position as u64 + 1)
            .await?;
    }
    // Publish no new metadata file until every referenced partition is complete.
    files.push(source.copy_index_file(METADATA_FILE, destination).await?);
    progress.stage_complete("query_metadata").await?;
    Ok(RewrittenQueryMetadata {
        files,
        partitions: partitions.len(),
        document_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, StringArray, UInt64Array};
    use async_trait::async_trait;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_core::cache::LanceCache;
    use lance_core::deepsize::DeepSizeOf;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;
    use roaring::RoaringBitmap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;
    use crate::progress::noop_progress;
    use crate::scalar::inverted::{
        DocSet, InvertedIndex, InvertedIndexBuilder, InvertedIndexParams, PostingListBuilder,
        TOKEN_FST_BYTES_COL, TOKEN_NEXT_ID_COL, TOKEN_TOTAL_LENGTH_COL,
        builder::{InnerBuilder, PositionRecorder},
        document_tokenizer::DocType,
        query::{FtsSearchParams, Operator, Tokens},
        token_lookup::LOOKUP_FST_COL,
    };
    use crate::scalar::{IndexReader, IndexWriter};

    #[derive(Debug, Default)]
    struct TokenReads {
        full_fst_bytes: AtomicUsize,
        lookup_rows: AtomicUsize,
    }

    struct CountingTokenReader {
        inner: Arc<dyn IndexReader>,
        reads: Arc<TokenReads>,
    }

    impl CountingTokenReader {
        fn record(&self, batch: &RecordBatch) {
            if let Some(array) = batch.column_by_name(TOKEN_FST_BYTES_COL) {
                self.reads.full_fst_bytes.fetch_add(
                    array.as_binary::<i64>().value_data().len(),
                    Ordering::Relaxed,
                );
            }
            if let Some(array) = batch.column_by_name(LOOKUP_FST_COL) {
                self.reads
                    .lookup_rows
                    .fetch_add(array.len(), Ordering::Relaxed);
            }
        }
    }

    #[async_trait]
    impl IndexReader for CountingTokenReader {
        fn native_memory(&self) -> lance_core::memory::NativeMemory {
            self.inner.native_memory()
        }

        async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
            let batch = self.inner.read_record_batch(n, batch_size).await?;
            self.record(&batch);
            Ok(batch)
        }

        async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
            self.inner.read_global_buffer(index).await
        }

        async fn read_range(
            &self,
            range: std::ops::Range<usize>,
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            let batch = self.inner.read_range(range, projection).await?;
            self.record(&batch);
            Ok(batch)
        }

        async fn read_ranges(
            &self,
            ranges: &[std::ops::Range<usize>],
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            let batch = self.inner.read_ranges(ranges, projection).await?;
            self.record(&batch);
            Ok(batch)
        }

        async fn num_batches(&self, batch_size: u64) -> u32 {
            self.inner.num_batches(batch_size).await
        }

        fn num_rows(&self) -> usize {
            self.inner.num_rows()
        }

        fn schema(&self) -> &lance_core::datatypes::Schema {
            self.inner.schema()
        }
    }

    #[derive(Clone, Debug)]
    struct CountingStore {
        inner: Arc<dyn IndexStore>,
        reads: Arc<TokenReads>,
    }

    impl DeepSizeOf for CountingStore {
        fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
            self.inner.deep_size_of_children(context)
        }
    }

    #[async_trait]
    impl IndexStore for CountingStore {
        fn native_memory(&self) -> lance_core::memory::NativeMemory {
            self.inner.native_memory()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn clone_arc(&self) -> Arc<dyn IndexStore> {
            Arc::new(self.clone())
        }
        fn io_parallelism(&self) -> usize {
            self.inner.io_parallelism()
        }
        fn with_io_priority(&self, priority: u64) -> Arc<dyn IndexStore> {
            Arc::new(Self {
                inner: self.inner.with_io_priority(priority),
                reads: self.reads.clone(),
            })
        }
        fn with_io_buffer_size(&self, bytes: u64) -> Arc<dyn IndexStore> {
            Arc::new(Self {
                inner: self.inner.with_io_buffer_size(bytes),
                reads: self.reads.clone(),
            })
        }
        async fn new_index_file(
            &self,
            name: &str,
            schema: Arc<Schema>,
        ) -> Result<Box<dyn IndexWriter>> {
            self.inner.new_index_file(name, schema).await
        }
        async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
            Ok(Arc::new(CountingTokenReader {
                inner: self.inner.open_index_file(name).await?,
                reads: self.reads.clone(),
            }))
        }
        async fn copy_index_file(
            &self,
            name: &str,
            destination: &dyn IndexStore,
        ) -> Result<IndexFile> {
            self.inner.copy_index_file(name, destination).await
        }
        async fn copy_index_file_to(
            &self,
            name: &str,
            new_name: &str,
            destination: &dyn IndexStore,
        ) -> Result<IndexFile> {
            self.inner
                .copy_index_file_to(name, new_name, destination)
                .await
        }
        async fn rename_index_file(&self, name: &str, new_name: &str) -> Result<IndexFile> {
            self.inner.rename_index_file(name, new_name).await
        }
        async fn delete_index_file(&self, name: &str) -> Result<()> {
            self.inner.delete_index_file(name).await
        }
        async fn list_files_with_sizes(&self) -> Result<Vec<IndexFile>> {
            self.inner.list_files_with_sizes().await
        }
    }

    async fn old_index(
        staging: &Arc<LanceIndexStore>,
        source: &Arc<LanceIndexStore>,
        block_size: usize,
    ) -> Result<()> {
        let params = InvertedIndexParams {
            block_size,
            with_position: true,
            ..Default::default()
        };
        for id in [7, 12] {
            let mut tokens = TokenSet::default();
            for term in ["needle", "ending"] {
                tokens.add(term.to_owned());
            }
            for i in 0..600 {
                tokens.add(format!("unused{i:04}"));
            }
            let mut postings: Vec<_> = (0..tokens.len())
                .map(|_| {
                    PostingListBuilder::new_with_posting_tail_codec_and_block_size(
                        true,
                        params.resolved_format_version().posting_tail_codec(),
                        block_size,
                    )
                })
                .collect();
            postings[0].add(0, PositionRecorder::Position(vec![0, 2].into()));
            postings[1].add(0, PositionRecorder::Position(vec![1].into()));
            let mut docs = DocSet::default();
            docs.append(id << 32, 3);
            docs.append((id << 32) + 1, 0);
            let old_docs = docs.to_batch()?;
            let token_batch = tokens.clone().to_batch(TokenSetFormat::Fst)?;
            let fields = [
                TOKEN_FST_BYTES_COL,
                TOKEN_NEXT_ID_COL,
                TOKEN_TOTAL_LENGTH_COL,
            ]
            .iter()
            .map(|name| token_batch.schema().index_of(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
            let original = token_batch.project(&fields)?.slice(0, 1);
            let old_tokens = RecordBatch::try_new(
                Arc::new(Schema::new(original.schema().fields().clone())),
                original.columns().to_vec(),
            )?;
            let mut partition = InnerBuilder::new_with_format_version_and_block_size(
                id,
                true,
                TokenSetFormat::Fst,
                params.resolved_format_version(),
                block_size,
            );
            partition.set_tokens(tokens);
            partition.set_docs(docs);
            partition.set_posting_lists(postings);
            partition.write(staging.as_ref()).await?;
            for (path, batch) in [
                (token_file_path(id), old_tokens),
                (doc_file_path(id), old_docs),
            ] {
                let mut writer = staging.new_index_file(&path, batch.schema()).await?;
                writer.write_record_batch(batch).await?;
                writer.finish().await?;
            }
        }
        let empty = RecordBatch::try_from_iter(vec![
            (
                "text",
                Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef,
            ),
            (
                "_rowid",
                Arc::new(UInt64Array::from(Vec::<u64>::new())) as ArrayRef,
            ),
        ])?;
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            empty.schema(),
            futures::stream::empty(),
        ));
        InvertedIndexBuilder::from_existing_index(
            params,
            Some(staging.clone()),
            vec![7, 12],
            TokenSetFormat::Fst,
            None,
            RoaringBitmap::from_iter([12]),
        )
        .update(stream, source.as_ref(), None)
        .await?;
        Ok(())
    }

    #[rstest::rstest]
    #[case::exact_lengths(128)]
    #[case::quantized_lengths(256)]
    #[tokio::test]
    async fn query_metadata_rewrite_preserves_physical_index(
        #[case] block_size: usize,
    ) -> Result<()> {
        let root = TempObjDir::default();
        let object_store = Arc::new(ObjectStore::local());
        let store = |name: &str| {
            Arc::new(LanceIndexStore::new(
                object_store.clone(),
                root.child(name),
                Arc::new(LanceCache::no_cache()),
            ))
        };
        let staging = store("staging");
        let source = store("source");
        let destination = store("converted");
        old_index(&staging, &source, block_size).await?;
        assert!(!query_metadata_ready(&source).await?);
        let original_files = source.list_files_with_sizes().await?;
        let mut original_bytes = Vec::with_capacity(original_files.len());
        for file in &original_files {
            original_bytes.push((
                file.path.clone(),
                object_store
                    .open(&root.child("source").child(file.path.as_str()))
                    .await?
                    .get_all()
                    .await?,
            ));
        }
        let original = InvertedIndex::load(source.clone(), None, &LanceCache::no_cache()).await?;
        let result = rewrite_query_metadata(
            &source,
            &destination,
            &QueryMetadataRewriteOptions {
                document_batch_rows: 1,
                ..Default::default()
            },
            noop_progress(),
        )
        .await?;
        assert_eq!(result.partitions, 2);
        assert_eq!(result.document_rows, 4);
        assert_eq!(result.files.len(), 7);
        assert!(query_metadata_ready(&destination).await?);
        assert_eq!(source.list_files_with_sizes().await?, original_files);
        for (path, bytes) in &original_bytes {
            assert_eq!(
                &object_store
                    .open(&root.child("source").child(path.as_str()))
                    .await?
                    .get_all()
                    .await?,
                bytes,
                "published source must remain byte-identical"
            );
        }
        let terms = vec![
            "needle".to_owned(),
            "ending".to_owned(),
            "absent".to_owned(),
        ];
        // A verbatim copy preserves scores but retains the old cold-query cost.
        // Exercise the exact-term reader with cold caches on both file formats.
        let old_reads = Arc::new(TokenReads::default());
        InvertedIndex::load_for_query(
            Arc::new(CountingStore {
                inner: source.clone(),
                reads: old_reads.clone(),
            }),
            None,
            None,
            &LanceCache::no_cache(),
            &terms,
        )
        .await?;
        assert!(
            old_reads.full_fst_bytes.load(Ordering::Relaxed) > 0,
            "the old-format fixture must require a full FST read"
        );
        assert_eq!(old_reads.lookup_rows.load(Ordering::Relaxed), 0);
        let converted_reads = Arc::new(TokenReads::default());
        let converted = InvertedIndex::load_for_query(
            Arc::new(CountingStore {
                inner: destination.clone(),
                reads: converted_reads.clone(),
            }),
            None,
            None,
            &LanceCache::no_cache(),
            &terms,
        )
        .await?;
        assert_eq!(
            converted_reads.full_fst_bytes.load(Ordering::Relaxed),
            0,
            "converted exact lookup read the full FST"
        );
        assert_eq!(
            converted_reads.lookup_rows.load(Ordering::Relaxed),
            2,
            "read only the requested block in each of the two partitions"
        );
        assert_eq!(converted.deleted_fragments(), original.deleted_fragments());
        assert_eq!(
            converted.bm25_stats_for_terms(&terms, None).await?,
            original.bm25_stats_for_terms(&terms, None).await?
        );
        let tokens = Arc::new(Tokens::new(
            vec!["needle".to_owned(), "ending".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(FtsSearchParams {
            phrase_slop: Some(0),
            limit: Some(10),
            ..Default::default()
        });
        let expected = original
            .bm25_search(
                tokens.clone(),
                params.clone(),
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .await?;
        let actual = converted
            .bm25_search(
                tokens,
                params,
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .await?;
        assert!(!expected.0.is_empty());
        // Concurrent partitions need not return equal-score row IDs in the same order.
        // Preserve score ordering and every row/score pair, including cardinality.
        assert_eq!(actual.1, expected.1, "score order must remain unchanged");
        assert_eq!(actual.0.len(), actual.1.len());
        assert_eq!(expected.0.len(), expected.1.len());
        let mut actual = actual.0.into_iter().zip(actual.1).collect::<Vec<_>>();
        let mut expected = expected.0.into_iter().zip(expected.1).collect::<Vec<_>>();
        actual.sort_by_key(|(id, _)| *id);
        expected.sort_by_key(|(id, _)| *id);
        assert_eq!(actual, expected);
        for id in [7, 12] {
            let before = source.open_index_file(&doc_file_path(id)).await?;
            let after = destination.open_index_file(&doc_file_path(id)).await?;
            assert!(!before.schema().metadata.contains_key(TOTAL_TOKENS_KEY));
            assert_eq!(
                after
                    .schema()
                    .metadata
                    .get(TOTAL_TOKENS_KEY)
                    .map(String::as_str),
                Some("3")
            );
            assert_eq!(
                before.read_range(0..2, None).await?.columns(),
                after.read_range(0..2, None).await?.columns()
            );
            for path in [posting_file_path(id), METADATA_FILE.to_owned()] {
                let before = object_store
                    .open(&root.child("source").child(path.as_str()))
                    .await?
                    .get_all()
                    .await?;
                let after = object_store
                    .open(&root.child("converted").child(path.as_str()))
                    .await?
                    .get_all()
                    .await?;
                assert_eq!(
                    before, after,
                    "copy must preserve all bytes and global buffers"
                );
            }
        }
        let error =
            rewrite_query_metadata(&source, &destination, &Default::default(), noop_progress())
                .await
                .unwrap_err();
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("must be empty"));
        Ok(())
    }

    #[derive(Debug)]
    struct StopAfterPartition;

    #[async_trait]
    impl IndexBuildProgress for StopAfterPartition {
        async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> Result<()> {
            Ok(())
        }
        async fn stage_progress(&self, _: &str, _: u64) -> Result<()> {
            Err(Error::execution("test cancellation after one partition"))
        }
        async fn stage_complete(&self, _: &str) -> Result<()> {
            Ok(())
        }
    }

    #[rstest::rstest]
    #[case::invalid_batch("batch")]
    #[case::dictionary_limit("size")]
    #[case::inconsistent_total("total")]
    #[case::cancelled("cancel")]
    #[tokio::test]
    async fn query_metadata_rewrite_failure_leaves_no_manifest(
        #[case] failure: &str,
    ) -> Result<()> {
        let root = TempObjDir::default();
        let store = |name: &str| {
            Arc::new(LanceIndexStore::new(
                ObjectStore::local().into(),
                root.child(name),
                Arc::new(LanceCache::no_cache()),
            ))
        };
        let source = store("source");
        let destination = store("destination");
        old_index(&store("staging"), &source, 128).await?;
        if failure == "total" {
            let path = doc_file_path(7);
            let reader = source.open_index_file(&path).await?;
            let batch = reader.read_range(0..reader.num_rows(), None).await?;
            let mut writer = source.new_index_file(&path, batch.schema()).await?;
            writer.write_record_batch(batch).await?;
            writer
                .finish_with_metadata(HashMap::from([(
                    TOTAL_TOKENS_KEY.to_owned(),
                    "999".to_owned(),
                )]))
                .await?;
        }
        let before = source.list_files_with_sizes().await?;
        let mut options = QueryMetadataRewriteOptions::default();
        if failure == "batch" {
            options.document_batch_rows = 0;
        }
        if failure == "size" {
            options.max_token_file_bytes = 1;
        }
        let progress: Arc<dyn IndexBuildProgress> = if failure == "cancel" {
            Arc::new(StopAfterPartition)
        } else {
            noop_progress()
        };
        let error = rewrite_query_metadata(&source, &destination, &options, progress)
            .await
            .unwrap_err();
        match failure {
            "batch" | "size" => assert!(matches!(&error, Error::InvalidInput { .. })),
            "total" => assert!(error.to_string().contains("disagrees")),
            "cancel" => assert!(error.to_string().contains("test cancellation")),
            _ => unreachable!(),
        }
        assert_eq!(source.list_files_with_sizes().await?, before);
        assert!(
            destination
                .list_files_with_sizes()
                .await?
                .iter()
                .all(|file| file.path != METADATA_FILE)
        );
        Ok(())
    }
}
