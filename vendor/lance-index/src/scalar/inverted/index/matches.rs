//! Unranked OR matching over existing inverted postings.
use std::sync::Arc;

use arrow::buffer::ScalarBuffer;
use futures::{StreamExt, stream, stream::BoxStream};
use lance_core::{Error, Result};

use super::{
    CompressedPostingListIterator, InvertedIndex, InvertedPartition, PartitionDocumentStore,
    PostingList, PostingReadPolicy, collect_query_tokens,
};
use crate::metrics::NoOpMetricsCollector;

enum Cursor {
    Plain {
        ids: ScalarBuffer<u64>,
        offset: usize,
    },
    Compressed(CompressedPostingListIterator),
}

impl Cursor {
    fn new(posting: PostingList) -> Self {
        match posting {
            PostingList::Plain(posting) => Self::Plain {
                ids: posting.row_ids,
                offset: 0,
            },
            PostingList::Compressed(posting) => Self::Compressed(posting.iter()),
        }
    }

    fn batch(&mut self, size: usize) -> Vec<u64> {
        match self {
            Self::Plain { ids, offset } => {
                let end = (*offset + size).min(ids.len());
                let result = ids[*offset..end].to_vec();
                *offset = end;
                result
            }
            Self::Compressed(iter) => iter
                .by_ref()
                .take(size)
                .map(|(id, _, _)| u64::from(id))
                .collect(),
        }
    }
}

struct Matches {
    partition: Arc<InvertedPartition>,
    terms: Arc<Vec<String>>,
    term: usize,
    cursor: Option<Cursor>,
    batch_size: usize,
}

impl InvertedIndex {
    /// Stream row IDs matching any analyzed query term without scores or corpus
    /// statistics. Batches contain at most `batch_size` IDs (1..=1024). Rows can
    /// repeat across terms; consumers deduplicate at their desired identity.
    /// Up to eight partitions advance concurrently; batches arrive without
    /// ordering between partitions so one slow read does not block the others.
    /// Dropping the stream cancels pending I/O. The caller must apply dataset
    /// deletions and its predicates when taking these rows from the snapshot.
    /// Legacy V1 indexes are rejected: their located postings do not retain the
    /// original-to-current row mapping needed after fragment reuse.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use lance_index::scalar::inverted::InvertedIndex;
    /// # fn example(index: Arc<InvertedIndex>) -> lance_core::Result<()> {
    /// let matches = index.stream_match_rows("needle", 32)?;
    /// drop(matches);
    /// # Ok(()) }
    /// ```
    pub fn stream_match_rows(
        &self,
        query: &str,
        batch_size: usize,
    ) -> Result<BoxStream<'static, Result<Vec<u64>>>> {
        if !(1..=1024).contains(&batch_size) {
            return Err(Error::invalid_input(
                "match batch_size must be between 1 and 1024",
            ));
        }
        if self.is_legacy() {
            return Err(Error::invalid_input(
                "streaming matches require a modern inverted index",
            ));
        }
        let tokens = collect_query_tokens(query, &mut self.tokenizer());
        let mut terms: Vec<String> = (&tokens).into_iter().cloned().collect();
        terms.sort_unstable();
        terms.dedup();
        if terms.is_empty() {
            return Ok(stream::empty().boxed());
        }
        let terms = Arc::new(terms);
        Ok(stream::iter(self.partitions.clone())
            .map(move |partition| partition_matches(partition, terms.clone(), batch_size))
            .flatten_unordered(8)
            .boxed())
    }
}

// Both preloaded and lazy openers use the same posting/remapping implementation.
pub(super) fn partition_matches(
    partition: Arc<InvertedPartition>,
    terms: Arc<Vec<String>>,
    batch_size: usize,
) -> BoxStream<'static, Result<Vec<u64>>> {
    let state = Matches {
        partition,
        terms,
        term: 0,
        cursor: None,
        batch_size,
    };
    stream::try_unfold(state, |mut state| async move {
        loop {
            let partition = &state.partition;
            if let Some(cursor) = state.cursor.as_mut() {
                let ids = cursor.batch(state.batch_size);
                if !ids.is_empty() {
                    let rows = match &partition.docs {
                        PartitionDocumentStore::Modern(docs) => docs.match_row_ids(&ids).await?,
                        PartitionDocumentStore::Legacy(_) => {
                            return Err(Error::invalid_input(
                                "streaming matches require a modern inverted index",
                            ));
                        }
                    };
                    if !rows.is_empty() {
                        return Ok(Some((rows, state)));
                    }
                    tokio::task::yield_now().await;
                    continue;
                }
                state.cursor = None;
            }
            if state.term == state.terms.len() {
                return Ok(None);
            }
            let token_id = partition.tokens.get(&state.terms[state.term]);
            state.term += 1;
            if let Some(token_id) = token_id {
                let posting = partition
                    .inverted_list
                    .posting_list_with_policy(
                        token_id,
                        false,
                        &NoOpMetricsCollector,
                        PostingReadPolicy::CacheAwareExact,
                    )
                    .await?;
                if !posting.is_empty() {
                    state.cursor = Some(Cursor::new(posting));
                }
            }
        }
    })
    .boxed()
}
