//! Byte-targeted encoding with one atomic append for the caller's input batch.
use super::*;

const TARGET_BYTES: usize = 4 * 1024 * 1024;
const MAX_ROWS: usize = 1024;

pub(super) struct DocumentBatches {
    documents:
        std::iter::Peekable<std::iter::Zip<std::vec::IntoIter<String>, std::vec::IntoIter<Value>>>,
    layout: ShredLayout,
    schema: Arc<Schema>,
}

impl DocumentBatches {
    pub(super) fn new(
        ids: Vec<String>,
        documents: Vec<Value>,
        layout: ShredLayout,
        schema: Arc<Schema>,
    ) -> Self {
        Self {
            documents: ids.into_iter().zip(documents).peekable(),
            layout,
            schema,
        }
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let mut ids = Vec::new();
        let mut documents = Vec::new();
        let mut bytes = 0usize;
        while let Some((_, document)) = self.documents.peek() {
            let size = estimated_bytes(document);
            if !documents.is_empty()
                && (documents.len() == MAX_ROWS || bytes.saturating_add(size) > TARGET_BYTES)
            {
                break;
            }
            let (id, document) = self.documents.next().expect("peeked document");
            bytes = bytes.saturating_add(size);
            ids.push(id);
            documents.push(self.layout.shred(document).map_err(Error::invalid_input)?);
        }
        if documents.is_empty() {
            return Ok(None);
        }
        encode(&ids, &documents, &self.layout, self.schema.clone()).map(Some)
    }
}

impl Iterator for DocumentBatches {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch().transpose().map(|result| {
            result.map_err(|error| arrow_schema::ArrowError::ExternalError(Box::new(error)))
        })
    }
}

// Conservative payload estimate without allocating serialized JSON. Escaped
// residual strings may need six bytes per source byte. A single oversized
// document is allowed, and caller-owned input / Lance buffers are not capped.
fn estimated_bytes(value: &Value) -> usize {
    let payload = match value {
        Value::String(text) => text.len().saturating_mul(6),
        Value::Array(values) => values
            .iter()
            .fold(0usize, |n, value| n.saturating_add(estimated_bytes(value))),
        Value::Object(values) => values.iter().fold(0usize, |n, (key, value)| {
            n.saturating_add(key.len().saturating_mul(6))
                .saturating_add(estimated_bytes(value))
        }),
        _ => 32,
    };
    payload.saturating_add(std::mem::size_of::<Value>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn encoding_splits_by_bytes_and_rows_but_accepts_one_oversized_document() {
        let dir = tempfile::tempdir().unwrap();
        let store =
            DocumentStore::create(dir.path().to_str().unwrap(), ShredPolicy::default(), None)
                .await
                .unwrap();
        let schema = Arc::new(Schema::from(store.dataset.schema()));
        let docs = vec![json!({"text": "x".repeat(400_000)}); 3];
        let batches: Vec<_> = DocumentBatches::new(
            vec!["a".into(), "b".into(), "c".into()],
            docs,
            store.layout.clone(),
            schema.clone(),
        )
        .collect::<std::result::Result<_, _>>()
        .unwrap();
        assert_eq!(
            batches.len(),
            3,
            "byte budget must split before the row limit"
        );
        let docs = vec![json!(null); 1025];
        let ids = (0..docs.len()).map(|i| i.to_string()).collect();
        let batches: Vec<_> = DocumentBatches::new(ids, docs, store.layout.clone(), schema.clone())
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![1024, 1]
        );
        let mut batches = DocumentBatches::new(
            vec!["large".into()],
            vec![json!("x".repeat(TARGET_BYTES + 1))],
            store.layout.clone(),
            schema,
        );
        assert_eq!(batches.next().unwrap().unwrap().num_rows(), 1);
        assert!(batches.next().is_none());
    }

    #[tokio::test]
    async fn multi_batch_add_remains_one_atomic_append_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
            .await
            .unwrap();
        // Establish schema and inspect every intermediate version after add.
        store.add(vec![json!({"body": "seed"})]).await.unwrap();
        let version = store.version();
        let docs = vec![json!({"body": "bounded text ".repeat(40_000)}); 6];
        let ids = store.add(docs.clone()).await.unwrap();
        let final_version = store.version();
        for snapshot in version..=final_version {
            store.checkout(snapshot).await.unwrap();
            let count = store.dataset.count_rows(None).await.unwrap();
            assert!(
                count == 1 || count == 7,
                "partial append visible at version {snapshot}"
            );
        }
        let reopened = DocumentStore::open(uri, None).await.unwrap();
        for (id, doc) in ids.iter().zip(docs) {
            assert_eq!(reopened.get(id).await.unwrap(), Some(doc));
        }
    }
}
