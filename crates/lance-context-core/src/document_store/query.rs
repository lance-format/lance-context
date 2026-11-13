//! Logical filtering and automatic text-search preparation.

use super::*;
use arrow_array::Float32Array;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::prelude::Expr;
use futures::StreamExt;
use lance::dataset::{CommitBuilder, InsertBuilder, WriteMode};
use lance::index::DatasetIndexExt;
use lance_index::scalar::inverted::Language;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::IndexType;
use serde::{Deserialize, Serialize};

const BACKFILLED_KEY: &str = "lance-context:shred-backfilled-columns";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSearchHit {
    pub id: String,
    pub score: f32,
}

impl DocumentStore {
    /// Equality over a logical object path, with residual fallback for old rows,
    /// nulls, containers, and type mismatches. Returns up to `limit` document IDs
    /// in unspecified order. Missing and explicit null are distinct.
    pub async fn filter_eq(
        &self,
        path: &[String],
        value: &Value,
        limit: usize,
    ) -> Result<Vec<String>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        let field = self.layout.fields.iter().find(|field| field.path == path);
        if let Some(field) = field.filter(|field| Some(field.data_type) == ShredType::of(value)) {
            let mut scanner = self.dataset.scan();
            scanner
                .project(&[ID])?
                .filter_expr(col(column_name(field)).eq(scalar_literal(value)?));
            scanner.limit(Some(i64::try_from(limit).map_err(invalid)?), None)?;
            let mut stream = scanner.try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                ids.extend(batch_ids(&batch)?);
            }
        }
        if ids.len() == limit {
            return Ok(ids);
        }
        let mut scanner = self.dataset.scan();
        // A non-null extracted scalar cannot match a different JSON type.
        if let Some(field) = field {
            scanner.filter_expr(col(column_name(field)).is_null());
        }
        // Reconstruct only the queried subtree and its scalar ancestors. Other
        // extracted fields can hold megabytes of text unrelated to this filter.
        let projected_layout = ShredLayout {
            fields: self
                .layout
                .fields
                .iter()
                .filter(|field| field.path.starts_with(path) || path.starts_with(&field.path))
                .cloned()
                .collect(),
        };
        let mut columns = vec![ID.to_string(), RESIDUAL.to_string()];
        columns.extend(projected_layout.fields.iter().map(column_name));
        scanner.project(&columns)?;
        scanner
            .batch_size(1024)
            .batch_size_bytes(4 * 1024 * 1024)
            .batch_readahead(1);
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let batch_ids = batch_ids(&batch)?;
            for (row, id) in batch_ids.into_iter().enumerate() {
                let document = decode(&batch, row, &projected_layout)?;
                if logical_value(&document, path) == Some(value) {
                    ids.push(id);
                    if ids.len() == limit {
                        return Ok(ids);
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Re-shred historical residual values with the current layout in a single
    /// overwrite commit. Streams batches targeting 4 MiB and at most 1024 rows;
    /// a single oversized record can exceed the byte target. IDs are retained.
    /// Historical dataset versions still retain their original bytes.
    pub async fn backfill(&mut self) -> Result<()> {
        if self.pinned {
            return Err(Error::invalid_input(
                "cannot backfill a pinned document snapshot",
            ));
        }
        let mut schema = Schema::from(self.dataset.schema());
        let mut metadata = schema.metadata().clone();
        metadata.insert(BACKFILLED_KEY.into(), self.layout.fields.len().to_string());
        schema = schema.with_metadata(metadata);
        let schema = Arc::new(schema);
        let mut scanner = self.dataset.scan();
        scanner
            .batch_size(1024)
            .batch_size_bytes(4 * 1024 * 1024)
            .batch_readahead(1);
        let source = scanner.try_into_stream().await?;
        let layout = self.layout.clone();
        let output_schema = schema.clone();
        let converted = source.map(move |batch| {
            let batch = batch?;
            let ids = batch_ids(&batch)?;
            let documents = (0..batch.num_rows())
                .map(|row| {
                    let document = decode(&batch, row, &layout)?;
                    layout.shred(document).map_err(Error::invalid_input)
                })
                .collect::<Result<Vec<_>>>()?;
            encode(&ids, &documents, &layout, output_schema.clone()).map_err(Into::into)
        });
        let stream: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(schema, converted));
        let params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let snapshot = Arc::new(self.dataset.clone());
        let transaction = InsertBuilder::new(snapshot.clone())
            .with_params(&params)
            .execute_uncommitted_stream(stream)
            .await?;
        // Strict overwrite must fail on a stale snapshot; rebasing a rewrite
        // would otherwise discard documents committed after the source scan.
        let dataset = CommitBuilder::new(snapshot)
            .with_max_retries(0)
            .execute(transaction)
            .await?;
        *self = Self::from_dataset(dataset, false)?;
        Ok(())
    }

    /// Index automatically selected text columns. Before building indexes,
    /// migrate historical residual values so searches cannot silently miss them.
    /// Called automatically by `search`; can be run ahead of latency-sensitive reads.
    pub async fn prepare_text_search(&mut self) -> Result<()> {
        let columns: Vec<_> = self
            .layout
            .fields
            .iter()
            .filter(|field| field.data_type == ShredType::String)
            .map(column_name)
            .collect();
        if columns.is_empty() {
            return Ok(());
        }
        let backfilled = self
            .dataset
            .schema()
            .metadata
            .get(BACKFILLED_KEY)
            .and_then(|value| value.parse::<usize>().ok());
        if backfilled != Some(self.layout.fields.len()) {
            self.backfill().await?;
        }
        let indices = self.dataset.load_indices().await?;
        for column in columns {
            let name = format!("document_fts_{column}");
            if indices.iter().any(|index| index.name == name) {
                continue;
            }
            if self.pinned {
                return Err(Error::invalid_input(
                    "pinned snapshot has no prepared text index",
                ));
            }
            let params = InvertedIndexParams::new("simple".into(), Language::English)
                .lower_case(true)
                .stem(false)
                .remove_stop_words(false)
                .max_token_length(None);
            self.dataset
                .create_index(
                    &[column.as_str()],
                    IndexType::Inverted,
                    Some(name),
                    &params,
                    true,
                )
                .await?;
        }
        Ok(())
    }

    /// Full-text search over automatically selected string fields. The first
    /// search after layout growth prepares historical data and indexes; later
    /// appends use Lance's indexed-plus-unindexed-tail search. Arrays and paths
    /// outside the column budget remain available through logical field queries.
    pub async fn search(&mut self, query: &str, limit: usize) -> Result<Vec<DocumentSearchHit>> {
        if limit == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(invalid)?;
        self.prepare_text_search().await?;
        let columns: Vec<_> = self
            .layout
            .fields
            .iter()
            .filter(|field| field.data_type == ShredType::String)
            .map(column_name)
            .collect();
        if columns.is_empty() {
            return Ok(Vec::new());
        }
        let mut scanner = self.dataset.scan();
        scanner.project(&[ID])?;
        scanner.full_text_search(
            FullTextSearchQuery::new(query.into())
                .with_columns(&columns)?
                .limit(Some(limit)),
        )?;
        let mut stream = scanner.try_into_stream().await?;
        let mut hits = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let scores = batch
                .column_by_name("_score")
                .and_then(|array| array.as_any().downcast_ref::<Float32Array>())
                .ok_or_else(|| Error::invalid_input("missing full-text score"))?;
            for (row, id) in batch_ids(&batch)?.into_iter().enumerate() {
                hits.push(DocumentSearchHit {
                    id,
                    score: scores.value(row),
                });
            }
        }
        Ok(hits)
    }
}

fn logical_value<'a>(document: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut value = document;
    for key in path {
        value = value.as_object()?.get(key)?;
    }
    Some(value)
}

fn batch_ids(batch: &RecordBatch) -> Result<Vec<String>> {
    let ids = batch
        .column_by_name(ID)
        .and_then(|array| array.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| Error::invalid_input("missing document IDs"))?;
    ids.iter()
        .map(|id| {
            id.ok_or_else(|| Error::invalid_input("null document ID"))
                .map(str::to_string)
        })
        .collect::<Result<_>>()
}

fn scalar_literal(value: &Value) -> Result<Expr> {
    match value {
        Value::String(value) => Ok(lit(value.as_str())),
        Value::Bool(value) => Ok(lit(*value)),
        Value::Number(value) if value.is_i64() => Ok(lit(value.as_i64().unwrap())),
        Value::Number(value) if value.is_u64() => Ok(lit(value.as_u64().unwrap())),
        Value::Number(value) => Ok(lit(value.as_f64().unwrap())),
        _ => Err(Error::invalid_input("expected a scalar filter value")),
    }
}
