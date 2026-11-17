//! Versioned documents backed by the standard Arrow Variant extension.
use std::{collections::HashMap, sync::Arc};

use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{
    physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream},
    prelude::{col, lit},
};
use futures::{StreamExt, TryStreamExt};
use lance::dataset::{
    builder::DatasetBuilder, CommitBuilder, InsertBuilder, WriteMode, WriteParams,
};
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Dataset, Error, Result};
use lance_index::{optimize::OptimizeOptions, scalar::ScalarIndexParams, IndexType};
use parquet_variant::{VariantPath, VariantPathElement};
use parquet_variant_compute::{
    unshred_variant, variant_get, GetOptions, VariantArray, VariantType,
};
use serde_json::Value;

use crate::{
    shredding::ShredPolicy,
    variant_shredding::{json_values_to_variant, variant_to_json_values, AutoVariantLayout},
};

const FORMAT: &str = "lance-context:variant-document-format";
const LAYOUT: &str = "lance-context:variant-layout";
const POLICY: &str = "lance-context:variant-policy";
const ID: &str = "id";
const DOCUMENT: &str = "document";
const ID_INDEX: &str = "variant_document_id";

/// One writer per dataset. Layout changes rewrite existing rows and append new
/// rows in one atomic, non-rebasing commit; pinned readers retain old layouts.
/// The custom JSON-residual DocumentStore remains an independent baseline.
pub struct VariantDocumentStore {
    dataset: Dataset,
    layout: AutoVariantLayout,
    policy: ShredPolicy,
    pinned: bool,
}

impl VariantDocumentStore {
    pub async fn create(
        uri: &str,
        policy: ShredPolicy,
        storage_options: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let layout = AutoVariantLayout::infer(&[], &policy)?;
        let schema = schema(&layout, &policy)?;
        let reader = RecordBatchIterator::new(
            Vec::<std::result::Result<RecordBatch, arrow_schema::ArrowError>>::new(),
            schema,
        );
        let params = WriteParams {
            store_params: storage_options.map(|options| ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            }),
            ..Default::default()
        };
        Self::from_dataset(Dataset::write(reader, uri, Some(params)).await?, false)
    }

    pub async fn open(uri: &str, storage_options: Option<HashMap<String, String>>) -> Result<Self> {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(options) = storage_options {
            builder = builder.with_storage_options(options);
        }
        Self::from_dataset(builder.load().await?, false)
    }

    fn from_dataset(dataset: Dataset, pinned: bool) -> Result<Self> {
        let stored = Schema::from(dataset.schema());
        if stored.metadata().get(FORMAT).map(String::as_str) != Some("1") {
            return Err(invalid("not an Arrow Variant document store"));
        }
        let layout: AutoVariantLayout = serde_json::from_str(
            stored
                .metadata()
                .get(LAYOUT)
                .ok_or_else(|| invalid("missing layout"))?,
        )
        .map_err(invalid)?;
        let policy: ShredPolicy = serde_json::from_str(
            stored
                .metadata()
                .get(POLICY)
                .ok_or_else(|| invalid("missing policy"))?,
        )
        .map_err(invalid)?;
        policy.validate().map_err(invalid)?;
        if layout.fields().len() > policy.max_columns
            || stored.fields() != schema(&layout, &policy)?.fields()
        {
            return Err(invalid(
                "Variant document schema does not match persisted layout",
            ));
        }
        if !stored.field(1).has_valid_extension_type::<VariantType>() {
            return Err(invalid("missing Variant extension"));
        }
        Ok(Self {
            dataset,
            layout,
            policy,
            pinned,
        })
    }

    pub fn layout(&self) -> &AutoVariantLayout {
        &self.layout
    }
    pub fn version(&self) -> u64 {
        self.dataset.version().version
    }

    /// Add one atomic batch. Incoming sampled string paths take priority when
    /// the layout budget is exhausted. An actual layout change streams a full
    /// rewrite in the same commit, so this may be expensive on an evolving schema.
    pub async fn add(&mut self, documents: Vec<Value>) -> Result<Vec<String>> {
        if self.pinned {
            return Err(invalid("cannot write through a pinned Variant snapshot"));
        }
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let next = self.layout.evolve(&documents, &self.policy)?;
        let rewrite = next != self.layout;
        let schema = schema(&next, &self.policy)?;
        let ids = documents
            .iter()
            .map(|_| crate::generate_id())
            .collect::<Vec<_>>();
        let batches = VariantBatches {
            rows: ids.clone().into_iter().zip(documents).peekable(),
            layout: next.clone(),
            schema: schema.clone(),
        };
        let incoming = futures::stream::iter(batches);
        let source: SendableRecordBatchStream = if rewrite {
            let mut scanner = self.dataset.scan();
            scanner
                .batch_size(1024)
                .batch_size_bytes(4 * 1024 * 1024)
                .batch_readahead(1);
            let source = scanner.try_into_stream().await?;
            let layout = next.clone();
            let output_schema = schema.clone();
            let converted = source.map(move |batch| {
                let batch = batch?;
                let input = VariantArray::try_new(
                    batch
                        .column_by_name(DOCUMENT)
                        .ok_or_else(|| {
                            datafusion::error::DataFusionError::Execution(
                                "missing Variant column".into(),
                            )
                        })?
                        .as_ref(),
                )?;
                let shredded = layout.shred(&input)?;
                Ok(RecordBatch::try_new(
                    output_schema.clone(),
                    vec![batch.column(0).clone(), Arc::new(shredded.into_inner())],
                )?)
            });
            Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                converted.chain(incoming),
            ))
        } else {
            Box::pin(RecordBatchStreamAdapter::new(schema.clone(), incoming))
        };
        let params = WriteParams {
            mode: if rewrite {
                WriteMode::Overwrite
            } else {
                WriteMode::Append
            },
            ..Default::default()
        };
        let snapshot = Arc::new(self.dataset.clone());
        let transaction = InsertBuilder::new(snapshot.clone())
            .with_params(&params)
            .execute_uncommitted_stream(source)
            .await?;
        let dataset = CommitBuilder::new(snapshot)
            .with_max_retries(0)
            .execute(transaction)
            .await?;
        *self = Self::from_dataset(dataset, false)?;
        // Never report an already durable append as failed: a retry could duplicate it.
        if let Err(error) = self.prepare_point_reads().await {
            tracing::warn!(%error, "Variant data committed; retry prepare_point_reads for ID indexing");
        }
        Ok(ids)
    }

    pub async fn prepare_point_reads(&mut self) -> Result<()> {
        let indices = self.dataset.load_indices().await?;
        let count = indices
            .iter()
            .filter(|index| index.name == ID_INDEX)
            .count();
        if count > 0 && self.dataset.unindexed_fragments(ID_INDEX).await?.is_empty() {
            return Ok(());
        }
        if self.pinned {
            return Err(invalid("cannot index a pinned Variant snapshot"));
        }
        if self.dataset.count_rows(None).await? == 0 {
            return Ok(());
        }
        if count == 0 {
            self.dataset
                .create_index_builder(&[ID], IndexType::BTree, &ScalarIndexParams::default())
                .name(ID_INDEX.to_string())
                .replace(true)
                .await?;
        } else {
            let options = if count >= 8 {
                OptimizeOptions::merge(count)
            } else {
                OptimizeOptions::append()
            };
            self.dataset
                .optimize_indices(&options.index_names(vec![ID_INDEX.to_string()]))
                .await?;
        }
        Ok(())
    }

    pub async fn get(&self, id: &str) -> Result<Option<Value>> {
        self.get_path(id, &[]).await
    }

    /// Read a path using Arrow's Variant path kernel, including standard residual
    /// fallback. Projection includes the requested subtree and ancestor residuals.
    pub async fn get_path(&self, id: &str, path: &[String]) -> Result<Option<Value>> {
        let mut scanner = self.dataset.scan();
        scanner
            .project_with_schema(&self.project_schema(&self.projection(path))?)?
            .filter_expr(col(ID).eq(lit(id)));
        scanner.limit(Some(1), None)?;
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            if batch.num_rows() == 0 {
                continue;
            }
            let input = batch
                .column_by_name(DOCUMENT)
                .ok_or_else(|| invalid("missing Variant column"))?;
            let value = if path.is_empty() {
                input.clone()
            } else {
                select_path(input, path)?
            };
            let array = VariantArray::try_new(value.as_ref())?;
            return Ok(variant_to_json_values(&array)?.into_iter().next().flatten());
        }
        Ok(None)
    }

    fn projection(&self, path: &[String]) -> Vec<Vec<String>> {
        if path.is_empty() {
            return vec![vec![DOCUMENT.to_owned()]];
        }
        let schema = Schema::from(self.dataset.schema());
        let mut wrapper = schema.field(1).data_type();
        let mut prefix = vec![DOCUMENT.to_string()];
        let mut columns = vec![
            vec![DOCUMENT.into(), "metadata".into()],
            vec![DOCUMENT.into(), "value".into()],
        ];
        for (i, key) in path.iter().enumerate() {
            let DataType::Struct(fields) = wrapper else {
                break;
            };
            let Some(typed) = fields.iter().find(|f| f.name() == "typed_value") else {
                break;
            };
            let DataType::Struct(children) = typed.data_type() else {
                break;
            };
            let Some(child) = children.iter().find(|f| f.name() == key) else {
                break;
            };
            prefix.extend(["typed_value".into(), key.clone()]);
            if i + 1 == path.len() {
                columns.push(prefix);
                break;
            }
            let mut residual = prefix.clone();
            residual.push("value".into());
            columns.push(residual);
            wrapper = child.data_type();
        }
        columns
    }

    // Resolve literal JSON keys directly, including empty keys and dots. Lance's
    // string path grammar is not SQL quoting and cannot represent empty keys.
    fn project_schema(&self, paths: &[Vec<String>]) -> Result<lance_core::datatypes::Schema> {
        let mut ids = Vec::with_capacity(paths.len());
        for path in paths {
            let mut fields = &self.dataset.schema().fields;
            for (position, key) in path.iter().enumerate() {
                let field = fields
                    .iter()
                    .find(|field| field.name == *key)
                    .ok_or_else(|| invalid(format!("missing physical Variant path {path:?}")))?;
                if position + 1 == path.len() {
                    ids.push(field.id);
                }
                fields = &field.children;
            }
        }
        Ok(self.dataset.schema().project_by_ids(&ids, true))
    }

    pub async fn checkout(&mut self, version: u64) -> Result<()> {
        *self = Self::from_dataset(self.dataset.checkout_version(version).await?, true)?;
        Ok(())
    }
    pub async fn refresh(&mut self) -> Result<()> {
        let mut dataset = self.dataset.clone();
        dataset.checkout_latest().await?;
        *self = Self::from_dataset(dataset, false)?;
        Ok(())
    }
}

fn schema(layout: &AutoVariantLayout, policy: &ShredPolicy) -> Result<Arc<Schema>> {
    let empty = layout.shred(&json_values_to_variant(std::iter::empty())?)?;
    Ok(Arc::new(
        Schema::new(vec![
            Field::new(ID, DataType::Utf8, false),
            empty.field(DOCUMENT),
        ])
        .with_metadata(HashMap::from([
            (FORMAT.into(), "1".into()),
            (
                LAYOUT.into(),
                serde_json::to_string(layout).map_err(invalid)?,
            ),
            (
                POLICY.into(),
                serde_json::to_string(policy).map_err(invalid)?,
            ),
        ])),
    ))
}

struct VariantBatches {
    rows:
        std::iter::Peekable<std::iter::Zip<std::vec::IntoIter<String>, std::vec::IntoIter<Value>>>,
    layout: AutoVariantLayout,
    schema: Arc<Schema>,
}
impl Iterator for VariantBatches {
    type Item = datafusion::error::Result<RecordBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut ids = Vec::new();
        let mut values = Vec::new();
        let mut bytes = 0usize;
        while let Some((_, value)) = self.rows.peek() {
            let size = estimate(value);
            if !values.is_empty()
                && (values.len() >= 1024 || bytes.saturating_add(size) > 4 * 1024 * 1024)
            {
                break;
            }
            let (id, value) = self.rows.next().unwrap();
            ids.push(id);
            values.push(value);
            bytes = bytes.saturating_add(size);
        }
        if values.is_empty() {
            return None;
        }
        Some((|| {
            let input = json_values_to_variant(values.iter().map(Some))?;
            let output = self.layout.shred(&input)?;
            Ok(RecordBatch::try_new(
                self.schema.clone(),
                vec![
                    Arc::new(StringArray::from(ids)),
                    Arc::new(output.into_inner()),
                ],
            )?)
        })())
    }
}
fn estimate(value: &Value) -> usize {
    match value {
        Value::String(value) => value.len().saturating_mul(6).saturating_add(32),
        Value::Array(values) => values
            .iter()
            .fold(32usize, |n, v| n.saturating_add(estimate(v))),
        Value::Object(values) => values.iter().fold(32usize, |n, (k, v)| {
            n.saturating_add(k.len().saturating_mul(6))
                .saturating_add(estimate(v))
        }),
        _ => 32,
    }
}
// Arrow 58's direct shredded path traversal can discard an ancestor residual
// when the requested child exists in the typed schema but is null for that row.
// Unshred only the projected subtree/residuals before the official path kernel.
fn select_path(input: &arrow_array::ArrayRef, path: &[String]) -> Result<arrow_array::ArrayRef> {
    let input = unshred_variant(&VariantArray::try_new(input.as_ref())?)?;
    let input: arrow_array::ArrayRef = Arc::new(input.into_inner());
    Ok(variant_get(
        &input,
        GetOptions::new_with_path(VariantPath::from_iter(
            path.iter()
                .map(|key| VariantPathElement::field(key.as_str())),
        )),
    )?)
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::invalid_input(error.to_string())
}

#[cfg(test)]
mod tests;
