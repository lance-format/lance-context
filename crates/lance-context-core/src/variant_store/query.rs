//! Search standard Variant typed children and residuals without a copied text column.
use super::*;
use crate::{shredding::ShredType, DocumentSearchHit};
use arrow_array::Float32Array;
use lance_index::scalar::inverted::{tokenizer::document_tokenizer::LanceTokenizer, Language};
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use std::collections::HashSet;

fn text_params() -> InvertedIndexParams {
    InvertedIndexParams::new("simple".into(), Language::English)
        .lower_case(true)
        .stem(false)
        .remove_stop_words(false)
        .max_token_length(None)
}

impl VariantDocumentStore {
    fn text_columns(&self) -> Vec<String> {
        self.layout
            .fields()
            .iter()
            .filter(|field| field.data_type == ShredType::String && indexable_path(&field.path))
            .map(|field| {
                let mut name = DOCUMENT.to_string();
                for key in &field.path {
                    name = format!("{name}.typed_value.{key}");
                }
                format!("{name}.typed_value")
            })
            .collect()
    }

    /// Index the standard typed text children directly. No duplicate text column
    /// and no preparation backfill: layout migrations already reshred old rows.
    pub async fn prepare_text_search(&mut self) -> Result<()> {
        let indices = self.dataset.load_indices().await?;
        for column in self.text_columns() {
            let name = format!(
                "variant_fts_{}",
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, column.as_bytes()).simple()
            );
            if indices.iter().any(|index| index.name == name) {
                continue;
            }
            if self.pinned {
                return Err(invalid(
                    "pinned Variant snapshot has an unprepared text index",
                ));
            }
            self.dataset
                .create_index(
                    &[column.as_str()],
                    IndexType::Inverted,
                    Some(name),
                    &text_params(),
                    true,
                )
                .await?;
        }
        Ok(())
    }

    /// Indexed hits first, then residual string matches with score 0 until limit.
    /// Both use the same OR token matching. This is not global BM25 ranking:
    /// rare/no-hit queries can scan the entire residual representation.
    pub async fn search(&mut self, query: &str, limit: usize) -> Result<Vec<DocumentSearchHit>> {
        if limit == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let native_limit = i64::try_from(limit).map_err(invalid)?;
        self.prepare_text_search().await?;
        let columns = self.text_columns();
        let mut hits = Vec::new();
        if !columns.is_empty() {
            let mut scanner = self.dataset.scan();
            scanner.project(&[ID])?;
            scanner.full_text_search(
                FullTextSearchQuery::new(query.into())
                    .with_columns(&columns)?
                    .limit(Some(native_limit)),
            )?;
            let mut stream = scanner.try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                let ids = batch
                    .column_by_name(ID)
                    .and_then(|a| a.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| invalid("missing IDs"))?;
                let scores = batch
                    .column_by_name("_score")
                    .and_then(|a| a.as_any().downcast_ref::<Float32Array>())
                    .ok_or_else(|| invalid("missing text scores"))?;
                hits.extend((0..batch.num_rows()).map(|row| DocumentSearchHit {
                    id: ids.value(row).into(),
                    score: scores.value(row),
                }));
            }
        }
        if hits.len() >= limit {
            return Ok(hits);
        }
        let mut tokenizer = text_params().build()?;
        let mut terms = HashSet::new();
        {
            let mut tokens = tokenizer.token_stream_for_search(query);
            while let Some(token) = tokens.next() {
                terms.insert(token.text.clone());
            }
        }
        if terms.is_empty() {
            return Ok(hits);
        }
        let mut seen: HashSet<_> = hits.iter().map(|hit| hit.id.clone()).collect();
        let mut scanner = self.dataset.scan();
        scanner.project_with_schema(&self.project_schema(&self.residual_projection())?)?;
        scanner
            .batch_size(1024)
            .batch_size_bytes(4 * 1024 * 1024)
            .batch_readahead(1);
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let ids = batch
                .column_by_name(ID)
                .and_then(|a| a.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| invalid("missing IDs"))?;
            let variant = VariantArray::try_new(
                batch
                    .column_by_name(DOCUMENT)
                    .ok_or_else(|| invalid("missing Variant"))?
                    .as_ref(),
            )?;
            let values = variant_to_json_values(&variant)?;
            for (row, value) in values.iter().enumerate() {
                if seen.contains(ids.value(row)) {
                    continue;
                }
                if value
                    .as_ref()
                    .is_some_and(|value| matches(value, tokenizer.as_mut(), &terms))
                {
                    seen.insert(ids.value(row).to_owned());
                    hits.push(DocumentSearchHit {
                        id: ids.value(row).into(),
                        score: 0.0,
                    });
                    if hits.len() == limit {
                        return Ok(hits);
                    }
                }
            }
        }
        Ok(hits)
    }

    fn residual_projection(&self) -> Vec<Vec<String>> {
        let mut columns = vec![
            vec![ID.into()],
            vec![DOCUMENT.into(), "metadata".into()],
            vec![DOCUMENT.into(), "value".into()],
        ];
        for field in self.layout.fields() {
            let mut prefix = vec![DOCUMENT.to_string()];
            for key in &field.path {
                prefix.extend(["typed_value".into(), key.clone()]);
                let mut column = prefix.clone();
                column.push("value".into());
                if !columns.contains(&column) {
                    columns.push(column);
                }
            }
            // Lance's current nested FTS traversal only accepts plain path
            // components. Keep other typed strings searchable via this scan.
            if field.data_type == ShredType::String && !indexable_path(&field.path) {
                prefix.push("typed_value".into());
                columns.push(prefix);
            }
        }
        columns
    }

    /// Equality with standard Variant path semantics. This correctness path
    /// scans only the projected subtree/residual columns; it does not yet push a
    /// typed predicate into Lance or use a scalar field index.
    pub async fn filter_eq(
        &self,
        path: &[String],
        value: &Value,
        limit: usize,
    ) -> Result<Vec<String>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut columns = vec![vec![ID.to_owned()]];
        columns.extend(self.projection(path));
        let mut scanner = self.dataset.scan();
        scanner
            .project_with_schema(&self.project_schema(&columns)?)?
            .batch_size(1024)
            .batch_size_bytes(4 * 1024 * 1024)
            .batch_readahead(1);
        let mut stream = scanner.try_into_stream().await?;
        let mut hits = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let ids = batch
                .column_by_name(ID)
                .and_then(|a| a.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| invalid("missing IDs"))?;
            let input = batch
                .column_by_name(DOCUMENT)
                .ok_or_else(|| invalid("missing Variant"))?;
            let selected = select_path(input, path)?;
            let selected = VariantArray::try_new(selected.as_ref())?;
            for (row, actual) in variant_to_json_values(&selected)?.iter().enumerate() {
                if actual.as_ref() == Some(value) {
                    hits.push(ids.value(row).to_owned());
                    if hits.len() == limit {
                        return Ok(hits);
                    }
                }
            }
        }
        Ok(hits)
    }
}

fn matches(value: &Value, tokenizer: &mut dyn LanceTokenizer, terms: &HashSet<String>) -> bool {
    match value {
        Value::String(value) => {
            let mut tokens = tokenizer.token_stream_for_doc(value);
            while let Some(token) = tokens.next() {
                if terms.contains(&token.text) {
                    return true;
                }
            }
            false
        }
        Value::Array(values) => values.iter().any(|value| matches(value, tokenizer, terms)),
        Value::Object(values) => values
            .values()
            .any(|value| matches(value, tokenizer, terms)),
        _ => false,
    }
}

fn indexable_path(path: &[String]) -> bool {
    path.iter().all(|key| {
        !key.is_empty()
            && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && key
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    })
}
