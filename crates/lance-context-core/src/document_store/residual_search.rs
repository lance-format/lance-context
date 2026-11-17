//! Complete search coverage without duplicating residual text into another column.

use super::*;
use lance_index::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use std::collections::HashSet;

impl DocumentStore {
    pub(super) async fn search_residual(
        &self,
        query: &str,
        limit: usize,
        hits: &mut Vec<DocumentSearchHit>,
    ) -> Result<()> {
        if hits.len() >= limit {
            return Ok(());
        }
        let mut tokenizer = super::query::text_index_params().build()?;
        let mut terms = HashSet::new();
        {
            let mut stream = tokenizer.token_stream_for_search(query);
            while let Some(token) = stream.next() {
                terms.insert(token.text.clone());
            }
        }
        if terms.is_empty() {
            return Ok(());
        }
        let mut seen: HashSet<_> = hits.iter().map(|hit| hit.id.clone()).collect();
        let mut scanner = self.dataset.scan();
        scanner.project(&[ID, RESIDUAL])?;
        scanner
            .batch_size(1024)
            .batch_size_bytes(4 * 1024 * 1024)
            .batch_readahead(1);
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::invalid_input("missing document IDs"))?;
            let residuals = batch
                .column(1)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| Error::invalid_input("missing document residual"))?;
            for row in 0..batch.num_rows() {
                if ids.is_null(row) || residuals.is_null(row) {
                    return Err(Error::invalid_input("null document ID or residual"));
                }
                let id = ids.value(row);
                if seen.contains(id) {
                    continue;
                }
                let value: Value = serde_json::from_str(residuals.value(row)).map_err(invalid)?;
                if matches_text(&value, tokenizer.as_mut(), &terms) {
                    seen.insert(id.to_owned());
                    hits.push(DocumentSearchHit {
                        id: id.to_owned(),
                        score: 0.0,
                    });
                    if hits.len() == limit {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

fn matches_text(
    value: &Value,
    tokenizer: &mut dyn LanceTokenizer,
    terms: &HashSet<String>,
) -> bool {
    match value {
        Value::String(text) => {
            let mut stream = tokenizer.token_stream_for_doc(text);
            while let Some(token) = stream.next() {
                if terms.contains(&token.text) {
                    return true;
                }
            }
            false
        }
        Value::Object(object) => object
            .values()
            .any(|value| matches_text(value, tokenizer, terms)),
        Value::Array(values) => values
            .iter()
            .any(|value| matches_text(value, tokenizer, terms)),
        _ => false,
    }
}
