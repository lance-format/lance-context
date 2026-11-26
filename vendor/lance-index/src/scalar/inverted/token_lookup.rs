//! Optional paged lookup beside the original FST in the same token file.
//!
//! The original columns retain their first-row values for older readers. Exact
//! query preparation reads only fence keys and the requested FST blocks.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{AsArray, LargeBinaryBuilder, StringBuilder};
use arrow_array::{Array, LargeBinaryArray, StringArray};
use fst::{IntoStreamer, Streamer};
use lance_core::cache::{CacheKey, LanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};

use crate::scalar::IndexReader;

pub(super) const LOOKUP_VERSION_KEY: &str = "token_lookup_version";
pub(super) const FIRST_TOKEN_COL: &str = "_lookup_first_token";
pub(super) const LOOKUP_FST_COL: &str = "_lookup_fst_bytes";
// Limit terms per block, independently of total vocabulary size.
pub(super) const BLOCK_TERMS: usize = 512;

struct FencesKey;
impl CacheKey for FencesKey {
    type ValueType = LookupFences;
    fn key(&self) -> Cow<'_, str> {
        "token-lookup-fences".into()
    }
    fn type_name() -> &'static str {
        "FTS/TokenLookupFences"
    }
}

// Retain the decoded Arrow buffers, including their native reservations. A
// separate String per fence would copy data and escape the reader's budget.
struct LookupFences(StringArray);

impl DeepSizeOf for LookupFences {
    fn deep_size_of_children(&self, _ctx: &mut Context) -> usize {
        self.0.get_buffer_memory_size()
    }
}

impl LookupFences {
    fn block_for(&self, term: &str) -> Option<usize> {
        let (mut lower, mut upper) = (0, self.0.len());
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            if self.0.value(middle) <= term {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        lower.checked_sub(1)
    }
}

struct LookupBytes(arrow::buffer::Buffer);
impl AsRef<[u8]> for LookupBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

struct LookupBlock(fst::Map<bytes::Bytes>);
impl DeepSizeOf for LookupBlock {
    fn deep_size_of_children(&self, _ctx: &mut Context) -> usize {
        self.0.as_fst().size()
    }
}

struct BlockKey(usize);
impl CacheKey for BlockKey {
    type ValueType = LookupBlock;
    fn key(&self) -> Cow<'_, str> {
        self.0.to_string().into()
    }
    fn type_name() -> &'static str {
        "FTS/TokenLookupBlock"
    }
}

pub(super) fn encode<T: AsRef<[u8]>>(map: &fst::Map<T>) -> Result<(StringArray, LargeBinaryArray)> {
    let mut first = StringBuilder::new();
    let mut blocks = LargeBinaryBuilder::new();
    let mut stream = map.stream();
    let mut builder = fst::MapBuilder::memory();
    let mut count = 0;
    while let Some((token, id)) = stream.next() {
        if count == 0 {
            first.append_value(std::str::from_utf8(token).map_err(|error| {
                Error::index(format!("token dictionary contains invalid UTF-8: {error}"))
            })?);
        }
        builder
            .insert(token, id)
            .map_err(|error| Error::index(format!("cannot encode token lookup block: {error}")))?;
        count += 1;
        if count == BLOCK_TERMS {
            blocks.append_value(builder.into_inner().map_err(|error| {
                Error::index(format!("cannot finish token lookup block: {error}"))
            })?);
            builder = fst::MapBuilder::memory();
            count = 0;
        }
    }
    if count != 0 || map.is_empty() {
        if map.is_empty() {
            first.append_value("");
        }
        blocks.append_value(
            builder.into_inner().map_err(|error| {
                Error::index(format!("cannot finish token lookup block: {error}"))
            })?,
        );
    }
    Ok((first.finish(), blocks.finish()))
}

pub(super) async fn lookup(
    reader: &Arc<dyn IndexReader>,
    terms: &[String],
    cache: &LanceCache,
) -> Result<Option<HashMap<String, u32>>> {
    let Some(version) = reader.schema().metadata.get(LOOKUP_VERSION_KEY) else {
        return Ok(None);
    };
    if version != "1" {
        // The original FST remains authoritative when a newer optional lookup
        // layout is not understood by this reader.
        return Ok(None);
    }
    let fences = cache
        .get_or_insert_with_key(FencesKey, || async {
            let batch = reader
                .read_range(0..reader.num_rows(), Some(&[FIRST_TOKEN_COL]))
                .await?;
            let first = batch
                .column_by_name(FIRST_TOKEN_COL)
                .and_then(|array| array.as_string_opt::<i32>())
                .ok_or_else(|| Error::index("missing UTF-8 token lookup fences"))?;
            if first.is_empty() || first.null_count() != 0 {
                return Err(Error::index(
                    "token lookup fences must be nonempty and nonnull",
                ));
            }
            if (1..first.len()).any(|row| first.value(row - 1) >= first.value(row)) {
                return Err(Error::index(
                    "token lookup fences are not strictly increasing",
                ));
            }
            Ok(LookupFences(first.clone()))
        })
        .await?;
    let mut selected: BTreeMap<usize, Vec<&String>> = BTreeMap::new();
    for term in terms {
        if let Some(block) = fences.block_for(term) {
            selected.entry(block).or_default().push(term);
        }
    }
    let mut tokens = HashMap::new();
    if selected.is_empty() {
        return Ok(Some(tokens));
    }
    let mut loaded = BTreeMap::new();
    let mut missing = Vec::new();
    for &block in selected.keys() {
        if let Some(map) = cache.get_with_key(&BlockKey(block)).await {
            loaded.insert(block, map);
        } else {
            missing.push(block);
        }
    }
    if !missing.is_empty() {
        let ranges: Vec<_> = missing.iter().map(|row| *row..*row + 1).collect();
        let batch = reader.read_ranges(&ranges, Some(&[LOOKUP_FST_COL])).await?;
        let blocks = batch
            .column_by_name(LOOKUP_FST_COL)
            .and_then(|array| array.as_binary_opt::<i64>())
            .ok_or_else(|| Error::index("missing token lookup block bytes"))?;
        if blocks.len() != missing.len() || blocks.null_count() != 0 {
            return Err(Error::index("token lookup returned an invalid block count"));
        }
        for (position, block) in missing.into_iter().enumerate() {
            let start = blocks.value_offsets()[position] as usize;
            let end = blocks.value_offsets()[position + 1] as usize;
            let bytes = bytes::Bytes::from_owner(LookupBytes(
                blocks.values().slice_with_length(start, end - start),
            ));
            let map = fst::Map::new(bytes).map_err(|error| {
                Error::index(format!("invalid token lookup block {block}: {error}"))
            })?;
            if map.len() > BLOCK_TERMS {
                return Err(Error::index("token lookup block exceeds the term limit"));
            }
            let mut entries = map.stream();
            match entries.next() {
                Some((key, _)) if key == fences.0.value(block).as_bytes() => {}
                None if fences.0.len() == 1 && fences.0.value(0).is_empty() => {}
                _ => {
                    return Err(Error::index(
                        "token lookup block does not match its first fence",
                    ));
                }
            }
            if block + 1 < fences.0.len()
                && (map.len() != BLOCK_TERMS
                    || map
                        .range()
                        .ge(fences.0.value(block + 1).as_bytes())
                        .into_stream()
                        .next()
                        .is_some())
            {
                return Err(Error::index("token lookup block crosses its next fence"));
            }
            let map = Arc::new(LookupBlock(map));
            cache.insert_with_key(&BlockKey(block), map.clone()).await;
            loaded.insert(block, map);
        }
    }
    for (block, terms) in selected {
        let map = &loaded[&block].0;
        for term in terms {
            if let Some(id) = map.get(term) {
                tokens.insert(
                    term.clone(),
                    u32::try_from(id)
                        .map_err(|error| Error::index(format!("token ID exceeds u32: {error}")))?,
                );
            }
        }
    }
    Ok(Some(tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_search_preserves_arrow_buffers_and_boundary_semantics() {
        let array = StringArray::from(vec!["a", "b", "é"]);
        let pointer = array.values().as_ptr();
        let fences = LookupFences(array.clone());
        drop(array);
        assert_eq!(fences.0.values().as_ptr(), pointer);
        assert_eq!(fences.block_for(""), None);
        assert_eq!(fences.block_for("a"), Some(0));
        assert_eq!(fences.block_for("az"), Some(0));
        assert_eq!(fences.block_for("b"), Some(1));
        assert_eq!(fences.block_for("é"), Some(2));
        assert_eq!(fences.block_for("界"), Some(2));
    }
}
