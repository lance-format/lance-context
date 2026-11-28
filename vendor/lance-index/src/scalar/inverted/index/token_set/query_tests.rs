use super::*;
use crate::scalar::{IndexStore, lance_format::LanceIndexStore};
use lance_core::utils::tempfile::TempObjDir;
use lance_io::object_store::ObjectStore;

#[tokio::test]
async fn paged_and_legacy_query_subsets_preserve_posting_ids() -> Result<()> {
    let terms = (0..1300)
        .map(|id| format!("term-{id:04}"))
        .collect::<Vec<_>>();
    let wanted = [0, 511, 512, 1023, 1024, 1299]
        .into_iter()
        .map(|id| terms[id].clone())
        .chain(["absent".to_owned(), "term-9999".to_owned()])
        .collect::<Vec<_>>();
    for mode in 0..3 {
        let directory = TempObjDir::default();
        let store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            directory.clone(),
            Arc::new(LanceCache::no_cache()),
        );
        let mut original = TokenSet::default();
        // IDs deliberately differ from dictionary order.
        for term in terms.iter().rev() {
            original.add(term.clone());
        }
        let format = if mode == 2 {
            TokenSetFormat::Arrow
        } else {
            TokenSetFormat::Fst
        };
        let mut batch = original.clone().to_batch(format)?;
        if mode == 1 {
            batch = batch.project(&[0, 1, 2])?.slice(0, 1);
            let schema = Arc::new(arrow_schema::Schema::new(batch.schema().fields().clone()));
            batch = RecordBatch::try_new(schema, batch.columns().to_vec())?;
        }
        let mut writer = store.new_index_file("tokens.lance", batch.schema()).await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;
        let reader = store.open_index_file("tokens.lance").await?;
        let cache = LanceCache::with_capacity(1 << 20);
        for _ in 0..2 {
            let selected =
                TokenSet::load_for_terms(reader.clone(), format, &wanted, &cache).await?;
            assert_eq!(selected.len(), 6);
            assert_eq!(selected.next_id, original.next_id);
            assert_eq!(selected.total_length, original.total_length);
            for term in &wanted {
                assert_eq!(selected.get(term), original.get(term));
            }
            assert_eq!(selected.get(&terms[100]), None);
        }
        // The fallback cache may already own the complete dictionary. This
        // path must preserve identical IDs and corpus statistics as paged I/O.
        cache
            .insert_with_key(&CompleteTokenSetKey, Arc::new(original.clone()))
            .await;
        let selected = TokenSet::load_for_terms(reader, format, &wanted, &cache).await?;
        for term in &wanted {
            assert_eq!(selected.get(term), original.get(term));
        }
    }
    Ok(())
}

#[test]
fn paged_output_keeps_the_original_fst_allocation() -> Result<()> {
    let mut tokens = TokenSet::default();
    for id in 0..1300 {
        tokens.add(format!("term-{id:04}"));
    }
    let TokenMap::HashMap(map) = tokens.tokens else {
        unreachable!()
    };
    let fst = TokenSet::build_fst_from_map(map)?;
    let pointer = fst.as_fst().as_bytes().as_ptr();
    let batch = fst_token_batch(fst, 1300, 11700)?;
    assert_eq!(batch.num_rows(), 3);
    let column = batch[TOKEN_FST_BYTES_COL].as_binary::<i64>();
    assert_eq!(column.value(0).as_ptr(), pointer);
    assert!(column.value(1).is_empty());
    let retained = Bytes::from_owner(FstBuffer(column.values().clone()));
    drop(batch);
    let fst = fst::Map::new(retained).unwrap();
    assert_eq!(fst.get("term-0512"), Some(512));
    Ok(())
}
