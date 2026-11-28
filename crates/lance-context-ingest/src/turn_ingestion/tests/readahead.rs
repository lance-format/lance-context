use super::*;
use lance::{
    dataset::{builder::DatasetBuilder, WriteParams},
    deps::datafusion::execution::runtime_env::RuntimeEnvBuilder,
    session::Session,
    Dataset,
};
use lance_context_core::MergeMemoryBudget;
use std::sync::atomic::Ordering;

mod memory;
use memory::Memory;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bounded_native_readahead_keeps_exact_keys_and_releases_admission() -> Result<()> {
    const ROWS: usize = 20 * 1024;
    const LIMIT: usize = 256 << 20;
    let root = tempfile::tempdir()?;
    let uri = root.path().join("body.lance");
    let payload = vec![b'x'; 1024];
    let input = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from_iter_values((0..ROWS).map(|row| {
                if row % 2 == 0 {
                    "a"
                } else {
                    "b"
                }
            }))),
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|row| format!("turn-{row:08}")),
            )),
            Arc::new(StringArray::from(vec![None::<&str>; ROWS])),
            Arc::new(LargeBinaryArray::from_iter_values(
                (0..ROWS).map(|_| payload.as_slice()),
            )),
            Arc::new(Float64Array::from(vec![None; ROWS])),
        ],
    )?;
    let dataset = Dataset::write(
        arrow_array::RecordBatchIterator::new(vec![Ok(input)], schema()),
        uri.to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 1024,
            max_rows_per_group: 1024,
            enable_stable_row_ids: true,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?;
    assert_eq!(dataset.fragments().len(), 20);
    drop(dataset);
    let memory = Arc::new(Memory::new(1));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(memory.clone())
        .build_arc()?;
    let base = Arc::new(
        DatasetBuilder::from_uri(uri.to_str().unwrap())
            .with_session(Arc::new(
                Session::default()
                    .with_native_memory(memory.clone())
                    .with_runtime_env(runtime),
            ))
            .load()
            .await?,
    );
    let spec = spec()?;
    let keys = [
        ("a", "turn-00000000"),
        ("b", "turn-00020479"),
        ("b", "turn-00000000"),
        ("z", "missing"),
    ]
    .into_iter()
    .map(|(session, id)| spec.business_key(&[session, id], 1024))
    .collect::<Result<Vec<_>>>()?;
    let refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
    let budget = MergeMemoryBudget::new(1 << 20);
    let refused = spec
        .lookup_existing(base.clone(), &refs, &budget)
        .await
        .err()
        .expect("native refusal must remain an error, never an absent-key result");
    assert!(refused.chain().any(|cause| cause
        .downcast_ref::<lance::Error>()
        .is_some_and(lance::Error::is_backpressure)));
    memory.released().await?;
    memory.limit.store(LIMIT, Ordering::SeqCst);
    let result = spec
        .lookup_existing(base.clone(), &refs, &budget)
        .await?
        .expect("key workspace admitted");
    for (key, expected) in keys.iter().zip([true, true, false, false]) {
        assert_eq!(result.get(key), Some(expected));
    }
    assert_eq!(result.observation().fragments_scanned, Some(20));
    assert_eq!(result.observation().scan_partitions, 8);
    assert!(result
        .observation()
        .fragment_streams_peak
        .is_some_and(|peak| (2..=8).contains(&peak)));
    assert!(result.observation().first_result_us.is_some());
    eprintln!("bounded native readahead scan: {:?}", result.observation());
    drop(result);
    // An all-hit lookup cancels its unfinished scan. Its borrowed buffers must
    // release their reservations just as an exhausted or refused scan does.
    let hit = spec
        .lookup_existing(base.clone(), &[refs[0]], &budget)
        .await?
        .unwrap();
    assert_eq!(hit.get(&keys[0]), Some(true));
    assert!(hit.observation().lance_bytes_read.is_some());
    assert!(hit.observation().lance_requests.is_some());
    drop(hit);
    drop(base);
    memory.released().await?;
    memory.report(LIMIT);
    Ok(())
}
