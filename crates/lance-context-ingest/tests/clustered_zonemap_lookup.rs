//! Sorted compaction plus a `session_id` ZoneMap makes the historical dedup
//! lookup prune. Sessions arrive interleaved, so every append fragment spans
//! almost the whole key range and a ZoneMap alone cannot skip anything.
#![cfg(feature = "materializer")]
use anyhow::Result;
use arrow_array::{BinaryArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::{
    dataset::{
        index::DatasetIndexRemapperOptions,
        optimize::{commit_compaction, CompactionOptions, CompactionTask, TaskData},
        WriteMode, WriteParams,
    },
    index::DatasetIndexExt,
    Dataset,
};
use lance_context_core::MergeMemoryBudget;
use lance_context_ingest::turn_ingestion::{TurnIngestionSpec, TurnLookupObservation};
use lance_index::{
    scalar::{zonemap::ZoneMapIndexBuilderParams, BuiltinIndexType, ScalarIndexParams},
    IndexType,
};
use std::sync::Arc;

const SESSIONS: usize = 512;
const TURNS: usize = 16;
const APPENDS: usize = 16;
const ROWS: usize = SESSIONS * TURNS;
const ROWS_PER_ZONE: u64 = 256;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("session_id", DataType::Utf8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
    ]))
}

fn session(n: usize) -> String {
    format!("session-{n:05}")
}

/// Each append holds one turn of every session, in a shuffled session order.
fn appends() -> Vec<RecordBatch> {
    (0..APPENDS)
        .map(|append| {
            let mut rows = Vec::new();
            for turn in (append * TURNS / APPENDS)..((append + 1) * TURNS / APPENDS) {
                for n in 0..SESSIONS {
                    let s = (n * 389 + append * 97) % SESSIONS;
                    rows.push((session(s), format!("turn-{s:05}-{turn:03}")));
                }
            }
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(StringArray::from_iter_values(rows.iter().map(|r| &r.0))),
                    Arc::new(StringArray::from_iter_values(rows.iter().map(|r| &r.1))),
                    Arc::new(BinaryArray::from_iter_values(
                        rows.iter().map(|_| [7u8; 64]),
                    )),
                ],
            )
            .unwrap()
        })
        .collect()
}

async fn write(uri: &str) -> Result<Dataset> {
    let mut dataset = None;
    for batch in appends() {
        let params = WriteParams {
            enable_stable_row_ids: true,
            mode: if dataset.is_some() {
                WriteMode::Append
            } else {
                WriteMode::Create
            },
            ..Default::default()
        };
        dataset = Some(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema()),
                uri,
                Some(params),
            )
            .await?,
        );
    }
    Ok(dataset.unwrap())
}

async fn index_sessions(dataset: &mut Dataset) -> Result<()> {
    dataset
        .create_index(
            &["session_id"],
            IndexType::ZoneMap,
            Some("lance_context_turn_routing".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
                .with_params(&ZoneMapIndexBuilderParams::new(ROWS_PER_ZONE)),
            true,
        )
        .await?;
    Ok(())
}

async fn lookup(
    spec: &TurnIngestionSpec,
    dataset: &Dataset,
    keys: &[String],
) -> Result<TurnLookupObservation> {
    let refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
    let found = spec
        .lookup_existing(
            Arc::new(dataset.clone()),
            &refs,
            &MergeMemoryBudget::new(16 << 20),
        )
        .await?
        .unwrap();
    for key in keys.iter().take(keys.len() - 1) {
        assert_eq!(found.get(key), Some(true), "{key}");
    }
    assert_eq!(found.get(keys.last().unwrap()), Some(false));
    Ok(found.observation().clone())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sorted_compaction_lets_session_zonemap_prune_the_dedup_lookup() -> Result<()> {
    let spec = TurnIngestionSpec::new(
        schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    // Four sessions' turns plus one miss in a known session: a small window,
    // as for a sparse duplicate check against a large history.
    let mut keys = [130, 170, 301, 388]
        .into_iter()
        .flat_map(|s| {
            let spec = &spec;
            (0..TURNS).step_by(5).map(move |turn| {
                spec.business_key(&[&session(s), &format!("turn-{s:05}-{turn:03}")], 4096)
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    keys.push(spec.business_key(&[&session(170), "turn-never"], 4096)?);

    let root = tempfile::tempdir()?;
    // Unclustered: interleaved appends, each with every session.
    let mut random = write(root.path().join("random").to_str().unwrap()).await?;
    index_sessions(&mut random).await?;
    let unclustered = lookup(&spec, &random, &keys).await?;

    // Clustered: the same appends, rewritten sorted by (session_id, id) into
    // several output fragments, then indexed again over the new fragments
    // (a rewrite withdraws ZoneMap coverage; maintenance rebuilds it).
    let mut sorted = write(root.path().join("sorted").to_str().unwrap()).await?;
    index_sessions(&mut sorted).await?;
    // One task over every fragment, as the materializer submits one bounded
    // task: a sort orders rows within a task, not across tasks.
    let options = CompactionOptions {
        target_rows_per_fragment: ROWS / 8,
        sort_by: vec!["session_id".into(), "id".into()],
        sort_memory_bytes: Some(32 << 20),
        ..Default::default()
    };
    let task = CompactionTask {
        task: TaskData {
            fragments: sorted.manifest().fragments.iter().cloned().collect(),
        },
        read_version: sorted.version_id(),
        options: options.clone(),
    };
    let result = task.execute(&sorted).await?;
    assert!(
        result.row_addrs.is_none(),
        "sorted rewrites record no address moves"
    );
    commit_compaction(
        &mut sorted,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await?;
    assert_eq!(sorted.get_fragments().len(), 8);
    let covered_after_rewrite = sorted
        .load_indices()
        .await?
        .iter()
        .filter(|i| i.name == "lance_context_turn_routing")
        .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
        .count();
    assert_eq!(
        covered_after_rewrite, 0,
        "a rewrite withdraws ZoneMap coverage"
    );
    let before_rebuild = lookup(&spec, &sorted, &keys).await?;
    index_sessions(&mut sorted).await?;
    let clustered = lookup(&spec, &sorted, &keys).await?;

    // Interleaved: every fragment and zone covers the whole key range.
    assert_eq!(unclustered.fragments_scanned, Some(APPENDS));
    assert!(unclustered.rows_scanned.unwrap() >= ROWS * 9 / 10);
    // Clustered but not yet re-covered: no index, so still a full scan.
    assert_eq!(before_rebuild.rows_scanned, Some(ROWS));
    // Clustered and covered: four sessions fall in at most four fragments and
    // four or eight zones each of 256 rows, instead of 8192 rows.
    let fragments = clustered.fragments_scanned.unwrap();
    let rows = clustered.rows_scanned.unwrap();
    assert!(fragments <= 4, "{clustered:?}");
    assert!(rows <= 8 * ROWS_PER_ZONE as usize, "{clustered:?}");
    assert!(
        rows * 4 <= unclustered.rows_scanned.unwrap(),
        "{clustered:?}"
    );
    eprintln!(
        "unclustered fragments={:?} rows={:?} bytes={:?}; clustered fragments={:?} rows={:?} bytes={:?}",
        unclustered.fragments_scanned,
        unclustered.rows_scanned,
        unclustered.lance_bytes_read,
        clustered.fragments_scanned,
        clustered.rows_scanned,
        clustered.lance_bytes_read
    );
    Ok(())
}
