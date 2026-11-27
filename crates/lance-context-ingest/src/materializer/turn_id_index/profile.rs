//! Cold existence lookup cost with and without a turn-ID ZoneMap, on key-ordered
//! and random-ordered fragments. Uses the production predicate shape
//! (`id IN (...)`, `id` projection, limit) and native scan statistics only.
//! Ignored by default: run explicitly to regenerate the PR evidence.
use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{WriteMode, WriteParams};
use lance::deps::datafusion::prelude::{col, lit};
use lance_datafusion::exec::ExecutionSummaryCounts;
use std::sync::Mutex;

const FRAGMENTS: usize = 32;
const ROWS_PER_FRAGMENT: usize = 16_384;
const WINDOW: usize = 64;

/// Canonical composite key shape used by managed turns: a JSON tuple.
fn key(session: usize, turn: usize) -> String {
    serde_json::to_string(&[format!("session-{session:08}"), format!("turn-{turn:04}")]).unwrap()
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    // Deterministic xorshift permutation; no external RNG dependency.
    let mut state = seed;
    let mut values: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.swap(i, (state % (i as u64 + 1)) as usize);
    }
    values
}

async fn write(uri: &str, ordered: bool) -> Result<Dataset> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let total = FRAGMENTS * ROWS_PER_FRAGMENT;
    // Ordered: each fragment holds one contiguous key range, as consecutive
    // sorted publications would. Random: keys are spread over every fragment.
    let order: Vec<usize> = if ordered {
        (0..total).collect()
    } else {
        shuffled(total, 0x5eed)
    };
    let mut dataset: Option<Dataset> = None;
    for fragment in order.chunks(ROWS_PER_FRAGMENT) {
        let mut keys: Vec<String> = fragment.iter().map(|&k| key(k / 4, k % 4)).collect();
        // Each publication batch is canonical-key sorted (BTreeMap order).
        keys.sort_unstable();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(keys))])?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let params = WriteParams {
            enable_stable_row_ids: true,
            mode: if dataset.is_some() {
                WriteMode::Append
            } else {
                WriteMode::Create
            },
            ..Default::default()
        };
        dataset = Some(Dataset::write(reader, uri, Some(params)).await?);
    }
    Ok(dataset.unwrap())
}

async fn index(dataset: &mut Dataset) -> Result<()> {
    let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();
    let segment = dataset
        .create_index_builder(
            &["id"],
            lance_index::IndexType::ZoneMap,
            &lance_index::scalar::ScalarIndexParams::for_builtin(
                lance_index::scalar::BuiltinIndexType::ZoneMap,
            ),
        )
        .name(Key::Turn.name().to_owned())
        .replace(true)
        .fragments(ids)
        .execute_uncommitted()
        .await?;
    let transaction = lance::dataset::transaction::Transaction::new_from_version(
        dataset.version_id(),
        lance::dataset::transaction::Operation::CreateIndex {
            new_indices: vec![segment],
            removed_indices: vec![],
        },
    );
    *dataset = lance::dataset::CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(transaction)
        .await?;
    Ok(())
}

/// One cold window: a fresh session, so no index or page cache is warm.
async fn lookup(uri: &str, keys: &[String]) -> Result<(usize, ExecutionSummaryCounts, f64)> {
    let dataset = lance::dataset::builder::DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(lance::session::Session::default()))
        .load()
        .await?;
    let stats = Arc::new(Mutex::new(None));
    let sink = stats.clone();
    let mut scan = dataset.scan();
    scan.project(&["id"])?;
    scan.filter_expr(col("id").in_list(keys.iter().map(|k| lit(k.as_str())).collect(), false));
    scan.limit(Some(keys.len() as i64 + 1), None)?;
    scan.batch_size(2048);
    scan.scan_stats_callback(Arc::new(move |counts: &ExecutionSummaryCounts| {
        *sink.lock().unwrap() = Some(counts.clone());
    }));
    let started = std::time::Instant::now();
    let found: usize = scan
        .try_into_stream()
        .await?
        .try_fold(0, |n, batch| async move { Ok(n + batch.num_rows()) })
        .await?;
    let seconds = started.elapsed().as_secs_f64();
    let counts = stats
        .lock()
        .unwrap()
        .take()
        .context("native scan statistics unavailable")?;
    Ok((found, counts, seconds))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "profile: run explicitly to regenerate cold-lookup evidence"]
async fn turn_id_zonemap_cold_lookup_profile() -> Result<()> {
    let root = tempfile::tempdir()?;
    let total = FRAGMENTS * ROWS_PER_FRAGMENT;
    // Half the window exists (spread over the table), half is absent but
    // inside the key range, as a duplicate-heavy registration batch would be.
    let mut window: Vec<String> = shuffled(total, 7)
        .into_iter()
        .take(WINDOW / 2)
        .map(|k| key(k / 4, k % 4))
        .collect();
    window.extend((0..WINDOW / 2).map(|i| key(total / 8 + i * 97, 9)));
    // A batch whose keys cluster in a few sessions, as one source call's turns
    // do: 4 sessions x 4 turns present, plus 4 sessions absent.
    let mut clustered: Vec<String> = (0..4)
        .flat_map(|s| (0..4).map(move |t| key(total / 4 * s / 4 + 1000, t)))
        .collect();
    clustered.extend((0..4).map(|s| key(total + s, 0)));
    for (ordered, label) in [(true, "ordered"), (false, "random")] {
        let uri = root
            .path()
            .join(format!("{label}.lance"))
            .to_str()
            .unwrap()
            .to_owned();
        let mut dataset = write(&uri, ordered).await?;
        let (found_plain, plain, plain_s) = lookup(&uri, &window).await?;
        index(&mut dataset).await?;
        let (found_zone, zone, zone_s) = lookup(&uri, &window).await?;
        assert_eq!(found_plain, found_zone, "ZoneMap changed the exact result");
        assert_eq!(found_plain, WINDOW / 2);
        // Clustered window, measured after the index exists and on a plain copy.
        let (found_c, cz, cz_s) = lookup(&uri, &clustered).await?;
        assert_eq!(found_c, 16);
        let plain_uri = format!("{uri}.plain");
        write(&plain_uri, ordered).await?;
        let (found_cp, cp, cp_s) = lookup(&plain_uri, &clustered).await?;
        assert_eq!(
            found_cp, found_c,
            "ZoneMap changed the exact clustered result"
        );
        println!(
            "PROFILE layout={label} clustered_window={} found={found_c} \
             | no_index: bytes_read={} iops={} seconds={cp_s:.4} \
             | zonemap: bytes_read={} iops={} seconds={cz_s:.4}",
            clustered.len(),
            cp.bytes_read,
            cp.iops,
            cz.bytes_read,
            cz.iops,
        );
        println!(
            "PROFILE layout={label} fragments={FRAGMENTS} rows={total} window={WINDOW} found={found_plain} \
             | no_index: bytes_read={} iops={} requests={} seconds={plain_s:.4} \
             | zonemap: bytes_read={} iops={} requests={} indices_loaded={} parts_loaded={} \
             index_comparisons={} seconds={zone_s:.4}",
            plain.bytes_read,
            plain.iops,
            plain.requests,
            zone.bytes_read,
            zone.iops,
            zone.requests,
            zone.indices_loaded,
            zone.parts_loaded,
            zone.index_comparisons,
        );
    }
    Ok(())
}
