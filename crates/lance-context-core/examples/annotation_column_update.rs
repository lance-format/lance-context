//! Native Lance experiment, not the distributed mutable staging protocol.
//! Run with --features lance-io-metrics; each invocation creates fresh datasets.
use std::{collections::BTreeSet, path::PathBuf, sync::Arc, time::Instant};

use arrow_array::{
    builder::LargeBinaryBuilder, Array, Float32Array, LargeBinaryArray, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use futures::TryStreamExt;
use lance::{
    dataset::{
        CommitBuilder, MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched,
        WriteParams,
    },
    Dataset,
};
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use serde_json::{json, Value};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    /// New local directory. Omit to use a temporary directory.
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    rows: usize,
    #[arg(long, default_value_t = 1_048_576)]
    body_bytes: usize,
    /// Update every Nth row in each round; 1 updates all rows.
    #[arg(long, default_value_t = 1)]
    stride: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
}

fn fixture(rows: usize, body_bytes: usize) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::LargeBinary, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    let mut bodies = LargeBinaryBuilder::with_capacity(rows, rows * body_bytes);
    let mut body = vec![0; body_bytes];
    // Deterministic, non-constant content so compression does not hide rewriting.
    for row in 0..rows {
        let mut state = row as u64 + 1;
        for byte in &mut body {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        bodies.append_value(&body);
    }
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|row| row.to_string()),
            )),
            Arc::new(bodies.finish()),
            Arc::new(StringArray::from(vec!["pending"; rows])),
            Arc::new(Float32Array::from(vec![None; rows])),
        ],
    )?)
}

fn score(round: usize) -> Option<f32> {
    // Includes assignment, clearing and assigning the same value again.
    (round > 0 && round.is_multiple_of(2)).then_some(0.9)
}

fn annotation_batch(base: &RecordBatch, stride: usize, round: usize) -> Result<RecordBatch> {
    let ids = (0..base.num_rows()).step_by(stride).collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        Arc::new(base.schema().project(&[0, 2, 3])?),
        vec![
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(usize::to_string),
            )),
            Arc::new(StringArray::from(vec!["reviewed"; ids.len()])),
            Arc::new(Float32Array::from(vec![score(round); ids.len()])),
        ],
    )?)
}

#[derive(Default, Clone, Copy)]
struct Io {
    read: u64,
    written: u64,
}

fn io(snapshotter: &Snapshotter) -> Io {
    let mut result = Io::default();
    for (key, _, _, value) in snapshotter.snapshot().into_vec() {
        if key.key().name() != "lance_object_store_request_bytes_total" {
            continue;
        }
        let DebugValue::Counter(bytes) = value else {
            continue;
        };
        for label in key.key().labels() {
            if label.key() == "operation" {
                match label.value() {
                    "get" => result.read += bytes,
                    "put" | "put_part" => result.written += bytes,
                    _ => {}
                }
            }
        }
    }
    result
}

fn io_delta(before: Io, after: Io) -> Value {
    json!({"read_bytes": after.read - before.read,
        "written_bytes": after.written - before.written})
}

fn body_files(dataset: &Dataset) -> Result<BTreeSet<String>> {
    let field = dataset.schema().field_id("body")?;
    Ok(dataset
        .manifest()
        .fragments
        .iter()
        .flat_map(|fragment| &fragment.files)
        .filter(|file| file.fields.contains(&field))
        .map(|file| file.path.clone())
        .collect())
}

async fn verify(
    dataset: &Dataset,
    original: &RecordBatch,
    stride: usize,
    round: usize,
) -> Result<()> {
    let original_bodies = original
        .column(1)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap();
    let mut scan = dataset.scan();
    scan.batch_size(8);
    let mut stream = scan.try_into_stream().await?;
    let mut seen = BTreeSet::new();
    while let Some(batch) = stream.try_next().await? {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bodies = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        let statuses = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let id: usize = ids.value(row).parse()?;
            assert!(
                id < original.num_rows() && seen.insert(id),
                "duplicate/foreign turn"
            );
            assert_eq!(bodies.value(row), original_bodies.value(id));
            let updated = round > 0 && id.is_multiple_of(stride);
            assert_eq!(
                statuses.value(row),
                if updated { "reviewed" } else { "pending" }
            );
            let expected = if updated { score(round) } else { None };
            assert_eq!((!scores.is_null(row)).then(|| scores.value(row)), expected);
        }
    }
    assert_eq!(seen.len(), original.num_rows());
    Ok(())
}

async fn run(args: Args) -> Result<()> {
    if args.rows == 0 || args.stride == 0 || args.rounds == 0 || args.body_bytes == 0 {
        return Err("rows, stride, rounds and body bytes must be positive".into());
    }
    if args
        .rows
        .checked_mul(args.body_bytes)
        .is_none_or(|n| n > 512 << 20)
    {
        return Err("fixture body buffers must fit within 512 MiB".into());
    }
    let temp = tempfile::tempdir()?;
    let root = args.output.as_deref().unwrap_or(temp.path());
    if args.output.is_some() {
        std::fs::create_dir(root)?;
    }
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::set_global_recorder(recorder)?;
    let original = fixture(args.rows, args.body_bytes)?;
    let mut results = Vec::new();
    for (name, mode) in [
        ("auto", MergeInsertWriteMode::Auto),
        ("rewrite_rows", MergeInsertWriteMode::RewriteRows),
        ("rewrite_columns", MergeInsertWriteMode::RewriteColumns),
    ] {
        let path = root.join(name);
        let uri = path.to_str().ok_or("non-UTF8 path")?;
        let reader = RecordBatchIterator::new(vec![Ok(original.clone())], original.schema());
        let mut current = Arc::new(
            Dataset::write(
                reader,
                uri,
                Some(WriteParams {
                    max_rows_per_file: 16,
                    max_rows_per_group: 16,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await?,
        );
        let original_body_files = body_files(&current)?;
        for round in 1..=args.rounds {
            let source = annotation_batch(&original, args.stride, round)?;
            let old = current.clone();
            let before = io(&snapshotter);
            let started = Instant::now();
            let prepared = MergeInsertBuilder::try_new(current.clone(), vec!["id".into()])?
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .write_mode(mode)
                .conflict_retries(0)
                .skip_auto_cleanup(true)
                .try_build()?
                .execute_uncommitted_batches(vec![source])
                .await?;
            let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
            let after_prepare = io(&snapshotter);
            assert_eq!(
                prepared.stats.num_updated_rows as usize,
                args.rows.div_ceil(args.stride)
            );
            assert_eq!(prepared.stats.num_inserted_rows, 0);
            assert_eq!(prepared.stats.num_deleted_rows, 0);
            // Writing data files alone must not change the current manifest.
            let before_publish = Dataset::open(uri).await?;
            assert_eq!(before_publish.version_id(), old.version_id());
            verify(&before_publish, &original, args.stride, round - 1).await?;
            let before_commit = io(&snapshotter);
            let started = Instant::now();
            let mut commit = CommitBuilder::new(current.clone())
                .with_max_retries(0)
                .with_skip_auto_cleanup(true);
            if let Some(affected) = prepared.affected_rows {
                commit = commit.with_affected_rows(affected);
            }
            current = Arc::new(commit.execute(prepared.transaction).await?);
            let commit_ms = started.elapsed().as_secs_f64() * 1000.0;
            let after_commit = io(&snapshotter);
            assert_eq!(current.version_id(), old.version_id() + 1);
            // An already captured reader keeps the old annotation and body.
            verify(&old, &original, args.stride, round - 1).await?;
            let reopened = Dataset::open(uri).await?;
            verify(&reopened, &original, args.stride, round).await?;
            let files = body_files(&current)?;
            if mode == MergeInsertWriteMode::RewriteColumns {
                assert_eq!(
                    files, original_body_files,
                    "narrow update rewrote body files"
                );
            }
            results.push(json!({"mode": name, "round": round,
                "updated_rows": prepared.stats.num_updated_rows,
                "prepare_ms": prepare_ms, "commit_ms": commit_ms,
                "prepare_io": io_delta(before, after_prepare),
                "commit_io": io_delta(before_commit, after_commit),
                "data_bytes_written": prepared.stats.bytes_written,
                "data_files_written": prepared.stats.num_files_written,
                "new_body_files": files.difference(&original_body_files).count(),
                "live_fragments": current.manifest().fragments.len(),
                "live_data_files": current.manifest().fragments.iter().map(|f| f.files.len()).sum::<usize>(),
                "version": current.version_id()}));
        }
    }
    let report = json!({"rows": args.rows, "body_bytes_per_row": args.body_bytes,
        "stride": args.stride, "rounds": args.rounds, "debug": cfg!(debug_assertions),
        "results": results,
        "verified": "one row per turn, all body bytes and current annotations, explicit null clearing, unpublished preparation, old snapshots, reopen, unchanged body files for RewriteColumns",
        "scope": "Local native Lance layout/publication experiment. Sequential isolated arms with warm OS caches. I/O counters include metadata and measure requested storage bytes, not device traffic. Verification I/O is outside the measured phases. No distributed staging, WAL ACK/recovery, cloud capacity, or end-to-end zero-copy claim."});
    let bytes = serde_json::to_vec_pretty(&report)?;
    if args.output.is_some() {
        std::fs::write(root.join("result.json"), &bytes)?;
    }
    println!("{}", String::from_utf8(bytes)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?
        .block_on(run(args))
}
