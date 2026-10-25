//! Opt-in single-host RF3/Lance experiment. Rates include cross-ingress reads.
use super::*;
use crate::{
    annotations::runtime::{Configuration, Runtime},
    bootstrap::annotations::Snapshot,
    materializer::{Acquisition, Options},
};
use arrow_array::{Float32Array, ListArray, RecordBatchIterator};
use futures::TryStreamExt;
use lance::dataset::WriteParams;
use lance_table::io::commit::ConditionalPutCommitHandler;
use tokio::time::Instant;

use super::pressure::Samples;

struct Load {
    acks: Samples,
    reads: Samples,
    refused: u64,
    acked_body_bytes: u64,
    last_index: u64,
    expected: Vec<u64>,
    elapsed: Duration,
}

struct Workload {
    seconds: u64,
    turns: usize,
    batch_rows: usize,
    working_set: usize,
}
impl Workload {
    fn read() -> Result<Self> {
        fn env(name: &str, default: usize) -> Result<usize> {
            Ok(std::env::var(name)
                .ok()
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(default))
        }
        let turns = env("ANNOTATION_LOAD_TURNS", 4096)?;
        let result = Self {
            seconds: env("ANNOTATION_LOAD_SECONDS", 30)? as u64,
            turns,
            batch_rows: env("ANNOTATION_LOAD_BATCH_ROWS", 32)?,
            working_set: env("ANNOTATION_LOAD_WORKING_SET", turns)?,
        };
        ensure!(
            (1..=600).contains(&result.seconds)
                && (256..=65_536).contains(&turns)
                && turns.is_multiple_of(2)
                && (1..=512).contains(&result.batch_rows)
                && (2..=turns).contains(&result.working_set)
                && result.working_set.is_multiple_of(2 * result.batch_rows),
            "invalid annotation batch experiment workload"
        );
        Ok(result)
    }
}

async fn frontend_for(
    c: &network::Cluster,
    binding: &TableBinding,
    limits: &Limits,
    uri: &str,
) -> Result<(Forwarder, Server)> {
    let ingress = annotations::connect_ingress(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding.clone(),
        limits,
        MergeMemoryBudget::new(16 << 20),
    )
    .await?;
    let gateway = Forwarder::new(
        vec![(ingress, uri.into())],
        keys()?,
        options(),
        TOKEN.into(),
    )?;
    let server = Server::start(gateway.router(), None).await?;
    Ok((gateway, server))
}

#[allow(clippy::too_many_arguments)]
async fn writer(
    client: &reqwest::Client,
    write_to: &Server,
    read_from: &Server,
    limits: &Limits,
    ids: &[String],
    workload: &Workload,
    slot: u32,
    until: Instant,
) -> Result<Load> {
    let producer = open(client, write_to, slot).await?;
    let mut load = Load {
        acks: Samples::new(),
        reads: Samples::new(),
        refused: 0,
        acked_body_bytes: 0,
        last_index: 0,
        expected: vec![0; ids.len()],
        elapsed: Duration::ZERO,
    };
    let beginning = Instant::now();
    let result = async {
        let mut sequence = 1u64;
        while Instant::now() < until {
            let start = (sequence as usize - 1) * workload.batch_rows % (workload.working_set / 2);
            let turns = ids[start..start + workload.batch_rows]
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let value = format!("{slot}:{sequence}");
            let body = crate::tests::annotations::wire(limits, &turns, 1, Some(&value), 0, None)?;
            let started = Instant::now();
            loop {
                let response = update(client, write_to, &producer, sequence, body.len())
                    .body(body.clone())
                    .send()
                    .await?;
                if response.status() == StatusCode::SERVICE_UNAVAILABLE {
                    let refusal: serde_json::Value = response.json().await?;
                    ensure!(
                        refusal["operation_outcome"] == "not_submitted",
                        "ambiguous mutation: {refusal}"
                    );
                    load.refused += 1;
                    ensure!(
                        started.elapsed() < Duration::from_secs(30),
                        "write admission stalled"
                    );
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    continue;
                }
                let status = response.status();
                ensure!(
                    status.is_success(),
                    "update failed: {status}: {}",
                    response.text().await?
                );
                let reply: MutationReply = response.json().await?;
                ensure!(
                    reply.completion.result == annotations::MutationResult::Applied,
                    "update rejected"
                );
                load.last_index = reply.completion.index;
                load.acks.record(started.elapsed());
                load.acked_body_bytes += body.len() as u64;
                load.expected[start..start + workload.batch_rows].fill(sequence);
                break;
            }
            // Probe both ends of every acknowledged batch via the other ingress.
            // The complete expected table is checked after drain below.
            for row in [0, workload.batch_rows - 1]
                .into_iter()
                .take(if workload.batch_rows == 1 { 1 } else { 2 })
            {
                let started = Instant::now();
                let response = post(client, read_from, "read")
                    .json(&PointRequest {
                        turn: turns[row].into(),
                    })
                    .send()
                    .await?;
                let status = response.status();
                ensure!(
                    status.is_success(),
                    "point read failed: {status}: {}",
                    response.text().await?
                );
                let current: TurnAnnotation = response.json().await?;
                load.reads.record(started.elapsed());
                ensure!(
                    current.status.as_deref() == Some(value.as_str())
                        && current.score == Some(0.5)
                        && current.labels.is_empty(),
                    "acknowledged annotation changed: {current:?}"
                );
            }
            sequence += 1;
        }
        ensure!(load.acks.count > 0, "writer made no progress");
        Ok::<_, anyhow::Error>(())
    }
    .await;
    load.elapsed = beginning.elapsed();
    eprintln!(
        "{}",
        serde_json::json!({"annotation_batch_writer": {
            "slot": slot, "batch_rows": workload.batch_rows, "working_set": workload.working_set,
            "table_turns": workload.turns, "offered_seconds": workload.seconds,
            "elapsed_ms": load.elapsed.as_millis(), "acks": load.acks.count,
            "updated_rows": load.acks.count * workload.batch_rows as u64, "refused": load.refused,
            "acked_body_bytes": load.acked_body_bytes,
            "distinct_updated_turns": load.expected.iter().filter(|&&s| s != 0).count(),
            "refusal_fraction": load.refused as f64 / (load.acks.count + load.refused).max(1) as f64,
            "ack_p99_ms_upper_bound": load.acks.p99_ms_upper_bound(),
            "reads": load.reads.count, "read_p99_ms_upper_bound": load.reads.p99_ms_upper_bound(),
            "error": result.as_ref().err().map(|e| format!("{e:#}")),
        }})
    );
    result?;
    Ok(load)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "single-host RF3/Lance batch and cold-turn qualification; run explicitly"]
async fn annotation_http_batch_load_preserves_values_and_drains() -> Result<()> {
    let workload = Workload::read()?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let ids = (0..workload.turns)
        .map(|n| format!("t{n:08}"))
        .collect::<Vec<_>>();
    // Real rollout schema and inline bodies. Fixture creation is outside timing.
    let batches = ids
        .chunks(256)
        .map(|ids| {
            let names = ids.iter().map(String::as_str).collect::<Vec<_>>();
            let original = batch(&names, &vec!["session"; ids.len()])?;
            let mut columns = original.columns().to_vec();
            columns[original.schema().index_of("score")?] =
                Arc::new(Float32Array::from(vec![Some(0.5); ids.len()]));
            Ok::<_, anyhow::Error>(RecordBatch::try_new(original.schema(), columns)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let schema = batches[0].schema();
    let base = Dataset::write(
        RecordBatchIterator::new(batches.into_iter().map(Ok), schema),
        &storage.uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            ..Default::default()
        }),
    )
    .await?;
    let mut binding = binding();
    binding.max_rows = 512; // Fixed across batch-size comparisons.
    let limits = Limits {
        retained_bytes: 16 << 20,
        retained_entries: 16_384,
        value_bytes: 4096,
        readers: 32,
    };
    crate::bootstrap::annotations::provision(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding.clone(),
        storage.clone(),
        crate::tests::bootstrap::identity(),
        Snapshot {
            version: base.version_id(),
            transaction: Uuid::parse_str(&base.read_transaction().await?.unwrap().uuid)?,
        },
        limits.clone(),
    )
    .await?;
    let original_fragments = base.manifest().fragments.len();
    drop(base);
    let state = annotation_publication::state(&c).await?;
    let (service, runtime) = Runtime::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding.clone(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(16 << 20),
            apply_budget: MergeMemoryBudget::new(16 << 20),
            publication_budget: MergeMemoryBudget::new(64 << 20),
            publication: Options {
                max_source_bytes: 8 << 20,
                min_source_bytes: 4 << 20,
                max_records: 64,
                max_batch_wait: Duration::from_secs(1),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await?;
    let owner = Owner::new(
        &runtime,
        keys()?,
        http::Options {
            data_requests: 4,
            ..options()
        },
    )?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend_for(&c, &binding, &limits, &server.uri()).await?;
    let (gb, b) = frontend_for(&c, &binding, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    eprintln!(
        "{}",
        serde_json::json!({"annotation_batch_configuration": {
            "processes": 1, "wal_replicas": 3, "owners": 1, "ingresses": 2, "writer_tasks": 2,
            "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "table_turns": workload.turns, "batch_rows": workload.batch_rows, "working_set": workload.working_set,
            "id_index": false, "inline_body_bytes_per_turn": 16 << 10, "staging": limits,
            "source_records": 64, "source_bytes": 8 << 20, "flush_wait_ms": 1000,
            "original_fragments": original_fragments,
        }})
    );
    let started = Instant::now();
    let until = started + Duration::from_secs(workload.seconds);
    let monitor = async {
        let mut maximum = (0, 0, 0);
        let mut last_report = Instant::now();
        while Instant::now() < until {
            let sample_started = Instant::now();
            let (view, pressured) = service.observation().await;
            let source = service.source_observation();
            ensure!(
                view.retained_entries <= limits.retained_entries
                    && view.retained_bytes <= limits.retained_bytes
                    && source.records <= 64,
                "annotation budget exceeded"
            );
            maximum.0 = maximum.0.max(view.retained_entries);
            maximum.1 = maximum.1.max(view.retained_bytes);
            maximum.2 = maximum.2.max(source.records);
            if last_report.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "{}",
                    serde_json::json!({"annotation_batch_sample": {
                        "elapsed_ms": started.elapsed().as_millis(), "applied": view.applied,
                        "materialized": view.materialized, "active_entries": view.active_entries,
                        "frozen_entries": view.frozen_entries, "retained_bytes": view.retained_bytes,
                        "source_records": source.records, "source_bytes": source.bytes,
                        "backpressured": pressured, "observation_wait_us": sample_started.elapsed().as_micros(),
                    }})
                );
                last_report = Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok::<_, anyhow::Error>(maximum)
    };
    let (one, two, observed) = tokio::join!(
        writer(
            &client,
            &a,
            &b,
            &limits,
            &ids[..ids.len() / 2],
            &workload,
            1,
            until
        ),
        writer(
            &client,
            &b,
            &a,
            &limits,
            &ids[ids.len() / 2..],
            &workload,
            2,
            until
        ),
        monitor
    );
    eprintln!(
        "annotation batch WAL admission: {:?}",
        c.control_admission()
    );
    let loads = [one?, two?];
    let maximum = observed?;
    let elapsed = started.elapsed();
    let drain = Instant::now();
    annotation_runtime::drained(&service, loads.iter().map(|s| s.last_index).max().unwrap())
        .await?;
    let drain_ms = drain.elapsed().as_millis();
    let selected = annotation_publication::read(&c, &storage, None).await?;
    let actual = selected
        .scan()
        .project(&["id", "labels", "status", "score"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut seen = vec![false; ids.len()];
    for batch in &actual {
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let labels = batch
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let status = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let score = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let n = id
                .value(row)
                .strip_prefix('t')
                .context("unknown turn")?
                .parse::<usize>()?;
            ensure!(n < ids.len() && !seen[n], "unknown or duplicate turn");
            seen[n] = true;
            let slot = n / (ids.len() / 2);
            let sequence = loads[slot].expected[n % (ids.len() / 2)];
            let expected = (sequence != 0).then(|| format!("{}:{sequence}", slot + 1));
            ensure!(
                status.is_null(row) == expected.is_none()
                    && expected.as_deref().is_none_or(|e| status.value(row) == e)
                    && !score.is_null(row)
                    && score.value(row) == 0.5
                    && (labels.is_null(row) || labels.value_length(row) == 0),
                "final annotation mismatch"
            );
        }
    }
    ensure!(seen.into_iter().all(|v| v), "final table lost rows");
    let rows = loads.iter().map(|l| l.acks.count).sum::<u64>() * workload.batch_rows as u64;
    eprintln!(
        "{}",
        serde_json::json!({"annotation_batch_result": {
            "batch_rows": workload.batch_rows, "working_set": workload.working_set,
            "elapsed_ms": elapsed.as_millis(), "updated_rows": rows,
            "updates_per_second": rows as f64 / elapsed.as_secs_f64(),
            "max_retained_entries": maximum.0, "max_retained_bytes": maximum.1,
            "max_source_records": maximum.2, "final_drain_ms": drain_ms,
            "verified_final_rows": ids.len(), "final_fragments": selected.manifest().fragments.len(),
        }})
    );
    ga.shutdown().await;
    gb.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
