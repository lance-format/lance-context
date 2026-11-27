use super::input::{Capture, Config, Record};
use anyhow::{bail, ensure, Result};
use lance::{dataset::builder::DatasetBuilder, Dataset};
use lance_context_core::MergeMemoryBudget;
use lance_context_ingest::{materializer::Storage, turn_ingestion::TurnLookupParallelism};
use lance_context_wal::replication::publication::lance::PinnedManifest;
use serde_json::{json, Value};
use std::{
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};

pub struct Options {
    pub key_workspace: Arc<MergeMemoryBudget>,
    pub rounds: usize,
    pub timeout: Duration,
}

pub fn emit(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

/// Reusable example entry. Preserve the caller's Storage, session, credential
/// accessors and wrappers. The independent CLI alone installs its experimental
/// pool before constructing Storage. This routine never replaces a pool.
pub async fn open(
    config: &Config,
    capture: &Capture,
    record: &Record,
    storage: &Storage,
) -> Result<Arc<Dataset>> {
    record.validate(config, capture)?;
    ensure!(
        storage.uri == config.uri && storage.session.native_memory().is_some(),
        "authorized storage/native pool required"
    );
    let handler = PinnedManifest::new(
        storage.base.clone(),
        record.cluster,
        record.layout,
        record.table,
        record.incarnation,
        &record.manifest,
    )?;
    let base = Arc::new(
        DatasetBuilder::from_uri(&storage.uri)
            .with_read_params(lance::dataset::ReadParams {
                store_options: Some(storage.params.clone()),
                ..Default::default()
            })
            .with_session(storage.session.clone())
            .with_commit_handler(Arc::new(handler))
            .with_version(capture.base_version)
            .load()
            .await?,
    );
    validate_base(capture, &base)?;
    Ok(base)
}

fn validate_base(capture: &Capture, base: &Dataset) -> Result<()> {
    ensure!(
        base.version_id() == capture.base_version
            && base
                .fragments()
                .iter()
                .map(|f| f.id)
                .eq(capture.base_fragment_ids.iter().copied()),
        "snapshot version or ordered fragment inventory differs"
    );
    Ok(())
}

/// Queries are sequential; caches persist. Outstanding cancelled I/O can finish
/// later. Store snapshots and host-pool counters are observations, not exclusive
/// per-query network or CPU accounting. Never emits keys, paths, or raw errors.
pub async fn compare(
    config: &Config,
    capture: &Capture,
    base: Arc<Dataset>,
    options: Options,
    out: &mut impl Write,
    native_snapshot: impl Fn() -> Value,
) -> Result<()> {
    config.validate_capture(capture)?;
    validate_base(capture, &base)?;
    ensure!(
        (1..=10).contains(&options.rounds)
            && (Duration::from_secs(1)..=Duration::from_secs(600)).contains(&options.timeout),
        "invalid probe bounds"
    );
    let store = base.object_store(None).await?;
    let keys: Vec<_> = capture.keys.iter().map(String::as_str).collect();
    let mut expected = None;
    // Include ordinary production lookup as the membership reference. Reorder
    // explicit settings each round, never call this a cold-cache comparison.
    for (ordinal, setting) in schedule(options.rounds).into_iter().enumerate() {
        let io_before = store.io_stats_snapshot();
        let native_before = native_snapshot();
        let start = Instant::now();
        let result = tokio::time::timeout(options.timeout, async {
            match setting {
                None => {
                    config
                        .spec
                        .lookup_existing(base.clone(), &keys, &options.key_workspace)
                        .await
                }
                Some((task_partitions, fragment_readahead)) => {
                    config
                        .spec
                        .lookup_existing_with_parallelism(
                            base.clone(),
                            &keys,
                            &options.key_workspace,
                            TurnLookupParallelism {
                                task_partitions,
                                fragment_readahead,
                            },
                        )
                        .await
                }
            }
        })
        .await;
        let elapsed_us = start.elapsed().as_micros();
        let io_after = store.io_stats_snapshot();
        let mut report = json!({
            "ordinal":ordinal, "requested_parallelism":setting, "elapsed_us":elapsed_us,
            "native_before":native_before, "native_at_completion":native_snapshot(),
            "store_read_iops_delta":io_after.read_iops.saturating_sub(io_before.read_iops),
            "store_read_bytes_delta":io_after.read_bytes.saturating_sub(io_before.read_bytes),
            "range_overlap":"not_instrumented", "network_bytes":"not_measured"
        });
        let lookup = match result {
            Ok(Ok(Some(value))) => value,
            other => {
                report["status"] = json!(match other {
                    Err(_) => "deadline",
                    Ok(Ok(None)) => "key_workspace_refused",
                    Ok(Err(ref error))
                        if error.chain().any(|e| e
                            .downcast_ref::<lance::Error>()
                            .is_some_and(lance::Error::is_backpressure)) =>
                        "native_backpressure",
                    _ => "lookup_error",
                });
                emit(out, &report)?;
                bail!("probe stopped after unsuccessful lookup; no miss or performance conclusion");
            }
        };
        ensure!(lookup.matches(&base), "lookup snapshot changed");
        let membership: Vec<_> = keys.iter().map(|key| lookup.get(key)).collect();
        let complete = membership.iter().all(Option::is_some);
        let equal = complete && expected.as_ref().is_none_or(|old| *old == membership);
        let o = lookup.observation();
        report["status"] = json!(if equal {
            "equal"
        } else {
            "membership_mismatch"
        });
        report["observation"] = json!({
            "requested_keys":o.requested_keys, "history_hits":o.history_hits, "history_misses":o.history_misses,
            "scans_started":o.scans_started, "projected_rows":o.projected_rows, "projected_array_bytes":o.projected_array_bytes,
            "elapsed_us":o.elapsed_us, "first_result_us":o.first_result_us, "scan_partitions":o.scan_partitions,
            "fragment_streams_peak":o.fragment_streams_peak, "lance_bytes_read":o.lance_bytes_read,
            "lance_requests":o.lance_requests, "lance_indices_loaded":o.lance_indices_loaded,
            "fragments_scanned":o.fragments_scanned, "ranges_scanned":o.ranges_scanned,
            "rows_scanned":o.rows_scanned, "task_wait_time_ns":o.task_wait_time_ns
        });
        drop(lookup); // Includes MiB-granular key admission; never hold across queries.
        emit(out, &report)?;
        ensure!(equal, "membership mismatch");
        if expected.is_none() {
            expected = Some(membership);
        }
        tokio::task::yield_now().await;
    }
    emit(
        out,
        &json!({"complete":true, "membership_equal":true, "queries":1 + 5 * options.rounds}),
    )?;
    Ok(())
}

fn schedule(rounds: usize) -> Vec<Option<(usize, usize)>> {
    let settings = [(8, 8), (16, 8), (32, 8), (8, 16), (8, 32)];
    let mut result = vec![None];
    for round in 0..rounds {
        let mut order = settings.to_vec();
        order.rotate_left(round % settings.len());
        if round % 2 == 1 {
            order.reverse();
        }
        result.extend(order.into_iter().map(Some));
    }
    result
}
