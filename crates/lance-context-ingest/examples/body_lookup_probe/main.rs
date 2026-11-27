//! Read-only fixed-input experiment. Never starts an ingestion owner or writer.
use anyhow::{bail, ensure, Result};
use lance_context_wal::replication::{
    publication::lance::ManifestCatalog,
    query::{Outcome, Query, Value},
    runtime::client::{Reply, Request},
};
use std::{fs::OpenOptions, io::Write, path::Path};

mod input;
mod pool;
mod probe;
use input::{Capture, Config, Record};
use std::{sync::Arc, time::Duration};

fn output(path: &Path) -> Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

async fn record(config: &Config, capture: &Capture) -> Result<Record> {
    config.validate_capture(capture)?;
    let client = config.wal.client()?;
    let reply = client
        .request_routed(Request::Query {
            group: capture.owner.group,
            query: Query::PublisherBinding {
                table: config.binding.table,
            },
        })
        .await?;
    let Reply::Query(Outcome::Observed {
        cut,
        value: Value::PublisherBinding(Some(binding)),
    }) = reply.reply
    else {
        bail!("publisher binding unavailable");
    };
    ensure!(
        cut.cluster == capture.owner.cluster
            && cut.layout == capture.owner.layout
            && cut.group == capture.owner.group
            && binding.authority.table == config.binding.table
            && binding.destination == config.binding.destination
            && binding.incarnation == Some(config.binding.incarnation)
            && binding.ingest
                == lance_context_ingest::annotations::single_body::contract(
                    &config.binding,
                    &config.spec,
                    &config.staging,
                    config.shard
                )?,
        "quorum publisher binding differs"
    );
    let outcome = client
        .manifest_catalog()
        .read_manifest(config.binding.table, Some(capture.base_version))
        .await?;
    let Outcome::Observed {
        cut,
        value: Value::Manifest(Some(manifest)),
    } = outcome
    else {
        bail!("exact manifest unavailable (retired, absent, or unknown); no latest fallback");
    };
    let record = Record {
        capture_id: capture.capture_id,
        cluster: capture.owner.cluster,
        layout: capture.owner.layout,
        table: capture.owner.table,
        incarnation: capture.incarnation,
        schema: capture.schema,
        source_group: capture.source_group,
        bucket: capture.bucket,
        observed: cut,
        manifest: (*manifest).clone(),
    };
    record.validate(config, capture)?;
    Ok(record)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
    if let Err(error) = run().await {
        // Backend errors may include storage credentials, paths or key literals.
        // Keep stderr safe; each successful stage is recorded in the private output.
        let _ = error;
        eprintln!("body_lookup_probe failed; check private inputs, exact snapshot availability and limits; no ingestion or publication requests were submitted");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !((args.len() == 4 && args[0] == "record") || (args.len() == 9 && args[0] == "run")) {
        eprintln!("usage: body_lookup_probe record CONFIG CAPTURE NEW_RECORD_JSON\n       body_lookup_probe run CONFIG CAPTURE RECORD NEW_REPORT_JSONL NATIVE_MIB KEY_MIB ROUNDS TIMEOUT_SECONDS");
        bail!("invalid arguments");
    }
    let config: Config = input::read(Path::new(&args[1]), 2 << 20)?;
    let capture: Capture = input::read(Path::new(&args[2]), 32 << 20)?;
    if args[0] == "record" {
        // Refuse existing outputs before issuing any RPC. Partial output is not valid JSON.
        let mut out = output(Path::new(&args[3]))?;
        let selected = record(&config, &capture).await?;
        serde_json::to_writer(&mut out, &selected)?;
        out.write_all(b"\n")?;
        out.sync_all()?;
        println!("saved exact quorum-selected manifest; this does not pin physical files");
    } else {
        let record: Record = input::read(Path::new(&args[3]), 64 << 10)?;
        record.validate(&config, &capture)?;
        let native_mib: usize = args[5].parse()?;
        let key_mib: usize = args[6].parse()?;
        let rounds = args[7].parse()?;
        let seconds = args[8].parse()?;
        ensure!(
            (1..=16384).contains(&native_mib)
                && (1..=1024).contains(&key_mib)
                && (1..=10).contains(&rounds)
                && (1..=600).contains(&seconds),
            "invalid experimental limits"
        );
        let mut out = output(Path::new(&args[4]))?;
        probe::emit(
            &mut out,
            &serde_json::json!({
                "kind":"standalone_fixed_capture_lookup", "capture_id":capture.capture_id,
                "base_version":capture.base_version, "fragments":capture.base_fragment_ids.len(),
                "keys":capture.keys.len(), "speculative_input":capture.window.speculative,
                "native_pool":"independent_bounded_experiment_not_production_shared_pool",
                "native_limit_bytes":native_mib << 20, "key_workspace_bytes":key_mib << 20,
                "rounds":rounds, "cache":"one_session_rotated_reversed_order_not_cold_cache",
                "limitations":"not online ACK throughput; outstanding I/O may finish after lookup; native peaks cumulative; logical/store reads are not cloud network bytes; pool excludes JSON/plan/cache overhead and is not RSS"
            }),
        )?;
        let pool = Arc::new(pool::Pool::new(native_mib << 20));
        let session = Arc::new(lance::session::Session::default().with_native_memory(pool.clone()));
        let storage = lance_context_ingest::materializer::Storage::resolve(
            config.uri.clone(),
            config.storage_options.clone(),
            session,
        )
        .await?;
        let base = probe::open(&config, &capture, &record, &storage).await?;
        probe::compare(
            &config,
            &capture,
            base,
            probe::Options {
                key_workspace: lance_context_core::MergeMemoryBudget::new(key_mib << 20),
                rounds,
                timeout: Duration::from_secs(seconds),
            },
            &mut out,
            || serde_json::to_value(pool.snapshot()).unwrap(),
        )
        .await?;
        out.sync_all()?;
        println!("completed fixed-input comparison; see private JSONL report");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
