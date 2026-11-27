//! Read-only fixed-input experiment. Never starts an ingestion owner or writer.
use anyhow::{bail, ensure, Context, Result};
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

/// Only these static labels may leave an error chain: backend/serde errors can
/// contain key literals, credential values or private paths.
#[derive(Debug)]
struct Failure(&'static str, &'static str);
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stage={} category={}", self.0, self.1)
    }
}
impl std::error::Error for Failure {}

fn output(path: &Path) -> Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).context(Failure("report", "create_new"))
}

fn record_output(path: &Path, selected: &Record) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .context(Failure("record_output", "temporary_create"))?;
    serde_json::to_writer(temporary.as_file_mut(), selected)
        .context(Failure("record_output", "encode"))?;
    temporary
        .as_file_mut()
        .write_all(b"\n")
        .context(Failure("record_output", "write"))?;
    temporary
        .as_file()
        .sync_all()
        .context(Failure("record_output", "sync"))?;
    temporary
        .persist_noclobber(path)
        .map_err(|_| Failure("record_output", "persist_new"))?;
    #[cfg(unix)]
    std::fs::File::open(parent)
        .and_then(|file| file.sync_all())
        .context(Failure("record_output", "directory_sync"))?;
    Ok(())
}

async fn record(config: &Config, capture: &Capture) -> Result<Record> {
    config
        .validate_capture(capture)
        .context(Failure("record", "capture_binding"))?;
    let client = config
        .wal
        .client()
        .context(Failure("record", "client_configuration"))?;
    let reply = client
        .request_routed(Request::Query {
            group: capture.owner.group,
            query: Query::PublisherBinding {
                table: config.binding.table,
            },
        })
        .await
        .context(Failure("record", "publisher_rpc"))?;
    let Reply::Query(Outcome::Observed {
        cut,
        value: Value::PublisherBinding(Some(binding)),
    }) = reply.reply
    else {
        bail!(Failure("record", "publisher_unavailable"));
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
        Failure("record", "publisher_binding_mismatch")
    );
    let outcome = client
        .manifest_catalog()
        .read_manifest(config.binding.table, Some(capture.base_version))
        .await
        .context(Failure("record", "manifest_rpc"))?;
    let (cut, manifest) = match outcome {
        Outcome::Observed {
            cut,
            value: Value::Manifest(Some(manifest)),
        } => (cut, manifest),
        Outcome::Observed {
            value: Value::ManifestRetired(_),
            ..
        } => bail!(Failure("record", "manifest_retired")),
        Outcome::Observed {
            value: Value::Manifest(None),
            ..
        } => bail!(Failure("record", "manifest_absent")),
        Outcome::Unknown => bail!(Failure("record", "manifest_unknown")),
        _ => bail!(Failure("record", "unexpected_manifest_reply")),
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
    record
        .validate(config, capture)
        .context(Failure("record", "selected_binding"))?;
    Ok(record)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("body_lookup_probe failed; no ingestion or publication requests submitted");
        let safe: Vec<_> = error
            .chain()
            .filter_map(|cause| cause.downcast_ref::<Failure>())
            .collect();
        if safe.is_empty() {
            eprintln!("stage=probe category=internal");
        }
        for failure in safe {
            eprintln!("{failure}");
        }
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !((args.len() == 4 && args[0] == "record") || (args.len() == 9 && args[0] == "run")) {
        eprintln!("usage: body_lookup_probe record CONFIG CAPTURE NEW_RECORD_JSON\n       body_lookup_probe run CONFIG CAPTURE RECORD NEW_REPORT_JSONL NATIVE_MIB KEY_MIB ROUNDS TIMEOUT_SECONDS");
        bail!(Failure("arguments", "usage"));
    }
    let config: Config = input::read(Path::new(&args[1])).context(Failure("config", "read"))?;
    let capture: Capture = input::read(Path::new(&args[2])).context(Failure("capture", "read"))?;
    if args[0] == "record" {
        let destination = Path::new(&args[3]);
        // Selection failure creates no destination; persistence still uses an
        // atomic create-new operation if another writer races this preflight.
        match std::fs::symlink_metadata(destination) {
            Ok(_) => bail!(Failure("record_output", "already_exists")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => bail!(Failure("record_output", "metadata")),
        }
        let selected = record(&config, &capture).await?;
        record_output(destination, &selected)?;
        println!("saved exact quorum-selected manifest; this does not pin physical files");
    } else {
        let record: Record =
            input::read(Path::new(&args[3])).context(Failure("saved_record", "read"))?;
        record
            .validate(&config, &capture)
            .context(Failure("saved_record", "binding"))?;
        let native_mib: usize = args[5]
            .parse()
            .context(Failure("arguments", "native_mib"))?;
        let key_mib: usize = args[6].parse().context(Failure("arguments", "key_mib"))?;
        let rounds = args[7].parse().context(Failure("arguments", "rounds"))?;
        let seconds = args[8].parse().context(Failure("arguments", "seconds"))?;
        ensure!(
            native_mib > 0 && key_mib > 0 && rounds > 0 && seconds > 0,
            Failure("arguments", "zero_budget_rounds_or_deadline")
        );
        let native_bytes = native_mib
            .checked_mul(1 << 20)
            .context(Failure("arguments", "native_bytes_overflow"))?;
        let key_bytes = key_mib
            .checked_mul(1 << 20)
            .context(Failure("arguments", "key_bytes_overflow"))?;
        let mut out = output(Path::new(&args[4]))?;
        probe::emit(
            &mut out,
            &serde_json::json!({
                "kind":"standalone_fixed_capture_lookup", "capture_id":capture.capture_id,
                "architecture":std::env::consts::ARCH, "debug_assertions":cfg!(debug_assertions),
                "runtime_workers":tokio::runtime::Handle::current().metrics().num_workers(),
                "base_version":capture.base_version, "fragments":capture.base_fragment_ids.len(),
                "keys":capture.keys.len(), "speculative_input":capture.window.speculative,
                "native_pool":"independent_bounded_experiment_not_production_shared_pool",
                "native_limit_bytes":native_bytes, "key_workspace_bytes":key_bytes,
                "rounds":rounds, "cache":"one_session_rotated_reversed_order_not_cold_cache",
                "limitations":"not online ACK throughput; outstanding I/O may finish after lookup; native peaks cumulative; logical/store reads are not cloud network bytes; pool excludes JSON/plan/cache overhead and is not RSS"
            }),
        ).context(Failure("report", "header_write"))?;
        let pool = Arc::new(pool::Pool::new(native_bytes));
        let session = Arc::new(lance::session::Session::default().with_native_memory(pool.clone()));
        let storage = lance_context_ingest::materializer::Storage::resolve(
            config.uri.clone(),
            config.storage_options.clone(),
            session,
        )
        .await
        .context(Failure("storage", "resolve"))?;
        let base = probe::open(&config, &capture, &record, &storage)
            .await
            .context(Failure("snapshot", "open_or_binding"))?;
        probe::compare(
            &config,
            &capture,
            base,
            probe::Options {
                key_workspace: lance_context_core::MergeMemoryBudget::new(key_bytes),
                rounds,
                timeout: Duration::from_secs(seconds),
            },
            &mut out,
            || serde_json::to_value(pool.snapshot()).unwrap(),
        )
        .await
        .context(Failure("comparison", "failed_see_report"))?;
        out.sync_all().context(Failure("report", "sync"))?;
        println!("completed fixed-input comparison; see private JSONL report");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
