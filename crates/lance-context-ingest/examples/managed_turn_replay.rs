//! Bounded Arrow-file smoke/replay client. Build inside this repository to use
//! its pinned Lance/Arrow patches. This does not advance external source state.
use anyhow::{bail, ensure, Context, Result};
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    new_null_array, Array, RecordBatch, StringArray, UInt8Array,
};
use arrow_schema::{DataType, Field};
use bytes::Bytes;
use lance_context_core::MergeMemoryBudget;
use lance_context_ingest::{
    annotations::{mixed, wire, MutationResult, Shard},
    http::journal::{Config as JournalConfig, DurableMutationClient},
    turn_ingestion::TurnIngestionSpec,
    ProducerOpen, TableBinding,
};
use lance_context_wal::{client_journal::Settings, placement::Layout};
use serde::Deserialize;
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

#[path = "managed_turn_replay/verify.rs"]
mod verify;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    layout: Layout,
    binding: TableBinding,
    spec: TurnIngestionSpec,
    limits: lance_context_core::annotation_staging::Limits,
    shard: Shard,
    open: ProducerOpen,
    endpoint: String,
    bearer_token_file: PathBuf,
    journal_directory: PathBuf,
}

fn encode(config: &Config, body: &RecordBatch) -> Result<Bytes> {
    let prepared = config
        .spec
        .prepare(body.clone(), config.limits.value_bytes)?;
    for row in 0..body.num_rows() {
        ensure!(
            prepared.bucket(
                row,
                &config.layout,
                &config.binding.tenant,
                &config.binding.qualified_table
            )? == config.shard.bucket,
            "input includes another shard"
        );
    }
    let mut labels = ListBuilder::new(StringBuilder::new()).with_field(Arc::new(Field::new(
        "item",
        DataType::Utf8,
        false,
    )));
    for _ in 0..body.num_rows() {
        labels.append(true);
    }
    let labels = Arc::new(labels.finish());
    let initial = RecordBatch::try_new(
        wire::schema(&config.limits),
        vec![
            Arc::new(StringArray::from(prepared.keys().to_vec())),
            labels.clone(),
            labels,
            Arc::new(UInt8Array::from(vec![0; body.num_rows()])),
            new_null_array(&DataType::Utf8, body.num_rows()),
            Arc::new(UInt8Array::from(vec![0; body.num_rows()])),
            new_null_array(&DataType::Float32, body.num_rows()),
        ],
    )?;
    let batch = config.shard.bind_batch(&mixed::wire::creates_for(
        &config.spec,
        body,
        &initial,
        &config.limits,
    )?)?;
    let mut bytes = Vec::new();
    let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut bytes, &batch.schema())?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(bytes.into())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("slot") {
        ensure!(
            args.len() == 4,
            "usage: managed_turn_replay slot OWNER_CONFIG GROUP SLOT"
        );
        let owner =
            lance_context_ingest::application::Config::read(std::path::Path::new(&args[1]))?;
        let group = args[2].parse::<u64>()?;
        let slot = args[3].parse::<u32>()?;
        ensure!(
            owner.layout.groups.contains_key(&group) && slot < owner.limits.producer_slots,
            "unknown group or slot"
        );
        use lance_context_wal::replication::{
            query,
            runtime::client::{Reply, Request},
        };
        let reply = owner
            .client()?
            .request_routed(Request::Query {
                group,
                query: query::Query::ProducerSlot { slot },
            })
            .await?;
        let Reply::Query(query::Outcome::Observed {
            cut,
            value: query::Value::ProducerSlot(Some(view)),
        }) = reply.reply
        else {
            bail!("producer slot quorum query unavailable");
        };
        ensure!(
            cut.cluster == owner.layout.cluster
                && cut.layout == owner.layout.fingerprint()?
                && cut.group == group
                && cut.term > 0
                && cut.applied >= cut.quorum_index,
            "foreign producer slot observation"
        );
        println!("{}", serde_json::json!({"cut":cut, "slot":view}));
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("verify") {
        ensure!(
            args.len() == 5,
            "usage: managed_turn_replay verify CONFIG ARROW_FILE REPLAY_LOG OWNER_CONFIG"
        );
        let config: Config = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        return verify::run(&config, &args[2], &args[3], &args[4]).await;
    }
    if args.first().map(String::as_str) == Some("spec") {
        ensure!(
            args.len() >= 4,
            "usage: managed_turn_replay spec ARROW_FILE ROUTING_COLUMN KEY_COLUMN..."
        );
        let reader = arrow_ipc::reader::FileReader::try_new(std::fs::File::open(&args[1])?, None)?;
        let spec = TurnIngestionSpec::new(reader.schema(), args[3..].to_vec(), &args[2])?;
        println!("{}", serde_json::to_string(&spec)?);
        return Ok(());
    }
    ensure!(
        matches!(args.len(), 2 | 3),
        "usage: managed_turn_replay <append|resume> CONFIG [ARROW_FILE]"
    );
    let config_bytes = std::fs::read(&args[1])?;
    ensure!(
        config_bytes.len() <= 1 << 20,
        "oversized replay configuration"
    );
    let config: Config = serde_json::from_slice(&config_bytes)?;
    ensure!(
        config.binding.max_record_bytes <= 8 << 20 && config.binding.max_rows <= 8192,
        "this bounded replay entry supports at most8MiB/8192rows per request"
    );
    if args[0] == "validate" && args.len() == 3 {
        let file = std::fs::File::open(&args[2])?;
        ensure!(file.metadata()?.len() <= 128 << 20, "input exceeds128MiB");
        let reader = arrow_ipc::reader::FileReader::try_new(file, None)?;
        ensure!(
            reader.schema() == *config.spec.body_schema(),
            "foreign input schema"
        );
        let mut rows = 0usize;
        let mut batches = 0usize;
        let mut invalid_components = 0usize;
        let mut oversized_keys = 0usize;
        let mut max_key_bytes = 0usize;
        for batch in reader {
            let batch = batch?;
            batches += 1;
            rows += batch.num_rows();
            let columns = config
                .spec
                .primary_key()
                .iter()
                .map(|name| {
                    batch
                        .column_by_name(name)
                        .context("missing key column")?
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .context("key is not UTF8")
                })
                .collect::<Result<Vec<_>>>()?;
            for row in 0..batch.num_rows() {
                let values = columns
                    .iter()
                    .map(|a| {
                        if a.is_null(row) || a.value(row).is_empty() {
                            invalid_components += 1;
                        }
                        a.value(row)
                    })
                    .collect::<Vec<_>>();
                let length = serde_json::to_string(&values)?.len();
                max_key_bytes = max_key_bytes.max(length);
                oversized_keys += usize::from(length > config.limits.value_bytes);
            }
        }
        println!(
            "{}",
            serde_json::json!({"event":"validated","rows":rows,"batches":batches,
            "null_or_empty_key_components":invalid_components,"oversized_keys":oversized_keys,
            "max_encoded_key_bytes":max_key_bytes,"key_limit_bytes":config.limits.value_bytes})
        );
        ensure!(
            invalid_components == 0 && oversized_keys == 0,
            "invalid business keys"
        );
        return Ok(());
    }
    let append = match args[0].as_str() {
        "append" if args.len() == 3 => true,
        "resume" if args.len() == 2 => false,
        _ => bail!("invalid mode/arguments"),
    };
    // A replay of the whole input is a distinct, explicitly configured producer
    // execution. Resume resolves only the pending journal intent; it does not
    // invent a source cursor or skip rows based on a producer sequence.
    if append {
        ensure!(
            !config.journal_directory.exists(),
            "append requires a fresh journal directory; use resume for an unresolved intent"
        );
    } else {
        ensure!(
            config.journal_directory.is_dir(),
            "resume requires the original journal"
        );
    }
    ensure!(
        config.open.bucket == config.shard.bucket,
        "foreign producer shard"
    );
    let budget = MergeMemoryBudget::new(1 << 30);
    let token = std::fs::read_to_string(&config.bearer_token_file)?;
    let mut headers = reqwest::header::HeaderMap::new();
    let mut authorization =
        reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token.trim()))?;
    authorization.set_sensitive(true);
    headers.insert(reqwest::header::AUTHORIZATION, authorization);
    let transport = reqwest::Client::builder()
        .default_headers(headers)
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .build()?;
    let client = DurableMutationClient::open(
        JournalConfig {
            directory: config.journal_directory.clone(),
            opens: vec![config.open.clone()],
            base: config.endpoint.parse()?,
            request_timeout: Duration::from_secs(60),
            journal: Settings {
                state_bytes: 64 << 20,
                queue_bytes: 16 << 20,
                batch_bytes: 16 << 20,
                file_bytes: 16 << 20,
                log_bytes: 512 << 20,
                ..Default::default()
            },
        },
        &config.layout,
        config.binding.clone(),
        &config.spec,
        &config.limits,
        config.shard,
        transport,
        budget.clone(),
    )
    .await?;
    let start = Instant::now();
    let mut through = 0;
    let mut rows = 0usize;
    let mut wire_bytes = 0usize;
    let mut latencies = Vec::new();
    if append {
        let file = std::fs::File::open(&args[2])?;
        let length = usize::try_from(file.metadata()?.len())?;
        ensure!(
            length <= 128 << 20,
            "split the input IPC file into files <=128MiB before this bounded smoke test"
        );
        let workspace = length
            .checked_mul(3)
            .and_then(|n| n.checked_add(64 << 20))
            .context("input workspace overflow")?;
        let _input = budget.reserve(workspace).await;
        let reader = arrow_ipc::reader::FileReader::try_new(file, None)?;
        ensure!(
            reader.schema() == *config.spec.body_schema(),
            "foreign input schema"
        );
        for (batch_index, batch) in reader.enumerate() {
            let batch = batch?;
            let mut offset = 0;
            while offset < batch.num_rows() {
                let mut count = (batch.num_rows() - offset).min(config.binding.max_rows as usize);
                let bytes = loop {
                    let bytes = encode(&config, &batch.slice(offset, count))?;
                    if bytes.len() <= config.binding.max_record_bytes as usize {
                        break bytes;
                    }
                    ensure!(
                        count > 1,
                        "one body exceeds the configured WAL record limit"
                    );
                    count = count.div_ceil(2);
                };
                let length = bytes.len();
                let sent = Instant::now();
                let reply = client.append(0, bytes).await?;
                ensure!(
                    matches!(reply.completion.result, MutationResult::Applied),
                    "committed mutation rejected: {:?}",
                    reply.completion.result
                );
                let elapsed = sent.elapsed().as_secs_f64();
                through = reply.completion.index;
                rows += count;
                wire_bytes += length;
                latencies.push(elapsed);
                println!(
                    "{}",
                    serde_json::json!({"event":"applied","input_batch":batch_index,"offset":offset,
                    "rows":count,"bytes":length,"sequence":reply.completion.meta.sequence,
                    "wal_index":through,"seconds":elapsed})
                );
                offset += count;
            }
        }
    } else if let Some(reply) = client.resume(0).await? {
        ensure!(
            matches!(reply.completion.result, MutationResult::Applied),
            "committed mutation rejected: {:?}",
            reply.completion.result
        );
        through = reply.completion.index;
        println!(
            "{}",
            serde_json::json!({"event":"resumed","sequence":reply.completion.meta.sequence,"wal_index":through})
        );
    } else if let Some(reply) = client.last_completed(0).await? {
        ensure!(
            matches!(reply.completion.result, MutationResult::Applied),
            "last completed mutation was rejected: {:?}",
            reply.completion.result
        );
        through = reply.completion.index;
    }
    let ack_seconds = start.elapsed().as_secs_f64();
    let selected = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            let selected = client.selected_body().await?;
            if selected.through >= through {
                return Ok::<_, anyhow::Error>(selected);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .context("publication did not cover the acknowledged cut in120s")??;
    latencies.sort_by(f64::total_cmp);
    let p99 = latencies
        .get(latencies.len().saturating_mul(99) / 100)
        .copied();
    println!(
        "{}",
        serde_json::json!({"event":"selected","candidate_rows":rows,"wire_bytes":wire_bytes,
        "requests":latencies.len(),"ack_seconds":ack_seconds,"published_seconds":start.elapsed().as_secs_f64(),
        "ack_p99_seconds":p99,"selected":selected,"source_checkpoint_advanced":false})
    );
    client.shutdown().await
}
