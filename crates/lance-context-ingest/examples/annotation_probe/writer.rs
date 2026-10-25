use super::*;
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    ArrayRef, UInt8Array,
};
use arrow_ipc::writer::FileWriter;
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::Bytes;
use lance_context_api::annotation::TurnAnnotation;
use lance_context_ingest::{
    annotations::{wire::schema, MutationResult},
    http::{self, annotations::PointRequest, Opened},
    ProducerOpen,
};

struct Hist {
    count: u64,
    milliseconds: Vec<u64>,
}
impl Hist {
    fn new() -> Self {
        Self {
            count: 0,
            milliseconds: vec![0; 60_001],
        }
    }
    fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.milliseconds[duration.as_micros().div_ceil(1000).min(60_000) as usize] += 1;
    }
    fn p99(&self) -> usize {
        let wanted = (self.count * 99).div_ceil(100);
        let mut count = 0;
        self.milliseconds
            .iter()
            .position(|n| {
                count += n;
                count >= wanted
            })
            .unwrap_or(60_000)
    }
}
fn body(config: &Config, ids: &[String], value: &str) -> Result<Bytes> {
    let schema = schema(&config.annotations.as_ref().unwrap().staging);
    let empty = || {
        let mut builder = ListBuilder::new(StringBuilder::new()).with_field(Arc::new(
            arrow_schema::Field::new("item", arrow_schema::DataType::Utf8, false),
        ));
        for _ in ids {
            builder.append(true);
        }
        Arc::new(builder.finish()) as ArrayRef
    };
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids.to_vec())),
            empty(),
            empty(),
            Arc::new(UInt8Array::from(vec![1; ids.len()])),
            Arc::new(StringArray::from(vec![Some(value); ids.len()])),
            Arc::new(UInt8Array::from(vec![0; ids.len()])),
            Arc::new(Float32Array::from(vec![None; ids.len()])),
        ],
    )?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(Bytes::from(bytes))
}
fn token(config: &Config) -> Result<String> {
    Ok(fs::read_to_string(
        &config
            .http
            .as_ref()
            .context("HTTP configuration required")?
            .bearer_token_file,
    )?
    .trim()
    .into())
}
fn post(
    client: &reqwest::Client,
    config: &Config,
    base: &str,
    token: &str,
    route: &str,
) -> reqwest::RequestBuilder {
    client
        .post(format!(
            "{}/api/v1/annotations/{}/{route}",
            base.trim_end_matches('/'),
            config.tables[0].binding.qualified_table
        ))
        .bearer_auth(token)
}
async fn read(
    client: &reqwest::Client,
    config: &Config,
    base: &str,
    token: &str,
    id: String,
) -> Result<TurnAnnotation> {
    Ok(post(client, config, base, token, "read")
        .json(&PointRequest { turn: id })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
async fn writer(
    config: &Config,
    w: &Workload,
    which: usize,
    client: &reqwest::Client,
    token: &str,
    until: Instant,
) -> Result<u64> {
    let directory = w.directory.join(format!("writer-{}", w.slots[which]));
    fs::create_dir(&directory).context(
        "writer directory already exists; reconcile original requests before another run",
    )?;
    let operation = ProducerOpen {
        bucket: 0,
        slot: w.slots[which],
        expected_generation: 0,
        operation: uuid::Uuid::new_v4(),
    };
    durable_json(&directory.join("open.json"), &operation)?;
    let producer: Opened = post(client, config, &w.ingresses[which], token, "producer")
        .json(&operation)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    durable_json(&directory.join("producer.json"), &producer)?;
    let mut journal = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(w.journal(which))?;
    journal.sync_all()?;
    fs::File::open(&directory)?.sync_all()?;
    let mut acks = Hist::new();
    let mut reads = Hist::new();
    let mut refused = 0u64;
    let mut acked_bytes = 0u64;
    let mut phases = [Duration::ZERO; 5];
    let started = Instant::now();
    let outcome = async {
        let mut sequence = 1u64;
        while Instant::now() < until && w.max_batches.is_none_or(|n| sequence <= n) {
            let encode_started = Instant::now();
            let first = which * (w.turns / 2) + ((sequence - 1) as usize * w.batch_rows) % (w.turns / 2);
            let ids = (first..first + w.batch_rows).map(|n| format!("t{n:08}")).collect::<Vec<_>>();
            let value = format!("{}:{}:{sequence}", w.phase, w.slots[which]);
            let body = body(config, &ids, &value)?;
            phases[0] += encode_started.elapsed();
            let begin = Instant::now();
            durable_json(&directory.join("pending.json"), &serde_json::json!({
                "producer":producer.producer,"sequence":sequence,"body_base64":STANDARD.encode(&body)
            }))?;
            phases[1] += begin.elapsed();
            let request_started = Instant::now();
            let reply: MutationReply = loop {
                let response = post(client, config, &w.ingresses[which], token, "update")
                    .header(http::PRODUCER, &producer.producer).header(http::SEQUENCE, sequence)
                    .header(reqwest::header::CONTENT_TYPE, http::ARROW_FILE).body(body.clone()).send().await?;
                if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                    let refusal: serde_json::Value = response.json().await?;
                    ensure!(refusal["operation_outcome"] == "not_submitted", "unknown mutation outcome; retain pending request: {refusal}");
                    refused += 1;
                    ensure!(begin.elapsed() < Duration::from_secs(30), "write admission stalled");
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    continue;
                }
                break response.error_for_status()?.json().await?;
            };
            phases[2] += request_started.elapsed();
            ensure!(reply.completion.result == MutationResult::Applied, "mutation was rejected");
            let journal_started = Instant::now();
            let ack = Ack { first, rows: w.batch_rows, status: value.clone(), index: reply.completion.index };
            serde_json::to_writer(&mut journal, &ack)?;
            journal.write_all(b"\n")?;
            journal.sync_data()?;
            phases[3] += journal_started.elapsed();
            acks.record(begin.elapsed());
            acked_bytes += body.len() as u64;
            for row in [0, w.batch_rows - 1].into_iter().take(if w.batch_rows == 1 { 1 } else { 2 }) {
                let begin = Instant::now();
                let value_read = read(client, config, &w.ingresses[1 - which], token, ids[row].clone()).await?;
                check_value(&value_read, &value)?;
                reads.record(begin.elapsed());
                phases[4] += begin.elapsed();
            }
            sequence += 1;
        }
        ensure!(acks.count > 0, "writer made no progress");
        Ok::<_, anyhow::Error>(())
    }.await;
    println!(
        "{}",
        serde_json::json!({"writer":w.slots[which],"acks":acks.count,
        "updated_rows":acks.count * w.batch_rows as u64,"ack_p99_ms_upper_bound":acks.p99(),
        "reads":reads.count,"read_p99_ms_upper_bound":reads.p99(),"refused":refused,
        "acked_arrow_bytes":acked_bytes,"elapsed_ms":started.elapsed().as_millis(),
        "client_phase_us": {"encode":phases[0].as_micros(),"pending_journal":phases[1].as_micros(),
        "update_request":phases[2].as_micros(),"ack_journal":phases[3].as_micros(),"point_reads":phases[4].as_micros()},
        "error":outcome.as_ref().err().map(|e| format!("{e:#}"))})
    );
    outcome?;
    Ok(acks.count * w.batch_rows as u64)
}
pub(super) async fn run(config: &Config, w: &Workload) -> Result<()> {
    fs::create_dir(&w.directory)
        .context("run directory already exists; do not launch a duplicate workload")?;
    durable_json(
        &w.directory.join("run.json"),
        &serde_json::json!({"phase":w.phase,"slots":w.slots,
        "turns":w.turns,"batch_rows":w.batch_rows,"seconds":w.seconds}),
    )?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let token = token(config)?;
    let started = Instant::now();
    let until = started + Duration::from_secs(w.seconds);
    let spawn_writer = |which| {
        let config = config.clone();
        let workload = w.clone();
        let client = client.clone();
        let token = token.clone();
        tokio::spawn(async move { writer(&config, &workload, which, &client, &token, until).await })
    };
    let (a, b) = tokio::join!(spawn_writer(0), spawn_writer(1));
    let rows = a?? + b??;
    println!(
        "{}",
        serde_json::json!({"updated_rows":rows,"elapsed_ms":started.elapsed().as_millis(),
        "updates_per_second":rows as f64 / started.elapsed().as_secs_f64(),
        "durable_client_pending_and_ack_journal":true})
    );
    Ok(())
}
pub(super) async fn read_expected(config: &Config, w: &Workload) -> Result<()> {
    use futures::{stream, StreamExt};
    let (values, _) = expected(w)?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let token = token(config)?;
    let rows = (0..w.turns).step_by(w.turns / 32).collect::<Vec<_>>();
    let mut pending = stream::iter(rows.iter().map(|&id| {
        let client = &client;
        let token = &token;
        let values = &values;
        async move {
            for base in &w.ingresses {
                let actual = read(client, config, base, token, format!("t{id:08}")).await?;
                ensure!(
                    actual.status == values[id]
                        && actual.score == Some(0.5)
                        && actual.labels.is_empty(),
                    "recovered point value mismatch"
                );
            }
            Ok::<_, anyhow::Error>(())
        }
    }))
    .buffer_unordered(2);
    while let Some(row) = pending.next().await {
        row?;
    }
    println!(
        "{}",
        serde_json::json!({"verified_point_reads":rows.len() * 2})
    );
    Ok(())
}
