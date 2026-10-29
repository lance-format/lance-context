use super::*;
use tokio::time::Instant;

pub(super) struct Samples {
    pub(super) count: u64,
    milliseconds: Vec<u64>,
}
impl Samples {
    pub(super) fn new() -> Self {
        Self {
            count: 0,
            milliseconds: vec![0; 60_001],
        }
    }
    pub(super) fn record(&mut self, elapsed: Duration) {
        self.count += 1;
        let bucket = elapsed.as_micros().div_ceil(1000).min(60_000) as usize;
        self.milliseconds[bucket] += 1;
    }
    pub(super) fn p99_ms_upper_bound(&self) -> usize {
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

struct Load {
    acknowledgments: Samples,
    reads: Samples,
    refused: u64,
    last_index: u64,
}

async fn writer(
    client: &reqwest::Client,
    write_to: &Server,
    read_from: &Server,
    limits: &Limits,
    turn: &str,
    slot: u32,
    until: Instant,
) -> Result<Load> {
    let producer = open(client, write_to, slot).await?;
    let mut load = Load {
        acknowledgments: Samples::new(),
        reads: Samples::new(),
        refused: 0,
        last_index: 0,
    };
    let mut sequence = 1;
    while Instant::now() < until {
        let value = sequence.to_string();
        let body = super::super::annotations::wire(limits, &[turn], 1, Some(&value), 0, None)?;
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
                    "ambiguous mutation during pressure probe: {refusal}"
                );
                load.refused += 1;
                ensure!(
                    started.elapsed() < Duration::from_secs(30),
                    "no write admission during pressure probe"
                );
                // Only a definite pre-submit refusal is resubmitted, preserving
                // the original producer, sequence and bytes. Unknowns fail.
                tokio::time::sleep(Duration::from_millis(1)).await;
                continue;
            }
            let reply: MutationReply = response.error_for_status()?.json().await?;
            ensure!(
                reply.completion.result == annotations::MutationResult::Applied,
                "update rejected"
            );
            load.last_index = reply.completion.index;
            load.acknowledgments.record(started.elapsed());
            break;
        }
        let started = Instant::now();
        let current: TurnAnnotation = post(client, read_from, "read")
            .json(&PointRequest { turn: turn.into() })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        load.reads.record(started.elapsed());
        ensure!(
            current.status.as_deref() == Some(value.as_str()),
            "acknowledged annotation regressed"
        );
        sequence += 1;
    }
    ensure!(load.acknowledgments.count > 0, "writer made no progress");
    Ok(load)
}

/// An opt-in long-running correctness/pressure probe over real RF3 and Lance.
/// The two-turn fixture measures churn/drain, not large-table lookup capacity.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "sustained real RF3/Lance HTTP pressure; run explicitly"]
async fn annotation_http_sustained_pressure_preserves_acked_reads_and_drains() -> Result<()> {
    let seconds = std::env::var("ANNOTATION_PRESSURE_SECONDS")
        .ok()
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(30);
    ensure!((1..=3600).contains(&seconds), "invalid probe duration");
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 3,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(
        &c,
        &storage,
        annotations::Service::contract(&binding(), &limits)?,
    )
    .await?;
    let (service, runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let owner = Owner::new(
        &runtime,
        keys()?,
        http::Options {
            data_requests: 1,
            ..options()
        },
    )?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let started = Instant::now();
    let until = started + Duration::from_secs(seconds);
    let monitor = async {
        let mut max_entries = 0;
        let mut max_bytes = 0;
        let mut pressured_samples = 0;
        let mut last_report = Instant::now();
        while Instant::now() < until {
            let (view, pressured) = service.observation().await;
            let source = service.source_observation();
            ensure!(
                view.retained_entries <= limits.retained_entries
                    && view.retained_bytes <= limits.retained_bytes
                    && source.records <= 2,
                "annotation budget exceeded"
            );
            max_entries = max_entries.max(view.retained_entries);
            max_bytes = max_bytes.max(view.retained_bytes);
            pressured_samples += u64::from(pressured);
            if last_report.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "{}",
                    serde_json::json!({"annotation_pressure_sample": {
                        "elapsed_ms": started.elapsed().as_millis(), "applied": view.applied,
                        "materialized": view.materialized, "source_records": source.records,
                        "retained_entries": view.retained_entries, "retained_bytes": view.retained_bytes,
                        "backpressured": pressured,
                    }})
                );
                last_report = Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok::<_, anyhow::Error>((max_entries, max_bytes, pressured_samples))
    };
    let (one, two, observed) = tokio::join!(
        writer(&client, &a, &b, &limits, "T", 1, until),
        writer(&client, &b, &a, &limits, "U", 2, until),
        monitor,
    );
    eprintln!("annotation WAL admission: {:?}", c.control_admission());
    let loads = [one?, two?];
    let (max_entries, max_bytes, pressured_samples) = observed?;
    let drain_started = Instant::now();
    annotation_runtime::drained(
        &c,
        &service,
        loads.iter().map(|s| s.last_index).max().unwrap(),
    )
    .await?;
    let final_view = service.observation().await.0;
    ensure!(
        final_view.retained_bytes == 0 && service.source_observation().records == 0,
        "final staging did not drain"
    );
    let final_drain_ms = drain_started.elapsed().as_millis();
    for (worker, load) in loads.into_iter().enumerate() {
        eprintln!(
            "{}",
            serde_json::json!({"annotation_pressure_result": {
                "worker": worker, "offered_seconds": seconds, "acks": load.acknowledgments.count,
                "refused": load.refused, "ack_p99_ms_upper_bound": load.acknowledgments.p99_ms_upper_bound(),
                "reads": load.reads.count, "read_p99_ms_upper_bound": load.reads.p99_ms_upper_bound(),
                "max_retained_entries": max_entries, "max_retained_bytes": max_bytes,
                "pressured_samples": pressured_samples, "final_drain_ms": final_drain_ms,
            }})
        );
    }
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
