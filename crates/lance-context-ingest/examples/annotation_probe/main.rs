//! Drive deployed annotation roles using the real HTTP, WAL and Lance protocols.
use anyhow::{bail, ensure, Context, Result};
use arrow_array::{Array, Float32Array, ListArray, RecordBatch, StringArray};
use lance::Dataset;
use lance_context_ingest::{application::Config, http::annotations::MutationReply};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::time::Instant;

mod storage;
mod writer;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    turns: usize,
    batch_rows: usize,
    seconds: u64,
    max_batches: Option<u64>,
    phase: String,
    slots: [u32; 2],
    ingresses: [String; 2],
    directory: PathBuf,
    #[serde(default)]
    prior_journals: Vec<PathBuf>,
}
impl Workload {
    fn validate(&self, config: &Config) -> Result<()> {
        ensure!(
            config.annotations.is_some(),
            "annotation configuration required"
        );
        ensure!(
            (256..=65_536).contains(&self.turns)
                && (1..=512).contains(&self.batch_rows)
                && self.turns.is_multiple_of(2 * self.batch_rows)
                && (1..=600).contains(&self.seconds)
                && self.max_batches.is_none_or(|v| v > 0)
                && !self.phase.is_empty()
                && self.phase.len() <= 64
                && self
                    .phase
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
                && self.slots[0] != self.slots[1]
                && self.slots.iter().all(|s| *s < config.limits.producer_slots)
                && self.prior_journals.len() <= 16,
            "invalid bounded workload"
        );
        ensure!(
            config.tables[0]
                .binding
                .qualified_table
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
            "invalid route name"
        );
        for url in &self.ingresses {
            let url = reqwest::Url::parse(url)?;
            ensure!(
                matches!(url.scheme(), "http" | "https"),
                "invalid ingress URL"
            );
        }
        Ok(())
    }
    fn journal(&self, slot: usize) -> PathBuf {
        self.directory
            .join(format!("writer-{}", self.slots[slot]))
            .join("acks.jsonl")
    }
}
#[derive(Serialize, Deserialize)]
struct Ack {
    first: usize,
    rows: usize,
    status: String,
    index: u64,
}
fn durable_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("pending");
    let mut file = fs::File::create(&temporary)?;
    serde_json::to_writer(&mut file, value)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    fs::File::open(path.parent().context("missing output parent")?)?.sync_all()?;
    Ok(())
}
fn expected(w: &Workload) -> Result<(Vec<Option<String>>, u64)> {
    let mut result = vec![None; w.turns];
    let mut through = 0;
    for file in w
        .prior_journals
        .iter()
        .cloned()
        .chain([w.journal(0), w.journal(1)])
    {
        let mut reader = std::io::BufReader::new(fs::File::open(file)?);
        loop {
            let mut line = String::new();
            let n = std::io::Read::take(&mut reader, 4097).read_line(&mut line)?;
            if n == 0 {
                break;
            }
            ensure!(
                n <= 4096 && line.ends_with('\n'),
                "incomplete/oversized ACK journal row"
            );
            let ack: Ack = serde_json::from_str(&line)?;
            ensure!(
                ack.rows > 0
                    && ack
                        .first
                        .checked_add(ack.rows)
                        .is_some_and(|n| n <= w.turns)
                    && ack.status.len() <= 128
                    && ack.index > 0,
                "invalid ACK journal row"
            );
            result[ack.first..ack.first + ack.rows].fill(Some(ack.status));
            through = through.max(ack.index);
        }
    }
    Ok((result, through))
}
fn check_value(
    value: &lance_context_api::annotation::TurnAnnotation,
    expected: &str,
) -> Result<()> {
    ensure!(
        value.status.as_deref() == Some(expected)
            && value.score == Some(0.5)
            && value.labels.is_empty(),
        "acknowledged annotation mismatch"
    );
    Ok(())
}
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: annotation_probe <seed|load|verify|read> CONFIG WORKLOAD"
    );
    let config = Config::read(Path::new(&args[1]))?;
    let w: Workload = serde_json::from_slice(&fs::read(&args[2])?)?;
    w.validate(&config)?;
    match args[0].as_str() {
        "seed" => storage::seed(&config, &w).await,
        "load" => writer::run(&config, &w).await,
        "verify" => storage::verify(&config, &w).await,
        "read" => writer::read_expected(&config, &w).await,
        _ => bail!("unknown probe operation"),
    }
}
