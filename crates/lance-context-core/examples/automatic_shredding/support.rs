use std::{collections::BTreeSet, future::Future, path::Path, time::Instant};

use lance::Dataset;
use metrics_util::debugging::{DebugValue, Snapshotter};
use serde_json::{json, Value};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn document(row: usize, rows: usize, bytes: usize) -> Value {
    let mut body = String::with_capacity(bytes + 16);
    body.push_str(if row == 0 {
        "shredneedle "
    } else {
        "ordinary "
    });
    let mut state = row as u64 + 832;
    while body.len() < bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        use std::fmt::Write;
        write!(&mut body, "word{:04x} ", state % 4096).unwrap();
    }
    body.truncate(bytes);
    let mut value = json!({"body": body, "sequence": row,
        "tags": [if row == rows - 1 { "target" } else { "other" }]});
    if row == rows - 1 {
        value["late_text"] = json!("historicalneedle residual");
    }
    value
}

pub fn fixture(rows: usize, bytes: usize) -> Vec<Value> {
    (0..rows).map(|row| document(row, rows, bytes)).collect()
}

fn io(snapshotter: &Snapshotter) -> (u64, u64) {
    let mut result = (0, 0);
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
                    "get" => result.0 += bytes,
                    "put" | "put_part" => result.1 += bytes,
                    _ => {}
                }
            }
        }
    }
    result
}

pub async fn measure<T>(
    snapshotter: &Snapshotter,
    future: impl Future<Output = Result<T>>,
) -> Result<(T, Value)> {
    let before = io(snapshotter);
    let start = Instant::now();
    let result = future.await?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let after = io(snapshotter);
    Ok((
        result,
        json!({"ms": elapsed, "read_bytes": after.0 - before.0, "written_bytes": after.1 - before.1}),
    ))
}

pub fn summarize(samples: Vec<Value>) -> Value {
    let mut times: Vec<_> = samples.iter().map(|v| v["ms"].as_f64().unwrap()).collect();
    times.sort_by(f64::total_cmp);
    json!({"p50_ms": times[times.len() / 2], "p95_ms": times[(times.len() * 95).div_ceil(100).saturating_sub(1)], "samples": samples})
}

fn directory_bytes(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let mut bytes = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            bytes += directory_bytes(&entry.path())?;
        } else {
            bytes += entry.metadata()?.len();
        }
    }
    Ok(bytes)
}

pub async fn storage(uri: &str) -> Result<Value> {
    let dataset = Dataset::open(uri).await?;
    let files: BTreeSet<_> = dataset
        .manifest()
        .fragments
        .iter()
        .flat_map(|f| f.files.iter().map(|f| &f.path))
        .collect();
    let mut live_bytes = 0;
    for file in files {
        live_bytes += std::fs::metadata(Path::new(uri).join("data").join(file))?.len();
    }
    Ok(
        json!({"live_data_bytes": live_bytes, "all_data_bytes": directory_bytes(&Path::new(uri).join("data"))?,
        "all_index_bytes": directory_bytes(&Path::new(uri).join("_indices"))?, "total_directory_bytes": directory_bytes(Path::new(uri))?,
        "version": dataset.version().version}),
    )
}

pub fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}
