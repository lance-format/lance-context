//! Run with `cargo run -p lance-context-core --example generic_batch_get`.
//! Supply `--uri` to benchmark an existing store without changing its data.

use std::time::Instant;

use clap::Parser;
use lance_context_api::{ColumnSpec, ColumnType, SchemaSpec};
use lance_context_core::{GenericStore, GenericStoreOptions};
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    uri: Option<String>,
    #[arg(long, default_value_t = 5)]
    samples: usize,
    /// Leave two generations pending in the synthetic store.
    #[arg(long)]
    pending_wal: bool,
}

fn percentiles(mut times: Vec<f64>) -> serde_json::Value {
    times.sort_by(f64::total_cmp);
    json!({"samples": times.len(), "p50_ms": times[(times.len() as f64 * 0.50).ceil() as usize - 1], "p95_ms": times[(times.len() as f64 * 0.95).ceil() as usize - 1]})
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.samples == 0 {
        return Err("samples must be positive".into());
    }
    let temp = tempfile::TempDir::new()?;
    let mut store = if let Some(uri) = &args.uri {
        GenericStore::open_existing(uri, GenericStoreOptions::default()).await?
    } else {
        let spec = SchemaSpec::new(vec![
            (
                "id".into(),
                ColumnSpec::required(ColumnType::String { large: false }),
            ),
            (
                "payload".into(),
                ColumnSpec::new(ColumnType::String { large: false }),
            ),
        ]);
        let mut store = GenericStore::open(
            temp.path().to_str().unwrap(),
            spec,
            GenericStoreOptions {
                seal_on_add: true,
                ..Default::default()
            },
        )
        .await?;
        let rows: Vec<_> = (0..2048)
            .map(|i| {
                json!({"id": format!("r{i:05}"), "payload": "x".repeat(2048)})
                    .as_object()
                    .unwrap()
                    .clone()
            })
            .collect();
        store.add(&rows).await?;
        store.cleanup_wal().await?;
        store.create_id_index().await?;
        if args.pending_wal {
            store.add(&rows[..256]).await?;
            store.add(&rows[128..384]).await?;
        }
        store
    };
    // Establish expected rows from this bounded fixture; no payloads are logged.
    let mut source_rows = store.list(None, None).await?;
    source_rows.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let mut results = Vec::new();
    for n in [1, 32, 512] {
        if source_rows.len() < n {
            continue;
        }
        let ids: Vec<String> = source_rows[..n]
            .iter()
            .map(|row| row["id"].as_str().unwrap().to_string())
            .collect();
        let filter = format!(
            "id IN ({})",
            ids.iter()
                .map(|id| format!("'{}'", id.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(",")
        );
        for mode in ["legacy_pk_in", "get_many", "get_many_ids_only"] {
            let mut times = Vec::new();
            let mut response_bytes = 0;
            // One warmup, then measured reads, including JSON row decoding.
            for sample in 0..=args.samples {
                let start = Instant::now();
                let rows = match mode {
                    "legacy_pk_in" => store.list_filtered(&filter, None, None).await?,
                    "get_many" => store.get_many(&ids, None).await?,
                    _ => store.get_many(&ids, Some(&["id".into()])).await?,
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(rows.len(), n);
                let mut sorted = rows;
                sorted.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
                if mode == "get_many_ids_only" {
                    let expected: Vec<_> = ids
                        .iter()
                        .map(|id| json!({"id":id}).as_object().unwrap().clone())
                        .collect();
                    assert!(
                        sorted == expected,
                        "ID projection differs from expected rows"
                    );
                } else {
                    assert!(
                        sorted == source_rows[..n],
                        "response differs from expected rows"
                    );
                }
                response_bytes = serde_json::to_vec(&sorted)?.len();
                if sample != 0 {
                    times.push(elapsed);
                }
            }
            results.push(json!({"ids": n, "mode": mode, "latency": percentiles(times), "response_bytes": response_bytes}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "fixture": if args.uri.is_some() { "existing_store" } else { "synthetic_2048_rows_2KiB_payload" },
            "debug": cfg!(debug_assertions),
            "pending_wal_generations": store.pending_wal_generations().await?,
            "results": results,
            "correctness": "all responses matched expected rows"
        }))?
    );
    store.close().await?;
    Ok(())
}
