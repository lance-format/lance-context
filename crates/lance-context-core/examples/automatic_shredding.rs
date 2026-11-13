//! Compare automatic shredding and JSON baselines in the same optimized binary.
//! Each invocation measures one arm in a fresh directory, including process RSS.
use std::{path::PathBuf, sync::Arc, time::Instant};

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, ValueEnum};
use datafusion::prelude::{col, lit};
use futures::TryStreamExt;
use lance::{dataset::WriteParams, index::DatasetIndexExt, Dataset};
use lance_context_core::{shredding::ShredPolicy, DocumentStore};
use lance_index::{
    scalar::{inverted::Language, FullTextSearchQuery, InvertedIndexParams},
    IndexType,
};
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
use serde_json::{json, Value};

#[path = "automatic_shredding/support.rs"]
mod support;
use support::{fixture, measure, storage, Result};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Json,
    Duplicated,
    Shredded,
}

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    /// A new directory; existing paths are rejected, never removed.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 256)]
    rows: usize,
    #[arg(long, default_value_t = 65_536)]
    body_bytes: usize,
    #[arg(long, default_value_t = 15)]
    samples: usize,
    /// Use one runtime size across all arms; independent of machine CPU count.
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

enum Store {
    Shredded(DocumentStore),
    Json { dataset: Dataset, duplicated: bool },
}

impl Store {
    async fn create(uri: &str, mode: Mode) -> Result<Self> {
        if matches!(mode, Mode::Shredded) {
            return Ok(Self::Shredded(
                DocumentStore::create(uri, ShredPolicy::default(), None).await?,
            ));
        }
        let duplicated = matches!(mode, Mode::Duplicated);
        let mut fields = vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("json", DataType::LargeUtf8, false),
        ];
        if duplicated {
            fields.extend([
                Field::new("body", DataType::LargeUtf8, true),
                Field::new("late_text", DataType::LargeUtf8, true),
            ]);
        }
        let schema = Arc::new(Schema::new(fields));
        let reader =
            RecordBatchIterator::new(Vec::<std::result::Result<RecordBatch, _>>::new(), schema);
        Ok(Self::Json {
            dataset: Dataset::write(reader, uri, Some(WriteParams::default())).await?,
            duplicated,
        })
    }

    async fn add(&mut self, documents: Vec<Value>) -> Result<Vec<String>> {
        match self {
            Self::Shredded(store) => Ok(store.add(documents).await?),
            Self::Json {
                dataset,
                duplicated,
            } => {
                let ids: Vec<_> = documents
                    .iter()
                    .map(|_| lance_context_core::generate_id())
                    .collect();
                let serialized = documents
                    .iter()
                    .map(serde_json::to_string)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let mut columns: Vec<ArrayRef> = vec![
                    Arc::new(StringArray::from_iter_values(&ids)),
                    Arc::new(LargeStringArray::from_iter_values(serialized)),
                ];
                if *duplicated {
                    for key in ["body", "late_text"] {
                        columns.push(Arc::new(LargeStringArray::from_iter(
                            documents
                                .iter()
                                .map(|doc| doc.get(key).and_then(Value::as_str)),
                        )));
                    }
                }
                let schema = Arc::new(Schema::from(dataset.schema()));
                let batch = RecordBatch::try_new(schema.clone(), columns)?;
                dataset
                    .append(RecordBatchIterator::new(vec![Ok(batch)], schema), None)
                    .await?;
                Ok(ids)
            }
        }
    }

    async fn get(&self, id: &str, field: Option<&str>) -> Result<Value> {
        match self {
            Self::Shredded(store) => Ok(match field {
                Some(key) => store.get_path(id, &[key.into()]).await?,
                None => store.get(id).await?,
            }
            .ok_or("missing document or path")?),
            Self::Json {
                dataset,
                duplicated,
            } => {
                let projected = if *duplicated { field } else { None };
                let mut scan = dataset.scan();
                scan.project(&[projected.unwrap_or("json")])?
                    .filter_expr(col("id").eq(lit(id)));
                let mut stream = scan.try_into_stream().await?;
                while let Some(batch) = stream.try_next().await? {
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    let text = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or("unexpected text type")?
                        .value(0);
                    if projected.is_some() {
                        return Ok(Value::from(text));
                    }
                    let mut value: Value = serde_json::from_str(text)?;
                    if let Some(key) = field {
                        value = value
                            .as_object_mut()
                            .and_then(|object| object.remove(key))
                            .ok_or("missing path")?;
                    }
                    return Ok(value);
                }
                Err("missing document".into())
            }
        }
    }

    async fn filter(&self) -> Result<Vec<String>> {
        match self {
            Self::Shredded(store) => Ok(store
                .filter_eq(&["tags".into()], &json!(["target"]), 1)
                .await?),
            Self::Json { dataset, .. } => {
                let mut scan = dataset.scan();
                scan.project(&["id", "json"])?;
                scan.batch_size(1024)
                    .batch_size_bytes(4 * 1024 * 1024)
                    .batch_readahead(1);
                let mut stream = scan.try_into_stream().await?;
                while let Some(batch) = stream.try_next().await? {
                    let documents = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or("unexpected JSON type")?;
                    let ids = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("unexpected ID type")?;
                    for row in 0..batch.num_rows() {
                        let doc: Value = serde_json::from_str(documents.value(row))?;
                        if doc["tags"] == json!(["target"]) {
                            return Ok(vec![ids.value(row).into()]);
                        }
                    }
                }
                Ok(Vec::new())
            }
        }
    }

    async fn prepare(&mut self) -> Result<()> {
        match self {
            Self::Shredded(store) => store.prepare_text_search().await?,
            Self::Json {
                dataset,
                duplicated: true,
            } => {
                let indices = dataset.load_indices().await?;
                for column in ["body", "late_text"] {
                    let name = format!("fts_{column}");
                    if indices.iter().any(|index| index.name == name) {
                        continue;
                    }
                    let params = InvertedIndexParams::new("simple".into(), Language::English)
                        .lower_case(true)
                        .stem(false)
                        .remove_stop_words(false)
                        .max_token_length(None);
                    dataset
                        .create_index(&[column], IndexType::Inverted, Some(name), &params, true)
                        .await?;
                }
            }
            Self::Json {
                duplicated: false, ..
            } => return Err("JSON-only arm has no native FTS".into()),
        }
        Ok(())
    }

    async fn search(&mut self, query: &str) -> Result<Vec<String>> {
        match self {
            Self::Shredded(store) => Ok(store
                .search(query, 10)
                .await?
                .into_iter()
                .map(|hit| hit.id)
                .collect()),
            Self::Json { dataset, .. } => {
                let mut scan = dataset.scan();
                scan.project(&["id"])?;
                scan.full_text_search(
                    FullTextSearchQuery::new(query.into())
                        .with_columns(&["body".into(), "late_text".into()])?
                        .limit(Some(10)),
                )?;
                let batches: Vec<RecordBatch> = scan.try_into_stream().await?.try_collect().await?;
                let mut result = Vec::new();
                for batch in batches {
                    let ids = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("unexpected ID type")?;
                    for value in ids.iter() {
                        result.push(value.ok_or("null ID")?.into());
                    }
                }
                Ok(result)
            }
        }
    }
}

async fn run(args: Args) -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("use cargo run --release for comparable results".into());
    }
    if args.rows < 4 || args.body_bytes < 64 || args.samples == 0 {
        return Err("rows >= 4, body-bytes >= 64 and samples >= 1 required".into());
    }
    std::fs::create_dir(&args.output)?;
    let uri = args.output.join("documents.lance");
    let uri = uri.to_str().ok_or("non-UTF8 path")?;
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::set_global_recorder(recorder)?;
    let documents = fixture(args.rows, args.body_bytes);
    let input_bytes: usize = documents
        .iter()
        .map(|doc| serde_json::to_vec(doc).unwrap().len())
        .sum();
    let mut store = Store::create(uri, args.mode).await?;
    let (ids, ingest) = measure(&snapshotter, store.add(documents)).await?;
    let mut result = json!({"mode": format!("{:?}", args.mode), "rows": args.rows,
        "body_bytes": args.body_bytes, "samples": args.samples, "threads": args.threads,
        "input_json_bytes": input_bytes, "ingest": ingest, "before_indexes": storage(uri).await?,
        "build": "release; all arms use this same executable and Lance dependency tree"});
    // Validate outside timing. Repeat deterministic generation without retaining the whole fixture.
    for row in [0, args.rows / 2, args.rows - 1] {
        assert_eq!(
            store.get(&ids[row], None).await?,
            support::document(row, args.rows, args.body_bytes)
        );
    }
    let expected_body = support::document(0, args.rows, args.body_bytes)["body"].take();
    assert_eq!(store.get(&ids[0], Some("body")).await?, expected_body);
    assert_eq!(store.filter().await?, vec![ids[args.rows - 1].clone()]);
    result["full_read"] = reads(&store, &ids, None, args.samples, &snapshotter).await?;
    result["projected_body_read"] =
        reads(&store, &ids, Some("body"), args.samples, &snapshotter).await?;
    let mut filters = Vec::new();
    for _ in 0..args.samples {
        let (found, sample) = measure(&snapshotter, store.filter()).await?;
        assert_eq!(found, vec![ids[args.rows - 1].clone()]);
        filters.push(sample);
    }
    result["residual_filter"] = support::summarize(filters);
    if !matches!(args.mode, Mode::Json) {
        let (_, prepare) = measure(&snapshotter, store.prepare()).await?;
        result["initial_fts_prepare"] = prepare;
        assert_eq!(store.search("shredneedle").await?, vec![ids[0].clone()]);
        let mut samples = Vec::new();
        for _ in 0..args.samples {
            let (found, sample) = measure(&snapshotter, store.search("shredneedle")).await?;
            assert_eq!(found, vec![ids[0].clone()]);
            samples.push(sample);
        }
        result["fts"] = support::summarize(samples);
        result["after_initial_indexes"] = storage(uri).await?;
        let later = json!({"body": "additional document", "sequence": args.rows, "tags": [], "late_text": "historicalneedle promoted"});
        let (new_ids, append) = measure(&snapshotter, store.add(vec![later.clone()])).await?;
        result["layout_growth_append"] = append;
        let (_, prepare) = measure(&snapshotter, store.prepare()).await?;
        result["layout_growth_prepare"] = prepare;
        let mut found = store.search("historicalneedle").await?;
        found.sort();
        let mut expected = vec![ids[args.rows - 1].clone(), new_ids[0].clone()];
        expected.sort();
        assert_eq!(found, expected);
        assert_eq!(store.get(&new_ids[0], None).await?, later);
    }
    result["final_storage"] = storage(uri).await?;
    drop(store);
    let reopened = match args.mode {
        Mode::Shredded => Store::Shredded(DocumentStore::open(uri, None).await?),
        mode => Store::Json {
            dataset: Dataset::open(uri).await?,
            duplicated: matches!(mode, Mode::Duplicated),
        },
    };
    let (value, reopen_read) = measure(&snapshotter, reopened.get(&ids[0], None)).await?;
    assert_eq!(value, support::document(0, args.rows, args.body_bytes));
    result["first_read_after_reopen"] = reopen_read;
    result["peak_rss_kib"] = json!(support::peak_rss_kib());
    result["scope"] = json!("Local synthetic single-writer workload. Warm OS cache; reopen resets handle caches, not the OS cache. I/O counters measure requested object-store bytes, not device traffic. RSS includes fixture generation, ingestion, queries and indexing in this fresh process. JSON-only FTS is intentionally unmeasured. Duplicated baseline materializes body and late_text upfront; shredding discovers late_text on later writes. No ID index in any arm. Validation is outside timed phases.");
    let output = serde_json::to_string_pretty(&result)?;
    std::fs::write(args.output.join("result.json"), &output)?;
    println!("{output}");
    Ok(())
}

async fn reads(
    store: &Store,
    ids: &[String],
    field: Option<&str>,
    samples: usize,
    snapshotter: &Snapshotter,
) -> Result<Value> {
    let mut results = Vec::new();
    for index in 0..samples {
        let (value, sample) = measure(
            snapshotter,
            store.get(&ids[(index * 17) % ids.len()], field),
        )
        .await?;
        std::hint::black_box(value);
        results.push(sample);
    }
    Ok(support::summarize(results))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build()?
        .block_on(run(args))?;
    eprintln!("Completed in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}
