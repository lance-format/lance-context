//! Offline expected rows versus the exact published snapshot. Outputs counts
//! and identities only; a mismatch never prints business keys or field values.
use super::*;
use futures::TryStreamExt;
use std::{collections::HashMap, io::BufRead};

pub(super) async fn run(config: &Config, input: &str, log: &str, owner_config: &str) -> Result<()> {
    let owner =
        lance_context_ingest::application::Config::read(std::path::Path::new(owner_config))?;
    let body = owner
        .annotations
        .as_ref()
        .and_then(|a| a.body.as_ref())
        .context("owner has no authorized body destination")?;
    ensure!(
        owner
            .tables
            .iter()
            .any(|t| t.binding.table == config.binding.table
                && t.binding.incarnation == config.binding.incarnation),
        "foreign owner binding"
    );
    let file = std::fs::File::open(log)?;
    ensure!(file.metadata()?.len() <= 8 << 20, "replay log exceeds8MiB");
    let mut selected = None;
    let mut acknowledged = 0u64;
    let mut acknowledged_rows = 0usize;
    for line in std::io::BufReader::new(file).lines() {
        let value: serde_json::Value = serde_json::from_str(&line?)?;
        match value["event"].as_str() {
            Some("applied") => {
                if let Some(start) = value.get("input_row_start") {
                    ensure!(
                        start.as_u64() == Some(acknowledged_rows as u64),
                        "noncontiguous input row interval"
                    );
                }
                let index = value["wal_index"].as_u64().context("missing ACK index")?;
                ensure!(index > acknowledged, "nonmonotonic replay ACK");
                acknowledged = index;
                acknowledged_rows +=
                    usize::try_from(value["rows"].as_u64().context("missing ACK rows")?)?;
            }
            Some("selected") => {
                ensure!(selected.is_none(), "multiple selected results");
                selected = Some(serde_json::from_value::<mixed::SelectedBody>(
                    value["selected"].clone(),
                )?);
            }
            _ => bail!("unexpected replay event"),
        }
    }
    let selected = selected.context("no successful selected result")?;
    ensure!(
        acknowledged > 0 && selected.through >= acknowledged,
        "uncovered replay ACKs"
    );
    ensure!(
        selected.cluster == config.layout.cluster
            && selected.table == config.binding.table
            && selected.source.shard == config.shard.bucket
            && selected.source.source_group
                == config.layout.group_for_bucket(config.shard.bucket)?
            && selected.source.table_instance == config.binding.incarnation.to_string()
            && selected.body.table_instance == selected.source.table_instance
            && selected.body.uri == body.uri,
        "foreign selected snapshot"
    );

    let file = std::fs::File::open(input)?;
    ensure!(file.metadata()?.len() <= 128 << 20, "input exceeds128MiB");
    let reader = arrow_ipc::reader::FileReader::try_new(file, None)?;
    ensure!(
        reader.schema() == *config.spec.body_schema(),
        "foreign input schema"
    );
    let mut batches = Vec::new();
    let mut expected = HashMap::new();
    let mut input_rows = 0usize;
    let mut allocations = HashMap::<usize, usize>::new();
    for batch in reader {
        let batch = batch?;
        // IPC columns can share one backing allocation. Counting capacity per
        // column would charge that allocation repeatedly.
        let mut arrays = batch
            .columns()
            .iter()
            .map(|a| a.to_data())
            .collect::<Vec<_>>();
        while let Some(data) = arrays.pop() {
            for buffer in data
                .buffers()
                .iter()
                .chain(data.nulls().map(|n| n.buffer()))
            {
                allocations
                    .entry(buffer.data_ptr().as_ptr() as usize)
                    .and_modify(|capacity| *capacity = (*capacity).max(buffer.capacity()))
                    .or_insert(buffer.capacity());
            }
            arrays.extend_from_slice(data.child_data());
        }
        let input_memory = allocations
            .values()
            .try_fold(0usize, |n, bytes| n.checked_add(*bytes))
            .context("input memory overflow")?;
        ensure!(input_memory <= 512 << 20, "decoded input exceeds512MiB");
        let prepared = config
            .spec
            .prepare(batch.clone(), config.limits.value_bytes)?;
        for (row, key) in prepared.keys().iter().enumerate() {
            expected.entry(key.clone()).or_insert((batches.len(), row));
        }
        input_rows += batch.num_rows();
        ensure!(input_rows <= 1_000_000, "input row bound exceeded");
        batches.push(batch);
    }
    ensure!(
        acknowledged_rows == input_rows,
        "replay did not acknowledge the entire input"
    );
    let unique_rows = expected.len();
    let started = Instant::now();
    let dataset = lance::dataset::builder::DatasetBuilder::from_uri(&body.uri)
        .with_storage_options(body.storage_options.clone())
        .with_version(selected.body.version)
        .load()
        .await?;
    ensure!(
        dataset
            .read_transaction()
            .await?
            .is_some_and(|t| t.uuid == selected.body.transaction),
        "selected native transaction mismatch"
    );
    let schema: arrow_schema::Schema = dataset.schema().into();
    ensure!(
        &schema == config.spec.body_schema().as_ref(),
        "published schema/metadata mismatch"
    );
    let mut scanner = dataset.scan();
    scanner.batch_size(1024);
    let mut stream = scanner.try_into_stream().await?;
    let mut actual_rows = 0usize;
    while let Some(batch) = stream.try_next().await? {
        // Dataset schema/metadata was checked above. Scanner batch metadata
        // need not repeat it; preserve and compare the actual returned arrays.
        let prepared = config.spec.prepare(
            RecordBatch::try_new(config.spec.body_schema().clone(), batch.columns().to_vec())?,
            config.limits.value_bytes,
        )?;
        for (row, key) in prepared.keys().iter().enumerate() {
            let (source, offset) = expected
                .remove(key)
                .context("unexpected or duplicate published key")?;
            for (column, expected_column) in batch.columns().iter().zip(batches[source].columns()) {
                ensure!(
                    column.slice(row, 1).to_data() == expected_column.slice(offset, 1).to_data(),
                    "published field differs from first occurrence"
                );
            }
            actual_rows += 1;
        }
    }
    ensure!(
        expected.is_empty() && actual_rows == unique_rows,
        "missing published rows"
    );
    println!(
        "{}",
        serde_json::json!({"event":"verified", "candidate_rows":input_rows,
        "unique_rows":unique_rows, "duplicate_candidates":input_rows-unique_rows,
        "fields":schema.fields().len(), "through":selected.through,
        "body_version":selected.body.version, "body_transaction":selected.body.transaction,
        "fragments":dataset.get_fragments().len(), "seconds":started.elapsed().as_secs_f64(),
        "all_values_match_first_occurrence":true, "schema_and_metadata_match":true})
    );
    Ok(())
}
