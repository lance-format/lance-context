use super::*;

#[derive(Default)]
pub(super) struct Report {
    pub through: u64,
    pub rows: usize,
    pub bytes: usize,
    pub latencies: Vec<f64>,
}

// Slice original Arrow arrays without concatenating body values. Keep order
// across source batch boundaries, including duplicate business keys.
fn split(batches: Vec<RecordBatch>, rows: usize) -> (Vec<RecordBatch>, Vec<RecordBatch>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut remaining = rows;
    for batch in batches {
        let count = batch.num_rows().min(remaining);
        if count > 0 {
            left.push(batch.slice(0, count));
        }
        if count < batch.num_rows() {
            right.push(batch.slice(count, batch.num_rows() - count));
        }
        remaining -= count;
    }
    (left, right)
}

async fn flush(
    config: &Config,
    client: &DurableMutationClient,
    batches: Vec<RecordBatch>,
    report: &mut Report,
) -> Result<()> {
    if batches.is_empty() {
        return Ok(());
    }
    let mut pending = vec![batches];
    while let Some(batches) = pending.pop() {
        let count = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        let bytes = encode(config, &batches)?;
        // The size target is only an estimate. Enforce the actual wire limit
        // before any network submission, splitting with shared Arrow slices.
        if bytes.len() > config.binding.max_record_bytes as usize {
            ensure!(
                count > 1,
                "one body exceeds the configured WAL record limit"
            );
            let (left, right) = split(batches, count.div_ceil(2));
            pending.push(right);
            pending.push(left);
            continue;
        }
        let length = bytes.len();
        let sent = Instant::now();
        let reply = client.append(0, bytes).await?;
        ensure!(
            matches!(reply.completion.result, MutationResult::Applied),
            "committed mutation rejected: {:?}",
            reply.completion.result
        );
        let seconds = sent.elapsed().as_secs_f64();
        report.through = reply.completion.index;
        println!(
            "{}",
            serde_json::json!({"event":"applied", "input_row_start":report.rows,
            "rows":count, "source_batch_slices":batches.len(), "bytes":length,
            "sequence":reply.completion.meta.sequence, "wal_index":report.through, "seconds":seconds})
        );
        report.rows += count;
        report.bytes += length;
        report.latencies.push(seconds);
    }
    Ok(())
}

pub(super) async fn run(
    config: &Config,
    client: &DurableMutationClient,
    reader: arrow_ipc::reader::FileReader<std::fs::File>,
    coalesce: bool,
) -> Result<Report> {
    let mut report = Report::default();
    let mut pending = Vec::new();
    let mut rows = 0usize;
    let mut estimated_bytes = 0usize;
    let target = config.binding.max_record_bytes as usize / 2;
    for batch in reader {
        let batch = batch?;
        let mut offset = 0;
        while offset < batch.num_rows() {
            let count = (batch.num_rows() - offset).min(config.binding.max_rows as usize - rows);
            let piece = batch.slice(offset, count);
            // Logical slices, not allocation capacity (IPC columns share their
            // backing buffer). Final encoded bytes are checked in flush.
            for column in piece.columns() {
                estimated_bytes = estimated_bytes
                    .checked_add(column.to_data().get_slice_memory_size()?)
                    .context("batch size overflow")?;
            }
            pending.push(piece);
            rows += count;
            offset += count;
            if !coalesce || rows == config.binding.max_rows as usize || estimated_bytes >= target {
                flush(config, client, std::mem::take(&mut pending), &mut report).await?;
                rows = 0;
                estimated_bytes = 0;
            }
        }
    }
    flush(config, client, pending, &mut report).await?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_keeps_source_order_and_shares_body_buffers() -> Result<()> {
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "id",
            DataType::Utf8,
            false,
        )]));
        let first = Arc::new(StringArray::from(vec!["a", "b", "b"]));
        let second = Arc::new(StringArray::from(vec!["c", "d"]));
        let batches = vec![
            RecordBatch::try_new(schema.clone(), vec![first.clone()])?,
            RecordBatch::try_new(schema, vec![second.clone()])?,
        ];
        let (left, right) = split(batches, 4);
        assert_eq!(
            left.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            vec![3, 1]
        );
        assert_eq!(
            right.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            vec![1]
        );
        let a = left[1]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let b = right[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(a.value(0), "c");
        assert_eq!(b.value(0), "d");
        assert_eq!(a.value_data().as_ptr(), second.value_data().as_ptr());
        assert_eq!(b.value_data().as_ptr(), second.value_data().as_ptr());
        Ok(())
    }
}
