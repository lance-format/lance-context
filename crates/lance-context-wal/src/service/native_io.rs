//! Reuse raft-engine's existing process-wide histograms at observation time.
//! These count completed encoded batches, not physical device writes or fsyncs.
//! A process restart resets them; multiple embedded engines share the totals.
use serde::Serialize;

#[derive(Default, Serialize)]
pub(super) struct Batches {
    batches: u64,
    bytes: f64,
}

#[derive(Default, Serialize)]
pub(super) struct Observation {
    append: Batches,
    /// Both categories write into the rewrite queue. The name identifies the
    /// source queue being reclaimed, not the destination file type.
    rewrite_from_append: Batches,
    rewrite_from_rewrite: Batches,
}

pub(super) fn observe() -> Observation {
    let mut observed = Observation::default();
    for family in prometheus::gather() {
        if !matches!(
            family.get_name(),
            "raft_engine_write_size" | "raft_engine_background_rewrite_bytes"
        ) {
            continue;
        }
        for metric in family.get_metric() {
            let target = match family.get_name() {
                "raft_engine_write_size" => &mut observed.append,
                _ => match metric
                    .get_label()
                    .iter()
                    .find(|label| label.get_name() == "type")
                    .map(|label| label.get_value())
                {
                    Some("append") => &mut observed.rewrite_from_append,
                    Some("rewrite") => &mut observed.rewrite_from_rewrite,
                    _ => continue,
                },
            };
            let histogram = metric.get_histogram();
            target.batches = histogram.get_sample_count();
            target.bytes = histogram.get_sample_sum();
        }
    }
    observed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_batches_distinguish_append_and_both_rewrite_sources() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let config = raft_engine::Config {
            dir: root.path().to_str().unwrap().to_owned(),
            target_file_size: raft_engine::ReadableSize(64 << 10),
            purge_threshold: raft_engine::ReadableSize(64 << 10),
            batch_compression_threshold: raft_engine::ReadableSize(0),
            enable_log_recycle: false,
            ..Default::default()
        };
        let engine = raft_engine::Engine::open(config.clone())?;
        let before = observe();
        let mut written = 0;
        for key in 0..4 {
            let mut batch = raft_engine::LogBatch::default();
            batch.put(1, vec![key], vec![key; 128 << 10])?;
            written += engine.write(&mut batch, true)?;
        }
        let appended = observe();
        assert!(appended.append.bytes - before.append.bytes >= written as f64);
        assert!(appended.append.batches - before.append.batches >= 4);
        engine.purge_manager().must_rewrite_append_queue(None, None);
        let rewritten = observe();
        assert!(rewritten.rewrite_from_append.bytes > appended.rewrite_from_append.bytes);
        assert!(rewritten.rewrite_from_append.batches > appended.rewrite_from_append.batches);
        engine.purge_manager().must_rewrite_rewrite_queue();
        let rewritten_again = observe();
        assert!(rewritten_again.rewrite_from_rewrite.bytes > rewritten.rewrite_from_rewrite.bytes);
        assert!(
            rewritten_again.rewrite_from_rewrite.batches > rewritten.rewrite_from_rewrite.batches
        );
        drop(engine);
        let reopened = raft_engine::Engine::open(config)?;
        for key in 0..4 {
            assert_eq!(reopened.get(1, &[key]), Some(vec![key; 128 << 10]));
        }
        Ok(())
    }
}
