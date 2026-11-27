//! Read-only recovery audit of the latest successful request per producer.
//! Stop writers, owners and GC first; this does not establish their quiescence.
//! Supply the original allocation and final acknowledged completion for each
//! producer. No open, close, retry, append or body scan is performed.
use anyhow::{bail, ensure, Context, Result};
use lance::dataset::builder::DatasetBuilder;
use lance_context_core::annotation_staging::{Scope, MAX_RESULTS_BYTES, RESULTS_PROPERTY};
use lance_context_ingest::{
    annotations::{Completed, MutationResult},
    application::Config,
    materializer::Storage,
    ProducerOpen,
};
use lance_context_wal::{
    record::RecordKind,
    replication::{
        producer,
        publication::lance::ManagedManifest,
        query::{Outcome, Query, Value},
        runtime::client::{Client, Reply, Request},
    },
};
use serde::Deserialize;
use std::{collections::BTreeSet, io::Read, path::Path, sync::Arc};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    through: u64,
    producers: Vec<ExpectedProducer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedProducer {
    open: ProducerOpen,
    completion: Completed,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Results {
    version: u8,
    slots: u32,
    rejected: Vec<(u32, u64, u32, u8)>,
}

fn checkpoint(cut: &str, results: &str, scope: &Scope, slots: u32, expected: u64) -> Result<u64> {
    ensure!(cut.len() <= 4096 && results.len() <= MAX_RESULTS_BYTES);
    let (actual_scope, through): (Scope, u64) = serde_json::from_str(cut)?;
    let results: Results = serde_json::from_str(results)?;
    ensure!(
        &actual_scope == scope && expected > 0 && through >= expected && through < u64::MAX,
        "selected application cut does not cover the acknowledged shard"
    );
    ensure!(
        results.version == 1
            && results.slots == slots
            && (1..=producer::MAX_SLOTS).contains(&slots)
            && results.rejected.is_empty(),
        "selected results differ from the expected all-applied workload"
    );
    Ok(through)
}

async fn query(config: &Config, client: &Client, group: u64, query: Query) -> Result<Value> {
    let reply = client
        .request_routed(Request::Query { group, query })
        .await?;
    let Reply::Query(Outcome::Observed { cut, value }) = reply.reply else {
        bail!("recovery quorum observation unavailable");
    };
    ensure!(
        cut.group == group
            && cut.cluster == config.layout.cluster
            && cut.layout == config.layout.fingerprint()?,
        "foreign quorum observation"
    );
    Ok(value)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: annotation_result_probe CONFIG EXPECTATIONS"
    );
    let config = Config::read(Path::new(&args[0]))?;
    ensure!(config.tables.len() == 1 && config.annotations.is_some());
    let mut bytes = Vec::new();
    std::fs::File::open(&args[1])?
        .take((1 << 20) + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 1 << 20, "oversized expectations");
    let expected: Expected = serde_json::from_slice(&bytes)?;
    let table = &config.tables[0];
    let bucket = config
        .annotations
        .as_ref()
        .unwrap()
        .shard
        .map_or(0, |s| s.bucket);
    let group = config.layout.group_for_bucket(bucket)?;
    let scope = Scope {
        table_instance: table.binding.incarnation.to_string(),
        source_group: group,
        shard: bucket,
    };
    ensure!(
        !expected.producers.is_empty()
            && expected.producers.len() <= config.limits.producer_slots as usize
            && expected
                .producers
                .iter()
                .map(|p| p.open.slot)
                .collect::<BTreeSet<_>>()
                .len()
                == expected.producers.len()
            && expected.producers.iter().map(|p| p.completion.index).max()
                == Some(expected.through),
        "invalid producer population or expected cut"
    );
    let client = config.client()?;
    let storage = Storage::resolve(
        table.uri.clone(),
        table.storage_options.clone(),
        Arc::default(),
    )
    .await?;
    let handler = Arc::new(ManagedManifest::reader(
        Arc::new(client.manifest_catalog()),
        storage.base.clone(),
        &config.layout,
        table.binding.table,
        table.binding.incarnation,
    )?);
    let selected = DatasetBuilder::from_uri(&storage.uri)
        .with_read_params(lance::dataset::ReadParams {
            store_options: Some(storage.params.clone()),
            ..Default::default()
        })
        .with_session(storage.session.clone())
        .with_commit_handler(handler.clone())
        .load()
        .await?;
    let transaction = selected
        .read_transaction()
        .await?
        .context("missing selected transaction")?;
    let properties = transaction
        .transaction_properties
        .as_ref()
        .context("missing application checkpoint")?;
    let through = checkpoint(
        properties
            .get("lance-context.annotation-cut")
            .context("missing application cut")?,
        properties
            .get(RESULTS_PROPERTY)
            .context("missing retained request results")?,
        &scope,
        config.limits.producer_slots,
        expected.through,
    )?;
    for producer in &expected.producers {
        let open = &producer.open;
        let completion = &producer.completion;
        let meta = &completion.meta;
        let generation = open
            .expected_generation
            .checked_add(1)
            .context("generation overflow")?;
        ensure!(
            open.slot < config.limits.producer_slots
                && open.bucket == bucket
                && completion.index > 0
                && completion.index <= through
                && completion.result == MutationResult::Applied
                && meta.kind == RecordKind::Data
                && meta.stream == table.binding.table
                && meta.bucket == bucket
                && meta.sequence > 0
                && meta.producer == producer::producer_id(open.slot, generation),
            "foreign or unsuccessful acknowledged request"
        );
        let Value::ProducerSlot(Some(view)) = query(
            &config,
            &client,
            group,
            Query::ProducerSlot { slot: open.slot },
        )
        .await?
        else {
            bail!("missing retained producer slot");
        };
        let lease = view.lease.as_ref().context("missing original allocation")?;
        ensure!(
            view.slot == open.slot
                && view.generation == generation
                && lease.table == table.binding.table
                && lease.bucket == bucket
                && lease.operation == open.operation
                && view.last
                    == Some(producer::Last {
                        sequence: meta.sequence,
                        index: completion.index
                    }),
            "retained producer prefix differs from the original ACK"
        );
        let Value::Producer(Some((retained, index))) = query(
            &config,
            &client,
            group,
            Query::Producer {
                table: table.binding.table,
                bucket,
                producer: meta.producer,
            },
        )
        .await?
        else {
            bail!("missing retained request receipt");
        };
        ensure!(
            retained == *meta && index == completion.index,
            "retained request metadata differs"
        );
    }
    let after = DatasetBuilder::from_uri(&storage.uri)
        .with_read_params(lance::dataset::ReadParams {
            store_options: Some(storage.params.clone()),
            ..Default::default()
        })
        .with_session(storage.session)
        .with_commit_handler(handler)
        .load()
        .await?;
    ensure!(
        after.version_id() == selected.version_id(),
        "publication moved during the audit"
    );
    println!(
        "{}",
        serde_json::json!({
            "verified":true, "selected_version":selected.version_id(), "selected_cut":through,
            "expected_through":expected.through, "producers":expected.producers.len(),
            "retained_result_slots":config.limits.producer_slots, "rejected_results":0,
            "client_mutations_submitted":0,
            "scope":"Latest acknowledged Applied request per supplied original producer; separate quorum observations with caller-established quiescence. No body/value audit or historical outcome claim."
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_must_cover_the_exact_shard_and_all_applied_outcomes() -> Result<()> {
        let scope = Scope {
            table_instance: "incarnation".into(),
            source_group: 4,
            shard: 2,
        };
        let cut = serde_json::to_string(&(scope.clone(), 91))?;
        let results = r#"{"version":1,"slots":8,"rejected":[]}"#;
        assert_eq!(checkpoint(&cut, results, &scope, 8, 90)?, 91);
        assert!(checkpoint(&cut, results, &scope, 8, 92).is_err());
        assert!(checkpoint(&cut, results, &scope, 9, 90).is_err());
        for foreign in [
            Scope {
                table_instance: "other".into(),
                ..scope.clone()
            },
            Scope {
                source_group: 5,
                ..scope.clone()
            },
            Scope {
                shard: 3,
                ..scope.clone()
            },
        ] {
            assert!(checkpoint(&cut, results, &foreign, 8, 90).is_err());
        }
        for invalid in [
            r#"{"version":2,"slots":8,"rejected":[]}"#,
            r#"{"version":1,"slots":8,"rejected":[[0,90,0,0]]}"#,
            r#"{"version":1,"slots":8}"#,
        ] {
            assert!(checkpoint(&cut, invalid, &scope, 8, 90).is_err());
        }
        Ok(())
    }
}
