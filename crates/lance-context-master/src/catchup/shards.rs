//! Bounded physical-shard coverage for local merges. Demand keys identify
//! immutable WAL locations; they do not authorize a commit or writer replay.
use crate::state::MasterState;
use etcd_client::GetOptions;
use lance_context_metrics::etcd::{ObserveEtcd, Operation};
use std::{collections::HashSet, time::Duration};

const MAX_SHARDS: usize = 256;

fn extend(known: &mut Vec<String>, discovered: &[String]) -> Result<bool, String> {
    let mut ids = HashSet::<uuid::Uuid>::new();
    for id in known.iter() {
        ids.insert(id.parse().map_err(|_| "invalid physical WAL shard UUID")?);
    }
    let mut additions = Vec::new();
    for id in discovered {
        let id = id.parse().map_err(|_| "invalid demand WAL shard UUID")?;
        if ids.insert(id) {
            additions.push(id.to_string());
        }
    }
    if ids.len() > MAX_SHARDS {
        return Err("local merge shard coverage exceeds 256; demand retained".into());
    }
    additions.sort();
    let changed = !additions.is_empty();
    known.extend(additions);
    Ok(changed)
}

pub(super) async fn refresh(
    state: &MasterState,
    target: &str,
    known: &mut Vec<String>,
) -> Result<bool, String> {
    let encoded: String = target.bytes().map(|b| format!("{b:02x}")).collect();
    let prefix = format!("{}/demand-events/{encoded}/", state.config.etcd.prefix());
    // One table only, keys only, no unbounded fleet LIST or event payloads.
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        state
            .task_store
            .etcd_client()
            .clone()
            .get(
                prefix.clone(),
                Some(
                    GetOptions::new()
                        .with_prefix()
                        .with_keys_only()
                        .with_limit((MAX_SHARDS + 1) as i64),
                ),
            )
            .observed(Operation::DemandRead),
    )
    .await
    .map_err(|_| "merge shard coverage read timed out")?
    .map_err(|e| e.to_string())?;
    if response.more() || response.kvs().len() > MAX_SHARDS {
        return Err("local merge shard coverage exceeds 256; demand retained".into());
    }
    let discovered = response
        .kvs()
        .iter()
        .map(|kv| {
            std::str::from_utf8(kv.key())
                .map_err(|_| "invalid demand shard key")?
                .strip_prefix(&prefix)
                .map(str::to_owned)
                .ok_or("unexpected demand shard prefix")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let before = known.len();
    let changed = extend(known, &discovered)?;
    metrics::counter!("master_merge_shard_coverage_additions_total")
        .increment((known.len() - before) as u64);
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn id(name: &str) -> String {
        lance_context_core::derive_shard_id(Some(name)).to_string()
    }

    #[test]
    fn retains_historical_order_and_discovers_new_writer_without_rehashing() {
        let mut known = vec![id("old-b"), id("old-a")];
        assert!(extend(&mut known, &[id("new"), id("old-a"), id("new")]).unwrap());
        assert_eq!(known, [id("old-b"), id("old-a"), id("new")]);
        assert!(!extend(&mut known, &[id("new")]).unwrap());
    }

    #[test]
    fn invalid_or_overflowing_coverage_never_truncates_to_success() {
        let original = vec![id("old")];
        for discovered in [
            vec!["not-a-uuid".into()],
            (0..256).map(|i| id(&i.to_string())).collect(),
        ] {
            let mut known = original.clone();
            assert!(extend(&mut known, &discovered).is_err());
            assert_eq!(known, original);
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn refresh_sees_writer_arriving_after_initial_snapshot_and_rejects_bad_keys() {
        let (_dir, state) = super::super::tests::fixture().await.unwrap();
        let mut known = vec![id("historical")];
        assert!(!refresh(&state, "hot", &mut known).await.unwrap());
        let prefix = format!("{}/demand-events/686f74/", state.config.etcd.prefix());
        let mut client = state.task_store.etcd_client().clone();
        // No value decoding: coverage is independent of the event payload size.
        client
            .put(
                format!("{prefix}{}", id("new")),
                vec![b'x'; 128 * 1024],
                None,
            )
            .await
            .unwrap();
        // Other targets must not leak into this table's coverage.
        client
            .put(
                format!(
                    "{}/demand-events/6f74686572/{}",
                    state.config.etcd.prefix(),
                    id("other")
                ),
                "ignored",
                None,
            )
            .await
            .unwrap();
        assert!(refresh(&state, "hot", &mut known).await.unwrap());
        assert_eq!(known, [id("historical"), id("new")]);
        client
            .put(format!("{prefix}invalid-shard"), "bad", None)
            .await
            .unwrap();
        let previous = known.clone();
        assert!(refresh(&state, "hot", &mut known).await.is_err());
        assert_eq!(known, previous);
    }
}
