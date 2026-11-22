use super::*;
use crate::annotations::runtime::reclamation::read_body_reference_for_test as read;
use lance::cache::LanceCache;

#[tokio::test]
async fn retained_body_reference_cache_keeps_exact_identity_without_pinning_datasets() -> Result<()>
{
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    let ingress = mixed::connect_ingress(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        &limits(),
        MergeMemoryBudget::new(8 << 20),
    )
    .await?;
    let state = annotation_publication::state(&c).await?;
    let manifests = state.manifests.as_ref().unwrap();
    let record = &manifests.versions[&manifests.head];
    let selected = annotation_publication::read(&c, &narrow, None).await?;
    let expected = BodySnapshot::read(&selected).await?.unwrap();
    drop(selected);
    let shared = LanceCache::with_capacity(64 << 10);
    let cache = shared.with_key_prefix("collector-a");
    let disabled = LanceCache::with_capacity(0);
    let first = read(&cache, &ingress, &narrow, record).await?;
    assert_eq!(first.0, expected);

    // Alternate real RF3 catalog/native Dataset opens and cache hits. Keep
    // timings as mechanism evidence, without asserting a wall-clock speedup.
    let mut uncached_micros = Vec::new();
    let mut cached_micros = Vec::new();
    for round in 0..16 {
        for cached in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let started = std::time::Instant::now();
            let value = read(
                if cached { &cache } else { &disabled },
                &ingress,
                &narrow,
                record,
            )
            .await?;
            let elapsed = started.elapsed().as_micros();
            assert_eq!(value.0, expected);
            assert_eq!(Arc::ptr_eq(&value, &first), cached);
            if cached {
                cached_micros.push(elapsed);
            } else {
                uncached_micros.push(elapsed);
            }
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({"retained_body_reference_probe": {
            "uncached_micros": uncached_micros,
            "cached_micros": cached_micros,
            "scope": "local RF3 immutable-reference reads, not end-to-end throughput"
        }})
    );
    assert_eq!(disabled.size().await, 0);
    assert!(shared.size_bytes().await <= 64 << 10);

    // Equal version/path cannot borrow a different catalog identity's entry.
    let mut foreign = (**record).clone();
    foreign.identity = Uuid::new_v4();
    assert!(read(&cache, &ingress, &narrow, &foreign).await.is_err());
    let other = shared.with_key_prefix("collector-b");
    let separate = read(&other, &ingress, &narrow, record).await?;
    assert_eq!(separate.0, expected);
    assert!(!Arc::ptr_eq(&separate, &first));

    c.shutdown().await?;
    // A hit needs no catalog RPC. A new collector or disabled cache still
    // resolves the original record and cannot pretend the stopped catalog works.
    assert!(Arc::ptr_eq(
        &read(&cache, &ingress, &narrow, record).await?,
        &first
    ));
    let deadline = lance_context_wal::replication::runtime::client::Options::default().timeout
        + Duration::from_secs(5);
    for missing in [disabled, shared.with_key_prefix("collector-c")] {
        // Let the real client's configured timeout report failure; an outer
        // five-second timeout would cancel its thirty-second request first.
        assert!(
            tokio::time::timeout(deadline, read(&missing, &ingress, &narrow, record))
                .await?
                .is_err()
        );
    }
    Ok(())
}
