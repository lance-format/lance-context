//! Attribute publication preparation to actual native body metadata reads.
use super::*;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;
use lance_table::io::commit::ManifestNamingScheme;
use object_store::path::Path;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn mixed_descriptor_body_reads_distinguish_warm_and_uncached_sessions() -> Result<()> {
    for cache_bytes in [0, 1 << 20] {
        let root = artifact_root()?;
        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let narrow = storage(root.path(), "annotations.lance").await?;
        let faults = Arc::new(fault_store::Faults::default());
        let body = fault_store::storage_with_cache(
            &root.path().join("body.lance"),
            faults.clone(),
            cache_bytes,
        )
        .await?;
        seed(&c, &narrow, &body).await?;
        let (a, driver) = connect(&c, &narrow, &body).await?;
        let (stop, task) = run(driver);
        let p = producer(&a, 1).await?;
        let initial = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
        a.submit(&initial, Duration::from_secs(10)).await?;
        let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
        let selection = annotation_publication::select_range(
            &c,
            frozen.sources().records.clone(),
            frozen.through(),
            Uuid::parse_str(frozen.identity())?,
        )
        .await?;
        let prepared = frozen
            .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
            .await?;
        let writes = faults.body_write_attempts.load(Ordering::SeqCst);
        let mut observations = Vec::new();
        for _ in 0..2 {
            faults.get_requests.lock().unwrap().clear();
            let descriptors = crate::annotation_publication::mixed::persist_descriptors(
                prepared.annotations(),
                &selection,
                &narrow,
                &body,
                c.client.max_descriptor_bytes(),
            )
            .await?;
            assert_eq!(descriptors.format, 4, "small metadata must remain inline");
            let requests = faults.get_requests.lock().unwrap();
            let gets = requests.iter().filter(|(_, head)| !head).count();
            let heads = requests.iter().filter(|(_, head)| *head).count();
            let data_reads = requests
                .iter()
                .filter(|(p, _)| p.as_ref().contains("/data/"))
                .count();
            assert_eq!(
                data_reads, 0,
                "identity validation must not read body payloads"
            );
            observations.push((gets, heads));
        }
        assert_eq!(faults.body_write_attempts.load(Ordering::SeqCst), writes);
        if cache_bytes == 0 {
            assert!(observations.iter().all(|(gets, _)| *gets > 0));
        } else {
            assert_eq!(
                observations[1].0, 0,
                "warm body validation must reuse cached manifests"
            );
        }
        eprintln!(
            "{}",
            serde_json::json!({"publication_body_read_probe": {
                "cache_bytes": cache_bytes, "first_get_head": observations[0],
                "repeat_get_head": observations[1], "body_payload_reads": 0,
                "descriptor_format": 4,
            }})
        );
        // A warmed manifest must not bypass transaction identity checks.
        let mut foreign = prepared.annotations().merge().transaction.clone();
        let properties = Arc::make_mut(foreign.transaction_properties.as_mut().unwrap());
        let mut reference: BodySnapshot =
            serde_json::from_str(&properties[BODY_SNAPSHOT_PROPERTY])?;
        let manifest_version = reference.version;
        reference.transaction = Uuid::new_v4().to_string();
        properties.insert(
            BODY_SNAPSHOT_PROPERTY.into(),
            serde_json::to_string(&reference)?,
        );
        let foreign = lance_context_wal::replication::publication::Descriptors::lance_annotations(
            bytes::Bytes::from(prost::Message::encode_to_vec(
                &lance_table::format::pb::Transaction::from(&foreign),
            )),
        )?;
        assert!(crate::annotation_publication::mixed::resolve(
            prepared.annotations().base(),
            &foreign,
            &selection,
            &narrow,
            &body,
        )
        .await
        .is_err());
        if cache_bytes > 0 {
            // Cache residency is not evidence that the durable manifest exists.
            let path = ManifestNamingScheme::V2.manifest_path(
                &Path::from_absolute_path(root.path().join("body.lance"))?,
                manifest_version,
            );
            object_store::ObjectStoreExt::delete(
                &object_store::local::LocalFileSystem::new(),
                &path,
            )
            .await?;
            assert!(crate::annotation_publication::mixed::persist_descriptors(
                prepared.annotations(),
                &selection,
                &narrow,
                &body,
                c.client.max_descriptor_bytes(),
            )
            .await
            .is_err());
        }
        drop((prepared, frozen));
        stop.send(()).unwrap();
        task.await??;
        c.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn mixed_descriptor_body_get_stall_preserves_concurrent_annotation() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let faults = Arc::new(fault_store::Faults::default());
    let body_path = root.path().join("body.lance");
    let body = fault_store::storage_with_cache(&body_path, faults.clone(), 0).await?;
    seed(&c, &narrow, &body).await?;
    let (a, driver) = connect(&c, &narrow, &body).await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    let initial = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
    a.submit(&initial, Duration::from_secs(10)).await?;
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    let selection = annotation_publication::select_range(
        &c,
        frozen.sources().records.clone(),
        frozen.through(),
        Uuid::parse_str(frozen.identity())?,
    )
    .await?;
    let prepared = frozen
        .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let reference: BodySnapshot = serde_json::from_str(
        &prepared
            .annotations()
            .merge()
            .transaction
            .transaction_properties
            .as_ref()
            .unwrap()[BODY_SNAPSHOT_PROPERTY],
    )?;
    let manifest = ManifestNamingScheme::V2
        .manifest_path(&Path::from_absolute_path(&body_path)?, reference.version);
    let (entered, resume) = faults.pause_manifest_get(manifest);
    let mut preparing = Box::pin(crate::annotation_publication::mixed::persist_descriptors(
        prepared.annotations(),
        &selection,
        &narrow,
        &body,
        c.client.max_descriptor_bytes(),
    ));
    tokio::time::timeout(Duration::from_secs(10), async {
        tokio::select! {
            entered = entered => entered.context("body read interception lost"),
            result = &mut preparing => anyhow::bail!("preparation bypassed paused body GET: {result:?}"),
        }
    }).await??;
    assert!(futures::poll!(preparing.as_mut()).is_pending());
    let update = request(
        &a,
        &p,
        2,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let updated = a.submit(&update, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    assert!(futures::poll!(preparing.as_mut()).is_pending());
    resume.send(()).unwrap();
    let descriptors = tokio::time::timeout(Duration::from_secs(10), preparing).await??;
    assert_eq!(
        descriptors.format, 4,
        "the stalled path did not need an external descriptor PUT"
    );
    let commit = annotation_publication::save(&c, &selection, descriptors.clone()).await?;
    let selected = crate::annotation_publication::mixed::publish(
        Arc::new(c.client.manifest_catalog()),
        &annotation_publication::one_shard(),
        &narrow,
        &body,
        commit,
        &selection,
        &descriptors,
        prepared.annotations().base().clone(),
    )
    .await?;
    annotation_publication::confirm(&c).await?;
    a.install_published(&prepared, Arc::new(selected)).await?;
    drop((prepared, frozen));
    annotation_publication::finish(&c).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    pressure::publish(&c, &a, &narrow, &body).await?;
    annotation_runtime::drained(&c, &a, updated.index).await?;
    assert_eq!(a.submit(&update, Duration::from_secs(10)).await?, updated);
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
