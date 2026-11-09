//! A confirmed detached commit already supplies the selected body identity.
use super::*;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;
use lance_table::io::commit::ManifestNamingScheme;
use object_store::path::Path;

#[tokio::test]
async fn mixed_body_preparation_does_not_read_back_its_successful_commit() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let faults = Arc::new(fault_store::Faults::default());
    // Disable metadata caching so an accidental readback cannot hide in cache.
    let body_path = root.path().join("body.lance");
    let body = fault_store::storage_with_cache(&body_path, faults.clone(), 0).await?;
    seed(&c, &narrow, &body).await?;
    let (a, driver) = connect(&c, &narrow, &body).await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    let first = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
    a.submit(&first, Duration::from_secs(10)).await?;
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    faults.manifest_gets.lock().unwrap().clear();
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
    assert_eq!(reference.transaction, frozen.identity());
    let manifest = ManifestNamingScheme::V2
        .manifest_path(&Path::from_absolute_path(&body_path)?, reference.version);
    assert!(
        !faults.manifest_gets.lock().unwrap().contains(&manifest),
        "successful body preparation must not GET its just-committed manifest"
    );

    // An independent session must still verify the stored identity. This also
    // proves that our interception observes actual reads of that manifest.
    let reopened = fault_store::storage_with_cache(&body_path, faults.clone(), 0).await?;
    let selected = mixed::load_body(&reopened, &reference, &reference.table_instance).await?;
    assert_eq!(
        selected.read_transaction().await?.unwrap().uuid,
        reference.transaction
    );
    assert_eq!(selected.count_rows(None).await?, 1);
    assert!(faults.manifest_gets.lock().unwrap().contains(&manifest));
    let mut wrong = reference.clone();
    wrong.transaction = Uuid::new_v4().to_string();
    assert!(mixed::load_body(&reopened, &wrong, &wrong.table_instance)
        .await
        .is_err());

    // The later annotation survives the first cut and an annotation-only cut
    // continues to select the same immutable body snapshot.
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
    a.submit(&update, Duration::from_secs(10)).await?;
    drop(prepared);
    pressure::publish_frozen(&c, &a, &narrow, &body, frozen).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    let update_only = a.freeze(Uuid::new_v4()).await?.unwrap();
    faults.manifest_gets.lock().unwrap().clear();
    let writes = faults
        .body_write_attempts
        .load(std::sync::atomic::Ordering::SeqCst);
    let prepared = update_only
        .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    assert!(
        faults.manifest_gets.lock().unwrap().is_empty(),
        "annotation-only preparation must not reopen the unchanged body snapshot"
    );
    assert_eq!(
        faults
            .body_write_attempts
            .load(std::sync::atomic::Ordering::SeqCst),
        writes,
        "annotation-only preparation must not write body files"
    );
    let unchanged: BodySnapshot = serde_json::from_str(
        &prepared
            .annotations()
            .merge()
            .transaction
            .transaction_properties
            .as_ref()
            .unwrap()[BODY_SNAPSHOT_PROPERTY],
    )?;
    assert_eq!(unchanged, reference);
    drop(prepared);
    pressure::publish_frozen(&c, &a, &narrow, &body, update_only).await?;
    assert!(
        faults.manifest_gets.lock().unwrap().is_empty(),
        "annotation-only publication must keep the original body reference without reopening it"
    );
    let latest = annotation_publication::read(&c, &narrow, None).await?;
    assert_eq!(BodySnapshot::read(&latest).await?, Some(reference));
    assert_eq!(latest.count_rows(None).await?, 1);
    assert_eq!(a.retained_body_frames(), (0, 0));
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
