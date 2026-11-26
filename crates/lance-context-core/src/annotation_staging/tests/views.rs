//! Physical deletes here are limited to disposable fixtures with explicitly
//! controlled readers/writers. Local roots alone cannot authorize distributed GC.
use super::*;

async fn publish(
    staging: &AnnotationStaging,
    position: u64,
) -> Result<(FrozenAnnotations, PreparedAnnotations)> {
    staging
        .apply(
            position - 1,
            position,
            "T",
            &TurnAnnotationPatch {
                score: Set(position as f32),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    Ok((frozen, prepared))
}

async fn versions(staging: &AnnotationStaging) -> Result<Vec<u64>> {
    Ok(staging
        .protected_bases()
        .await?
        .datasets()
        .map(Dataset::version_id)
        .collect())
}

#[tokio::test]
async fn bases_remain_protected_until_both_reader_and_preparation_release() -> Result<()> {
    let root = tempfile::tempdir()?;
    let staging = AnnotationStaging::new(
        fixture(&root.path().join("turns.lance")).await?,
        scope(),
        0,
        Limits::default(),
    )?;
    let initial = staging.observation().await.installed_base;
    assert!(!initial.immutable_body);
    assert_eq!(initial.source, scope());
    assert_eq!(initial.rows, Some(3));
    drop(publish(&staging, 1).await?);
    let held = staging.capture("T").await?;
    let (frozen, prepared) = publish(&staging, 2).await?;
    assert_eq!(versions(&staging).await?, vec![2, 3]);
    drop(frozen);
    drop(held);
    assert_eq!(versions(&staging).await?, vec![2, 3]);
    drop(prepared);
    assert_eq!(versions(&staging).await?, vec![3]);

    // A single snapshot pins its finite set, without pausing reads/publication.
    // Callers cannot accumulate old snapshots to evade admission accounting.
    let protected = staging.protected_bases().await?;
    assert!(staging.protected_bases().await.is_err());
    drop(publish(&staging, 3).await?);
    assert_eq!(current(&staging, "T").await?.score, Some(3.0));
    let old = protected.datasets().next().unwrap();
    assert_eq!(old.version_id(), 3);
    assert_eq!(read_base(old, "T", 4096).await?.unwrap().score, Some(2.0));
    drop(protected);
    assert_eq!(versions(&staging).await?, vec![4]);
    Ok(())
}

async fn reclaim_local_fixture(
    staging: &AnnotationStaging,
    path: &std::path::Path,
) -> Result<usize> {
    let roots = staging.protected_bases().await?;
    let live: BTreeSet<_> = roots
        .datasets()
        .flat_map(|base| base.manifest().fragments.iter())
        .flat_map(|fragment| fragment.referenced_lance_files())
        .map(|file| file.path.clone())
        .collect();
    for file in std::fs::read_dir(path.join("data"))? {
        let file = file?;
        if !live.contains(file.file_name().to_str().unwrap()) {
            std::fs::remove_file(file.path())?;
        }
    }
    assert!(live
        .iter()
        .all(|file| path.join("data").join(file).is_file()));
    Ok(live.len())
}

#[tokio::test]
async fn tracked_cold_reader_allows_repeated_physical_drain_then_releases_old_data() -> Result<()> {
    for stable in [false, true] {
        let root = tempfile::tempdir()?;
        let path = root.path().join("turns.lance");
        let original = fixture_with_row_ids(&path, stable).await?;
        let body_files = body_paths(&original)?;
        let staging = AnnotationStaging::new(original, scope(), 0, Limits::default())?;
        drop(publish(&staging, 1).await?);
        let held = staging.capture("T").await?;
        for position in 2..=8 {
            drop(publish(&staging, position).await?);
            assert_eq!(versions(&staging).await?, vec![2, position + 1]);
            assert_eq!(reclaim_local_fixture(&staging, &path).await?, 4);
            assert_eq!(held.resolve().await?.unwrap().score, Some(1.0));
            assert_eq!(current(&staging, "T").await?.score, Some(position as f32));
            assert_eq!(staging.observation().await.retained_entries, 0);
        }
        drop(held);
        assert_eq!(versions(&staging).await?, vec![9]);
        assert_eq!(reclaim_local_fixture(&staging, &path).await?, 3);
        let latest = Dataset::open(path.to_str().unwrap()).await?;
        assert_eq!(body_paths(&latest)?, body_files);
        // Reopen with a cold Session and verify every byte after physical GC.
        let batches = latest
            .scan()
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        for batch in batches {
            let bodies = batch
                .column_by_name("body")
                .unwrap()
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap();
            for body in bodies.iter().flatten() {
                assert_eq!(body.len(), 1 << 20);
                assert!(body
                    .iter()
                    .enumerate()
                    .all(|(i, b)| *b == ((i * 31 + i / 17) % 251) as u8));
            }
        }
    }
    Ok(())
}
