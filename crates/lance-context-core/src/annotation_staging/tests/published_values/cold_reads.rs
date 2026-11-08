//! Repeated reads reuse only values from their captured immutable snapshot.
use super::*;

#[tokio::test]
async fn repeated_cold_point_reads_avoid_data_io_within_the_same_snapshot() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let staging = Arc::new(AnnotationStaging::new(base, scope(), 0, Limits::default())?);
    let expected = current(&staging, "T").await?;
    assert!(staging.capture("missing").await?.resolve().await?.is_none());
    let before = throttle.ops.load(Ordering::Relaxed);
    assert!(
        before > 0,
        "first reads must use the actual native data store"
    );

    throttle.frozen.store(true, Ordering::Relaxed);
    let mut reads = tokio::spawn({
        let staging = staging.clone();
        async move {
            for _ in 0..8 {
                assert_eq!(current(&staging, "T").await?, expected);
                assert!(staging.capture("missing").await?.resolve().await?.is_none());
            }
            Ok::<_, Error>(())
        }
    });
    let completed = tokio::time::timeout(Duration::from_secs(1), &mut reads).await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let reused = completed.is_ok();
    // Join the original task even in the negative control; release native IO first.
    let result = match completed {
        Ok(result) => result,
        Err(_) => reads.await,
    };
    result.map_err(|e| invalid(e.to_string()))??;
    let after = throttle.ops.load(Ordering::Relaxed);
    println!(
        "cold_point_repeat reads=16 additional_data_operations={}",
        after - before
    );
    assert!(reused, "repeated point reads waited for blocked data IO");
    assert_eq!(after, before);
    let observed = staging.observation().await;
    assert_eq!(observed.applied, 0);
    assert_eq!(observed.retained_bytes, 0);
    assert_eq!(observed.retained_entries, 0);
    Ok(())
}
