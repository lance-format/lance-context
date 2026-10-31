//! Existing inline native encoding preserves the exact legacy rewrite and cut.
use super::*;

pub(super) async fn verify(
    base: &lance::Dataset,
    storage: &Storage,
    pending: &p::PendingPlan,
    legacy: &p::Descriptors,
) -> Result<()> {
    assert_eq!(legacy.format, 2);
    let selection = annotations::Selection {
        table: fixture_binding().table,
        incarnation: fixture_binding().incarnation,
        pending: pending.clone(),
    };
    let inline = annotations::compaction::descriptors(
        base,
        serde_json::from_slice(&legacy.bytes)?,
        &selection,
        storage,
        p::MAX_DESCRIPTOR_BYTES as u64,
    )
    .await?;
    assert_eq!(inline.format, 4);
    let expected =
        materializer::stage::annotation_compaction_transaction(base, pending, legacy).await?;
    assert_eq!(
        annotations::resolve_transaction(base, &inline, &selection, storage).await?,
        expected
    );
    for alteration in 0..8 {
        let mut changed = expected.clone();
        match alteration {
            0 => changed.uuid = Uuid::new_v4().to_string(),
            1 => changed.read_version += 1,
            2 => changed.tag = Some("foreign".into()),
            3 => {
                Arc::make_mut(changed.transaction_properties.as_mut().unwrap())
                    .remove(lance_context_core::annotation_staging::RESULTS_PROPERTY);
            }
            4..=6 => {
                let properties = Arc::make_mut(changed.transaction_properties.as_mut().unwrap());
                let (mut scope, through): (AnnotationScope, u64) =
                    serde_json::from_str(&properties["lance-context.annotation-cut"])?;
                match alteration {
                    4 => scope.shard += 1,
                    5 => scope.source_group += 1,
                    _ => scope.table_instance = Uuid::new_v4().to_string(),
                }
                properties.insert(
                    "lance-context.annotation-cut".into(),
                    serde_json::to_string(&(scope, through))?,
                );
            }
            _ => {
                changed.operation = lance::dataset::transaction::Operation::Append {
                    fragments: Vec::new(),
                };
            }
        }
        let descriptors = p::Descriptors::lance_annotations(Bytes::from(
            pb::Transaction::from(&changed).encode_to_vec(),
        ))?;
        assert!(
            annotations::resolve_transaction(base, &descriptors, &selection, storage)
                .await
                .is_err(),
            "accepted invalid native maintenance alteration {alteration}"
        );
    }
    Ok(())
}
