//! Inline native encoding preserves exact index remaps and the annotation cut.
use super::*;

pub(super) async fn verify(
    base: &lance::Dataset,
    storage: &Storage,
    pending: &p::PendingPlan,
    inline: &p::Descriptors,
) -> Result<()> {
    assert_eq!(inline.format, 4);
    let selection = annotations::Selection {
        table: fixture_binding().table,
        incarnation: fixture_binding().incarnation,
        pending: pending.clone(),
    };
    let expected = lance::dataset::transaction::Transaction::try_from(pb::Transaction::decode(
        inline.bytes.clone(),
    )?)?;
    let lance::dataset::transaction::Operation::Rewrite {
        rewritten_indices, ..
    } = &expected.operation
    else {
        anyhow::bail!("expected indexed compaction");
    };
    assert!(!rewritten_indices.is_empty());
    assert_eq!(
        annotations::resolve_transaction(base, inline, &selection, storage).await?,
        expected
    );
    for alteration in 0..13 {
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
            7 => {
                changed.operation = lance::dataset::transaction::Operation::Append {
                    fragments: Vec::new(),
                };
            }
            _ => {
                let lance::dataset::transaction::Operation::Rewrite {
                    rewritten_indices, ..
                } = &mut changed.operation
                else {
                    unreachable!();
                };
                match alteration {
                    8 => rewritten_indices.clear(),
                    9 => rewritten_indices[0].old_id = Uuid::new_v4(),
                    10 => rewritten_indices[0].new_id = Uuid::nil(),
                    11 => rewritten_indices[0].new_index_files = None,
                    _ => rewritten_indices[0].new_index_version += 1,
                }
            }
        }
        let descriptors = p::Descriptors::lance_annotations(Bytes::from(
            pb::Transaction::try_from(&changed)?.encode_to_vec(),
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
