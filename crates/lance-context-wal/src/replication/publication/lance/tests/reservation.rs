//! Real manifest objects with deterministic reservation response loss.
use super::*;
use std::collections::VecDeque;

#[derive(Debug)]
enum ReplyMode {
    Commit,
    CommitUnknown,
    CommitThenRejected,
    Unknown,
    TransportError,
    Rejected,
}

#[derive(Debug)]
struct Uncertain {
    inner: Arc<Catalog>,
    replies: Mutex<VecDeque<ReplyMode>>,
}

#[async_trait::async_trait]
impl ManifestCatalog for Uncertain {
    async fn read_manifest(&self, table: Uuid, version: Option<u64>) -> Result<query::Outcome> {
        self.inner.read_manifest(table, version).await
    }

    async fn reserve_manifest(&self, request: ReserveManifest) -> Result<Outcome> {
        self.inner
            .reservations
            .lock()
            .unwrap()
            .push(request.clone());
        let mode = self.replies.lock().unwrap().pop_front().unwrap();
        match mode {
            ReplyMode::Unknown => return Ok(Outcome::Unknown),
            ReplyMode::TransportError => anyhow::bail!("original response transport lost"),
            ReplyMode::Rejected => {
                return Ok(Outcome::Committed(Applied::Publisher(
                    super::super::super::Reply::Rejected(super::super::super::Rejection::Authority),
                )))
            }
            _ => {}
        }
        let record = Arc::new(ManifestRecord {
            version: request.target,
            identity: manifest::identity(
                &request.authority,
                request.incarnation,
                request.target,
                99,
            ),
            object: request.object,
            transaction: request.transaction,
            index: 99,
            plan_index: request.plan_index,
            prepared_index: request.prepared_index,
            descriptor_digest: request.descriptor_digest,
        });
        self.inner
            .records
            .lock()
            .unwrap()
            .insert(request.target, record.clone());
        if matches!(mode, ReplyMode::CommitThenRejected) {
            return Ok(Outcome::Committed(Applied::Publisher(
                super::super::super::Reply::Rejected(super::super::super::Rejection::Authority),
            )));
        }
        Ok(if matches!(mode, ReplyMode::CommitUnknown) {
            Outcome::Unknown
        } else {
            Outcome::Committed(Applied::Publisher(
                super::super::super::Reply::ManifestReserved(record),
            ))
        })
    }
}

async fn commit(handler: &ManagedManifest, store: &ObjectStore) -> Result<ManifestLocation> {
    let mut manifest = Manifest::new(
        Default::default(),
        Arc::new(vec![]),
        Default::default(),
        Default::default(),
    );
    let scope = handler.scope.as_ref().unwrap();
    manifest.version = scope.target;
    let transaction = Transaction {
        inner: lance_table::format::pb::Transaction {
            uuid: scope.transaction.to_string(),
            ..Default::default()
        },
    };
    handler
        .commit(
            &mut manifest,
            None,
            &handler.base,
            store,
            lance_table::io::commit::write_manifest_file_to_path,
            ManifestNamingScheme::V1,
            Some(transaction),
        )
        .await
        .map_err(|error| anyhow::anyhow!("manifest commit: {error:?}"))
}

async fn exercise(modes: Vec<ReplyMode>, success: bool, reservations: usize) -> Result<()> {
    let (mut handler, catalog, store) = fixture()?;
    handler.catalog = Arc::new(Uncertain {
        inner: catalog.clone(),
        replies: Mutex::new(modes.into()),
    });
    let result = commit(&handler, &store).await;
    assert_eq!(result.is_ok(), success, "{result:?}");
    let requests = catalog.reservations.lock().unwrap().clone();
    assert_eq!(requests.len(), reservations);
    for request in &requests {
        assert_eq!(request, &requests[0]);
    }
    let objects = store.inner.list(None).try_collect::<Vec<_>>().await?;
    assert_eq!(
        objects.len(),
        1,
        "response loss must not upload another manifest"
    );
    let expected = Path::parse(format!("{}/{}", handler.base, requests[0].object.path))?;
    assert_eq!(objects[0].location, expected);
    if let Ok(selected) = result {
        assert_eq!(selected.path, expected);
        assert_eq!(selected.version, handler.scope.as_ref().unwrap().target);
    }
    Ok(())
}

#[tokio::test]
async fn reserved_manifest_is_observed_after_lost_reply_without_resubmission() -> Result<()> {
    exercise(vec![ReplyMode::CommitUnknown], true, 1).await
}

#[tokio::test]
async fn absent_manifest_resubmits_one_identical_reservation_without_upload() -> Result<()> {
    exercise(vec![ReplyMode::Unknown, ReplyMode::Commit], true, 2).await
}

#[tokio::test]
async fn lost_second_reservation_reply_resolves_the_original_object() -> Result<()> {
    exercise(vec![ReplyMode::Unknown, ReplyMode::CommitUnknown], true, 2).await
}

#[tokio::test]
async fn unavailable_second_reservation_remains_unknown_and_retains_object() -> Result<()> {
    exercise(
        vec![ReplyMode::Unknown, ReplyMode::TransportError],
        false,
        2,
    )
    .await
}

#[tokio::test]
async fn initial_reservation_rejection_does_not_resubmit() -> Result<()> {
    exercise(vec![ReplyMode::Rejected], false, 1).await
}

#[tokio::test]
async fn transport_loss_can_resolve_one_exact_reservation() -> Result<()> {
    exercise(vec![ReplyMode::TransportError, ReplyMode::Commit], true, 2).await
}

#[tokio::test]
async fn rejected_second_reservation_still_observes_late_original_commit() -> Result<()> {
    exercise(
        vec![ReplyMode::Unknown, ReplyMode::CommitThenRejected],
        true,
        2,
    )
    .await
}

#[tokio::test]
async fn normal_reservation_needs_no_manifest_observation() -> Result<()> {
    let (mut handler, catalog, store) = fixture()?;
    *catalog.override_read.lock().unwrap() = Some(query::Outcome::Unknown);
    handler.catalog = Arc::new(Uncertain {
        inner: catalog.clone(),
        replies: Mutex::new(vec![ReplyMode::Commit].into()),
    });
    assert_eq!(commit(&handler, &store).await?.version, 4);
    assert_eq!(catalog.reservations.lock().unwrap().len(), 1);
    Ok(())
}

#[tokio::test]
async fn foreign_manifest_observation_cannot_authorize_resubmission() -> Result<()> {
    let (mut handler, catalog, store) = fixture()?;
    *catalog.override_read.lock().unwrap() = Some(query::Outcome::Observed {
        cut: catalog.cut.clone(),
        value: query::Value::Manifest(Some(catalog.records.lock().unwrap()[&3].clone())),
    });
    handler.catalog = Arc::new(Uncertain {
        inner: catalog.clone(),
        replies: Mutex::new(vec![ReplyMode::Unknown].into()),
    });
    assert!(commit(&handler, &store).await.is_err());
    assert_eq!(catalog.reservations.lock().unwrap().len(), 1);
    Ok(())
}

#[tokio::test]
async fn unavailable_manifest_observation_does_not_resubmit() -> Result<()> {
    let (mut handler, catalog, store) = fixture()?;
    *catalog.override_read.lock().unwrap() = Some(query::Outcome::Unknown);
    handler.catalog = Arc::new(Uncertain {
        inner: catalog.clone(),
        replies: Mutex::new(vec![ReplyMode::Unknown].into()),
    });
    assert!(commit(&handler, &store).await.is_err());
    assert_eq!(catalog.reservations.lock().unwrap().len(), 1);
    assert_eq!(
        store.inner.list(None).try_collect::<Vec<_>>().await?.len(),
        1
    );
    Ok(())
}
