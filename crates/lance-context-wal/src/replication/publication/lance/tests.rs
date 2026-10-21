use super::*;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::ObjectStoreExt;
use std::{collections::BTreeMap, sync::Mutex};

#[derive(Debug)]
struct Catalog {
    cut: query::Cut,
    records: Mutex<BTreeMap<u64, Arc<ManifestRecord>>>,
    override_read: Mutex<Option<query::Outcome>>,
    reservations: Mutex<Vec<ReserveManifest>>,
}
#[async_trait::async_trait]
impl ManifestCatalog for Catalog {
    async fn read_manifest(&self, _: Uuid, version: Option<u64>) -> Result<query::Outcome> {
        if let Some(value) = self.override_read.lock().unwrap().clone() {
            return Ok(value);
        }
        let records = self.records.lock().unwrap();
        Ok(query::Outcome::Observed {
            cut: self.cut.clone(),
            value: query::Value::Manifest(version.map_or_else(
                || records.last_key_value().map(|(_, r)| r.clone()),
                |v| records.get(&v).cloned(),
            )),
        })
    }
    async fn reserve_manifest(&self, request: ReserveManifest) -> Result<Outcome> {
        self.reservations.lock().unwrap().push(request);
        Ok(Outcome::Unknown)
    }
}
fn fixture() -> Result<(ManagedManifest, Arc<Catalog>, ObjectStore)> {
    let layout = Layout {
        cluster: Uuid::new_v4(),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3])]),
        buckets: vec![10],
    };
    let scope = ManifestCommit {
        authority: Authority {
            cluster: layout.cluster,
            layout: layout.fingerprint()?,
            group: 10,
            table: Uuid::new_v4(),
            owner: Uuid::new_v4(),
            epoch: 1,
            grant_index: 1,
        },
        incarnation: Uuid::new_v4(),
        transaction: Uuid::new_v4(),
        plan_index: 20,
        prepared_index: 21,
        descriptor_digest: [1; 32],
        predecessor: Uuid::new_v4(),
        target: 4,
    };
    let records = (1..=3)
        .map(|v| {
            (
                v,
                Arc::new(ManifestRecord {
                    version: v,
                    identity: manifest::identity(&scope.authority, scope.incarnation, v, 10 + v),
                    object: ManifestObject {
                        path: format!("_versions/{v}.manifest-{}", Uuid::new_v4()),
                        bytes: 64,
                        v2_names: false,
                    },
                    transaction: Uuid::new_v4(),
                    index: 10 + v,
                    plan_index: 2,
                    prepared_index: 3,
                    descriptor_digest: [1; 32],
                }),
            )
        })
        .collect();
    let catalog = Arc::new(Catalog {
        cut: query::Cut {
            cluster: layout.cluster,
            layout: scope.authority.layout,
            group: 10,
            term: 1,
            quorum_index: 100,
            applied: 100,
        },
        records: Mutex::new(records),
        override_read: Mutex::new(None),
        reservations: Mutex::new(vec![]),
    });
    Ok((
        ManagedManifest::new(catalog.clone(), Path::from("dataset"), &layout, scope)?,
        catalog,
        ObjectStore::memory(),
    ))
}

#[tokio::test]
async fn catalog_reads_reject_foreign_scope_identity_and_unknown_outcomes() -> Result<()> {
    let (h, c, store) = fixture()?;
    assert_eq!(h.resolve_latest_location(&h.base, &store).await?.version, 3);
    assert!(h
        .resolve_latest_location(&Path::from("foreign"), &store)
        .await
        .is_err());
    for case in 0..5 {
        let mut cut = c.cut.clone();
        let mut record = c.records.lock().unwrap()[&3].as_ref().clone();
        match case {
            0 => cut.group = 20,
            1 => cut.layout = [0; 32],
            2 => cut.applied = cut.quorum_index - 1,
            3 => record.identity = Uuid::new_v4(),
            _ => record.object.path = "../foreign/3.manifest".into(),
        }
        *c.override_read.lock().unwrap() = Some(query::Outcome::Observed {
            cut,
            value: query::Value::Manifest(Some(Arc::new(record))),
        });
        assert!(h.resolve_latest_location(&h.base, &store).await.is_err());
    }
    *c.override_read.lock().unwrap() = Some(query::Outcome::Unknown);
    assert!(h
        .version_exists(&h.base, 3, store.inner.as_ref(), ManifestNamingScheme::V1)
        .await
        .is_err());
    *c.override_read.lock().unwrap() = Some(query::Outcome::Observed {
        cut: c.cut.clone(),
        value: query::Value::ManifestRetired(super::super::HistoryBoundary {
            incarnation: h.binding.incarnation,
            bootstrap_version: 1,
            retained_from: 3,
        }),
    });
    assert!(h
        .version_exists(&h.base, 2, store.inner.as_ref(), ManifestNamingScheme::V1)
        .await
        .is_err());
    *c.override_read.lock().unwrap() = None;
    assert!(
        !h.version_exists(&h.base, 4, store.inner.as_ref(), ManifestNamingScheme::V1)
            .await?
    );
    Ok(())
}

#[tokio::test]
async fn history_uses_selected_objects_and_refuses_gaps_or_unsafe_collection() -> Result<()> {
    let (h, c, store) = fixture()?;
    let ghost = Path::from(format!("dataset/_versions/2.manifest-{}", Uuid::new_v4()));
    store
        .inner
        .put(&ghost, Bytes::from_static(b"unselected").into())
        .await?;
    let history: Vec<_> = h
        .list_manifest_locations_since(&h.base, &store, 1)
        .try_collect()
        .await?;
    assert_eq!(
        history.iter().map(|r| r.version).collect::<Vec<_>>(),
        vec![3, 2]
    );
    for item in &history {
        assert_ne!(item.path, ghost);
        assert_eq!(
            item.path,
            h.location(&c.records.lock().unwrap()[&item.version])?.path
        );
    }
    c.records.lock().unwrap().remove(&2);
    assert!(h
        .list_manifest_locations_since(&h.base, &store, 1)
        .try_collect::<Vec<_>>()
        .await
        .is_err());
    assert!(h
        .list_manifest_locations(&h.base, &store, false)
        .try_collect::<Vec<_>>()
        .await
        .is_err());
    assert!(h
        .list_detached_manifest_locations(&h.base, &store)
        .try_collect::<Vec<_>>()
        .await
        .is_err());
    assert!(h.forget_version(&h.base, 3, "identity").await.is_err());
    assert!(h.delete(&h.base).await.is_err());
    Ok(())
}

#[tokio::test]
async fn fixed_target_rejects_before_upload_and_ambiguous_reservation_retains_staging() -> Result<()>
{
    let (h, c, store) = fixture()?;
    let mut manifest = Manifest::new(
        Default::default(),
        Arc::new(vec![]),
        Default::default(),
        Default::default(),
    );
    let transaction = Transaction {
        inner: lance_table::format::pb::Transaction {
            uuid: h.scope.as_ref().unwrap().transaction.to_string(),
            ..Default::default()
        },
    };
    let writer = lance_table::io::commit::write_manifest_file_to_path;
    for (version, tx) in [(5, Some(transaction.clone())), (4, None)] {
        manifest.version = version;
        assert!(matches!(
            h.commit(
                &mut manifest,
                None,
                &h.base,
                &store,
                writer,
                ManifestNamingScheme::V1,
                tx
            )
            .await,
            Err(CommitError::CommitConflict)
        ));
    }
    assert!(c.reservations.lock().unwrap().is_empty());
    assert!(store
        .inner
        .list(None)
        .try_collect::<Vec<_>>()
        .await?
        .is_empty());
    manifest.version = 4;
    assert!(h
        .commit(
            &mut manifest,
            None,
            &h.base,
            &store,
            writer,
            ManifestNamingScheme::V1,
            Some(transaction)
        )
        .await
        .is_err());
    let request = c.reservations.lock().unwrap()[0].clone();
    assert_eq!(request.target, 4);
    let path = Path::parse(format!("{}/{}", h.base, request.object.path))?;
    assert_eq!(store.inner.head(&path).await?.size, request.object.bytes);
    Ok(())
}

#[tokio::test]
async fn reader_opens_bootstrap_without_a_publication_plan_and_cannot_upload() -> Result<()> {
    let (writer, c, store) = fixture()?;
    c.records.lock().unwrap().retain(|version, _| *version == 1);
    let layout = Layout {
        cluster: writer.binding.cluster,
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3])]),
        buckets: vec![10],
    };
    let reader = ManagedManifest::reader(
        c.clone(),
        writer.base.clone(),
        &layout,
        writer.binding.table,
        writer.binding.incarnation,
    )?;
    assert!(reader.scope.is_none());
    assert_eq!(
        reader
            .resolve_latest_location(&reader.base, &store)
            .await?
            .version,
        1
    );
    let versions: Vec<_> = reader
        .list_manifest_locations_since(&reader.base, &store, 0)
        .try_collect()
        .await?;
    assert_eq!(versions.len(), 1);
    assert_eq!(versions[0].version, 1);
    let mut manifest = Manifest::new(
        Default::default(),
        Arc::new(vec![]),
        Default::default(),
        Default::default(),
    );
    manifest.version = writer.scope.as_ref().unwrap().target;
    let transaction = Transaction {
        inner: lance_table::format::pb::Transaction {
            uuid: writer.scope.as_ref().unwrap().transaction.to_string(),
            ..Default::default()
        },
    };
    assert!(reader
        .commit(
            &mut manifest,
            None,
            &reader.base,
            &store,
            lance_table::io::commit::write_manifest_file_to_path,
            ManifestNamingScheme::V1,
            Some(transaction)
        )
        .await
        .is_err());
    assert!(c.reservations.lock().unwrap().is_empty());
    assert!(store
        .inner
        .list(None)
        .try_collect::<Vec<_>>()
        .await?
        .is_empty());
    let foreign = ManagedManifest::reader(
        c,
        reader.base.clone(),
        &layout,
        Uuid::new_v4(),
        reader.binding.incarnation,
    )?;
    assert!(foreign
        .resolve_latest_location(&foreign.base, &store)
        .await
        .is_err());
    Ok(())
}
