//! Actual Lance writer/reader adapter over the six-process catalog fixture.
//! This qualifies the durable index, not production transport or coordinated GC.
use super::*;
#[path = "manifest_retention.rs"]
mod manifest_retention;
#[path = "producer_lifecycle.rs"]
pub(super) mod producer_lifecycle;
#[path = "source_closure.rs"]
mod source_closure;
use lance_context_wal::replication::publication::{
    Authority, BootstrapManifest, ConfirmPublication, ManifestObject, ManifestRecord, Prepared,
    ReserveManifest,
};
use object_store::path::Path as ObjectPath;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Mutex,
};

pub(super) async fn bootstrap(
    c: &mut Cluster,
    leader: u64,
    dataset: &Dataset,
    authority: &Authority,
) -> Result<(Uuid, Arc<ManifestRecord>)> {
    let location = dataset.manifest_location();
    let filename = location.path.filename().context("manifest filename")?;
    let incarnation = Uuid::from_u128(930);
    let genesis = dataset
        .read_transaction_by_version(1)
        .await?
        .context("actual genesis transaction")?;
    c.command(
        leader,
        Request::BootstrapManifest(
            2009,
            BootstrapManifest {
                authority: authority.clone(),
                incarnation,
                request: Uuid::from_u128(931),
                version: 1,
                object: ManifestObject {
                    path: format!("_versions/{filename}"),
                    bytes: location.size.context("known written size")?,
                    v2_names: location.naming_scheme == ManifestNamingScheme::V2,
                },
                transaction: Uuid::parse_str(&genesis.uuid)?,
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::ManifestReserved(record)))) =
        c.outcome(2009)
    else {
        anyhow::bail!("manifest bootstrap missing");
    };
    Ok((incarnation, record))
}

pub(super) struct Attempt {
    pub uri: String,
    pub base: Dataset,
    pub tx: Transaction,
    pub prepared: Arc<Prepared>,
    pub authority: Authority,
    pub bootstrap: (Uuid, Arc<ManifestRecord>),
    pub input: Batch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Normal,
    LostReply,
    Takeover,
    Retention,
    ProducerSlots,
}

#[derive(Default)]
struct Pause {
    uploaded: tokio::sync::Notify,
    resume: tokio::sync::Notify,
    path: Mutex<Option<ObjectPath>>,
}

#[derive(Clone)]
struct Managed {
    cluster: Arc<Mutex<Cluster>>,
    base: ObjectPath,
    scope: ReserveManifest,
    tokens: Arc<AtomicU64>,
    lose_reply: Arc<AtomicBool>,
    pause: Option<Arc<Pause>>,
}
impl std::fmt::Debug for Managed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ManagedCatalogFixture")
    }
}
fn error(error: impl std::fmt::Display) -> lance::Error {
    lance::Error::io(error.to_string())
}
impl Managed {
    fn read(&self, version: Option<u64>) -> lance::Result<Option<Arc<ManifestRecord>>> {
        let mut c = self.cluster.lock().unwrap();
        let leader = c.elect(1, 10, &[1, 2, 3]).map_err(error)?;
        let token = self.tokens.fetch_add(1, Ordering::Relaxed);
        c.command(
            leader,
            Request::Query(
                10,
                token,
                Query::Manifest {
                    table: self.scope.authority.table,
                    version,
                },
            ),
        )
        .map_err(error)?;
        c.pump(&[1, 2, 3]).map_err(error)?;
        match c.read_outcome(token) {
            Some(ReadOutcome::Observed {
                value: Value::Manifest(record),
                ..
            }) => Ok(record),
            other => Err(error(format!(
                "manifest quorum read unavailable: {other:?}"
            ))),
        }
    }
    fn location(&self, record: &ManifestRecord) -> ManifestLocation {
        ManifestLocation {
            version: record.version,
            path: ObjectPath::parse(format!("{}/{}", self.base, record.object.path))
                .expect("validated relative manifest path"),
            size: Some(record.object.bytes),
            naming_scheme: if record.object.v2_names {
                ManifestNamingScheme::V2
            } else {
                ManifestNamingScheme::V1
            },
            e_tag: None,
            identity: Some(record.identity.to_string()),
        }
    }
}

impl Managed {
    fn reader(
        &self,
    ) -> lance::Result<lance_context_wal::replication::publication::lance::ManagedManifest> {
        let layout: Layout = {
            let c = self.cluster.lock().unwrap();
            serde_json::from_slice(
                &std::fs::read(c.root.path().join("layout.json")).map_err(error)?,
            )
            .map_err(error)?
        };
        lance_context_wal::replication::publication::lance::ManagedManifest::reader(
            Arc::new(self.clone()),
            self.base.clone(),
            &layout,
            self.scope.authority.table,
            self.scope.incarnation,
        )
        .map_err(error)
    }
    fn adapter(
        &self,
    ) -> lance::Result<lance_context_wal::replication::publication::lance::ManagedManifest> {
        let layout: Layout = {
            let c = self.cluster.lock().unwrap();
            serde_json::from_slice(
                &std::fs::read(c.root.path().join("layout.json")).map_err(error)?,
            )
            .map_err(error)?
        };
        lance_context_wal::replication::publication::lance::ManagedManifest::new(
            Arc::new(self.clone()),
            self.base.clone(),
            &layout,
            (&self.scope).into(),
        )
        .map_err(error)
    }
}
#[async_trait::async_trait]
impl lance_context_wal::replication::publication::lance::ManifestCatalog for Managed {
    async fn read_manifest(&self, table: Uuid, version: Option<u64>) -> Result<ReadOutcome> {
        ensure!(table == self.scope.authority.table, "foreign fixture table");
        let mut c = self.cluster.lock().unwrap();
        let leader = c.elect(1, 10, &[1, 2, 3])?;
        let token = self.tokens.fetch_add(1, Ordering::Relaxed);
        c.command(
            leader,
            Request::Query(10, token, Query::Manifest { table, version }),
        )?;
        c.pump(&[1, 2, 3])?;
        c.read_outcome(token)
            .context("manifest quorum read unavailable")
    }
    async fn reserve_manifest(&self, request: ReserveManifest) -> Result<Outcome> {
        if let Some(pause) = &self.pause {
            *pause.path.lock().unwrap() = Some(ObjectPath::parse(format!(
                "{}/{}",
                self.base, request.object.path
            ))?);
            pause.uploaded.notify_one();
            pause.resume.notified().await;
        }
        let outcome = {
            let mut c = self.cluster.lock().unwrap();
            let leader = c.elect(1, 10, &[1, 2, 3])?;
            let token = self.tokens.fetch_add(1, Ordering::Relaxed);
            c.command(leader, Request::ReserveManifest(token, request))?;
            c.pump(&[1, 2, 3])?;
            c.outcome(token).context("reservation completion missing")?
        };
        if self.lose_reply.swap(false, Ordering::SeqCst) {
            anyhow::bail!("injected lost reply after durable catalog commit");
        }
        Ok(outcome)
    }
}

// The fixture supplies transport/faults; all Lance behavior delegates to the
// reusable library implementation exercised by the real-process qualifications.
#[async_trait::async_trait]
#[allow(clippy::too_many_arguments)]
impl CommitHandler for Managed {
    async fn resolve_latest_location(
        &self,
        base: &ObjectPath,
        store: &lance_io::object_store::ObjectStore,
    ) -> lance::Result<ManifestLocation> {
        self.adapter()?.resolve_latest_location(base, store).await
    }
    async fn resolve_version_location(
        &self,
        base: &ObjectPath,
        version: u64,
        store: &dyn object_store::ObjectStore,
    ) -> lance::Result<ManifestLocation> {
        self.adapter()?
            .resolve_version_location(base, version, store)
            .await
    }
    async fn version_exists(
        &self,
        base: &ObjectPath,
        version: u64,
        store: &dyn object_store::ObjectStore,
        naming: ManifestNamingScheme,
    ) -> lance::Result<bool> {
        self.adapter()?
            .version_exists(base, version, store, naming)
            .await
    }
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base: &ObjectPath,
        store: &lance_io::object_store::ObjectStore,
        writer: ManifestWriter,
        naming: ManifestNamingScheme,
        transaction: Option<ManifestTransaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        self.adapter()?
            .commit(manifest, indices, base, store, writer, naming, transaction)
            .await
    }
    fn list_detached_manifest_locations<'a>(
        &self,
        base: &ObjectPath,
        store: &'a lance_io::object_store::ObjectStore,
    ) -> futures::stream::BoxStream<'a, lance::Result<ManifestLocation>> {
        match self.adapter() {
            Ok(h) => h.list_detached_manifest_locations(base, store),
            Err(e) => Box::pin(futures::stream::once(async { Err(e) })),
        }
    }
    fn list_manifest_locations<'a>(
        &self,
        base: &ObjectPath,
        store: &'a lance_io::object_store::ObjectStore,
        sorted: bool,
    ) -> futures::stream::BoxStream<'a, lance::Result<ManifestLocation>> {
        match self.adapter() {
            Ok(h) => h.list_manifest_locations(base, store, sorted),
            Err(e) => Box::pin(futures::stream::once(async { Err(e) })),
        }
    }
    fn list_manifest_locations_since<'a>(
        &self,
        base: &ObjectPath,
        store: &'a lance_io::object_store::ObjectStore,
        version: u64,
    ) -> futures::stream::BoxStream<'a, lance::Result<ManifestLocation>> {
        match self.adapter() {
            Ok(h) => h.list_manifest_locations_since(base, store, version),
            Err(e) => Box::pin(futures::stream::once(async { Err(e) })),
        }
    }
    async fn forget_version(
        &self,
        base: &ObjectPath,
        version: u64,
        identity: &str,
    ) -> lance::Result<()> {
        self.adapter()?
            .forget_version(base, version, identity)
            .await
    }
    async fn delete(&self, base: &ObjectPath) -> lance::Result<()> {
        self.adapter()?.delete(base).await
    }
}

pub(super) async fn publish_and_recover(c: Cluster, attempt: Attempt, mode: Mode) -> Result<()> {
    let base_path = attempt
        .base
        .manifest_location()
        .path
        .as_ref()
        .rsplit_once("/_versions/")
        .context("dataset base path")?
        .0;
    let base_path = ObjectPath::parse(base_path)?;
    let cluster = Arc::new(Mutex::new(c));
    let pause = (mode == Mode::Takeover).then(|| Arc::new(Pause::default()));
    let mut handler = Arc::new(Managed {
        cluster: cluster.clone(),
        base: base_path,
        scope: ReserveManifest {
            authority: attempt.authority.clone(),
            incarnation: attempt.bootstrap.0,
            transaction: attempt.prepared.transaction,
            plan_index: attempt.prepared.plan_index,
            prepared_index: attempt.prepared.index,
            descriptor_digest: attempt.prepared.descriptors.digest,
            predecessor: attempt.bootstrap.1.identity,
            target: 2,
            object: attempt.bootstrap.1.object.clone(),
        },
        tokens: Arc::new(AtomicU64::new(6000)),
        lose_reply: Arc::new(AtomicBool::new(mode == Mode::LostReply)),
        pause: pause.clone(),
    });
    if let Some(pause) = pause {
        let old = tokio::spawn(commit(
            Arc::new(attempt.base),
            attempt.tx.clone(),
            handler.clone(),
        ));
        tokio::time::timeout(Duration::from_secs(20), pause.uploaded.notified()).await?;
        let old_path = pause
            .path
            .lock()
            .unwrap()
            .clone()
            .context("paused staging path")?;
        assert!(Path::new(&attempt.uri)
            .join("_versions")
            .join(old_path.filename().unwrap())
            .is_file());
        let authority = takeover(&handler)?;
        pause.resume.notify_one();
        let old_result = tokio::time::timeout(Duration::from_secs(20), old).await??;
        assert!(
            old_result.is_err(),
            "old owner selected a path after takeover"
        );
        assert!(handler.read(Some(2))?.is_none());
        assert_eq!(handler.read(None)?.unwrap().version, 1);
        let mut scope = handler.scope.clone();
        scope.authority = authority;
        handler = Arc::new(Managed {
            cluster: cluster.clone(),
            base: handler.base.clone(),
            scope,
            tokens: Arc::new(AtomicU64::new(10000)),
            lose_reply: Arc::new(AtomicBool::new(false)),
            pause: None,
        });
        let base = lance::dataset::builder::DatasetBuilder::from_uri(&attempt.uri)
            .with_commit_handler(Arc::new(handler.reader()?))
            .load()
            .await?;
        commit(Arc::new(base), attempt.tx, handler.clone()).await?;
        assert_ne!(
            handler.location(&handler.read(Some(2))?.unwrap()).path,
            old_path
        );
    } else {
        let result = commit(Arc::new(attempt.base), attempt.tx, handler.clone()).await;
        if mode == Mode::LostReply {
            assert!(
                result.is_err(),
                "lost response should remain ambiguous to the caller"
            );
        } else {
            result?;
        }
    }
    let selected = handler.read(Some(2))?.context("missing committed target")?;
    assert!(selected.object.path.contains(".manifest-"));
    let confirmation = ConfirmPublication {
        authority: handler.scope.authority.clone(),
        incarnation: handler.scope.incarnation,
        transaction: selected.transaction,
        plan_index: selected.plan_index,
        prepared_index: selected.prepared_index,
        descriptor_digest: selected.descriptor_digest,
        target: selected.version,
        manifest_identity: selected.identity,
    };
    {
        let mut c = cluster.lock().unwrap();
        let leader = c.elect(1, 10, &[1, 2, 3])?;
        c.command(
            leader,
            Request::ConfirmPublication(500_000, confirmation.clone()),
        )?;
        c.pump(&[1, 2, 3])?;
        // Lose the confirmation reply before any caller consumes it. Recovery
        // must discover the outcome from durable state, including its index.
        let before = c.completions.len();
        c.completions
            .retain(|completion| completion.request != 500_000);
        assert_eq!(before - c.completions.len(), 1);
        c.peers
            .get_mut(&leader)
            .unwrap()
            .call(Request::PrepareCheckpoint(10, Uuid::new_v4()))?;
        c.command(leader, Request::InstallCheckpoint)?;
        c.pump(&[1, 2, 3])?;
        c.queue.clear();
        c.peers.clear();
        for id in 1..=6 {
            c.start(id)?;
        }
        for id in 4..=6 {
            verify_records(
                c.peers.get_mut(&id).unwrap(),
                20,
                std::slice::from_ref(&attempt.input),
            )?;
        }
    }
    assert_eq!(handler.read(Some(2))?.as_deref(), Some(selected.as_ref()));
    {
        let mut c = cluster.lock().unwrap();
        let leader = c.elect(1, 10, &[1, 2, 3])?;
        c.command(
            leader,
            Request::Query(
                10,
                500_001,
                Query::Publisher {
                    table: confirmation.authority.table,
                },
            ),
        )?;
        c.pump(&[1, 2, 3])?;
        let Some(ReadOutcome::Observed {
            value: Value::Publisher(Some(view)),
            ..
        }) = c.read_outcome(500_001)
        else {
            anyhow::bail!("publication outcome query missing");
        };
        let published = view
            .progress
            .published
            .context("durable outcome missing after restart")?;
        assert_eq!(published.manifest, selected);
        assert!(published.index > selected.index);
        assert_eq!(
            view.progress
                .pending
                .context("source selection lost")?
                .index,
            selected.plan_index
        );
        assert_eq!(
            view.progress.prepared.context("descriptors lost")?,
            attempt.prepared
        );
        assert!(view.progress.checkpoint.is_empty());
        c.command(leader, Request::ConfirmPublication(500_002, confirmation))?;
        c.pump(&[1, 2, 3])?;
        assert_eq!(
            c.outcome(500_002),
            Some(Outcome::Committed(Applied::Publisher(
                PublisherReply::Published(published)
            )))
        );
    }
    source_closure::close_and_restart(
        &cluster,
        handler.scope.authority.table,
        &selected,
        &attempt.input,
    )?;
    if mode == Mode::ProducerSlots {
        producer_lifecycle::reuse_after_retirement(&cluster, &attempt.input)?;
    }
    // Both version lookup and latest lookup are strict catalog reads. A raw
    // URI reader cannot discover this unique path and is not a managed reader.
    let latest = lance::dataset::builder::DatasetBuilder::from_uri(&attempt.uri)
        .with_commit_handler(Arc::new(handler.reader()?))
        .load()
        .await?;
    assert_eq!(latest.version().version, 2);
    assert_eq!(
        latest.manifest_location().identity,
        Some(selected.identity.to_string())
    );
    let transaction = latest
        .read_transaction_by_version(2)
        .await?
        .context("missing actual transaction")?;
    assert_eq!(transaction.uuid, attempt.prepared.transaction.to_string());
    verify_dataset(latest, &[0, 1, 2, 3, 4]).await?;
    assert!(handler.read(Some(3))?.is_none());
    assert_eq!(handler.read(None)?.unwrap().version, 2);
    if mode == Mode::Retention {
        manifest_retention::qualify(&attempt.uri, handler).await?;
    }
    Ok(())
}

async fn commit(
    base: Arc<Dataset>,
    tx: Transaction,
    handler: Arc<Managed>,
) -> lance::Result<Dataset> {
    CommitBuilder::new(base)
        .with_commit_handler(handler)
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(tx)
        .await
}

fn takeover(handler: &Managed) -> Result<Authority> {
    let mut c = handler.cluster.lock().unwrap();
    let leader = c.elect(1, 10, &[1, 2, 3])?;
    c.command(
        leader,
        Request::Query(
            10,
            100_000,
            Query::Publisher {
                table: handler.scope.authority.table,
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Publisher(Some(view)),
        ..
    }) = c.read_outcome(100_000)
    else {
        anyhow::bail!("takeover view missing");
    };
    c.command(
        leader,
        Request::Acquire(
            100_001,
            Acquire {
                table: view.authority.table,
                destination: view.destination,
                ingest: view.ingest,
                owner: Uuid::from_u128(940),
                request: Uuid::from_u128(941),
                expected_epoch: view.authority.epoch,
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(authority)))) =
        c.outcome(100_001)
    else {
        anyhow::bail!("takeover did not commit");
    };
    Ok(authority)
}
