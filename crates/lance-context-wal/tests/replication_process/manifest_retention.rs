//! More real Lance publications than the catalog history cap, with bounded
//! retained metadata, a lost commit reply, takeover and six-process restart.
use super::*;
use lance_context_wal::replication::publication::{AdvancePublication, RetireManifestHistory};

const CATALOG: &[u64] = &[1, 2, 3];
const SOURCE: &[u64] = &[4, 5, 6];
const LAST_VERSION: u64 = 162;

struct Run {
    handler: Arc<Managed>,
    token: u64,
}

impl Run {
    fn command(&mut self, group: u64, request: impl FnOnce(u64) -> Request) -> Result<Outcome> {
        self.token += 1;
        let mut c = self.handler.cluster.lock().unwrap();
        let members = if group == 10 { CATALOG } else { SOURCE };
        let leader = c.elect(members[0], group, members)?;
        c.command(leader, request(self.token))?;
        c.pump(members)?;
        c.outcome(self.token).context("missing command completion")
    }

    fn query(&mut self, group: u64, query: Query) -> Result<Value> {
        self.token += 1;
        let mut c = self.handler.cluster.lock().unwrap();
        let members = if group == 10 { CATALOG } else { SOURCE };
        let leader = c.elect(members[0], group, members)?;
        c.command(leader, Request::Query(group, self.token, query))?;
        c.pump(members)?;
        let Some(ReadOutcome::Observed { value, .. }) = c.read_outcome(self.token) else {
            anyhow::bail!("missing quorum observation");
        };
        Ok(value)
    }

    fn view(&mut self) -> Result<Table> {
        let table = self.handler.scope.authority.table;
        let Value::Publisher(Some(view)) = self.query(10, Query::Publisher { table })? else {
            anyhow::bail!("missing publisher view");
        };
        Ok(*view)
    }

    fn restart(&self) -> Result<()> {
        let mut c = self.handler.cluster.lock().unwrap();
        // Every member must reopen from its own installed image and durable log.
        for id in 1..=6 {
            let group = if id <= 3 { 10 } else { 20 };
            c.peers
                .get_mut(&id)
                .unwrap()
                .call(Request::PrepareCheckpoint(group, Uuid::new_v4()))?;
            c.command(id, Request::InstallCheckpoint)?;
        }
        c.pump(&[1, 2, 3, 4, 5, 6])?;
        c.queue.clear();
        c.peers.clear();
        c.completions.clear();
        c.reads.clear();
        for id in 1..=6 {
            c.start(id)?;
        }
        Ok(())
    }

    fn confirm_and_advance(&mut self, record: &ManifestRecord) -> Result<()> {
        let authority = self.handler.scope.authority.clone();
        let table = authority.table;
        let confirmation = ConfirmPublication {
            authority: authority.clone(),
            incarnation: self.handler.scope.incarnation,
            transaction: record.transaction,
            plan_index: record.plan_index,
            prepared_index: record.prepared_index,
            descriptor_digest: record.descriptor_digest,
            target: record.version,
            manifest_identity: record.identity,
        };
        let Outcome::Committed(Applied::Publisher(PublisherReply::Published(published))) =
            self.command(10, |token| Request::ConfirmPublication(token, confirmation))?
        else {
            anyhow::bail!("publication confirmation missing");
        };
        let Value::ClosePermit(Some(permit)) = self.query(
            10,
            Query::ClosePermit {
                table,
                transaction: record.transaction,
                plan_index: record.plan_index,
                source_group: 20,
            },
        )?
        else {
            anyhow::bail!("source closure permit missing");
        };
        assert!(matches!(
            self.command(20, |token| Request::CloseSource(token, *permit))?,
            Outcome::Committed(Applied::Source(SourceReply::Closed { .. }))
        ));
        let Value::ClosedProof(Some(proof)) = self.query(20, Query::ClosedProof { table })? else {
            anyhow::bail!("durable source closure proof missing");
        };
        let request = AdvancePublication {
            authority,
            published_index: published.index,
            proofs: BTreeMap::from([(20, *proof)]),
        };
        let Outcome::Committed(Applied::Publisher(PublisherReply::Advanced(advanced))) =
            self.command(10, |token| Request::AdvancePublication(token, request))?
        else {
            anyhow::bail!("catalog advancement missing");
        };
        assert_eq!(advanced.publication.manifest.as_ref(), record);
        Ok(())
    }
}

pub(super) async fn qualify(uri: &str, handler: Arc<Managed>) -> Result<()> {
    let mut run = Run {
        handler,
        token: 1_000_000,
    };
    let bootstrap = run.handler.read(Some(1))?.context("bootstrap missing")?;
    let mut old_reservation = None;
    for version in 3..=LAST_VERSION {
        if version == 90 {
            let authority = takeover(&run.handler)?;
            let mut scope = run.handler.scope.clone();
            scope.authority = authority;
            run.handler = Arc::new(Managed {
                cluster: run.handler.cluster.clone(),
                base: run.handler.base.clone(),
                scope,
                tokens: Arc::new(AtomicU64::new(9_000_000)),
                lose_reply: Arc::new(AtomicBool::new(false)),
                pause: None,
            });
        }
        let view = run.view()?;
        let authority = view.authority;
        let table = authority.table;
        let predecessor = run.handler.read(None)?.context("head missing")?;
        assert_eq!(predecessor.version, version - 1);
        let rows = batch(&[version as i64 + 2])?;
        let mut wire = Vec::new();
        {
            let mut encoder = arrow_ipc::writer::FileWriter::try_new(&mut wire, &rows.schema())?;
            encoder.write(&rows)?;
            encoder.finish()?;
        }
        let input = Batch::new(
            table,
            1,
            Uuid::from_u128(10_000 + version as u128),
            1,
            Bytes::from(wire),
        );
        let Outcome::Committed(Applied::Accepted { index }) =
            run.command(20, |token| Request::Propose(20, token, input.clone()))?
        else {
            anyhow::bail!("source append missing at version {version}");
        };
        let transaction = Uuid::new_v4();
        let plan = PlanRequest {
            authority: authority.clone(),
            plan: Plan {
                transaction,
                read_version: version - 1,
                next_group: 10,
                sources: BTreeMap::from([(
                    20,
                    SourceRange {
                        after: view.progress.checkpoint[&20],
                        through: index,
                        records: vec![(index, input.meta.clone())],
                    },
                )]),
            },
        };
        let Outcome::Committed(Applied::Publisher(PublisherReply::Planned { index: plan_index })) =
            run.command(10, |token| Request::Plan(token, plan))?
        else {
            anyhow::bail!("plan missing at version {version}");
        };
        let Value::Pin(Some(permit)) = run.query(
            10,
            Query::Pin {
                table,
                transaction,
                plan_index,
                source_group: 20,
            },
        )?
        else {
            anyhow::bail!("source pin permit missing");
        };
        assert!(matches!(
            run.command(20, |token| Request::Pin(token, *permit))?,
            Outcome::Committed(Applied::Source(SourceReply::Pinned { .. }))
        ));
        let recovered = {
            let mut c = run.handler.cluster.lock().unwrap();
            let source = c.elect(4, 20, SOURCE)?;
            let Response::Record(Some(record)) = c
                .peers
                .get_mut(&source)
                .unwrap()
                .call(Request::Read(20, index))?
            else {
                anyhow::bail!("pinned source body missing");
            };
            record
        };
        assert_eq!(recovered, input);
        let decoded = lance_context_wal::ipc::decode(recovered.body, rows.schema())?;
        let base = Arc::new(
            lance::dataset::builder::DatasetBuilder::from_uri(uri)
                .with_commit_handler(Arc::new(run.handler.reader()?))
                .load()
                .await?,
        );
        let staged = InsertBuilder::new(base.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(decoded)
            .await?;
        let Operation::Append { fragments } = staged.operation else {
            anyhow::bail!("not append fragments");
        };
        let descriptors =
            Descriptors::lance_fragments(Bytes::from(serde_json::to_vec(&fragments)?))?;
        let prepare = PrepareRequest {
            authority: authority.clone(),
            transaction,
            plan_index,
            descriptors: descriptors.clone(),
        };
        let Outcome::Committed(Applied::Publisher(PublisherReply::Prepared {
            index: prepared_index,
        })) = run.command(10, |token| Request::Prepare(token, prepare))?
        else {
            anyhow::bail!("preparation missing");
        };
        let mut tx = Transaction::new(version - 1, Operation::Append { fragments }, None);
        tx.uuid = transaction.to_string();
        run.handler = Arc::new(Managed {
            cluster: run.handler.cluster.clone(),
            base: run.handler.base.clone(),
            scope: ReserveManifest {
                authority,
                incarnation: run.handler.scope.incarnation,
                transaction,
                plan_index,
                prepared_index,
                descriptor_digest: descriptors.digest,
                predecessor: predecessor.identity,
                target: version,
                object: predecessor.object.clone(),
            },
            tokens: Arc::new(AtomicU64::new(10_000_000 + version * 1000)),
            lose_reply: Arc::new(AtomicBool::new(version == 64)),
            pause: None,
        });
        let committed = commit(base, tx, run.handler.clone()).await;
        if version == 64 {
            assert!(committed.is_err(), "injected lost reply was not observed");
            run.restart()?;
        } else {
            committed?;
        }
        let selected = run
            .handler
            .read(Some(version))?
            .context("durable selection missing")?;
        assert_eq!(selected.transaction, transaction);
        if version == 3 {
            let mut request = run.handler.scope.clone();
            request.object = selected.object.clone();
            old_reservation = Some((
                request,
                std::fs::read(Path::new(uri).join(&selected.object.path))?,
            ));
        }
        run.confirm_and_advance(&selected)?;
        if version >= 16 && version % 8 == 0 {
            let retirement = RetireManifestHistory {
                authority: run.handler.scope.authority.clone(),
                incarnation: run.handler.scope.incarnation,
                retain_from: version - 7,
            };
            let Outcome::Committed(Applied::Publisher(PublisherReply::ManifestHistory(boundary))) =
                run.command(10, |token| {
                    Request::RetireManifestHistory(token, retirement)
                })?
            else {
                anyhow::bail!("history retirement missing");
            };
            assert_eq!(boundary.retained_from, version - 7);
        }
        let view = run.view()?;
        let manifests = view.manifests.context("manifest directory missing")?;
        assert!(manifests.versions.len() <= 16);
        assert_eq!(manifests.head, version);
        assert_eq!(manifests.versions[&1], bootstrap);
        assert!(
            view.progress.pending.is_none()
                && view.progress.prepared.is_none()
                && view.progress.published.is_none()
        );
    }
    run.restart()?;
    let (old, bytes) = old_reservation.context("original version-three request missing")?;
    assert_ne!(old.authority, run.handler.scope.authority);
    let old_path = Path::new(uri).join(&old.object.path);
    let Outcome::Committed(Applied::Publisher(PublisherReply::ManifestRetired(boundary))) =
        run.command(10, |token| Request::ReserveManifest(token, old))?
    else {
        anyhow::bail!("retired retry must resolve before stale authority rejection");
    };
    assert_eq!(boundary.retained_from, 153);
    let table = run.handler.scope.authority.table;
    assert_eq!(
        run.query(
            10,
            Query::Manifest {
                table,
                version: Some(3)
            }
        )?,
        Value::ManifestRetired(boundary)
    );
    assert_eq!(
        std::fs::read(old_path)?,
        bytes,
        "metadata retirement must not delete or replace immutable objects"
    );
    assert_eq!(run.handler.read(Some(1))?.as_ref(), Some(&bootstrap));
    let current = run.handler.read(None)?.context("head lost after restart")?;
    assert_eq!(current.version, LAST_VERSION);
    {
        let mut c = run.handler.cluster.lock().unwrap();
        for id in 1..=3 {
            let Response::Publisher(Some(view)) = c
                .peers
                .get_mut(&id)
                .unwrap()
                .call(Request::Publisher(table))?
            else {
                anyhow::bail!("replica catalog missing");
            };
            let manifests = view
                .manifests
                .context("replica manifest directory missing")?;
            assert_eq!(manifests.versions.len(), 11);
            assert_eq!(manifests.versions[&LAST_VERSION], current);
        }
    }
    let latest = lance::dataset::builder::DatasetBuilder::from_uri(uri)
        .with_commit_handler(run.handler)
        .load()
        .await?;
    assert_eq!(latest.version().version, LAST_VERSION);
    verify_dataset(latest, &(0..=LAST_VERSION as i64 + 2).collect::<Vec<_>>()).await?;
    Ok(())
}
