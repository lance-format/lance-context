//! Real RF3 catalog/source publication; this does not qualify an annotation API.
use super::*;
use crate::{annotation_publication as annotations, materializer::Storage};
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    Float32Array, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{builder::DatasetBuilder, WriteParams};
use lance_context_api::annotation::{AnnotationFieldUpdate::Set, TurnAnnotationPatch};
use lance_context_core::annotation_staging::{
    AnnotationStaging, Limits, Mutation, Scope as AnnotationScope,
};
use lance_table::{
    format::pb,
    io::commit::{ConditionalPutCommitHandler, ManifestNamingScheme},
};
use prost::Message;

mod large;

pub(super) fn one_shard() -> Arc<Layout> {
    let mut value = (*layout()).clone();
    value.groups.remove(&20);
    value.buckets = vec![10];
    Arc::new(value)
}

fn patch_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]))
}

async fn command(c: &network::Cluster, command: Command) -> Result<Applied> {
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(command)))
        .await?;
    let Reply::Commit(Outcome::Committed(applied)) = response.reply else {
        bail!("test command outcome unknown");
    };
    Ok(applied)
}

async fn query(c: &network::Cluster, query: query::Query) -> Result<query::Value> {
    let response = c
        .client
        .request_routed(Request::Query { group: 10, query })
        .await?;
    let Reply::Query(query::Outcome::Observed { value, .. }) = response.reply else {
        bail!("test quorum query unavailable");
    };
    Ok(value)
}

pub(super) async fn state(c: &network::Cluster) -> Result<Box<p::Table>> {
    let query::Value::Publisher(Some(state)) = query(
        c,
        query::Query::Publisher {
            table: binding().table,
        },
    )
    .await?
    else {
        bail!("test catalog binding missing");
    };
    Ok(state)
}

pub(super) async fn read(
    c: &network::Cluster,
    storage: &Storage,
    version: Option<u64>,
) -> Result<Arc<Dataset>> {
    let handler = p::lance::ManagedManifest::reader(
        Arc::new(c.client.manifest_catalog()),
        storage.base.clone(),
        &one_shard(),
        binding().table,
        binding().incarnation,
    )?;
    let mut builder = DatasetBuilder::from_uri(&storage.uri).with_commit_handler(Arc::new(handler));
    if let Some(version) = version {
        builder = builder.with_version(version);
    }
    Ok(Arc::new(builder.load().await?))
}

fn staging(base: Arc<Dataset>, materialized: u64) -> Result<AnnotationStaging> {
    Ok(AnnotationStaging::new(
        base,
        AnnotationScope {
            table_instance: binding().incarnation.to_string(),
            source_group: 10,
            shard: 0,
        },
        materialized,
        Limits::default(),
    )?)
}

async fn seed(c: &network::Cluster, storage: &Storage) -> Result<()> {
    let contract = IngestContract {
        schema: arrow_schema_fingerprint(&patch_schema())?,
        policy: WritePolicy::TurnAnnotations,
        ..binding().contract()?
    };
    seed_contract(c, storage, contract).await
}

pub(super) async fn seed_contract(
    c: &network::Cluster,
    storage: &Storage,
    contract: IngestContract,
) -> Result<()> {
    let mut labels = ListBuilder::new(StringBuilder::new());
    for _ in 0..2 {
        labels.values().append_value("original");
        labels.append(true);
    }
    let labels = labels.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::LargeBinary, false),
        Field::new("labels", labels.data_type().clone(), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    let payload: Vec<_> = (0..256 << 10)
        .map(|i| ((i * 31 + i / 17) % 251) as u8)
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["T", "U"])),
            Arc::new(LargeBinaryArray::from_iter_values([payload.as_slice(); 2])),
            Arc::new(labels),
            Arc::new(StringArray::from(vec!["pending"; 2])),
            Arc::new(Float32Array::from(vec![None; 2])),
        ],
    )?;
    let base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &storage.uri,
        Some(WriteParams {
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            enable_stable_row_ids: true,
            max_rows_per_file: 1,
            max_rows_per_group: 1,
            ..Default::default()
        }),
    )
    .await?;
    seed_base(c, contract, &base).await
}

async fn seed_base(c: &network::Cluster, contract: IngestContract, base: &Dataset) -> Result<()> {
    let Applied::Publisher(p::Reply::Acquired(authority)) = command(
        c,
        Command::Acquire(p::Acquire {
            table: binding().table,
            destination: binding().destination,
            ingest: contract,
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: 0,
        }),
    )
    .await?
    else {
        bail!("test acquire rejected");
    };
    let location = base.manifest_location();
    assert!(matches!(
        command(
            c,
            Command::Bootstrap(p::BootstrapManifest {
                authority,
                incarnation: binding().incarnation,
                request: Uuid::new_v4(),
                version: base.version_id(),
                object: p::ManifestObject {
                    path: format!("_versions/{}", location.path.filename().unwrap()),
                    bytes: location.size.unwrap(),
                    v2_names: location.naming_scheme == ManifestNamingScheme::V2,
                },
                transaction: Uuid::parse_str(&base.read_transaction().await?.unwrap().uuid)?,
            })
        )
        .await?,
        Applied::Publisher(p::Reply::ManifestReserved(_))
    ));
    let layout = one_shard();
    assert!(matches!(
        command(
            c,
            Command::Producer(producer::Command {
                cluster: layout.cluster,
                layout: layout.fingerprint()?,
                group: 10,
                table: binding().table,
                slot: 0,
                action: producer::Action::Open {
                    expected_generation: 0,
                    operation: Uuid::new_v4(),
                    bucket: 0
                },
            })
        )
        .await?,
        Applied::Producer(producer::Reply::Opened { .. })
    ));
    Ok(())
}

async fn append(c: &network::Cluster, sequence: u64) -> Result<(u64, BatchMeta)> {
    // Two turns in one IPC record; the test exercises an actual opaque WAL
    // source. This fixture schema is not the public mutation wire contract.
    let schema = patch_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["T", "U"])),
            Arc::new(StringArray::from(vec![
                if sequence == 1 {
                    Some("reviewed")
                } else {
                    None
                };
                2
            ])),
            Arc::new(Float32Array::from(vec![
                if sequence == 2 {
                    Some(0.9)
                } else {
                    None
                };
                2
            ])),
        ],
    )?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    let batch = Batch::new(
        binding().table,
        0,
        producer::producer_id(0, 1),
        sequence,
        bytes.into(),
    );
    let meta = batch.meta.clone();
    let response = c
        .client
        .request_routed(Request::Append { group: 10, batch })
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply else {
        bail!("test append rejected");
    };
    Ok((index, meta))
}

async fn apply(
    c: &network::Cluster,
    staging: &AnnotationStaging,
    record: &(u64, BatchMeta),
) -> Result<()> {
    let response = c
        .client
        .request_routed(Request::Fetch {
            group: 10,
            index: record.0,
        })
        .await?;
    assert!(matches!(&response.reply, Reply::Record(Some(meta)) if meta == &record.1));
    let batches = ipc::decode(response.body.clone(), patch_schema())?;
    let mut mutations = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let status = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let mut patch = TurnAnnotationPatch::default();
            if status.is_valid(row) {
                patch.status = Set(status.value(row).to_owned());
            }
            if scores.is_valid(row) {
                patch.score = Set(scores.value(row));
            }
            mutations.push((ids.value(row), patch));
        }
    }
    staging
        .apply_batch(
            staging.observation().await.applied,
            record.0,
            &mutations
                .iter()
                .map(|(turn, patch)| Mutation { turn, patch })
                .collect::<Vec<_>>(),
        )
        .await?;
    Ok(())
}

async fn select(c: &network::Cluster, record: &(u64, BatchMeta)) -> Result<annotations::Selection> {
    select_range(c, vec![record.clone()], record.0, Uuid::new_v4()).await
}

pub(super) async fn select_range(
    c: &network::Cluster,
    records: Vec<(u64, BatchMeta)>,
    through: u64,
    transaction: Uuid,
) -> Result<annotations::Selection> {
    let view = state(c).await?;
    let plan = p::Plan {
        transaction,
        read_version: view
            .manifests
            .as_ref()
            .context("test manifest missing")?
            .head,
        sources: BTreeMap::from([(
            10,
            p::SourceRange {
                after: view.progress.checkpoint.get(&10).copied().unwrap_or(0),
                through,
                records,
            },
        )]),
        next_group: 10,
    };
    let Applied::Publisher(p::Reply::Planned { index }) = command(
        c,
        Command::Plan(p::PlanRequest {
            authority: view.authority,
            plan: plan.clone(),
        }),
    )
    .await?
    else {
        bail!("test plan rejected");
    };
    let query::Value::Pin(Some(permit)) = query(
        c,
        query::Query::Pin {
            table: binding().table,
            transaction: plan.transaction,
            plan_index: index,
            source_group: 10,
        },
    )
    .await?
    else {
        bail!("test pin missing");
    };
    assert!(matches!(
        command(c, Command::Pin(*permit)).await?,
        Applied::Source(_)
    ));
    Ok(annotations::Selection {
        table: binding().table,
        incarnation: binding().incarnation,
        pending: p::PendingPlan { index, plan },
    })
}

pub(super) async fn save(
    c: &network::Cluster,
    selection: &annotations::Selection,
    descriptors: p::Descriptors,
) -> Result<p::lance::ManifestCommit> {
    let view = state(c).await?;
    let Applied::Publisher(p::Reply::Prepared { index }) = command(
        c,
        Command::Prepare(p::PrepareRequest {
            authority: view.authority.clone(),
            transaction: selection.pending.plan.transaction,
            plan_index: selection.pending.index,
            descriptors: descriptors.clone(),
        }),
    )
    .await?
    else {
        bail!("test prepare rejected");
    };
    Ok(p::lance::ManifestCommit {
        authority: view.authority,
        incarnation: selection.incarnation,
        transaction: selection.pending.plan.transaction,
        plan_index: selection.pending.index,
        prepared_index: index,
        descriptor_digest: descriptors.digest,
        predecessor: view.manifests.as_ref().unwrap().versions
            [&selection.pending.plan.read_version]
            .identity,
        target: selection.pending.plan.target_version()?,
    })
}

async fn confirm(c: &network::Cluster) -> Result<Arc<p::Published>> {
    let view = state(c).await?;
    let pending = view.progress.pending.as_ref().unwrap();
    let prepared = view.progress.prepared.as_ref().unwrap();
    let target = pending.plan.target_version()?;
    let selected = &view.manifests.as_ref().unwrap().versions[&target];
    let Applied::Publisher(p::Reply::Published(published)) = command(
        c,
        Command::Confirm(p::ConfirmPublication {
            authority: view.authority.clone(),
            incarnation: binding().incarnation,
            transaction: pending.plan.transaction,
            plan_index: pending.index,
            prepared_index: prepared.index,
            descriptor_digest: prepared.descriptors.digest,
            target,
            manifest_identity: selected.identity,
        }),
    )
    .await?
    else {
        bail!("test confirm rejected");
    };
    Ok(published)
}

pub(super) async fn finish(c: &network::Cluster) -> Result<()> {
    let published = confirm(c).await?;
    let view = state(c).await?;
    let pending = view.progress.pending.as_ref().unwrap();
    let query::Value::ClosePermit(Some(permit)) = query(
        c,
        query::Query::ClosePermit {
            table: binding().table,
            transaction: pending.plan.transaction,
            plan_index: pending.index,
            source_group: 10,
        },
    )
    .await?
    else {
        bail!("test close permit missing");
    };
    assert!(matches!(
        command(c, Command::Close(*permit)).await?,
        Applied::Source(_)
    ));
    let query::Value::ClosedProof(Some(proof)) = query(
        c,
        query::Query::ClosedProof {
            table: binding().table,
        },
    )
    .await?
    else {
        bail!("test closure proof missing");
    };
    assert!(matches!(
        command(
            c,
            Command::Advance(p::AdvancePublication {
                authority: view.authority,
                published_index: published.index,
                proofs: BTreeMap::from([(10, *proof)])
            })
        )
        .await?,
        Applied::Publisher(p::Reply::Advanced(_))
    ));
    Ok(())
}

#[derive(Debug)]
pub(super) struct LostReply(pub(super) Arc<dyn p::lance::ManifestCatalog>);
#[async_trait::async_trait]
impl p::lance::ManifestCatalog for LostReply {
    async fn read_manifest(&self, table: Uuid, version: Option<u64>) -> Result<query::Outcome> {
        self.0.read_manifest(table, version).await
    }
    async fn reserve_manifest(&self, request: p::ReserveManifest) -> Result<Outcome> {
        assert!(matches!(
            self.0.reserve_manifest(request).await?,
            Outcome::Committed(Applied::Publisher(p::Reply::ManifestReserved(_)))
        ));
        Ok(Outcome::Unknown)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn frozen_native_update_recovers_unknown_publication_and_preserves_wal_suffix() -> Result<()>
{
    restart_at(Stop::Selected).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn frozen_native_update_reuses_prepared_files_after_restart_before_publication() -> Result<()>
{
    restart_at(Stop::Prepared).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn frozen_native_update_recovers_confirmed_cut_before_source_closure() -> Result<()> {
    restart_at(Stop::Confirmed).await
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stop {
    Prepared,
    Selected,
    Confirmed,
}

async fn restart_at(stop: Stop) -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = Storage::resolve(
        root.path()
            .join("annotations.lance")
            .to_str()
            .unwrap()
            .into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    seed(&c, &storage).await?;
    let base = read(&c, &storage, None).await?;
    let view = staging(base.clone(), 0)?;
    let first = append(&c, 1).await?;
    apply(&c, &view, &first).await?;
    let selection = select(&c, &first).await?;
    let frozen = view
        .freeze(selection.pending.plan.transaction)
        .await?
        .unwrap();
    let prepared = frozen
        .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let descriptors = annotations::descriptors(&prepared, &selection)?;
    assert_eq!(
        annotations::transaction(&base, &descriptors, &selection)?,
        prepared.merge().transaction
    );
    rejects_foreign_preparation(&base, &descriptors, &selection)?;
    let commit = save(&c, &selection, descriptors.clone()).await?;
    let second = append(&c, 2).await?;
    apply(&c, &view, &second).await?;
    if stop != Stop::Prepared {
        assert!(annotations::publish(
            Arc::new(LostReply(Arc::new(c.client.manifest_catalog()))),
            &one_shard(),
            &storage,
            commit.clone(),
            &selection,
            &descriptors,
            base.clone()
        )
        .await
        .is_err());
    }
    if stop == Stop::Confirmed {
        confirm(&c).await?;
    }
    assert_eq!(view.observation().await.materialized, 0);
    // No confirmation/closure on an unknown response. The selected manifest
    // exists, but both source records and the frozen generation remain.
    assert_eq!(
        state(&c).await?.progress.published.is_some(),
        stop == Stop::Confirmed
    );
    assert_eq!(
        view.capture("T").await?.resolve().await?.unwrap().score,
        Some(0.9)
    );
    drop((view, prepared, frozen, base, commit));
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    let mut selected = read(&c, &storage, None).await?;
    let mut recovered = state(&c).await?;
    assert!(recovered.progress.checkpoint.is_empty());
    assert_eq!(
        recovered.progress.published.is_some(),
        stop == Stop::Confirmed
    );
    let saved = recovered.progress.prepared.as_ref().unwrap();
    assert_eq!(saved.descriptors, descriptors);
    let recovered_selection = annotations::Selection {
        table: binding().table,
        incarnation: binding().incarnation,
        pending: recovered
            .progress
            .pending
            .as_ref()
            .unwrap()
            .as_ref()
            .clone(),
    };
    let original_base = read(
        &c,
        &storage,
        Some(recovered_selection.pending.plan.read_version),
    )
    .await?;
    if stop == Stop::Prepared {
        assert_eq!(selected.version_id(), 1);
        let commit = p::lance::ManifestCommit {
            authority: recovered.authority.clone(),
            incarnation: recovered_selection.incarnation,
            transaction: recovered_selection.pending.plan.transaction,
            plan_index: recovered_selection.pending.index,
            prepared_index: saved.index,
            descriptor_digest: saved.descriptors.digest,
            predecessor: recovered.manifests.as_ref().unwrap().versions
                [&recovered_selection.pending.plan.read_version]
                .identity,
            target: recovered_selection.pending.plan.target_version()?,
        };
        selected = Arc::new(
            annotations::publish(
                Arc::new(c.client.manifest_catalog()),
                &one_shard(),
                &storage,
                commit,
                &recovered_selection,
                &saved.descriptors,
                original_base.clone(),
            )
            .await?,
        );
        recovered = state(&c).await?;
    }
    assert_eq!(selected.version_id(), 2);
    let saved = recovered.progress.prepared.as_ref().unwrap();
    let cut = annotations::selected_cut(
        &selected,
        &original_base,
        &saved.descriptors,
        &recovered_selection,
        &recovered.manifests.as_ref().unwrap().versions[&2],
        &storage,
    )
    .await?;
    assert_eq!(cut, first.0);
    // Recover from the exact selected base/cut, without rerunning preparation
    // or replaying the already materialized patch. Its source is still pinned.
    let view = staging(selected.clone(), cut)?;
    let query::Value::Accepted(page) = query(
        &c,
        query::Query::AcceptedBucket {
            table: binding().table,
            bucket: 0,
            after: cut,
            limit: 128,
        },
    )
    .await?
    else {
        bail!("recovery source suffix unavailable");
    };
    assert_eq!(page.entries, vec![second.clone()]);
    let second = page.entries.into_iter().next().unwrap();
    apply(&c, &view, &second).await?;
    let current = view.capture("T").await?.resolve().await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    assert!(current.labels.contains("original"));
    finish(&c).await?;
    assert_eq!(state(&c).await?.progress.checkpoint[&10], first.0);
    assert_eq!(view.capture("T").await?.resolve().await?.unwrap(), current);
    let removed = c
        .client
        .request_routed(Request::Fetch {
            group: 10,
            index: first.0,
        })
        .await?;
    assert!(matches!(removed.reply, Reply::Record(None)));
    let retained = c
        .client
        .request_routed(Request::Fetch {
            group: 10,
            index: second.0,
        })
        .await?;
    assert!(matches!(retained.reply, Reply::Record(Some(_))));
    let selection = select(&c, &second).await?;
    let frozen = view
        .freeze(selection.pending.plan.transaction)
        .await?
        .unwrap();
    let prepared = frozen
        .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let descriptors = annotations::descriptors(&prepared, &selection)?;
    let commit = save(&c, &selection, descriptors.clone()).await?;
    let selected = annotations::publish(
        Arc::new(c.client.manifest_catalog()),
        &one_shard(),
        &storage,
        commit,
        &selection,
        &descriptors,
        selected,
    )
    .await?;
    finish(&c).await?;
    view.install_published(&prepared, Arc::new(selected))
        .await?;
    drop((prepared, frozen));
    assert_eq!(view.observation().await.retained_entries, 0);
    assert_eq!(view.observation().await.retained_bytes, 0);
    assert_eq!(view.capture("T").await?.resolve().await?.unwrap(), current);
    let final_base = read(&c, &storage, None).await?;
    assert_eq!(final_base.count_rows(None).await?, 2);
    // Native manifest construction drops fully superseded narrow files. Each
    // fragment keeps its original body file and one current annotation file.
    assert!(final_base
        .manifest()
        .fragments
        .iter()
        .all(|f| f.files.len() == 2));
    let body_id = original_base.schema().field_id("body")?;
    let paths = |base: &Dataset| {
        base.manifest()
            .fragments
            .iter()
            .flat_map(|f| &f.files)
            .filter(|f| f.fields.contains(&body_id))
            .map(|f| f.path.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(paths(&final_base), paths(&original_base));
    let expected: Vec<_> = (0..256 << 10)
        .map(|i| ((i * 31 + i / 17) % 251) as u8)
        .collect();
    let batches: Vec<_> = final_base
        .scan()
        .project(&["id", "body"])?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    let mut ids = std::collections::BTreeSet::new();
    for batch in batches {
        let keys = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bodies = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(ids.insert(keys.value(row).to_owned()));
            assert_eq!(bodies.value(row), expected);
        }
    }
    assert_eq!(ids, ["T".into(), "U".into()].into_iter().collect());
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    let recovered = state(&c).await?;
    assert_eq!(recovered.progress.checkpoint[&10], second.0);
    assert!(recovered.progress.pending.is_none());
    let final_base = read(&c, &storage, None).await?;
    assert_eq!(final_base.version_id(), 3);
    let view = staging(final_base, second.0)?;
    assert_eq!(view.capture("T").await?.resolve().await?.unwrap(), current);
    for index in [first.0, second.0] {
        let response = c
            .client
            .request_routed(Request::Fetch { group: 10, index })
            .await?;
        assert!(matches!(response.reply, Reply::Record(None)));
    }
    c.shutdown().await
}

fn rejects_foreign_preparation(
    base: &Dataset,
    descriptors: &p::Descriptors,
    selection: &annotations::Selection,
) -> Result<()> {
    let original = annotations::transaction(base, descriptors, selection)?;
    for case in 0..4 {
        let mut changed = annotations::Selection {
            table: selection.table,
            incarnation: selection.incarnation,
            pending: selection.pending.clone(),
        };
        match case {
            0 => changed.incarnation = Uuid::new_v4(),
            1 => changed.table = Uuid::new_v4(),
            2 => changed.pending.plan.sources.get_mut(&10).unwrap().through += 1,
            _ => {
                changed.pending.plan.sources.get_mut(&10).unwrap().records[0]
                    .1
                    .bucket = 1
            }
        }
        assert!(annotations::transaction(base, descriptors, &changed).is_err());
    }
    for case in 0..9 {
        let mut changed = original.clone();
        match case {
            0 => changed.uuid = Uuid::new_v4().to_string(),
            1 => changed.read_version += 1,
            2 => changed.transaction_properties = None,
            3 => {
                if let lance_table::transaction::Operation::Update {
                    fields_modified, ..
                } = &mut changed.operation
                {
                    fields_modified.push(base.schema().field_id("body")? as u32);
                }
            }
            4 => {
                if let lance_table::transaction::Operation::Update {
                    updated_fragments, ..
                } = &mut changed.operation
                {
                    updated_fragments[0].files[0].path = "foreign.lance".into();
                }
            }
            5 => {
                if let lance_table::transaction::Operation::Update { new_fragments, .. } =
                    &mut changed.operation
                {
                    new_fragments.push(base.manifest().fragments[0].clone());
                }
            }
            _ => {
                if let lance_table::transaction::Operation::Update {
                    updated_fragments, ..
                } = &mut changed.operation
                {
                    let file = updated_fragments[0].files.last_mut().unwrap();
                    let mut fields = file.fields.to_vec();
                    match case {
                        6 => fields.fill(-2),
                        7 => fields[0] = -1,
                        _ => fields[0] = base.schema().field_id("body")?,
                    }
                    file.fields = fields.into();
                }
            }
        }
        let bad = p::Descriptors::lance_annotations(
            pb::Transaction::from(&changed).encode_to_vec().into(),
        )?;
        assert!(annotations::transaction(base, &bad, selection).is_err());
    }
    Ok(())
}
