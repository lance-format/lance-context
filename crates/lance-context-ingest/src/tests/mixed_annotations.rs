use super::annotations::{producer, request, run};
use super::*;
mod compaction;
mod id_index;
#[cfg(feature = "service")]
mod lost_prepare;
mod prefetch;
mod pressure;
mod replay;
mod runtime;
mod status_index;
#[cfg(feature = "http")]
mod upload_failure;
use crate::{
    annotations::{self as service, mixed, Service},
    materializer::{Acquisition, Options, Storage},
};
use arrow_array::{Float32Array, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{
    transaction::{Operation, Transaction},
    CommitBuilder, WriteParams,
};
use lance_context_core::annotation_staging::{
    BodySnapshot, Limits, Scope as AnnotationScope, BODY_SNAPSHOT_PROPERTY,
};

pub(super) fn limits() -> Limits {
    Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    }
}

pub(super) async fn storage(root: &std::path::Path, name: &str) -> Result<Storage> {
    Storage::resolve(
        root.join(name).to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await
}

pub(super) async fn seed(c: &network::Cluster, narrow: &Storage, body: &Storage) -> Result<()> {
    seed_with_binding(c, narrow, body, binding()).await
}

pub(super) async fn seed_with_binding(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    binding: TableBinding,
) -> Result<()> {
    seed_with_limits(c, narrow, body, binding, &limits()).await
}

pub(super) async fn seed_with_limits(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    binding: TableBinding,
    limits: &Limits,
) -> Result<()> {
    let write = |schema: SchemaRef, uri: String| async move {
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema),
            &uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                skip_auto_cleanup: true,
                ..Default::default()
            }),
        )
        .await
    };
    let body = write(Arc::new(rollout_schema()), body.uri.clone()).await?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("rollout_id", DataType::Utf8, false),
        Field::new(
            "labels",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    let base = Arc::new(write(schema, narrow.uri.clone()).await?);
    seed_pair_with_limits(c, binding, base, &body, limits).await
}

async fn seed_pair_with_limits(
    c: &network::Cluster,
    binding: TableBinding,
    base: Arc<Dataset>,
    body: &Dataset,
    limits: &Limits,
) -> Result<()> {
    let reference = BodySnapshot::capture(binding.incarnation.to_string(), body).await?;
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: Default::default(),
        },
    );
    let scope = AnnotationScope {
        table_instance: binding.incarnation.to_string(),
        source_group: 10,
        shard: 0,
    };
    transaction.transaction_properties = Some(Arc::new(std::collections::HashMap::from([
        (
            BODY_SNAPSHOT_PROPERTY.into(),
            serde_json::to_string(&reference)?,
        ),
        (
            "lance-context.annotation-cut".into(),
            serde_json::to_string(&(scope, 0u64))?,
        ),
    ])));
    let base = CommitBuilder::new(base)
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    annotation_publication::seed_base(c, Service::mixed_contract(&binding, limits)?, &base).await
}

pub(super) fn encoded(batch: RecordBatch) -> Result<Bytes> {
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &batch.schema())?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(bytes.into())
}

pub(super) fn patch(ids: &[&str], status: Option<&str>, score: Option<f32>) -> Result<RecordBatch> {
    patch_with_limits(ids, status, score, &limits())
}

pub(super) fn patch_with_limits(
    ids: &[&str],
    status: Option<&str>,
    score: Option<f32>,
    limits: &Limits,
) -> Result<RecordBatch> {
    Ok(ipc::decode(
        super::annotations::wire(
            limits,
            ids,
            u8::from(status.is_some()),
            status,
            u8::from(score.is_some()),
            score,
        )?,
        service::wire::schema(limits),
    )?
    .remove(0))
}

pub(super) fn create(ids: &[&str], groups: &[&str], bytes: &[u8]) -> Result<RecordBatch> {
    create_with_limits(ids, groups, bytes, &limits())
}

pub(super) fn create_with_limits(
    ids: &[&str],
    groups: &[&str],
    bytes: &[u8],
    limits: &Limits,
) -> Result<RecordBatch> {
    let source = batch(ids, groups)?;
    let mut columns = source.columns().to_vec();
    columns[source.schema().index_of("binary_payload")?] = Arc::new(
        LargeBinaryArray::from_iter_values(bytes.iter().map(|b| vec![*b; 16 << 10])),
    );
    let body = RecordBatch::try_new(source.schema(), columns)?;
    let initial = patch_with_limits(ids, Some("pending"), None, limits)?;
    let mut labels =
        arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new())
            .with_field(Arc::new(Field::new("item", DataType::Utf8, false)));
    for _ in ids {
        labels.values().append_value("imported");
        labels.append(true);
    }
    let mut columns = initial.columns().to_vec();
    columns[1] = Arc::new(labels.finish());
    let initial = RecordBatch::try_new(initial.schema(), columns)?;
    mixed::wire::creates(&body, &initial, limits)
}

async fn connect(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
) -> Result<(Service, service::Driver)> {
    Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow,
        body.clone(),
        limits(),
        MergeMemoryBudget::new(8 << 20),
        MergeMemoryBudget::new(8 << 20),
    )
    .await
}

pub(super) async fn check(
    service: &Service,
    id: &str,
    group: &str,
    byte: u8,
    status: &str,
    score: Option<f32>,
) -> Result<()> {
    let turn = service
        .get_turn(id, Duration::from_secs(10))
        .await?
        .context("turn missing")?;
    assert_eq!(turn.annotation.status.as_deref(), Some(status));
    assert_eq!(
        turn.annotation.labels,
        std::collections::BTreeSet::from(["imported".into()])
    );
    assert_eq!(turn.annotation.score, score);
    let body = turn.body();
    let groups = body
        .column_by_name("rollout_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(groups.value(0), group);
    let payloads = body
        .column_by_name("binary_payload")
        .unwrap()
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap();
    assert_eq!(payloads.value(0), vec![byte; 16 << 10]);
    let scores = body
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(scores.iter().next().unwrap(), score);
    Ok(())
}

#[tokio::test]
async fn mixed_held_body_read_outlives_publication_source_closure_and_frame_retirement(
) -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    let apply = MergeMemoryBudget::new(8 << 20);
    let (a, driver) = Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        &narrow,
        body.clone(),
        limits(),
        MergeMemoryBudget::new(8 << 20),
        apply.clone(),
    )
    .await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    let initial = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
    a.submit(&initial, Duration::from_secs(10)).await?;
    assert_eq!(
        a.phase_observation()
            .iter()
            .find(|p| p.phase == "annotation_fetch")
            .unwrap()
            .started,
        0
    );
    // The local upload can disappear: staged bodies and held readers retain
    // their own apply reservation through publication and source retirement.
    drop(initial);
    let held = a.get_turn("A", Duration::from_secs(10)).await?.unwrap();
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    let selection = annotation_publication::select_range(
        &c,
        frozen.sources().records.clone(),
        frozen.through(),
        Uuid::parse_str(frozen.identity())?,
    )
    .await?;
    let prepared = frozen
        .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    assert!(
        crate::annotation_publication::descriptors(prepared.annotations(), &selection).is_err()
    );
    let descriptors = crate::annotation_publication::mixed::persist_descriptors(
        prepared.annotations(),
        &selection,
        &narrow,
        &body,
        c.client.max_descriptor_bytes(),
    )
    .await?;
    let commit = annotation_publication::save(&c, &selection, descriptors.clone()).await?;
    let suffix = request(
        &a,
        &p,
        2,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    a.submit(&suffix, Duration::from_secs(10)).await?;
    // Classification comes from the already validated mixed Arrow operation,
    // including the one-row create above; row count is not an operation tag.
    let phases = a.phase_observation();
    for name in [
        "annotation_create_append",
        "annotation_patch_append",
        "annotation_create_apply_wait",
        "annotation_patch_apply_wait",
    ] {
        let phase = phases.iter().find(|p| p.phase == name).unwrap();
        assert_eq!(phase.started, 1, "{name}");
        assert_eq!(phase.completed, 1, "{name}");
        assert_eq!(phase.failed + phase.cancelled, 0, "{name}");
    }
    let selected = crate::annotation_publication::mixed::publish(
        Arc::new(c.client.manifest_catalog()),
        &annotation_publication::one_shard(),
        &narrow,
        &body,
        commit,
        &selection,
        &descriptors,
        prepared.annotations().base().clone(),
    )
    .await?;
    annotation_publication::confirm(&c).await?;
    a.install_published(&prepared, Arc::new(selected)).await?;
    drop((prepared, frozen));
    annotation_publication::finish(&c).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert!(apply.reserved() > 0);
    assert_eq!(held.annotation.status.as_deref(), Some("pending"));
    assert_eq!(held.annotation.score, None);
    assert_eq!(
        held.body()
            .column_by_name("binary_payload")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .value(0),
        vec![1; 16 << 10]
    );
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    drop(held);
    assert_eq!(apply.reserved(), 0);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}

fn artifact_root() -> Result<tempfile::TempDir> {
    if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        Ok(tempfile::Builder::new()
            .prefix("mixed-driver-")
            .disable_cleanup(true)
            .tempdir_in(parent)?)
    } else {
        Ok(tempfile::tempdir()?)
    }
}

pub(super) async fn owner(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
) -> Result<(Service, service::runtime::Runtime)> {
    let state = annotation_publication::state(c).await?;
    service::runtime::Runtime::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow.clone(),
        body.clone(),
        service::runtime::Configuration {
            limits: limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 4,
                max_batch_wait: Duration::from_millis(20),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await
}
