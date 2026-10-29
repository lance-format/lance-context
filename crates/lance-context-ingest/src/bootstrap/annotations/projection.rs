//! Offline projection into a fresh annotation destination. The source must stay
//! quiesced through handoff; it is never modified, collected, or adopted here.
use super::*;
use arrow_array::new_null_array;
use datafusion_common::DataFusionError;
use datafusion_physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream};
use futures::TryStreamExt;
use lance::{dataset::InsertBuilder, index::DatasetIndexExt};
use lance_file::writer::FileWriterOptions;

const PROJECTION: &str = "lance-context.annotation-projection.v1";
const COLUMNS: [&str; 5] = ["id", "rollout_id", "labels", "status", "score"];
const INDEX: &str = "annotation_point_id";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub uri: String,
    #[serde(default)]
    pub storage_options: HashMap<String, String>,
    pub snapshot: Snapshot,
}

impl Source {
    pub fn validate(&self, workspace_bytes: usize) -> Result<()> {
        ensure!(
            !self.uri.is_empty()
                && self.uri.len() <= 8192
                && self.storage_options.len() <= 128
                && self.snapshot.version > 0
                && self.snapshot.version < (1 << 63) - 1
                && !self.snapshot.transaction.is_nil()
                && workspace_bytes >= 1 << 20,
            "invalid annotation projection source or workspace"
        );
        Ok(())
    }
}

/// Retry with the original source, destination and bootstrap identity. Completed
/// handoffs resolve through the managed catalog, never by reimporting old values.
#[allow(clippy::too_many_arguments)]
pub async fn provision(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    storage: Storage,
    identity: Identity,
    source: &Source,
    limits: Limits,
    workspace_bytes: usize,
) -> Result<Storage> {
    source.validate(workspace_bytes)?;
    ensure!(
        layout.buckets.len() == 1 && source.uri != storage.uri,
        "annotation projection requires one shard and a separate destination"
    );
    let scope = Scope {
        cluster: layout.cluster,
        layout: layout.fingerprint()?,
        table: binding.table,
        destination: binding.destination.clone(),
        incarnation: binding.incarnation,
        contract: Service::contract(&binding, &limits)?,
    };
    ensure!(
        client.identity() == (scope.cluster, scope.layout)
            && !scope.table.is_nil()
            && !scope.incarnation.is_nil()
            && !identity.owner.is_nil()
            && !identity.acquisition.is_nil()
            && !identity.manifest.is_nil()
            && scope.contract.max_record_bytes as usize <= client.max_record_bytes(),
        "invalid annotation projection identity"
    );
    // Credentials are deliberately excluded from persisted provenance.
    let marker = serde_json::to_string(&(
        &scope,
        &identity,
        &storage.uri,
        &source.uri,
        &source.snapshot,
    ))?;
    ensure!(marker.len() <= 32 << 10, "projection identity too large");
    let existing = state(&client, &scope).await?;
    if let Some(manifests) = existing.as_ref().and_then(|s| s.manifests.as_ref()) {
        ensure!(
            manifests.incarnation == scope.incarnation
                && manifests.bootstrap_request == identity.manifest
                && manifests.initial_version == 3,
            "annotation projection handoff identity changed"
        );
        let reader = p::lance::ManagedManifest::reader(
            Arc::new(client.manifest_catalog()),
            storage.base.clone(),
            &layout,
            scope.table,
            scope.incarnation,
        )?;
        let selected = DatasetBuilder::from_uri(&storage.uri)
            .with_storage_options(storage.options.clone())
            .with_session(storage.session.clone())
            .with_commit_handler(Arc::new(reader))
            .load()
            .await?;
        ensure!(
            selected.schema().metadata.get(PROJECTION) == Some(&marker)
                && selected.schema().metadata.contains_key(HANDOFF),
            "annotation projection source or destination changed"
        );
        return Ok(storage);
    }
    let origin = Storage::resolve(
        source.uri.clone(),
        source.storage_options.clone(),
        storage.session.clone(),
    )
    .await?;
    let (source_store, source_path) = ObjectStore::from_uri_and_params(
        origin.session.store_registry(),
        &origin.uri,
        &parameters(&origin.options),
    )
    .await?;
    let (target_store, target_path) = ObjectStore::from_uri_and_params(
        storage.session.store_registry(),
        &storage.uri,
        &parameters(&storage.options),
    )
    .await?;
    ensure!(
        source_store.store_prefix != target_store.store_prefix
            || (!source_path.prefix_matches(&target_path)
                && !target_path.prefix_matches(&source_path)),
        "annotation source and destination storage roots overlap"
    );
    let base = source_snapshot(&origin, &source.snapshot).await?;
    let schema = schema(&base, &scope, &identity, &marker)?;
    let acquired = commit(
        &client,
        Command::Acquire(p::Acquire {
            table: scope.table,
            destination: scope.destination.clone(),
            ingest: scope.contract.clone(),
            owner: identity.owner,
            request: identity.acquisition,
            expected_epoch: 0,
        }),
    )
    .await?;
    ensure!(
        matches!(acquired, Applied::Publisher(p::Reply::Acquired(_))),
        "annotation projection acquisition rejected: {acquired:?}"
    );
    let initial = project(&base, &storage, schema, &marker, workspace_bytes).await?;
    // A changed source cannot silently become the initial annotation state.
    source_snapshot(&origin, &source.snapshot).await?;
    let indexed = index(&storage, &initial, &marker).await?;
    source_snapshot(&origin, &source.snapshot).await?;
    super::provision(
        client,
        layout,
        binding,
        storage,
        identity,
        Snapshot {
            version: 2,
            transaction: indexed
                .read_transaction()
                .await?
                .context("projected index transaction missing")?
                .uuid
                .parse()?,
        },
        limits,
    )
    .await
}

async fn source_snapshot(storage: &Storage, snapshot: &Snapshot) -> Result<Dataset> {
    let base = canonical(storage, None).await?;
    ensure!(
        base.version_id() == snapshot.version
            && base
                .read_transaction()
                .await?
                .is_some_and(|t| t.uuid == snapshot.transaction.to_string()),
        "annotation projection source advanced or was replaced"
    );
    Ok(base)
}

pub(crate) fn schema(
    base: &Dataset,
    scope: &Scope,
    identity: &Identity,
    marker: &str,
) -> Result<SchemaRef> {
    // Reuse the handoff's type/nullability validation and nullable additions,
    // without committing anything to the source table.
    let Operation::Merge { schema, .. } = prepare(base, scope, identity, marker)?.operation else {
        unreachable!()
    };
    let group = schema.field("rollout_id").context("rollout_id missing")?;
    ensure!(
        group.data_type() == DataType::Utf8 && !group.nullable,
        "projection requires a non-null UTF8 rollout_id"
    );
    let projected: Schema = (&schema.project(&COLUMNS)?).into();
    Ok(Arc::new(projected.with_metadata(HashMap::from([(
        PROJECTION.into(),
        marker.into(),
    )]))))
}

pub(crate) async fn project(
    base: &Dataset,
    storage: &Storage,
    schema: SchemaRef,
    marker: &str,
    workspace_bytes: usize,
) -> Result<Dataset> {
    match canonical(storage, None).await {
        Ok(current) => {
            ensure!(
                (1..=3).contains(&current.version_id()),
                "projection destination advanced"
            );
            let initial = canonical(storage, Some(1)).await?;
            verify_initial(&initial, &schema, marker).await?;
            return Ok(initial);
        }
        Err(error)
            if matches!(
                error.downcast_ref::<lance::Error>(),
                Some(lance::Error::DatasetNotFound { .. })
            ) => {}
        Err(error) => return Err(error),
    }
    let buffer = workspace_bytes / 4;
    let names = COLUMNS
        .iter()
        .copied()
        .filter(|name| base.schema().field(name).is_some())
        .collect::<Vec<_>>();
    let mut scan = base.scan();
    scan.project(&names)?;
    scan.batch_size(8192.min(buffer / 128))
        .batch_size_bytes((buffer / 2) as u64)
        .batch_readahead(1)
        // Overlap fragment I/O through the scan's shared byte budget without
        // changing its decode settings.
        .fragment_readahead(4)
        .io_buffer_size(buffer as u64);
    let output_schema = schema.clone();
    let stream = scan
        .try_into_stream()
        .await?
        .map_err(|e| DataFusionError::External(Box::new(e)))
        .and_then(move |batch| {
            let schema = output_schema.clone();
            async move {
                let columns = schema
                    .fields()
                    .iter()
                    .map(|field| {
                        batch
                            .column_by_name(field.name())
                            .cloned()
                            .unwrap_or_else(|| new_null_array(field.data_type(), batch.num_rows()))
                    })
                    .collect();
                let output = RecordBatch::try_new(schema, columns)?;
                if output.get_array_memory_size() > buffer {
                    return Err(DataFusionError::Execution(
                        "annotation projection batch exceeds workspace".into(),
                    ));
                }
                Ok(output)
            }
        });
    let stream: SendableRecordBatchStream =
        Box::pin(RecordBatchStreamAdapter::new(schema.clone(), stream));
    let params = WriteParams {
        mode: WriteMode::Create,
        enable_stable_row_ids: true,
        max_bytes_per_file: workspace_bytes,
        store_params: Some(parameters(&storage.options)),
        session: Some(storage.session.clone()),
        commit_handler: Some(Arc::new(ExactVersion(1))),
        skip_auto_cleanup: true,
        file_writer_options: Some(FileWriterOptions {
            data_cache_bytes: Some(buffer as u64),
            max_page_bytes: Some(buffer as u64),
            ..Default::default()
        }),
        transaction_properties: Some(Arc::new(HashMap::from([(
            PROJECTION.into(),
            marker.into(),
        )]))),
        ..Default::default()
    };
    let initial = InsertBuilder::new(storage.uri.as_str())
        .with_params(&params)
        .execute_stream(stream)
        .await?;
    verify_initial(&initial, &schema, marker).await?;
    Ok(initial)
}

async fn verify_initial(dataset: &Dataset, schema: &SchemaRef, marker: &str) -> Result<()> {
    let actual: Schema = dataset.schema().into();
    ensure!(
        dataset.version_id() == 1
            && actual == **schema
            && dataset.manifest().uses_stable_row_ids()
            && dataset.read_transaction().await?.is_some_and(|t| t
                .transaction_properties
                .as_ref()
                .and_then(|p| p.get(PROJECTION))
                .is_some_and(|v| v == marker)),
        "refusing to adopt a foreign annotation projection"
    );
    Ok(())
}

pub(crate) async fn index(storage: &Storage, initial: &Dataset, marker: &str) -> Result<Dataset> {
    let current = canonical(storage, None).await?;
    let indexed = if current.version_id() == 1 {
        let mut target = DatasetBuilder::from_uri(&storage.uri)
            .with_version(1)
            .with_storage_options(storage.options.clone())
            .with_session(storage.session.clone())
            .with_commit_handler(Arc::new(ExactVersion(2)))
            .load()
            .await?;
        target
            .create_index_builder(
                &["id"],
                lance_index::IndexType::BTree,
                &lance_index::scalar::ScalarIndexParams::default(),
            )
            .name(INDEX.into())
            .transaction_properties(HashMap::from([(PROJECTION.into(), marker.into())]))
            .await?;
        target
    } else {
        ensure!(
            (2..=3).contains(&current.version_id()),
            "projection destination advanced"
        );
        canonical(storage, Some(2)).await?
    };
    let indices = indexed.load_indices().await?;
    let transaction = indexed
        .read_transaction()
        .await?
        .context("projected index transaction missing")?;
    ensure!(
        indexed.version_id() == 2
            && indexed.schema() == initial.schema()
            && indexed.manifest().fragments == initial.manifest().fragments
            && transaction
                .transaction_properties
                .as_ref()
                .and_then(|p| p.get(PROJECTION))
                .is_some_and(|v| v == marker)
            && matches!(&transaction.operation, Operation::CreateIndex { new_indices, removed_indices }
                // Native index timestamps persist at millisecond precision.
                // Compare every persisted field, not transient nanoseconds.
                if removed_indices.is_empty() && new_indices.iter().map(lance_table::format::pb::IndexMetadata::from)
                    .eq(indices.iter().map(lance_table::format::pb::IndexMetadata::from)))
            && indices.len() == 1
            && indices[0].name == INDEX
            && !indices[0].uuid.is_nil()
            && indices[0].fields == vec![indexed.schema().field("id").unwrap().id]
            && indices[0]
                .fragment_bitmap
                .as_ref()
                .is_some_and(|coverage| indexed
                    .fragments()
                    .iter()
                    .all(|f| u32::try_from(f.id).is_ok_and(|id| coverage.contains(id)))),
        "projected ID index or source files changed"
    );
    Ok(indexed)
}
