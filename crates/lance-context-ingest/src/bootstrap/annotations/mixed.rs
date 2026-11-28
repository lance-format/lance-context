//! Explicit creation of a fresh body/narrow pair. Persist the configuration and
//! bootstrap identity before invoking this; ordinary role startup never calls it.
use super::*;
use lance_context_core::annotation_staging::{
    BodySnapshot, Scope as AnnotationScope, BODY_SNAPSHOT_PROPERTY,
};

const GENESIS: &str = "lance-context.mixed-genesis.v1";

/// Retry an interrupted bootstrap with the same identity and two destinations.
/// Existing unmanaged tables and patch-only bindings are never adopted.
pub async fn provision(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    narrow: Storage,
    body: Storage,
    identity: Identity,
    limits: Limits,
) -> Result<Storage> {
    provision_mode(
        client, layout, binding, narrow, body, identity, limits, None, None,
    )
    .await
}

/// Provision an independent physical table pair for one immutable WAL bucket.
#[allow(clippy::too_many_arguments)] // Same bootstrap identities plus an explicit shard.
pub async fn provision_shard(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    narrow: Storage,
    body: Storage,
    identity: Identity,
    limits: Limits,
    shard: crate::annotations::Shard,
) -> Result<Storage> {
    provision_mode(
        client,
        layout,
        binding,
        narrow,
        body,
        identity,
        limits,
        Some(shard),
        None,
    )
    .await
}

/// Provision a fresh managed pair with the caller's immutable body schema,
/// composite business key and session routing. Existing tables are not migrated.
#[allow(clippy::too_many_arguments)]
pub async fn provision_turn_shard(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    narrow: Storage,
    body: Storage,
    identity: Identity,
    limits: Limits,
    shard: crate::annotations::Shard,
    spec: &crate::turn_ingestion::TurnIngestionSpec,
) -> Result<Storage> {
    provision_mode(
        client,
        layout,
        binding,
        narrow,
        body,
        identity,
        limits,
        Some(shard),
        Some(spec),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn provision_mode(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    narrow: Storage,
    body: Storage,
    identity: Identity,
    limits: Limits,
    shard: Option<crate::annotations::Shard>,
    spec: Option<&crate::turn_ingestion::TurnIngestionSpec>,
) -> Result<Storage> {
    ensure!(
        shard.is_some() || layout.buckets.len() == 1,
        "mixed bootstrap requires an explicit shard"
    );
    let bucket = shard.map_or(0, |shard| shard.bucket);
    let group = layout.group_for_bucket(bucket)?;
    let scope = Scope {
        cluster: layout.cluster,
        layout: layout.fingerprint()?,
        table: binding.table,
        destination: binding.destination.clone(),
        incarnation: binding.incarnation,
        contract: if let Some(spec) = spec {
            shard
                .context("generic turns need an explicit shard")?
                .turn_contract(&binding, spec, &limits)?
        } else if let Some(shard) = shard {
            shard.contract(&binding, &limits, true)?
        } else {
            Service::mixed_contract(&binding, &limits)?
        },
    };
    ensure!(
        client.identity() == (scope.cluster, scope.layout)
            && !scope.table.is_nil()
            && !scope.incarnation.is_nil()
            && !identity.owner.is_nil()
            && !identity.acquisition.is_nil()
            && !identity.manifest.is_nil()
            && scope.contract.max_record_bytes as usize <= client.max_record_bytes(),
        "invalid mixed bootstrap identity"
    );
    let (narrow_store, narrow_path) = ObjectStore::from_uri_and_params(
        narrow.session.store_registry(),
        &narrow.uri,
        &narrow.params,
    )
    .await?;
    let (body_store, body_path) =
        ObjectStore::from_uri_and_params(body.session.store_registry(), &body.uri, &body.params)
            .await?;
    ensure!(
        narrow_store.store_prefix != body_store.store_prefix
            || (!narrow_path.prefix_matches(&body_path) && !body_path.prefix_matches(&narrow_path)),
        "mixed storage roots overlap"
    );
    // Credentials never enter the durable marker or WAL.
    let marker = serde_json::to_string(&(&scope, &identity, &narrow.uri, &body.uri))?;
    ensure!(
        marker.len() <= 32 << 10,
        "mixed bootstrap identity too large"
    );
    let existing = state(&client, &scope).await?;
    if let Some(manifests) = existing.as_ref().and_then(|s| s.manifests.as_ref()) {
        ensure!(
            manifests.incarnation == scope.incarnation
                && manifests.bootstrap_request == identity.manifest
                && manifests.initial_version == 1,
            "mixed bootstrap identity changed"
        );
        let reader = p::lance::ManagedManifest::reader(
            Arc::new(client.manifest_catalog()),
            narrow.base.clone(),
            &layout,
            scope.table,
            scope.incarnation,
        )?;
        let selected = DatasetBuilder::from_uri(&narrow.uri)
            .with_read_params(narrow.read_params(Default::default()))
            .with_session(narrow.session.clone())
            .with_commit_handler(Arc::new(reader))
            .load()
            .await?;
        ensure!(
            selected.schema().metadata.get(GENESIS) == Some(&marker),
            "mixed bootstrap destinations changed"
        );
        let reference = BodySnapshot::read(&selected)
            .await?
            .context("mixed body reference missing")?;
        crate::annotations::mixed::load_body_with_schema(
            &body,
            &reference,
            &scope.incarnation.to_string(),
            spec.map(|s| s.body_schema().as_ref())
                .unwrap_or(&rollout_schema()),
        )
        .await?;
        return Ok(narrow);
    }
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
    let Applied::Publisher(p::Reply::Acquired(authority)) = acquired else {
        bail!("mixed bootstrap acquisition rejected: {acquired:?}");
    };
    let selected =
        create_pair_for_shard(&narrow, &body, &scope, &identity, group, bucket, spec).await?;
    let transaction = selected
        .read_transaction()
        .await?
        .context("mixed genesis transaction missing")?;
    let location = selected.manifest_location();
    ensure!(
        location.path.parent().and_then(|p| p.parent()).as_ref() == Some(&narrow.base),
        "mixed manifest outside configured storage"
    );
    let object = p::ManifestObject {
        path: format!(
            "_versions/{}",
            location
                .path
                .filename()
                .context("mixed manifest name missing")?
        ),
        bytes: location.size.context("mixed manifest size missing")?,
        v2_names: location.naming_scheme == ManifestNamingScheme::V2,
    };
    let transaction = Uuid::parse_str(&transaction.uuid)?;
    let reply = commit(
        &client,
        Command::Bootstrap(p::BootstrapManifest {
            authority,
            incarnation: scope.incarnation,
            request: identity.manifest,
            version: 1,
            object: object.clone(),
            transaction,
        }),
    )
    .await?;
    ensure!(
        matches!(reply, Applied::Publisher(p::Reply::ManifestReserved(ref r))
        if r.version == 1 && r.object == object && r.transaction == transaction),
        "mixed genesis selection not confirmed: {reply:?}"
    );
    Ok(narrow)
}

#[cfg(test)]
pub(crate) async fn create_pair(
    narrow: &Storage,
    body: &Storage,
    scope: &Scope,
    identity: &Identity,
    group: u64,
) -> Result<Dataset> {
    create_pair_for_shard(narrow, body, scope, identity, group, 0, None).await
}

async fn create_pair_for_shard(
    narrow: &Storage,
    body: &Storage,
    scope: &Scope,
    identity: &Identity,
    group: u64,
    bucket: u32,
    spec: Option<&crate::turn_ingestion::TurnIngestionSpec>,
) -> Result<Dataset> {
    let marker = serde_json::to_string(&(scope, identity, &narrow.uri, &body.uri))?;
    let properties = HashMap::from([(GENESIS.to_owned(), marker.clone())]);
    let body_base = empty(
        body,
        spec.map_or_else(|| Arc::new(rollout_schema()), |s| s.body_schema().clone()),
        properties.clone(),
    )
    .await?;
    let reference = BodySnapshot::capture(scope.incarnation.to_string(), &body_base).await?;
    ensure!(
        reference.uri == body.uri,
        "body URI must match the persisted native URI"
    );
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("rollout_id", DataType::Utf8, false),
            Field::new(
                "labels",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
            Field::new("score", DataType::Float32, true),
        ],
        properties.clone(),
    ));
    let mut properties = properties;
    let schema = match spec {
        Some(spec) => spec.bind_schema(&schema)?,
        None => schema,
    };
    properties.insert(
        BODY_SNAPSHOT_PROPERTY.into(),
        serde_json::to_string(&reference)?,
    );
    properties.insert(
        "lance-context.annotation-cut".into(),
        serde_json::to_string(&(
            AnnotationScope {
                table_instance: scope.incarnation.to_string(),
                source_group: group,
                shard: bucket,
            },
            0u64,
        ))?,
    );
    empty(narrow, schema, properties).await
}

async fn empty(
    storage: &Storage,
    schema: SchemaRef,
    properties: HashMap<String, String>,
) -> Result<Dataset> {
    let dataset = match Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(RecordBatch::new_empty(schema.clone()))],
            schema.clone(),
        ),
        &storage.uri,
        Some(WriteParams {
            mode: WriteMode::Create,
            enable_stable_row_ids: true,
            skip_auto_cleanup: true,
            store_params: Some(storage.params.clone()),
            session: Some(storage.session.clone()),
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            transaction_properties: Some(Arc::new(properties.clone())),
            ..Default::default()
        }),
    )
    .await
    {
        Ok(dataset) => dataset,
        Err(lance::Error::DatasetAlreadyExists { .. }) => canonical(storage, None).await?,
        Err(error) => return Err(error.into()),
    };
    let actual_schema: Schema = dataset.schema().into();
    let transaction = dataset
        .read_transaction()
        .await?
        .context("mixed genesis transaction missing")?;
    ensure!(
        dataset.version_id() == 1
            && actual_schema == *schema
            && dataset.manifest().uses_stable_row_ids()
            && dataset.count_rows(None).await? == 0
            && transaction.transaction_properties.as_deref() == Some(&properties),
        "refusing foreign or advanced mixed genesis"
    );
    Ok(dataset)
}
