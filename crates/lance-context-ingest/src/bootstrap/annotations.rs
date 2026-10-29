//! Explicit handoff of a fixed, quiesced Lance snapshot to annotation ownership.
//! Stop old writers before invoking this operation. It does not migrate another
//! live WAL policy or discover pending rows from an unmanaged writer.
use super::*;
use crate::annotations::Service;
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{
    transaction::{Operation, Transaction},
    CommitBuilder,
};
use lance_context_core::annotation_staging::Limits;
use lance_table::{
    format::{IndexMetadata, Manifest},
    io::commit::{CommitError, CommitHandler, ManifestLocation, ManifestWriter},
};
use object_store::path::Path;

const HANDOFF: &str = "lance-context.annotation-handoff.v1";

pub mod projection;

// Lance can rebase even on its first attempt with max_retries=0. Check the
// target immediately before the conditional write, not only after publication.
#[derive(Debug)]
pub(crate) struct ExactVersion(pub u64);

#[async_trait::async_trait]
impl CommitHandler for ExactVersion {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        if manifest.version != self.0 {
            return Err(CommitError::OtherError(lance::Error::invalid_input(
                "annotation handoff cannot rebase onto a changed snapshot",
            )));
        }
        ConditionalPutCommitHandler
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub version: u64,
    pub transaction: Uuid,
}

/// Persist this snapshot identity and the normal bootstrap identity before use.
/// All new columns are nullable metadata-only columns; existing score/labels/
/// status and the original data files remain unchanged.
pub async fn provision(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    storage: Storage,
    identity: Identity,
    snapshot: Snapshot,
    limits: Limits,
) -> Result<Storage> {
    ensure!(
        layout.buckets.len() == 1,
        "annotation handoff requires one shard"
    );
    ensure!(
        snapshot.version > 0
            && snapshot.version < (1 << 63) - 1
            && !snapshot.transaction.is_nil()
            && !identity.owner.is_nil()
            && !identity.acquisition.is_nil()
            && !identity.manifest.is_nil(),
        "invalid annotation handoff identity"
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
            && scope.contract.max_record_bytes as usize <= client.max_record_bytes(),
        "invalid annotation handoff scope"
    );
    let marker = serde_json::to_string(&(&scope, &identity, &snapshot))?;
    ensure!(
        marker.len() <= 16 << 10,
        "annotation handoff marker too large"
    );
    // Check policy binding before modifying a table already owned by this WAL.
    let existing = state(&client, &scope).await?;
    if let Some(manifests) = existing.as_ref().and_then(|s| s.manifests.as_ref()) {
        ensure!(
            manifests.incarnation == scope.incarnation
                && manifests.bootstrap_request == identity.manifest
                && manifests.initial_version == snapshot.version + 1,
            "annotation handoff identity changed"
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
            selected.schema().metadata.get(HANDOFF) == Some(&marker),
            "annotation handoff snapshot changed"
        );
        return Ok(storage);
    }
    let base = canonical(&storage, Some(snapshot.version)).await?;
    ensure!(
        base.read_transaction()
            .await?
            .is_some_and(|t| t.uuid == snapshot.transaction.to_string()),
        "annotation source transaction changed"
    );
    let expected = prepare(&base, &scope, &identity, &marker)?;
    let current = canonical(&storage, None).await?;
    let selected = if current.version_id() == snapshot.version {
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
            "annotation handoff acquisition rejected: {acquired:?}"
        );
        // Fixed native transaction identity, no rebase onto concurrent changes.
        // An unknown outcome is resolved by reopening this exact version later.
        CommitBuilder::new(Arc::new(base.clone()))
            .with_commit_handler(Arc::new(ExactVersion(snapshot.version + 1)))
            .with_max_retries(0)
            .with_skip_auto_cleanup(true)
            .execute(expected.clone())
            .await?
    } else {
        ensure!(
            current.version_id() == snapshot.version + 1,
            "annotation source advanced; quiesce and choose an explicit snapshot"
        );
        current
    };
    ensure!(
        selected.version_id() == snapshot.version + 1
            && selected.read_transaction().await?.as_ref() == Some(&expected)
            && selected.manifest().fragments == base.manifest().fragments
            && selected.schema().metadata.get(HANDOFF) == Some(&marker),
        "annotation handoff differs from the original preparation"
    );
    // Repeating the same acquisition resolves its original result; never take
    // another epoch after an ambiguous bootstrap result.
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
        bail!("annotation handoff acquisition rejected: {acquired:?}");
    };
    let location = selected.manifest_location();
    let object = p::ManifestObject {
        path: format!(
            "_versions/{}",
            location
                .path
                .filename()
                .context("handoff manifest name missing")?
        ),
        bytes: location.size.context("handoff manifest size missing")?,
        v2_names: location.naming_scheme == ManifestNamingScheme::V2,
    };
    ensure!(
        location.path.parent().and_then(|p| p.parent()).as_ref() == Some(&storage.base),
        "handoff manifest outside configured storage"
    );
    let reply = commit(
        &client,
        Command::Bootstrap(p::BootstrapManifest {
            authority,
            incarnation: scope.incarnation,
            request: identity.manifest,
            version: snapshot.version + 1,
            object: object.clone(),
            transaction: identity.manifest,
        }),
    )
    .await?;
    ensure!(
        matches!(reply, Applied::Publisher(p::Reply::ManifestReserved(ref r))
        if r.version == snapshot.version + 1 && r.object == object && r.transaction == identity.manifest),
        "annotation handoff selection not confirmed: {reply:?}"
    );
    Ok(storage)
}

async fn canonical(storage: &Storage, version: Option<u64>) -> Result<Dataset> {
    let mut builder = DatasetBuilder::from_uri(&storage.uri)
        .with_storage_options(storage.options.clone())
        .with_session(storage.session.clone())
        .with_commit_handler(Arc::new(ConditionalPutCommitHandler));
    if let Some(version) = version {
        builder = builder.with_version(version);
    }
    Ok(builder.load().await?)
}

pub(crate) fn prepare(
    base: &Dataset,
    scope: &Scope,
    identity: &Identity,
    marker: &str,
) -> Result<Transaction> {
    ensure!(
        base.manifest()
            .data_storage_format
            .lance_file_format()
            .to_data_file_numbers()
            .0
            == 2,
        "annotation handoff requires Lance v2 data files"
    );
    ensure!(
        !base.schema().metadata.contains_key(HANDOFF),
        "snapshot already belongs to an annotation handoff"
    );
    let id = base
        .schema()
        .field("id")
        .context("annotation table has no turn id")?;
    ensure!(
        id.data_type() == DataType::Utf8 && !id.nullable,
        "annotation turn id must be non-null UTF8"
    );
    let mut missing = Vec::new();
    for field in [
        Field::new(
            "labels",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ] {
        if let Some(existing) = base.schema().field(field.name()) {
            ensure!(
                existing.data_type() == *field.data_type(),
                "annotation column type differs: {}",
                field.name()
            );
            ensure!(
                field.name() == "labels" || existing.nullable,
                "annotation column must allow clearing: {}",
                field.name()
            );
        } else {
            missing.push(field);
        }
    }
    let additions =
        Schema::new_with_metadata(missing, HashMap::from([(HANDOFF.into(), marker.into())]));
    let mut schema = base.schema().merge(&additions)?;
    schema.set_field_id(Some(base.manifest().max_field_id()));
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::Merge {
            fragments: base.manifest().fragments.as_ref().clone(),
            schema,
            preserves_nullability: true,
        },
    );
    transaction.uuid = identity.manifest.to_string();
    transaction.transaction_properties = Some(Arc::new(HashMap::from([
        (HANDOFF.into(), marker.into()),
        (
            "lance-context.annotation-table".into(),
            scope.table.to_string(),
        ),
    ])));
    Ok(transaction)
}
