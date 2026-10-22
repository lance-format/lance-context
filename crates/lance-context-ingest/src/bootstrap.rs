//! Idempotent provisioning of a NEW managed rollout generation. Persist the
//! binding, URI and complete operation identity before the first call. This is
//! not an importer for existing datasets or a legacy writer migration.
use super::*;
use crate::materializer::Storage;
use arrow_array::RecordBatchIterator;
use lance::{
    dataset::{builder::DatasetBuilder, WriteMode, WriteParams},
    Dataset,
};
use lance_context_wal::replication::publication as p;
use lance_io::{
    object_store::StorageOptionsAccessor,
    object_store::{ObjectStore, ObjectStoreParams},
};
use lance_table::io::commit::{ConditionalPutCommitHandler, ManifestNamingScheme};
use std::collections::HashMap;

const GENESIS: &str = "lance-context.ingest.genesis.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub owner: Uuid,
    pub acquisition: Uuid,
    pub manifest: Uuid,
}

impl Storage {
    /// Resolve object-store paths with Lance's own URI/credential handling.
    /// Storage options may select workload identity; no credentials enter WAL.
    pub async fn resolve(
        uri: String,
        options: HashMap<String, String>,
        session: Arc<lance::session::Session>,
    ) -> Result<Self> {
        ensure!(!uri.is_empty() && uri.len() <= 8192, "invalid dataset URI");
        let (_, base) =
            ObjectStore::from_uri_and_params(session.store_registry(), &uri, &parameters(&options))
                .await?;
        Ok(Self {
            uri,
            base,
            options,
            session,
        })
    }
}
fn parameters(options: &HashMap<String, String>) -> ObjectStoreParams {
    ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            options.clone(),
        ))),
        ..Default::default()
    }
}
async fn state(client: &Client, scope: &Scope) -> Result<Option<Box<p::Table>>> {
    let group = client.catalog_group();
    let response = client
        .request_routed(Request::Query {
            group,
            query: query::Query::Publisher { table: scope.table },
        })
        .await?;
    let Reply::Query(query::Outcome::Observed {
        cut,
        value: query::Value::Publisher(view),
    }) = response.reply
    else {
        bail!("bootstrap quorum observation unavailable");
    };
    ensure!(
        cut.cluster == scope.cluster
            && cut.layout == scope.layout
            && cut.group == group
            && cut.term > 0
            && cut.quorum_index > 0
            && cut.applied >= cut.quorum_index,
        "foreign bootstrap observation"
    );
    if let Some(view) = &view {
        ensure!(
            view.destination == scope.destination && view.ingest == scope.contract,
            "dataset generation already bound differently"
        );
    }
    Ok(view)
}
async fn commit(client: &Client, command: Command) -> Result<Applied> {
    let response = client
        .request_routed(Request::Commit(Box::new(command)))
        .await?;
    let Reply::Commit(Outcome::Committed(applied)) = response.reply else {
        bail!("bootstrap outcome unknown; retry the same persisted identity");
    };
    Ok(applied)
}

/// Provision the reserved generation at an application-authorized URI. Only a
/// newly created empty dataset carrying this exact bootstrap identity can become
/// the catalog's genesis. A timeout never generates replacement operation IDs.
pub async fn provision(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    storage: Storage,
    identity: Identity,
) -> Result<Storage> {
    let scope = Scope {
        cluster: layout.cluster,
        layout: layout.fingerprint()?,
        table: binding.table,
        destination: binding.destination.clone(),
        incarnation: binding.incarnation,
        contract: binding.contract()?,
    };
    ensure!(
        client.identity() == (scope.cluster, scope.layout)
            && !scope.table.is_nil()
            && !scope.incarnation.is_nil()
            && !identity.owner.is_nil()
            && !identity.acquisition.is_nil()
            && !identity.manifest.is_nil()
            && !scope.destination.namespace.is_nil()
            && !scope.destination.dataset.is_empty()
            && client.producer_slots() > 0
            && scope.contract.max_record_bytes as usize <= client.max_record_bytes(),
        "invalid bootstrap identity"
    );
    let existing = state(&client, &scope).await?;
    if let Some(manifests) = existing.as_ref().and_then(|v| v.manifests.as_ref()) {
        ensure!(
            manifests.incarnation == scope.incarnation
                && manifests.bootstrap_request == identity.manifest
                && manifests.initial_version == 1,
            "bootstrap identity differs from durable generation"
        );
        // After initialization, use only the catalog-selected manifest. The
        // canonical genesis path may be old; never rediscover visibility there.
        let reader = p::lance::ManagedManifest::reader(
            Arc::new(client.manifest_catalog()),
            storage.base.clone(),
            &layout,
            scope.table,
            scope.incarnation,
        )?;
        let dataset = DatasetBuilder::from_uri(&storage.uri)
            .with_storage_options(storage.options.clone())
            .with_session(storage.session.clone())
            .with_commit_handler(Arc::new(reader))
            .load()
            .await?;
        let schema: arrow_schema::Schema = dataset.schema().into();
        ensure!(
            schema == rollout_schema(),
            "managed generation schema differs"
        );
        return Ok(storage);
    }
    let applied = commit(
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
    let Applied::Publisher(p::Reply::Acquired(authority)) = applied else {
        bail!("bootstrap acquisition rejected: {applied:?}");
    };
    ensure!(
        authority.owner == identity.owner
            && authority.table == scope.table
            && authority.epoch == 1
            && authority.cluster == scope.cluster
            && authority.layout == scope.layout
            && authority.group == client.catalog_group(),
        "foreign bootstrap authority"
    );
    let (object, transaction) = genesis(&storage, &scope, &identity).await?;
    let applied = commit(
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
    let Applied::Publisher(p::Reply::ManifestReserved(selected)) = applied else {
        bail!("genesis selection not confirmed: {applied:?}");
    };
    ensure!(
        selected.version == 1 && selected.object == object && selected.transaction == transaction,
        "catalog selected another genesis"
    );
    Ok(storage)
}

pub(crate) async fn genesis(
    storage: &Storage,
    scope: &Scope,
    identity: &Identity,
) -> Result<(p::ManifestObject, Uuid)> {
    let marker = serde_json::to_string(&(scope, identity))?;
    let parameters = WriteParams {
        mode: WriteMode::Create,
        store_params: Some(parameters(&storage.options)),
        session: Some(storage.session.clone()),
        commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
        transaction_properties: Some(Arc::new(HashMap::from([(
            GENESIS.to_owned(),
            marker.clone(),
        )]))),
        ..Default::default()
    };
    let schema = Arc::new(rollout_schema());
    let empty = RecordBatch::new_empty(schema.clone());
    let dataset = match Dataset::write(
        RecordBatchIterator::new(vec![Ok(empty)], schema),
        &storage.uri,
        Some(parameters),
    )
    .await
    {
        Ok(dataset) => dataset,
        Err(lance::Error::DatasetAlreadyExists { .. }) => {
            DatasetBuilder::from_uri(&storage.uri)
                .with_version(1)
                .with_storage_options(storage.options.clone())
                .with_session(storage.session.clone())
                .with_commit_handler(Arc::new(ConditionalPutCommitHandler))
                .load()
                .await?
        }
        Err(error) => return Err(error.into()),
    };
    let schema: arrow_schema::Schema = dataset.schema().into();
    let transaction = dataset
        .read_transaction_by_version(1)
        .await?
        .context("genesis transaction missing")?;
    ensure!(
        dataset.version().version == 1
            && schema == rollout_schema()
            && dataset.count_rows(None).await? == 0
            && transaction
                .transaction_properties
                .as_ref()
                .and_then(|p| p.get(GENESIS))
                == Some(&marker),
        "refusing to adopt a foreign or nonempty genesis"
    );
    let location = dataset.manifest_location();
    ensure!(
        location.path.parent().and_then(|p| p.parent()).as_ref() == Some(&storage.base),
        "genesis path differs from configured storage"
    );
    let object = p::ManifestObject {
        path: format!(
            "_versions/{}",
            location
                .path
                .filename()
                .context("genesis filename missing")?
        ),
        bytes: location.size.context("genesis size missing")?,
        v2_names: location.naming_scheme == ManifestNamingScheme::V2,
    };
    let transaction = Uuid::parse_str(&transaction.uuid)?;
    Ok((object, transaction))
}
