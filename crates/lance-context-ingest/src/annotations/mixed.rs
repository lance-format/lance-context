//! First-create bodies and annotation patches in one ordered source stream.
use super::*;

pub(super) mod frames;
mod prepare;
mod read;
pub mod wire;
pub use read::Turn;

/// A catalog-selected, exact body snapshot and its covered WAL prefix. This is
/// not a reader lease or permission to advance an unrelated source checkpoint:
/// callers must also account for their associations and all source outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectedBody {
    pub cluster: Uuid,
    pub layout: [u8; 32],
    pub table: Uuid,
    pub schema: [u8; 32],
    pub source: lance_context_core::annotation_staging::Scope,
    pub through: u64,
    pub narrow_version: u64,
    pub narrow_transaction: String,
    pub body: lance_context_core::annotation_staging::BodySnapshot,
}

/// A forwarding ingress validates the explicit mixed contract but opens no
/// Lance dataset or apply state. Requests retain their original Arrow bytes.
pub async fn connect_ingress(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    limits: &Limits,
    budget: Arc<MergeMemoryBudget>,
) -> Result<Ingress> {
    let contract = Service::mixed_contract(&binding, limits)?;
    Ingress::connect_with_schema(
        client,
        layout,
        binding,
        budget,
        contract,
        wire::schema(limits),
    )
    .await
}

/// Resolve one configured physical shard without opening a Lance read view.
pub async fn connect_shard_ingress(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    limits: &Limits,
    budget: Arc<MergeMemoryBudget>,
    shard: Shard,
) -> Result<Ingress> {
    layout.group_for_bucket(shard.bucket)?;
    let contract = shard.contract(&binding, limits, true)?;
    Ingress::connect_with_schema(
        client,
        layout,
        binding,
        budget,
        contract,
        shard.wire_schema(limits, true)?,
    )
    .await
}

#[allow(clippy::too_many_arguments)] // Explicit body specification augments shard routing.
pub async fn connect_turn_ingress(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    limits: &Limits,
    budget: Arc<MergeMemoryBudget>,
    shard: Shard,
    spec: &crate::turn_ingestion::TurnIngestionSpec,
) -> Result<Ingress> {
    layout.group_for_bucket(shard.bucket)?;
    let contract = shard.turn_contract(&binding, spec, limits)?;
    Ingress::connect_with_schema(
        client,
        layout,
        binding,
        budget,
        contract,
        shard.turn_schema(spec, limits)?,
    )
    .await
}

pub(super) struct Created {
    pub turn: String,
    pub rollout_id: String,
    pub annotation: TurnAnnotation,
}

pub(super) enum Decoded {
    Patches(Vec<super::wire::Mutation>),
    Creates {
        rows: Vec<Created>,
        bodies: Vec<RecordBatch>,
    },
}

impl Decoded {
    pub(super) fn rows(&self) -> usize {
        match self {
            Self::Patches(rows) => rows.len(),
            Self::Creates { rows, .. } => rows.len(),
        }
    }
}

impl Service {
    #[allow(clippy::too_many_arguments)]
    pub async fn connect_turn_shard(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: &Storage,
        body: Storage,
        limits: Limits,
        budget: Arc<MergeMemoryBudget>,
        apply_budget: Arc<MergeMemoryBudget>,
        shard: Shard,
        spec: &crate::turn_ingestion::TurnIngestionSpec,
    ) -> Result<(Self, Driver)> {
        let ingress = Self::validated_ingress_mode(
            client,
            layout,
            binding,
            &limits,
            budget,
            &apply_budget,
            true,
            Some(shard),
            Some(spec),
        )
        .await?;
        Self::open_mode(ingress, storage, Some(body), limits, apply_budget).await
    }

    /// Read the managed selected version, never the body URI's latest version.
    /// The body may itself be detached. External readers require physical file
    /// reclamation to remain disabled until their retention is integrated.
    pub async fn selected_body(&self, deadline: Duration) -> Result<SelectedBody> {
        let body = self
            .shared
            .body
            .as_ref()
            .context("not a mixed turn service")?;
        tokio::time::timeout(deadline, async {
            let (narrow, snapshot) = checkpoint::open(
                &self.ingress,
                &self.shared.narrow,
                Some(body),
                self.bucket(),
            )
            .await?;
            let reference = lance_context_core::annotation_staging::BodySnapshot::read(&narrow)
                .await?
                .context("selected snapshot has no body reference")?;
            ensure!(
                reference.uri == body.uri
                    && reference.table_instance == self.ingress.scope.incarnation.to_string(),
                "foreign selected body reference"
            );
            Ok(SelectedBody {
                cluster: self.ingress.scope.cluster,
                layout: self.ingress.scope.layout,
                table: self.ingress.scope.table,
                schema: self.ingress.scope.contract.schema,
                source: lance_context_core::annotation_staging::Scope {
                    table_instance: self.ingress.scope.incarnation.to_string(),
                    source_group: self.group()?,
                    shard: self.bucket(),
                },
                through: snapshot.through,
                narrow_version: narrow.version_id(),
                narrow_transaction: narrow
                    .read_transaction()
                    .await?
                    .context("selected transaction missing")?
                    .uuid,
                body: reference,
            })
        })
        .await
        .context("selected body read deadline exceeded")?
    }

    pub fn mixed_contract(binding: &TableBinding, limits: &Limits) -> Result<IngestContract> {
        limits.validate()?;
        Ok(IngestContract {
            policy: WritePolicy::TurnIngestion,
            schema: arrow_schema_fingerprint(&wire::schema(limits))?,
            ..binding.contract()?
        })
    }

    /// Connect only an explicitly provisioned mixed binding. A durable initial
    /// body reference must already be present in its selected narrow snapshot.
    #[allow(clippy::too_many_arguments)] // Mirrors connect with an authorized body destination.
    pub async fn connect_mixed(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: &Storage,
        body: Storage,
        limits: Limits,
        budget: Arc<MergeMemoryBudget>,
        apply_budget: Arc<MergeMemoryBudget>,
    ) -> Result<(Self, Driver)> {
        let ingress = Self::validated_ingress_mode(
            client,
            layout,
            binding,
            &limits,
            budget,
            &apply_budget,
            true,
            None,
            None,
        )
        .await?;
        Self::open_mode(ingress, storage, Some(body), limits, apply_budget).await
    }

    /// Active/frozen window only. Retired reader-held frames still charge the
    /// shared apply budget, but are no longer members of these two maps.
    pub fn retained_body_frames(&self) -> (usize, usize) {
        self.shared.frames.lock().unwrap().retained()
    }
}

pub(super) async fn open_body(
    storage: &Storage,
    narrow: &lance::Dataset,
    incarnation: &str,
) -> Result<lance::Dataset> {
    let reference = lance_context_core::annotation_staging::BodySnapshot::read(narrow)
        .await?
        .context("mixed snapshot has no durable body reference")?;
    let schema: arrow_schema::Schema = narrow.schema().into();
    let spec = crate::turn_ingestion::TurnIngestionSpec::from_schema(&schema)?;
    load_body_with_schema(
        storage,
        &reference,
        incarnation,
        spec.as_ref()
            .map(|s| s.body_schema().as_ref())
            .unwrap_or(&rollout_schema()),
    )
    .await
}

pub(crate) async fn load_body(
    storage: &Storage,
    reference: &lance_context_core::annotation_staging::BodySnapshot,
    incarnation: &str,
) -> Result<lance::Dataset> {
    load_body_with_schema(storage, reference, incarnation, &rollout_schema()).await
}

pub(crate) async fn load_body_with_schema(
    storage: &Storage,
    reference: &lance_context_core::annotation_staging::BodySnapshot,
    incarnation: &str,
    expected: &arrow_schema::Schema,
) -> Result<lance::Dataset> {
    ensure!(
        reference.uri == storage.uri && reference.table_instance == incarnation,
        "body snapshot differs from the authorized destination"
    );
    let body = lance::dataset::builder::DatasetBuilder::from_uri(&storage.uri)
        .with_read_params(storage.read_params(Default::default()))
        .with_session(storage.session.clone())
        .with_version(reference.version)
        .load()
        .await?;
    ensure!(
        body.read_transaction()
            .await?
            .is_some_and(|tx| tx.uuid == reference.transaction),
        "body native identity differs from selected reference"
    );
    let schema: arrow_schema::Schema = body.schema().into();
    ensure!(&schema == expected, "body snapshot has a foreign schema");
    Ok(body)
}
