//! First-create bodies and annotation patches in one ordered source stream.
use super::*;

pub(super) mod frames;
mod prepare;
mod read;
pub mod wire;
pub(super) use prepare::BodyWriteFailed;
pub use read::Turn;

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
    load_body(storage, &reference, incarnation).await
}

pub(crate) async fn load_body(
    storage: &Storage,
    reference: &lance_context_core::annotation_staging::BodySnapshot,
    incarnation: &str,
) -> Result<lance::Dataset> {
    ensure!(
        reference.uri == storage.uri && reference.table_instance == incarnation,
        "body snapshot differs from the authorized destination"
    );
    let body = lance::dataset::builder::DatasetBuilder::from_uri(&storage.uri)
        .with_storage_options(storage.options.clone())
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
    ensure!(
        schema == rollout_schema(),
        "body snapshot has a foreign schema"
    );
    Ok(body)
}
