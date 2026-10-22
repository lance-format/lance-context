use super::*;
use lance::dataset::{
    transaction::{Operation, Transaction},
    CommitBuilder, InsertBuilder, WriteMode, WriteParams,
};
use lance_table::format::Fragment;

/// A cancelled caller detaches this bounded job rather than cancelling its
/// admitted file writers. The reservation stays with those writers until done.
pub(super) async fn materialize(
    ingress: Ingress,
    options: Options,
    pending: Arc<p::PendingPlan>,
    dataset: Dataset,
    reservation: Arc<MergeReservation>,
    phases: Arc<PhaseCounters>,
) -> Result<p::Descriptors> {
    let inputs = pending
        .plan
        .sources
        .iter()
        .flat_map(|(&group, range)| {
            range
                .records
                .iter()
                .map(move |(index, meta)| (group, *index, meta.clone()))
        })
        .collect::<Vec<_>>();
    materialize_inputs(ingress, options, inputs, dataset, reservation, phases).await
}

pub(crate) async fn materialize_inputs(
    ingress: Ingress,
    options: Options,
    inputs: Vec<(u64, u64, BatchMeta)>,
    dataset: Dataset,
    reservation: Arc<MergeReservation>,
    phases: Arc<PhaseCounters>,
) -> Result<p::Descriptors> {
    tokio::spawn(async move {
        let _reservation = reservation.clone();
        if options.pipeline_preparation {
            return super::pipeline::materialize(
                ingress,
                options,
                inputs,
                dataset,
                reservation,
                phases,
            )
            .await;
        }
        let decoded = stream::iter(inputs.into_iter().map(|(group, index, meta)| {
            fetch(
                ingress.clone(),
                (group, index, meta),
                options.source_read_policy,
                reservation.clone(),
                phases.clone(),
                None,
            )
        }))
        .buffered(options.fetch_parallelism)
        .try_collect::<Vec<_>>()
        .await?;
        // Share Arrow buffers across a bounded set of file writers. Each source
        // record remains whole; writer count cannot change retry/plan identity.
        let writers = options.write_parallelism.min(decoded.len());
        ensure!(writers > 0, "empty pinned source selection");
        let mut partitions = (0..writers).map(|_| Vec::new()).collect::<Vec<_>>();
        for (i, batches) in decoded.into_iter().enumerate() {
            partitions[i % writers].extend(batches);
        }
        let dataset = Arc::new(dataset);
        let tasks = partitions.into_iter().map(|batches| {
            let dataset = dataset.clone();
            let phases = phases.clone();
            let reservation = reservation.clone();
            async move {
                let _reservation = reservation;
                let staged = phases
                    .measure(
                        Phase::BlobWrite,
                        InsertBuilder::new(dataset)
                            .with_params(&WriteParams {
                                mode: WriteMode::Append,
                                ..Default::default()
                            })
                            .execute_uncommitted(batches),
                    )
                    .await?;
                let Operation::Append { fragments } = staged.operation else {
                    bail!("materialization did not produce append fragments");
                };
                Ok::<_, anyhow::Error>(fragments)
            }
        });
        // Join every admitted writer even if another writer fails. Completed
        // orphan files are retained; no unverified deletion is attempted here.
        let results = futures::future::join_all(tasks).await;
        let fragments = results
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        p::Descriptors::lance_fragments(Bytes::from(serde_json::to_vec(&fragments)?))
    })
    .await?
}

/// The optional shared permit covers both the RPC and blocking decode. Moving
/// it into the blocking job prevents cancelled streams from releasing capacity
/// while that job still uses the admitted workspace.
pub(super) async fn fetch(
    ingress: Ingress,
    (group, index, meta): (u64, u64, BatchMeta),
    policy: SourceReadPolicy,
    reservation: Arc<MergeReservation>,
    phases: Arc<PhaseCounters>,
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> Result<Vec<RecordBatch>> {
    let response = phases
        .measure(Phase::Fetch, async {
            let mut fetched = None;
            for node in policy.candidates(&ingress.layout.groups[&group], group, index) {
                if let Ok(response) = ingress
                    .client
                    .request(node, Request::Fetch { group, index })
                    .await
                {
                    if let Reply::Record(Some(observed)) = &response.reply {
                        ensure!(
                            observed == &meta,
                            "fetched record differs from durable source pin"
                        );
                        fetched = Some(response);
                        break;
                    }
                }
            }
            fetched.context("pinned source unavailable on every voter")
        })
        .await?;
    tokio::task::spawn_blocking(move || {
        let _reservation = reservation;
        let _permit = permit;
        phases.blocking(Phase::DecodeValidate, || {
            meta.validate_payload(
                &response.body,
                ingress.scope.contract.max_record_bytes as usize,
            )?;
            let batches = ipc::decode(response.body, ingress.schema.clone())?;
            validate_rows(
                &batches,
                &ingress.scope.contract,
                &ingress.layout,
                meta.bucket,
            )?;
            Ok(batches)
        })
    })
    .await?
}

pub(super) async fn publish(
    publisher: &Publisher,
    pending: &p::PendingPlan,
    prepared: &p::Prepared,
    predecessor: &p::ManifestRecord,
    base: Dataset,
) -> Result<()> {
    prepared
        .descriptors
        .validate(publisher.ingress.client.max_descriptor_bytes())?;
    let fragments: Vec<Fragment> = serde_json::from_slice(&prepared.descriptors.bytes)?;
    ensure!(!fragments.is_empty(), "empty prepared fragment list");
    let mut transaction = Transaction::new(
        pending.plan.read_version,
        Operation::Append { fragments },
        None,
    );
    transaction.uuid = pending.plan.transaction.to_string();
    let handler = p::lance::ManagedManifest::new(
        Arc::new(publisher.ingress.client.manifest_catalog()),
        publisher.storage.base.clone(),
        &publisher.ingress.layout,
        p::lance::ManifestCommit {
            authority: publisher.authority.clone(),
            incarnation: publisher.ingress.scope.incarnation,
            transaction: pending.plan.transaction,
            plan_index: pending.index,
            prepared_index: prepared.index,
            descriptor_digest: prepared.descriptors.digest,
            predecessor: predecessor.identity,
            target: pending.plan.target_version()?,
        },
    )?;
    CommitBuilder::new(Arc::new(base))
        .with_commit_handler(Arc::new(handler))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    Ok(())
}
