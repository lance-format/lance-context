//! One bounded physical rewrite that leaves annotation publication available.
use super::*;
use lance::dataset::optimize::RewriteResult;
use lance::index::DatasetIndexExt;

mod outputs;

pub(crate) struct Preparation {
    pub(crate) source: Dataset,
    outputs: Arc<outputs::Outputs>,
    job: tokio::task::JoinHandle<Result<(RewriteResult, MergeReservation)>>,
}

pub(crate) struct Ready {
    base: Dataset,
    result: RewriteResult,
    _reservation: MergeReservation,
}

impl Preparation {
    pub(crate) fn protect_files(&self, candidates: &mut BTreeMap<object_store::path::Path, u64>) {
        self.outputs.protect(candidates);
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.job.is_finished()
    }

    pub(crate) async fn finish(self, current: Dataset) -> Result<Option<Ready>> {
        let (mut result, reservation) = self.job.await??;
        ensure!(
            result.read_version == self.source.version_id(),
            "background compaction returned a foreign source version"
        );
        // A concurrent update may replace data inside a fragment without
        // changing its ID. Compare the complete metadata before reusing files.
        if self.source.uri() != current.uri()
            || self.source.schema() != current.schema()
            || !current.manifest().uses_stable_row_ids()
            || result
                .original_fragments
                .iter()
                .any(|fragment| !current.manifest().fragments.contains(fragment))
        {
            return Ok(None);
        }
        let indices = current.load_indices().await?;
        for index in indices.iter() {
            let Some(coverage) = &index.fragment_bitmap else {
                return Ok(None);
            };
            let mut covered = result
                .original_fragments
                .iter()
                .map(|fragment| coverage.contains(fragment.id as u32));
            let first = covered.next().context("empty background compaction")?;
            if covered.any(|value| value != first)
                || lance_index::is_system_index(index)
                || (first
                    && index.results_are_row_addrs()
                    && (!index.supports_eager_remap() || result.row_addrs.is_none()))
            {
                return Ok(None);
            }
        }
        // Data publications may have consumed IDs while native IO ran. Final
        // IDs and index remaps belong to the current selected version.
        for (ordinal, fragment) in result.new_fragments.iter_mut().enumerate() {
            ensure!(fragment.id == 0, "background rewrite assigned fragment IDs");
            fragment.id = compaction::next_fragment_id(&current, ordinal)?;
        }
        result.read_version = current.version_id();
        compaction::validate_result(&current, &result).await?;
        Ok(Some(Ready {
            base: current,
            result,
            _reservation: reservation,
        }))
    }
}

impl Ready {
    pub(crate) async fn descriptors(
        self,
        publisher: &Publisher,
        pending: &p::PendingPlan,
    ) -> Result<p::Descriptors> {
        let Self {
            base,
            result,
            _reservation,
        } = self;
        ensure!(
            pending.plan.is_maintenance() && pending.plan.read_version == base.version_id(),
            "background rewrite differs from selected maintenance"
        );
        let selection = crate::annotation_publication::Selection {
            table: publisher.authority.table,
            incarnation: publisher.ingress.scope.incarnation,
            pending: pending.clone(),
        };
        crate::annotation_publication::compaction::descriptors(
            &base,
            result,
            &selection,
            &publisher.storage,
            publisher.ingress.client.max_descriptor_bytes(),
            &publisher.phases,
        )
        .await
    }
}

pub(crate) async fn start(publisher: &Publisher, version: u64) -> Result<Option<Preparation>> {
    let Some(options) = &publisher.options.compaction else {
        return Ok(None);
    };
    let output_bytes = usize::try_from(publisher.ingress.client.max_descriptor_bytes())
        .context("compaction output path budget exceeds address space")?;
    let workspace = options
        .workspace(publisher.ingress.scope.contract.max_record_bytes as usize)?
        .checked_add(output_bytes)
        .context("compaction output workspace overflow")?;
    let publication = publisher.options.workspace(
        &publisher.ingress.scope.contract,
        publisher.options.max_source_bytes,
        publisher.options.max_records,
    )?;
    if workspace
        .checked_add(publication)
        .is_none_or(|required| required > publisher.budget.limit())
    {
        return Ok(None);
    }
    let source = compaction::read_compaction_base(publisher, version).await?;
    if !source.manifest().uses_stable_row_ids()
        || source.object_store(None).await?.has_direct_local_paths()
    {
        // Direct local writers bypass ObjectStore. Keep their existing serial
        // path; concurrent object-store writes must all register their outputs.
        return Ok(None);
    }
    let Some(task) = compaction::prepare_task(publisher, &source).await? else {
        return Ok(None);
    };
    if task.task.fragments.iter().any(|f| !f.overlays.is_empty()) {
        return Ok(None);
    }
    let Some(reservation) = publisher.budget.try_reserve(workspace) else {
        return Ok(None);
    };
    let outputs = outputs::Outputs::new(publisher.storage.base.clone(), output_bytes);
    // Pin only this bounded task's inputs, not every old data file in the table.
    for file in task
        .task
        .fragments
        .iter()
        .flat_map(|f| f.referenced_lance_files())
    {
        ensure!(
            file.base_id.is_none(),
            "background compaction has an external source"
        );
        outputs.register(
            &publisher
                .storage
                .base
                .clone()
                .join("data")
                .join(file.path.as_str()),
        )?;
    }
    let wrapper: Arc<dyn lance_io::object_store::WrappingObjectStore> =
        Arc::new(outputs::Wrapper(outputs.clone()));
    let dataset = source.with_object_store_wrappers([wrapper]);
    let phases = publisher.phases.clone();
    let job = tokio::spawn(async move {
        let result = phases
            .measure(
                Phase::BlobWrite,
                phases.measure(Phase::NativeCompaction, task.execute(&dataset)),
            )
            .await?;
        // Keep native admission even if the owner is cancelled or the result
        // waits for another publication. No manifest is selected by this job.
        Ok((result, reservation))
    });
    Ok(Some(Preparation {
        source,
        outputs,
        job,
    }))
}
