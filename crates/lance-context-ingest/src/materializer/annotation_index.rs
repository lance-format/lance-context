//! One bounded native index preparation that leaves data publication available.
use super::*;
use session_index::{Action, Key};

/// Native writers retain their workspace until they finish, including after
/// owner cancellation. Only the owning publisher can select their descriptors.
pub(crate) struct Preparation {
    pub(crate) source: Dataset,
    key: Key,
    job: tokio::task::JoinHandle<Result<p::Descriptors>>,
}

impl Preparation {
    pub(crate) fn is_finished(&self) -> bool {
        self.job.is_finished()
    }

    pub(crate) async fn finish(self, current: &Dataset) -> Result<Option<p::Descriptors>> {
        let descriptors = self.job.await??;
        match self.key {
            Key::Turn => Ok(Some(
                session_index::rebase_turn(&self.source, current, &descriptors).await?,
            )),
            Key::Status => session_index::rebase_status(&self.source, current, &descriptors).await,
            Key::Session => unreachable!("annotation preparation selects only turn or status"),
        }
    }
}

pub(crate) async fn start(publisher: &Publisher, version: u64) -> Result<Option<Preparation>> {
    if publisher.options.turn_id_index.is_none() && publisher.options.status_index.is_none() {
        return Ok(None);
    }
    // Keep an ordinary publication's workspace available while the one native
    // job is running. Smaller configurations retain the serial bounded path.
    let publication_workspace = publisher.options.workspace(
        &publisher.ingress.scope.contract,
        publisher.options.max_source_bytes,
        publisher.options.max_records,
    )?;
    if publisher.budget.limit() < session_index::WORKSPACE + publication_workspace {
        return Ok(None);
    }
    if Publisher::compaction_first(version) && compaction::has_work(publisher, version).await? {
        return Ok(None);
    }
    let source = compaction::read_compaction_base(publisher, version).await?;
    let status_first = Publisher::status_index_first(version);
    let mut selected: Option<(Action, Key)> = None;
    if status_first {
        if let Some(options) = &publisher.options.status_index {
            selected = options.action(&source, 1).await?.map(|a| (a, Key::Status));
        }
    }
    if selected.is_none() {
        if let Some(options) = &publisher.options.turn_id_index {
            selected = options.action(&source).await?.map(|a| (a, Key::Turn));
        }
    }
    if selected.is_none() && !status_first {
        if let Some(options) = &publisher.options.status_index {
            selected = options.action(&source, 1).await?.map(|a| (a, Key::Status));
        }
    }
    let Some((action, key)) = selected else {
        return Ok(None);
    };
    let Some(reservation) = publisher.budget.try_reserve(session_index::WORKSPACE) else {
        return Ok(None);
    };
    let job = session_index::spawn_action(
        source.clone(),
        action,
        key,
        reservation,
        publisher.phases.clone(),
    );
    Ok(Some(Preparation { source, key, job }))
}
