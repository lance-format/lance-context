//! Prepare one native body branch and bind it to the original narrow cut.
use super::*;
use crate::materializer::observation::Phase;
use lance::dataset::{
    transaction::Operation, CommitBuilder, InsertBuilder, WriteMode, WriteParams,
};
use lance_context_core::annotation_staging::BodySnapshot;
use lance_file::writer::FileWriterOptions;
use std::collections::BTreeMap;

impl FrozenCheckpoint {
    /// The detached body snapshot is private until the managed narrow manifest
    /// selects its exact identity. Retain the original preparation across an
    /// ambiguous catalog result; this method does not authorize source closure.
    pub async fn prepare_mixed(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        batch_rows: usize,
    ) -> Result<PreparedCheckpoint> {
        ensure!(
            (1..=65_536).contains(&batch_rows),
            "invalid mixed publication batch rows"
        );
        let storage = self
            .origin
            .body
            .as_ref()
            .context("not a mixed checkpoint")?;
        let mut prepared = self.body_preparation.lock().await;
        if prepared.is_none() {
            let phases = self.origin.phases.clone();
            let snapshot = phases
                .measure(Phase::AnnotationBodyPrepare, async {
                    let prior = BodySnapshot::read(self.annotations.base())
                        .await?
                        .context("mixed base has no body reference")?;
                    let base = Arc::new(
                        open_body(storage, self.annotations.base(), &prior.table_instance).await?,
                    );
                    let reservation = workspace.reserve(workspace.limit() / 2).await;
                    let mut selection = BTreeMap::<u64, Vec<(u32, &str)>>::new();
                    for (turn, source) in self.annotations.new_turns() {
                        selection
                            .entry(source.record_index)
                            .or_default()
                            .push((source.row, turn));
                    }
                    let mut batches = Vec::new();
                    for (index, mut rows) in selection {
                        rows.sort_unstable_by_key(|r| r.0);
                        batches.extend(
                            self.bodies
                                .get(&index)
                                .context("frozen first-create body frame is missing")?
                                .select(&rows)?,
                        );
                    }
                    let selected = if batches.is_empty() {
                        base
                    } else {
                        // The input frames own their apply reservations. This separate
                        // writer allowance remains with the admitted task if its
                        // caller is cancelled, including any Arrow encoder children.
                        let frames = self.bodies.clone();
                        let buffer = workspace.limit() / 8;
                        let maximum = workspace.limit();
                        let original = self
                            .annotations
                            .base()
                            .read_transaction()
                            .await?
                            .context("mixed base has no native transaction")?;
                        let cut = original
                            .transaction_properties
                            .as_ref()
                            .and_then(|p| p.get("lance-context.annotation-cut"))
                            .context("mixed base has no application scope")?;
                        let (scope, _): (StagingScope, u64) = serde_json::from_str(cut)?;
                        let cut = serde_json::to_string(&(scope, self.through()))?;
                        let identity = self.identity().to_owned();
                        let phases = phases.clone();
                        Arc::new(
                            tokio::spawn(async move {
                                let _frames = frames;
                                let _reservation = reservation;
                                let mut transaction = phases
                                    .measure(
                                        Phase::AnnotationBodyWrite,
                                        InsertBuilder::new(base.clone())
                                            .with_params(&WriteParams {
                                                mode: WriteMode::Append,
                                                max_bytes_per_file: maximum,
                                                skip_auto_cleanup: true,
                                                file_writer_options: Some(FileWriterOptions {
                                                    data_cache_bytes: Some(buffer as u64),
                                                    max_page_bytes: Some(buffer as u64),
                                                    ..Default::default()
                                                }),
                                                ..Default::default()
                                            })
                                            .execute_uncommitted(batches),
                                    )
                                    .await?;
                                ensure!(
                                    matches!(transaction.operation, Operation::Append { .. }),
                                    "body preparation is not append-only"
                                );
                                transaction.uuid = identity;
                                transaction.transaction_properties =
                                    Some(Arc::new(std::collections::HashMap::from([(
                                        "lance-context.annotation-cut".into(),
                                        cut,
                                    )])));
                                // Detached Lance commits count attempts: one is one
                                // attempt, with no conflict retry or mainline selection.
                                Ok::<_, anyhow::Error>(
                                    phases
                                        .measure(
                                            Phase::AnnotationBodyCommit,
                                            CommitBuilder::new(base)
                                                .with_detached(true)
                                                .with_max_retries(1)
                                                .with_skip_auto_cleanup(true)
                                                .execute(transaction),
                                        )
                                        .await?,
                                )
                            })
                            .await??,
                        )
                    };
                    BodySnapshot::capture(prior.table_instance, &selected)
                        .await
                        .map_err(anyhow::Error::from)
                })
                .await?;
            *prepared = Some(snapshot);
        }
        self.origin
            .phases
            .measure(
                Phase::AnnotationNarrowPrepare,
                self.prepare_with_body(workspace, batch_rows, prepared.as_ref().unwrap().clone()),
            )
            .await
    }
}
