use super::*;
use futures::TryStreamExt;
use lance_context_core::annotation_staging::{TurnBodySource, TurnRead};

enum BodyHold {
    Staged { _row: frames::HeldRow },
    Materialized { _reservation: MergeReservation },
}

/// Keep this response alive through encoding/streaming. It retains the frame
/// or read reservation and the compatible base; body columns share Arrow arrays.
pub struct Turn {
    pub annotation: TurnAnnotation,
    body: RecordBatch,
    _hold: BodyHold,
    _view: TurnRead,
}

impl Turn {
    /// Generic bodies preserve their original fields. Legacy rollout bodies
    /// expose the current annotation score in their existing score column.
    pub fn body(&self) -> &RecordBatch {
        &self.body
    }
}

impl Service {
    #[cfg(test)]
    pub(crate) async fn capture_turn_for_test(&self, turn: &str) -> Result<TurnRead> {
        let _ordered = self.shared.ordered.lock().await;
        Ok(self.shared.staging.capture_turn(turn).await?)
    }

    pub async fn get_turn(&self, turn: &str, deadline: Duration) -> Result<Option<Turn>> {
        self.validate_turn(turn)?;
        ensure!(
            self.shared.body.is_some(),
            "full mixed reads require body storage"
        );
        tokio::time::timeout(deadline, async {
            let through = self.barrier().await?;
            self.wait_through(through).await?;
            self.get_turn_applied(turn).await
        })
        .await
        .context("mixed turn read deadline exceeded")?
    }

    async fn get_turn_applied(&self, turn: &str) -> Result<Option<Turn>> {
        let (view, staged) = {
            // Only capture/clone under the gate; never hold it for body I/O.
            let _ordered = self.shared.ordered.lock().await;
            let view = self.shared.staging.capture_turn(turn).await?;
            let staged = match view.body_source() {
                TurnBodySource::Unmaterialized(source) => {
                    Some(self.shared.frames.lock().unwrap().row(source, turn)?)
                }
                TurnBodySource::Materialized(_) => None,
            };
            (view, staged)
        };
        let Some(annotation) = view.annotations().resolve().await? else {
            return Ok(None);
        };
        let (batch, hold) = if let Some(row) = staged {
            (row.batch.clone(), BodyHold::Staged { _row: row })
        } else {
            let TurnBodySource::Materialized(base) = view.body_source() else {
                bail!("mixed read lost its captured body location");
            };
            let bytes = (self.ingress.scope.contract.max_record_bytes as usize)
                .checked_mul(3)
                .context("body read workspace overflow")?;
            let reservation = self.ingress.budget.reserve(bytes).await;
            let body = open_body(
                self.shared.body.as_ref().unwrap(),
                base,
                &self.ingress.scope.incarnation.to_string(),
            )
            .await?;
            let mut scan = body.scan();
            let filter = if let Some(spec) = &self.ingress.turn_spec {
                spec.primary_key()
                    .iter()
                    .zip(spec.key_values(turn)?)
                    .map(|(name, value)| {
                        format!(
                            "`{}` = '{}'",
                            name.replace('`', "``"),
                            value.replace('\'', "''")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" AND ")
            } else {
                format!("id = '{}'", turn.replace('\'', "''"))
            };
            scan.filter(&filter)?;
            scan.limit(Some(2), None)?;
            scan.batch_size(128)
                .batch_readahead(1)
                .fragment_readahead(1)
                .io_buffer_size(self.ingress.scope.contract.max_record_bytes as u64);
            let mut stream = scan.try_into_stream().await?;
            let mut found = None;
            while let Some(batch) = stream.try_next().await? {
                if batch.num_rows() == 0 {
                    continue;
                }
                ensure!(
                    batch.num_rows() == 1 && found.is_none(),
                    "duplicate materialized body ID"
                );
                found = Some(batch);
            }
            (
                found.context("selected annotation has no materialized body")?,
                BodyHold::Materialized {
                    _reservation: reservation,
                },
            )
        };
        let mut columns = batch.columns().to_vec();
        if self.ingress.turn_spec.is_none() {
            columns[batch.schema().index_of("score")?] =
                Arc::new(arrow_array::Float32Array::from(vec![annotation.score]));
        }
        let body = RecordBatch::try_new(batch.schema(), columns)?;
        Ok(Some(Turn {
            annotation,
            body,
            _hold: hold,
            _view: view,
        }))
    }
}
