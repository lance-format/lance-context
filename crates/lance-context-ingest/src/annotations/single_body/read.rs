use super::*;
use futures::TryStreamExt;
use lance_context_core::annotation_staging::{ImmutableBodyRead, TurnBodySource};

enum Hold {
    Staged { _row: mixed::frames::HeldRow },
    Materialized { _reservation: MergeReservation },
}

/// Retain through response encoding. Staged data shares its original Arrow
/// frame; materialized data retains its compatible base and scan reservation.
pub struct Body {
    batch: RecordBatch,
    _hold: Hold,
    _view: ImmutableBodyRead,
}

impl Body {
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

impl Service {
    /// Quorum barrier then local applied view. This read never scans WAL data
    /// and never consults an independently opened latest Lance snapshot.
    pub async fn get_full_body(&self, key: &str, deadline: Duration) -> Result<Option<Body>> {
        self.get_full_body_at(key, None, deadline).await
    }

    /// The HTTP owner fences the completed result. Its recovery cut covers
    /// predecessor writes; its own ACKs already await local application.
    #[cfg(feature = "http")]
    pub(crate) async fn get_full_body_owned(
        &self,
        key: &str,
        recovery_cut: u64,
        deadline: Duration,
    ) -> Result<Option<Body>> {
        self.get_full_body_at(key, Some(recovery_cut), deadline)
            .await
    }

    async fn get_full_body_at(
        &self,
        key: &str,
        owned_cut: Option<u64>,
        deadline: Duration,
    ) -> Result<Option<Body>> {
        ensure!(self.is_single_body(), "not a single body service");
        self.validate_turn(key)?;
        tokio::time::timeout(deadline, async {
            let through = match owned_cut {
                Some(cut) => cut,
                None => self.barrier().await?,
            };
            self.wait_through(through).await?;
            let (view, staged) = {
                let _ordered = self.shared.ordered.lock().await;
                let view = self.shared.staging.capture_immutable_body(key).await?;
                let staged = match view.source() {
                    TurnBodySource::Unmaterialized(source) => {
                        Some(self.shared.frames.lock().unwrap().row(source, key)?)
                    }
                    TurnBodySource::Materialized(_) => None,
                };
                (view, staged)
            };
            let (batch, hold) = if let Some(row) = staged {
                (row.batch.clone(), Hold::Staged { _row: row })
            } else {
                let TurnBodySource::Materialized(base) = view.source() else {
                    bail!("body read lost its captured base")
                };
                let bytes = (self.ingress.scope.contract.max_record_bytes as usize)
                    .checked_mul(3)
                    .context("body read budget overflow")?;
                let reservation = if owned_cut.is_some() {
                    // A forwarder or another body read may already retain
                    // this shared pool. Refuse rather than wait while holding
                    // a captured view and blocking its eventual reclamation.
                    self.ingress
                        .budget
                        .try_reserve(bytes)
                        .context("body read memory unavailable")?
                } else {
                    self.ingress.budget.reserve(bytes).await
                };
                let spec = self.ingress.turn_spec.as_ref().unwrap();
                let filter = spec
                    .primary_key()
                    .iter()
                    .zip(spec.key_values(key)?)
                    .map(|(name, value)| {
                        format!(
                            "`{}` = '{}'",
                            name.replace('`', "``"),
                            value.replace('\'', "''")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" AND ");
                let mut scan = base.scan();
                scan.filter(&filter)?.limit(Some(2), None)?;
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
                        "duplicate immutable body key"
                    );
                    found = Some(batch);
                }
                let Some(batch) = found else { return Ok(None) };
                (
                    batch,
                    Hold::Materialized {
                        _reservation: reservation,
                    },
                )
            };
            // Return the user's exact original schema, sharing all columns.
            let batch = RecordBatch::try_new(
                self.ingress
                    .turn_spec
                    .as_ref()
                    .unwrap()
                    .body_schema()
                    .clone(),
                batch.columns().to_vec(),
            )?;
            Ok(Some(Body {
                batch,
                _hold: hold,
                _view: view,
            }))
        })
        .await
        .context("full body read deadline exceeded")?
    }
}
