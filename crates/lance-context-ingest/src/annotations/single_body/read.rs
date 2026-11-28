use super::*;
use futures::TryStreamExt;
use lance_context_core::annotation_staging::{ImmutableBodyRead, TurnBodySource};

enum Hold {
    Staged {
        _row: mixed::frames::HeldRow,
    },
    /// Owner batch: the staged body was refetched from its WAL record.
    Refetched {
        _reservation: MergeReservation,
    },
    Materialized,
}

/// Retain through response encoding. Staged data shares its original Arrow
/// frame; materialized data retains its compatible base and scan reservation.
pub struct Body {
    batch: RecordBatch,
    _hold: Hold,
    _view: ImmutableBodyRead,
    _workspace: Option<MergeReservation>,
}

impl Body {
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

impl Service {
    /// The staged source at `index` was retired from the WAL. Recapture this
    /// owner's installed compatible view and require it to cover `index` for
    /// exactly this table, group and incarnation (the source closure's cut
    /// and version). Uncovered or foreign metadata stays an error.
    async fn recapture_retired(&self, view: &mut ImmutableBodyRead, index: u64) -> Result<()> {
        let (_, value) = crate::reader::observe(
            &self.ingress,
            self.group()?,
            query::Query::SourceClosure {
                table: self.ingress.scope.table,
            },
        )
        .await?;
        let query::Value::SourceClosure(closed) = value else {
            bail!("source closure unavailable");
        };
        let closed = closed.context("retired WAL payload without a source closure")?;
        let closure = closed.permit.closure();
        ensure!(
            closure.table == self.ingress.scope.table
                && closure.group == self.group()?
                && closure.incarnation == self.ingress.scope.incarnation,
            "foreign source closure"
        );
        ensure!(
            index <= closure.through,
            "unpublished WAL payload unavailable"
        );
        let materialized = {
            let _ordered = self.shared.ordered.lock().await;
            self.shared
                .staging
                .recapture_immutable_installed(view)
                .await?
        };
        let TurnBodySource::Materialized(base) = view.source() else {
            bail!("installed recapture is not a materialized base")
        };
        ensure!(
            materialized >= index && base.version().version >= closure.version,
            "installed view does not cover the retired WAL payload"
        );
        Ok(())
    }

    /// Quorum barrier then local applied view. This read never scans WAL data
    /// and never consults an independently opened latest Lance snapshot.
    pub async fn get_full_body(&self, key: &str, deadline: Duration) -> Result<Option<Body>> {
        self.read_full_body(key, None, deadline, None).await
    }

    /// The HTTP owner fences the result after encoding. Admit read and output
    /// workspace together before entering, so response encoding never waits
    /// for more credit while retaining a read from that same budget.
    #[cfg(feature = "http")]
    pub(crate) async fn get_full_body_owned(
        &self,
        key: &str,
        recovery_cut: u64,
        deadline: Duration,
        workspace: MergeReservation,
    ) -> Result<Option<Body>> {
        self.read_full_body(key, Some(recovery_cut), deadline, Some(workspace))
            .await
    }

    async fn read_full_body(
        &self,
        key: &str,
        recovery_cut: Option<u64>,
        deadline: Duration,
        mut workspace: Option<MergeReservation>,
    ) -> Result<Option<Body>> {
        ensure!(self.is_single_body(), "not a single body service");
        self.validate_turn(key)?;
        tokio::time::timeout(deadline, async {
            let through = match recovery_cut {
                Some(cut) => cut,
                None => self.barrier().await?,
            };
            self.wait_through(through).await?;
            let (mut view, staged, deferred) = {
                let _ordered = self.shared.ordered.lock().await;
                let view = self.shared.staging.capture_immutable_body(key).await?;
                let mut deferred = None;
                let staged = match view.source() {
                    TurnBodySource::Unmaterialized(source) => {
                        let meta = self
                            .shared
                            .write_batch
                            .lock()
                            .unwrap()
                            .as_ref()
                            .and_then(|batch| batch.deferred(source.record_index));
                        match meta {
                            Some(meta) => {
                                deferred = Some((source, meta));
                                None
                            }
                            None => Some(self.shared.frames.lock().unwrap().row(source, key)?),
                        }
                    }
                    TurnBodySource::Materialized(_) => None,
                };
                (view, staged, deferred)
            };
            let mut refetched = None;
            let mut recaptured = false;
            if let Some((source, meta)) = deferred {
                // Capture -> Close -> Fetch: publication can retire the WAL
                // record after the ordered capture. The row is then in the
                // installed base; recapture it below and read from Lance.
                match self.refetch_bodies(source.record_index, meta).await {
                    Ok(fetched) => refetched = Some((source, fetched)),
                    Err(error) if error.is::<super::prefetch::PayloadRetired>() => {
                        self.recapture_retired(&mut view, source.record_index)
                            .await?;
                        recaptured = true;
                    }
                    Err(error) => return Err(error),
                }
            }
            let (batch, hold) = if let Some(row) = staged {
                (row.batch.clone(), Hold::Staged { _row: row })
            } else if let Some((source, (bodies, reservation))) = refetched {
                let spec = self.ingress.turn_spec.as_ref().unwrap();
                let mut row = source.row as usize;
                let mut found = None;
                for batch in &bodies {
                    if row < batch.num_rows() {
                        let original = RecordBatch::try_new(
                            spec.body_schema().clone(),
                            batch.columns().to_vec(),
                        )?;
                        ensure!(
                            spec.row_key(&original, row, key.len())? == key,
                            "refetched body row differs from turn origin"
                        );
                        found = Some(batch.slice(row, 1));
                        break;
                    }
                    row -= batch.num_rows();
                }
                (
                    found.context("refetched body row is outside its source record")?,
                    Hold::Refetched {
                        _reservation: reservation,
                    },
                )
            } else {
                let TurnBodySource::Materialized(base) = view.source() else {
                    bail!("body read lost its captured base")
                };
                if workspace.is_none() {
                    let bytes = (self.ingress.scope.contract.max_record_bytes as usize)
                        .checked_mul(3)
                        .context("body read budget overflow")?;
                    workspace = Some(self.ingress.budget.reserve(bytes).await);
                }
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
                let Some(batch) = found else {
                    // A recaptured base was proven to cover the retired WAL
                    // record that staged this key; absence is corruption.
                    ensure!(
                        !recaptured,
                        "installed base covers the retired WAL payload but lacks its body"
                    );
                    return Ok(None);
                };
                (batch, Hold::Materialized)
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
                _workspace: workspace,
            }))
        })
        .await
        .context("full body read deadline exceeded")?
    }
}
