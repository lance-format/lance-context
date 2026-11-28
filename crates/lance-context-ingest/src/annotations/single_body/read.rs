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
    #[cfg(all(test, feature = "http-client", feature = "service"))]
    pub(crate) fn apply_budget_for_test(&self) -> &Arc<MergeMemoryBudget> {
        &self.apply_budget
    }

    #[cfg(all(test, feature = "http-client", feature = "service"))]
    pub(crate) fn pause_body_fetch_for_test(
        &self,
    ) -> (
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
    ) {
        let (entered, captured) = tokio::sync::oneshot::channel();
        let (resume, wait) = tokio::sync::oneshot::channel();
        assert!(self
            .shared
            .pause_body_fetch
            .lock()
            .unwrap()
            .replace((entered, wait))
            .is_none());
        (captured, resume)
    }

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
        let covers_index = {
            let _ordered = self.shared.ordered.lock().await;
            self.shared
                .staging
                .recapture_immutable_installed(view, index)
                .await?
        };
        ensure!(
            covers_index,
            "installed view does not cover the retired WAL payload"
        );
        let TurnBodySource::Materialized(base) = view.source() else {
            bail!("installed recapture is not a materialized base")
        };
        ensure!(
            base.version().version >= closure.version,
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
                #[cfg(all(test, feature = "http-client", feature = "service"))]
                {
                    let pause = self.shared.pause_body_fetch.lock().unwrap().take();
                    if let Some((entered, wait)) = pause {
                        let _ = entered.send(());
                        wait.await
                            .context("test released the body fetch gate without resuming")?;
                    }
                }
                // Capture -> Close -> Fetch: publication can retire the WAL
                // record after the ordered capture. The row is then in the
                // installed base; recapture it below and read from Lance.
                let retry_until = tokio::time::Instant::now() + Duration::from_secs(2);
                loop {
                    match self
                        .try_refetch_bodies(source.record_index, meta.clone())
                        .await
                    {
                        Ok(Some(fetched)) => {
                            refetched = Some((source, fetched));
                            break;
                        }
                        Ok(None) => {
                            // Admission can refuse before the Fetch reaches the
                            // WAL. Use the already-installed cut when it covers
                            // the captured record, even if the WAL is unreadable.
                            if self
                                .shared
                                .staging
                                .recapture_immutable_installed(&mut view, source.record_index)
                                .await?
                            {
                                recaptured = true;
                                break;
                            }
                            ensure!(
                                tokio::time::Instant::now() < retry_until,
                                "owner batch body refetch memory unavailable"
                            );
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                        Err(error) if error.is::<super::prefetch::PayloadRetired>() => {
                            self.recapture_retired(&mut view, source.record_index)
                                .await?;
                            recaptured = true;
                            break;
                        }
                        Err(error) => return Err(error),
                    }
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
