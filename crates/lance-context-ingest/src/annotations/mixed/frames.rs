//! Decoded body buffers share the existing apply reservation through publication
//! and admitted reads. Dropping a window never frees a reader's reservation.
use super::*;
use std::collections::BTreeMap;
mod memory;

pub(crate) struct Frame {
    pub batches: Vec<RecordBatch>,
    pub reservation: MergeReservation,
    spec: Option<Arc<crate::turn_ingestion::TurnIngestionSpec>>,
}

pub(crate) struct HeldRow {
    pub batch: RecordBatch,
    _frame: Arc<Frame>,
}

impl Frame {
    fn matches(&self, batch: &RecordBatch, row: usize, turn: &str) -> Result<bool> {
        if let Some(spec) = &self.spec {
            return Ok(spec.row_key(batch, row, turn.len())? == turn);
        }
        let ids = batch
            .column_by_name("id")
            .context("missing body IDs")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("invalid body IDs")?;
        Ok(!ids.is_null(row) && ids.value(row) == turn)
    }
    /// Validate first-create origins and slice consecutive rows together. The
    /// common all-new import keeps each original Arrow batch intact; duplicate
    /// imports only split at omitted rows, without copying body buffers.
    pub fn select(&self, rows: &[(u32, &str)]) -> Result<Vec<RecordBatch>> {
        ensure!(
            rows.windows(2).all(|w| w[0].0 < w[1].0),
            "body selection repeats or reorders a source row"
        );
        let mut selected = Vec::new();
        let mut remaining = rows.iter().peekable();
        let mut offset = 0usize;
        for batch in &self.batches {
            let end = offset
                .checked_add(batch.num_rows())
                .context("body ordinal overflow")?;
            let mut run: Option<(usize, usize)> = None;
            while remaining
                .peek()
                .is_some_and(|(row, _)| (*row as usize) < end)
            {
                let &(row, turn) = remaining.next().unwrap();
                let row = (row as usize)
                    .checked_sub(offset)
                    .context("body ordinal regressed")?;
                ensure!(
                    self.matches(batch, row, turn)?,
                    "body selection differs from first-create identity"
                );
                match &mut run {
                    Some((_, last)) if row == *last + 1 => *last = row,
                    _ => {
                        if let Some((first, last)) = run.take() {
                            selected.push(batch.slice(first, last - first + 1));
                        }
                        run = Some((row, row));
                    }
                }
            }
            if let Some((first, last)) = run {
                selected.push(batch.slice(first, last - first + 1));
            }
            offset = end;
        }
        ensure!(
            remaining.next().is_none(),
            "body selection exceeds original source rows"
        );
        Ok(selected)
    }

    pub fn row(self: &Arc<Self>, ordinal: u32, turn: &str) -> Result<HeldRow> {
        let mut row = ordinal as usize;
        for batch in &self.batches {
            if row < batch.num_rows() {
                ensure!(
                    self.matches(batch, row, turn)?,
                    "retained body row differs from turn origin"
                );
                return Ok(HeldRow {
                    batch: batch.slice(row, 1),
                    _frame: self.clone(),
                });
            }
            row -= batch.num_rows();
        }
        bail!("retained body row is outside the original source record")
    }
}

pub(crate) type Frozen = Arc<BTreeMap<u64, Arc<Frame>>>;

#[derive(Default)]
pub(crate) struct Window {
    active: BTreeMap<u64, Arc<Frame>>,
    frozen: Option<Frozen>,
}

pub(crate) struct RetainedFrameObservation {
    pub reservation_bytes: usize,
    pub arrow_backing_bytes: Option<usize>,
}

impl Window {
    #[cfg(test)]
    pub fn record(
        &mut self,
        index: u64,
        batches: Vec<RecordBatch>,
        reservation: MergeReservation,
        input_bytes: usize,
        input_allocation: Option<(usize, usize)>,
    ) {
        self.record_with_spec(
            index,
            batches,
            reservation,
            input_bytes,
            input_allocation,
            None,
        );
    }

    pub fn record_with_spec(
        &mut self,
        index: u64,
        batches: Vec<RecordBatch>,
        mut reservation: MergeReservation,
        input_bytes: usize,
        input_allocation: Option<(usize, usize)>,
        spec: Option<Arc<crate::turn_ingestion::TurnIngestionSpec>>,
    ) -> RetainedFrameObservation {
        let retention =
            memory::retained_bytes(&batches, input_bytes, reservation.bytes(), input_allocation);
        if let Some(retention) = &retention {
            reservation.shrink_to(retention.charge);
        }
        let observed = RetainedFrameObservation {
            reservation_bytes: reservation.bytes(),
            arrow_backing_bytes: retention.map(|retention| retention.backing_bytes),
        };
        assert!(self
            .active
            .insert(
                index,
                Arc::new(Frame {
                    batches,
                    reservation,
                    spec,
                })
            )
            .is_none());
        observed
    }

    pub fn row(
        &self,
        source: lance_context_core::annotation_staging::NewTurnSource,
        turn: &str,
    ) -> Result<HeldRow> {
        self.active
            .get(&source.record_index)
            .or_else(|| {
                self.frozen
                    .as_ref()
                    .and_then(|f| f.get(&source.record_index))
            })
            .context("pending body frame is unavailable at the captured cut")?
            .row(source.row, turn)
    }

    pub fn freeze(&mut self) -> Frozen {
        assert!(self.frozen.is_none());
        let frozen = Arc::new(std::mem::take(&mut self.active));
        self.frozen = Some(frozen.clone());
        frozen
    }

    pub fn is_current(&self, frozen: &Frozen) -> bool {
        self.frozen
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, frozen))
    }

    pub fn retire(&mut self, frozen: &Frozen) {
        assert!(self.is_current(frozen));
        self.frozen = None;
    }

    pub fn retained(&self) -> (usize, usize) {
        self.active
            .values()
            .chain(self.frozen.iter().flat_map(|f| f.values()))
            .fold((0, 0), |(records, bytes), frame| {
                (records + 1, bytes + frame.reservation.bytes())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::LargeBinaryArray;
    use arrow_schema::{DataType, Field, Schema};

    #[tokio::test]
    async fn shared_ipc_credit_is_released_only_after_the_last_reader() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("binary_payload", DataType::LargeBinary, false),
        ]));
        let payload = vec![7; 256 << 10];
        let mut encoded = Vec::new();
        let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut encoded, &schema)?;
        for ids in [["a", "b"], ["c", "d"]] {
            writer.write(&RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(Vec::from(ids))),
                    Arc::new(LargeBinaryArray::from_iter_values([payload.as_slice(); 2])),
                ],
            )?)?;
        }
        writer.finish()?;
        drop(writer);
        for (prefix, known) in [(0, true), (1, true), (0, false), (1, false)] {
            let mut input =
                Vec::with_capacity((if known { 1 } else { 2 }) * encoded.len() + prefix);
            input.resize(prefix, 0);
            input.extend_from_slice(&encoded);
            let capacity = input.capacity();
            let input = Bytes::from(input).slice(prefix..);
            let begin = input.as_ptr() as usize;
            let length = input.len();
            let batches = ipc::decode(input, schema.clone())?;
            let budget = MergeMemoryBudget::new(4 << 20);
            let original = 2 * length + if known { capacity } else { length } + (64 << 10);
            let reservation = budget.reserve(original).await;
            assert!(budget.try_reserve(1 << 20).is_none());
            let mut window = Window::default();
            window.record(
                12,
                batches,
                reservation,
                length,
                known.then_some((begin, capacity)),
            );
            let retained = window.retained().1;
            assert!(
                retained >= length,
                "the full input backing must remain charged"
            );
            if known {
                assert!(retained < 2 * length, "shared input counted more than once");
            } else {
                assert_eq!(retained, original, "unknown input must not shrink credit");
            }
            // The returned transient capacity can admit independent work even
            // while the original body is still active and queryable.
            let other = budget.try_reserve(2 << 20);
            assert_eq!(other.is_some(), known);
            let other_bytes = other.as_ref().map_or(0, |r| r.bytes());
            let held = window.row(
                lance_context_core::annotation_staging::NewTurnSource {
                    record_index: 12,
                    row: 2,
                },
                "c",
            )?;
            let frozen = window.freeze();
            window.retire(&frozen);
            drop(frozen);
            assert_eq!(window.retained(), (0, 0));
            assert_eq!(budget.reserved(), retained + other_bytes);
            assert!(budget.try_reserve(1).is_none());
            let value = held
                .batch
                .column(1)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap();
            let pointer = value.value_data().as_ptr() as usize;
            assert!(pointer >= begin && pointer + value.value_data().len() <= begin + length);
            assert_eq!(value.value(0), payload);
            drop(held);
            assert_eq!(budget.reserved(), other_bytes);
            drop(other);
            assert_eq!(budget.reserved(), 0);
        }
        Ok(())
    }

    #[tokio::test]
    async fn sliced_body_read_retains_buffers_and_budget_after_window_retirement() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("binary_payload", DataType::LargeBinary, false),
        ]));
        let payload = vec![3; 4096];
        let bodies = Arc::new(LargeBinaryArray::from_iter_values([payload.as_slice(); 4]));
        let pointer = bodies.value_data().as_ptr();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                bodies,
            ],
        )?;
        let budget = MergeMemoryBudget::new(1 << 20);
        let mut window = Window::default();
        window.record(
            12,
            vec![batch.slice(0, 2), batch.slice(2, 2)],
            budget.reserve(1 << 20).await,
            0,
            None,
        );
        let held = window.row(
            lance_context_core::annotation_staging::NewTurnSource {
                record_index: 12,
                row: 2,
            },
            "c",
        )?;
        let frozen = window.freeze();
        let frame = &frozen[&12];
        let slices = frame.select(&[(0, "a"), (1, "b"), (3, "d")])?;
        assert_eq!(
            slices.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            vec![2, 1]
        );
        for slice in &slices {
            assert_eq!(
                slice
                    .column(1)
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .unwrap()
                    .value_data()
                    .as_ptr(),
                pointer
            );
        }
        assert!(frame.select(&[(0, "b")]).is_err());
        assert!(frame.select(&[(1, "b"), (1, "b")]).is_err());
        assert!(frame.select(&[(5, "e")]).is_err());
        drop(slices);
        window.retire(&frozen);
        drop(frozen);
        assert_eq!(window.retained(), (0, 0));
        assert_eq!(budget.reserved(), 1 << 20);
        assert!(budget.try_reserve(1).is_none());
        let values = held
            .batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        assert_eq!(values.value(0), payload);
        assert_eq!(values.value_data().as_ptr(), pointer);
        drop(held);
        assert_eq!(budget.reserved(), 0);
        assert!(budget.try_reserve(1).is_some());
        Ok(())
    }
}
