use std::sync::Arc;

use arrow_array::{Array, ArrayRef, LargeBinaryArray, ListArray, RecordBatch, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_arrow::{RecordBatchExt, deepcopy::deep_copy_array_sliced};
use lance_core::Result;
use lance_core::memory::{NativeMemory, NativeMemoryKind, retain_buffer, try_reserve};

fn copy_buffer(bytes: &[u8], memory: &NativeMemory) -> Result<Buffer> {
    // Arrow's MutableBuffer allocations round up to a 64-byte boundary.
    let reservation = try_reserve(
        memory,
        NativeMemoryKind::Index,
        bytes.len().div_ceil(64) * 64,
    )?;
    Ok(retain_buffer(Buffer::from(bytes), reservation))
}

fn copy_nulls(nulls: Option<&NullBuffer>, memory: &NativeMemory) -> Result<Option<NullBuffer>> {
    nulls
        .map(|nulls| {
            let reservation = try_reserve(
                memory,
                NativeMemoryKind::Index,
                nulls.len().div_ceil(512) * 64,
            )?;
            let bits = BooleanBuffer::collect_bool(nulls.len(), |i| nulls.is_valid(i));
            let buffer = retain_buffer(bits.into_inner(), reservation);
            Ok(NullBuffer::new(BooleanBuffer::new(buffer, 0, nulls.len())))
        })
        .transpose()
}

fn copy_offsets<T: arrow_array::OffsetSizeTrait>(
    offsets: &[T],
    memory: &NativeMemory,
) -> Result<OffsetBuffer<T>> {
    let reservation = try_reserve(
        memory,
        NativeMemoryKind::Index,
        std::mem::size_of_val(offsets),
    )?;
    let base = offsets[0];
    let values = offsets
        .iter()
        .map(|value| *value - base)
        .collect::<Vec<_>>();
    Ok(OffsetBuffer::new(ScalarBuffer::from(retain_buffer(
        Buffer::from_vec(values),
        reservation,
    ))))
}

fn compact_array(array: &dyn Array, memory: &NativeMemory) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::LargeBinary => {
            let binary = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            let offsets = binary.value_offsets();
            let start = offsets[0] as usize;
            let end = offsets[binary.len()] as usize;
            Ok(Arc::new(LargeBinaryArray::new(
                copy_offsets(offsets, memory)?,
                copy_buffer(&binary.values()[start..end], memory)?,
                copy_nulls(binary.nulls(), memory)?,
            )))
        }
        DataType::List(field) => {
            let list = array.as_any().downcast_ref::<ListArray>().unwrap();
            let offsets = list.value_offsets();
            let start = offsets[0] as usize;
            let end = offsets[list.len()] as usize;
            Ok(Arc::new(ListArray::try_new(
                field.clone(),
                copy_offsets(offsets, memory)?,
                compact_array(list.values().slice(start, end - start).as_ref(), memory)?,
                copy_nulls(list.nulls(), memory)?,
            )?))
        }
        DataType::UInt32 | DataType::Float32 | DataType::UInt64 => {
            let data = array.to_data();
            let width = if matches!(array.data_type(), DataType::UInt64) {
                8
            } else {
                4
            };
            let start = data.offset() * width;
            let buffer = copy_buffer(
                &data.buffers()[0][start..start + data.len() * width],
                memory,
            )?;
            let copied = ArrayData::builder(data.data_type().clone())
                .len(data.len())
                .buffers(vec![buffer])
                .nulls(copy_nulls(data.nulls(), memory)?)
                .build()?;
            Ok(make_array(copied))
        }
        // Other index layouts retain their existing copy implementation. The
        // native admission above covers modern packed posting column layouts.
        _ => Ok(deep_copy_array_sliced(array)),
    }
}

/// Copy the compact modern posting buffers with admission before allocation.
/// Each resulting Arrow buffer retains its own reservation through slices.
pub(super) fn compact_posting_batch(
    batch: &RecordBatch,
    memory: &NativeMemory,
) -> Result<RecordBatch> {
    if memory.is_none() {
        return Ok(batch.shrink_to_fit()?);
    }
    let arrays = batch
        .columns()
        .iter()
        .map(|array| compact_array(array.as_ref(), memory))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(batch.schema(), arrays)?)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use lance_core::memory::{NativeMemoryPool, NativeMemoryReservation};

    use super::*;

    #[derive(Debug)]
    struct Pool(Arc<AtomicUsize>, usize);

    #[derive(Debug)]
    struct Reservation(Arc<AtomicUsize>, usize);

    impl NativeMemoryReservation for Reservation {
        fn bytes(&self) -> usize {
            self.1
        }
        fn set_backed_bytes(&mut self, _: usize) {}
    }

    impl Drop for Reservation {
        fn drop(&mut self) {
            self.0.fetch_sub(self.1, Ordering::SeqCst);
        }
    }

    impl NativeMemoryPool for Pool {
        fn reserve(
            &self,
            kind: NativeMemoryKind,
            bytes: usize,
        ) -> futures::future::BoxFuture<'_, Result<Box<dyn NativeMemoryReservation>>> {
            Box::pin(async move { self.try_reserve(kind, bytes) })
        }

        fn try_reserve(
            &self,
            _: NativeMemoryKind,
            bytes: usize,
        ) -> Result<Box<dyn NativeMemoryReservation>> {
            if self.0.load(Ordering::SeqCst) + bytes > self.1 {
                return Err(lance_core::Error::execution("test allocation refused"));
            }
            self.0.fetch_add(bytes, Ordering::SeqCst);
            Ok(Box::new(Reservation(self.0.clone(), bytes)))
        }
    }

    #[test]
    fn compact_posting_values_keep_admission_after_batch_drop() {
        let live = Arc::new(AtomicUsize::new(0));
        let memory: NativeMemory = Some(Arc::new(Pool(live.clone(), 1024)));
        let input = LargeBinaryArray::from_vec(vec![b"unselected", b"kept", b"last"]);
        let batch =
            RecordBatch::try_from_iter([("posting", Arc::new(input.slice(1, 1)) as ArrayRef)])
                .unwrap();
        let output = compact_posting_batch(&batch, &memory).unwrap();
        let binary = output
            .column(0)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        assert_eq!(binary.value(0), b"kept");
        assert_eq!(binary.value_offsets(), &[0, 4]);
        let retained = binary.values().clone();
        drop(output);
        assert_eq!(live.load(Ordering::SeqCst), 64);
        drop(retained);
        assert_eq!(live.load(Ordering::SeqCst), 0);

        let denied: NativeMemory = Some(Arc::new(Pool(live.clone(), 16)));
        assert!(compact_posting_batch(&batch, &denied).is_err());
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }
}
