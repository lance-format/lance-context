//! Retained IPC allocations after decoded mutation rows have been released.
use arrow_array::RecordBatch;
use std::{
    collections::BTreeMap,
    mem::{size_of, size_of_val},
};

/// Keep the existing row workspace allowance as well as Arrow's retained
/// allocations. Do not sum per-array buffer sizes: every slice of an IPC input
/// reports the capacity of the same complete input allocation.
///
/// This is only used for the fixed mixed-body schema decoded from bounded,
/// uncompressed IPC. Unknown allocation capacity or overflow keeps the original
/// reservation. Array/schema bookkeeping is deliberately counted even when
/// shared; only the large backing allocations are deduplicated.
pub(super) fn retained_bytes(
    batches: &Vec<RecordBatch>,
    input_bytes: usize,
    original: usize,
) -> Option<usize> {
    let mut bytes = original.checked_sub(input_bytes.checked_mul(3)?)?;
    bytes = bytes.checked_add(batches.capacity().checked_mul(size_of::<RecordBatch>())?)?;
    let mut allocations = BTreeMap::<usize, usize>::new();
    for batch in batches {
        let schema = batch.schema();
        bytes = bytes.checked_add(size_of_val(schema.as_ref()))?;
        bytes = bytes.checked_add(schema.fields().size())?;
        bytes = bytes.checked_add(
            schema
                .metadata()
                .capacity()
                .checked_mul(size_of::<(String, String)>() * 2)?,
        )?;
        for (key, value) in schema.metadata() {
            bytes = bytes
                .checked_add(key.capacity())?
                .checked_add(value.capacity())?;
        }
        bytes = bytes.checked_add(batch.num_columns().checked_mul(size_of::<usize>() * 2)?)?;
        let mut pending = batch
            .columns()
            .iter()
            .map(|a| a.to_data())
            .collect::<Vec<_>>();
        while let Some(data) = pending.pop() {
            // ArrayData includes the Arrow container's inline layout; count its
            // datatype and Arc bookkeeping too, including nested/dictionary data.
            let bookkeeping = size_of_val(&data).checked_add(size_of::<usize>() * 2)?;
            bytes = bytes
                .checked_add(bookkeeping)?
                .checked_add(data.data_type().size())?;
            for buffer in data
                .buffers()
                .iter()
                .chain(data.nulls().map(|n| n.buffer()))
            {
                let capacity = buffer.capacity();
                if capacity == 0 {
                    if !buffer.is_empty() {
                        return None;
                    }
                    continue;
                }
                let allocation = capacity.checked_add(bookkeeping)?;
                allocations
                    .entry(buffer.data_ptr().as_ptr() as usize)
                    .and_modify(|old| *old = (*old).max(allocation))
                    .or_insert(allocation);
            }
            pending.extend(data.child_data().iter().cloned());
        }
    }
    // Keep at least the full input even if a projection owns only empty slices.
    // The metadata-only walk has dropped all temporary ArrayData before return.
    let buffers = allocations
        .values()
        .try_fold(0usize, |n, b| n.checked_add(*b))?;
    bytes.checked_add(buffers.max(input_bytes))
}
