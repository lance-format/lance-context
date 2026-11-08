//! Reuse an in-flight upload only after its exact metadata is committed.
use super::*;
use std::{collections::BTreeMap, sync::Weak};

/// One weak entry per bounded producer slot, never a retained payload cache.
#[derive(Default)]
pub(super) struct Registry(Mutex<BTreeMap<usize, Weak<Batch>>>);

pub(super) struct Registration<'a> {
    registry: &'a Registry,
    slot: usize,
    batch: Arc<Batch>,
}

impl Registry {
    pub(super) fn register(&self, slot: usize, prepared: &Batch) -> Registration<'_> {
        // Batch::clone shares Bytes; PreparedAppend's upload reservation remains
        // alive for this entire submission. No new payload buffer is allocated.
        let batch = Arc::new(prepared.clone());
        self.0.lock().unwrap().insert(slot, Arc::downgrade(&batch));
        Registration {
            registry: self,
            slot,
            batch,
        }
    }

    /// Call only after reserving the apply workspace. The returned Bytes may
    /// then outlive the upload, backed by that workspace or retained body frame.
    pub(super) fn body(&self, slot: usize, committed: &BatchMeta) -> Option<Bytes> {
        let batch = self.0.lock().unwrap().get(&slot)?.upgrade()?;
        // A producer slot can be reused, and a conflicting same-sequence
        // request can race the accepted one. Match the complete WAL identity,
        // including its existing digest, without rehashing or copying payloads.
        (batch.meta == *committed).then(|| batch.body.clone())
    }
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        let mut entries = self.registry.0.lock().unwrap();
        if entries
            .get(&self.slot)
            .is_some_and(|entry| entry.ptr_eq(&Arc::downgrade(&self.batch)))
        {
            entries.remove(&self.slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_bytes_require_exact_committed_identity() {
        let registry = Registry::default();
        let input = Batch::new(
            Uuid::new_v4(),
            0,
            producer::producer_id(1, 1),
            1,
            Bytes::from(vec![7; 4096]),
        );
        let registration = registry.register(1, &input);
        let body = registry.body(1, &input.meta).unwrap();
        assert_eq!(body.as_ptr(), input.body.as_ptr());
        for field in 0..7 {
            let mut other = input.meta.clone();
            match field {
                0 => other.stream = Uuid::new_v4(),
                1 => other.bucket += 1,
                2 => other.producer = producer::producer_id(1, 2),
                3 => other.sequence += 1,
                4 => other.digest[0] ^= 1,
                5 => other.body_bytes += 1,
                _ => other.kind = RecordKind::Source,
            }
            assert!(registry.body(1, &other).is_none());
        }
        drop(registration);
        assert!(registry.body(1, &input.meta).is_none());
        assert!(registry.0.lock().unwrap().is_empty());
        drop(input);
        assert_eq!(&body[..], &[7; 4096]);
    }

    #[test]
    fn older_submission_drop_cannot_remove_replacement_slot() {
        let registry = Registry::default();
        let first = Batch::new(
            Uuid::new_v4(),
            0,
            producer::producer_id(1, 1),
            1,
            Bytes::from(vec![1; 16]),
        );
        let mut next = first.clone();
        next.meta.producer = producer::producer_id(1, 2);
        let old = registry.register(1, &first);
        let new = registry.register(1, &next);
        assert_eq!(registry.0.lock().unwrap().len(), 1);
        drop(old);
        assert!(registry.body(1, &first.meta).is_none());
        assert!(registry.body(1, &next.meta).is_some());
        // Even identical duplicate submissions have distinct lifetimes.
        let duplicate = registry.register(1, &next);
        drop(new);
        assert!(registry.body(1, &next.meta).is_some());
        drop(duplicate);
        assert!(registry.0.lock().unwrap().is_empty());
    }
}
