//! Only preparation metadata is externalized; mutation records remain in RF3.
use super::*;
use object_store::{path::Path, ObjectStoreExt};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};

// This bounds one metadata fetch and protobuf decode. It is separate from dirty
// staging, upload admission and Lance's own execution memory.
pub(super) const MAX_NATIVE_BYTES: usize = 64 << 20;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    table: Uuid,
    incarnation: Uuid,
    transaction: Uuid,
    bytes: u64,
    digest: [u8; 32],
}

fn checksum(bytes: &[u8]) -> [u8; 32] {
    digest(&SHA256, bytes).as_ref().try_into().unwrap()
}

impl Reference {
    fn path(&self, storage: &Storage, selection: &Selection) -> Result<Path> {
        selection.scope()?;
        ensure!(
            self.table == selection.table
                && self.incarnation == selection.incarnation
                && self.transaction == selection.pending.plan.transaction
                && self.bytes > 0
                && self.bytes <= MAX_NATIVE_BYTES as u64,
            "foreign or oversized annotation metadata reference"
        );
        let name: String = self.digest.iter().map(|b| format!("{b:02x}")).collect();
        // The authorized destination supplies the root. No URI or arbitrary
        // object path is accepted from the descriptor.
        Ok(storage
            .base
            .clone()
            .join("_annotation_preparations")
            .join(self.incarnation.to_string())
            .join(self.transaction.to_string())
            .join(format!("{name}.pb")))
    }
}

pub(super) async fn persist(
    base: &Dataset,
    storage: &Storage,
    selection: &Selection,
    bytes: Bytes,
    maximum: u64,
) -> Result<p::Descriptors> {
    let reference = Reference {
        table: selection.table,
        incarnation: selection.incarnation,
        transaction: selection.pending.plan.transaction,
        bytes: bytes.len() as u64,
        digest: checksum(&bytes),
    };
    let path = reference.path(storage, selection)?;
    let descriptors =
        p::Descriptors::lance_annotation_reference(Bytes::from(serde_json::to_vec(&reference)?))?;
    descriptors.validate(maximum)?;
    let store = base.object_store(None).await?;
    match store.put_if_absent(&path, bytes.into()).await {
        Ok(()) => {}
        Err(
            object_store::Error::AlreadyExists { .. } | object_store::Error::Precondition { .. },
        ) => {
            // A conditional-create conflict is resolved against the exact
            // immutable content. Other errors, including unknowns, propagate.
            load(base, storage, selection, &descriptors).await?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(descriptors)
}

pub(super) async fn load(
    base: &Dataset,
    storage: &Storage,
    selection: &Selection,
    descriptors: &p::Descriptors,
) -> Result<Bytes> {
    ensure!(
        base.uri() == storage.uri && descriptors.format == 5,
        "foreign metadata storage"
    );
    let reference: Reference = serde_json::from_slice(&descriptors.bytes)?;
    let path = reference.path(storage, selection)?;
    let store = base.object_store(None).await?;
    let object = store.inner.get(&path).await?;
    ensure!(
        object.meta.size == reference.bytes,
        "annotation metadata size changed"
    );
    let bytes = object.bytes().await?;
    ensure!(
        bytes.len() as u64 == reference.bytes && checksum(&bytes) == reference.digest,
        "annotation metadata digest changed"
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use lance_context_wal::record::Batch;
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn annotation_metadata_create_conflict_requires_exact_bytes() -> Result<()> {
        let root = tempfile::tempdir()?;
        let storage = Storage::resolve(
            root.path().join("test.lance").to_str().unwrap().into(),
            Default::default(),
            Arc::default(),
        )
        .await?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(vec!["T"]))])?;
        let base = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &storage.uri,
            None,
        )
        .await?;
        let table = Uuid::new_v4();
        let selection = Selection {
            table,
            incarnation: Uuid::new_v4(),
            pending: p::PendingPlan {
                index: 2,
                plan: p::Plan {
                    transaction: Uuid::new_v4(),
                    read_version: base.version_id(),
                    next_group: 10,
                    sources: BTreeMap::from([(
                        10,
                        p::SourceRange {
                            after: 0,
                            through: 1,
                            records: vec![(
                                1,
                                Batch::new(
                                    table,
                                    0,
                                    Uuid::new_v4(),
                                    1,
                                    Bytes::from_static(b"source"),
                                )
                                .meta,
                            )],
                        },
                    )]),
                },
            },
        };
        // Test the immutable object helper with exactly the same encoded bytes.
        // Re-encoding a native protobuf map need not reproduce byte order.
        let bytes = Bytes::from(vec![7; p::MAX_DESCRIPTOR_BYTES + 1]);
        let maximum = p::MAX_DESCRIPTOR_BYTES as u64;
        assert!(persist(&base, &storage, &selection, bytes.clone(), 1)
            .await
            .is_err());
        let descriptors = persist(&base, &storage, &selection, bytes.clone(), maximum).await?;
        assert_eq!(
            persist(&base, &storage, &selection, bytes.clone(), maximum).await?,
            descriptors
        );
        let reference: Reference = serde_json::from_slice(&descriptors.bytes)?;
        let path = reference.path(&storage, &selection)?;
        let store = base.object_store(None).await?;
        for damaged in [
            bytes.slice(..bytes.len() - 1),
            Bytes::from(vec![8; bytes.len()]),
        ] {
            store.inner.put(&path, damaged.into()).await?;
            assert!(persist(&base, &storage, &selection, bytes.clone(), maximum)
                .await
                .is_err());
            assert!(load(&base, &storage, &selection, &descriptors)
                .await
                .is_err());
        }
        store.inner.put(&path, bytes.clone().into()).await?;
        assert_eq!(
            load(&base, &storage, &selection, &descriptors).await?,
            bytes
        );
        Ok(())
    }
}
