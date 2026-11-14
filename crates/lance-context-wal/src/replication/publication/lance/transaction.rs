//! Validate the existing Lance transaction identity when it is not inlined.
use super::*;
use bytes::Bytes;
use prost::Message;

// Match the native preparation metadata bound. Ordinary WAL replicas do not
// enable this adapter or fetch transaction metadata from object storage.
pub(super) const MAX_EXTERNAL_BYTES: u64 = 64 << 20;

// Project only these fields of Lance's existing Transaction protobuf. UUID
// bytes share the fetched buffer; operation/properties are skipped, avoiding
// construction of a second full native transaction during a large compaction.
#[derive(Message)]
struct Identity {
    #[prost(uint64, tag = "1")]
    read_version: u64,
    #[prost(bytes = "bytes", tag = "2")]
    uuid: Bytes,
}

pub(super) async fn matches(
    scope: &ManifestCommit,
    manifest: &Manifest,
    base: &Path,
    store: &ObjectStore,
    inline: Option<&Transaction>,
) -> Result<bool> {
    let uuid = scope.transaction.to_string();
    if let Some(inline) = inline {
        return Ok(inline.as_pb().uuid == uuid);
    }
    let read_version = scope.target - 1;
    let expected = format!("{read_version}-{uuid}.txn");
    if manifest.transaction_file.as_deref() != Some(expected.as_str()) {
        return Ok(false);
    }
    // Construct the path from the bound publication, never an arbitrary path
    // supplied by the manifest. The filename alone does not prove its contents.
    let path = base.child("_transactions").child(expected);
    let object = store.inner.get(&path).await?;
    let size = object.meta.size;
    ensure!(
        (1..=MAX_EXTERNAL_BYTES).contains(&size) && object.range == (0..size),
        "external managed transaction exceeds metadata bounds"
    );
    let bytes = object.bytes().await?;
    ensure!(
        bytes.len() as u64 == size,
        "external transaction size changed"
    );
    let identity = Identity::decode(bytes)?;
    Ok(identity.read_version == read_version && identity.uuid.as_ref() == uuid.as_bytes())
}
