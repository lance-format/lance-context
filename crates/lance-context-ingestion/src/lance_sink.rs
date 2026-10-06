//! Immutable file staging plus a single version-pinned table publisher.
//! The caller holds the table's durable maintenance ownership across publication
//! and supplies its guarded commit handler. This module does not acquire leases.
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::Schema;
use async_trait::async_trait;
use futures::stream::BoxStream;
use lance::dataset::{
    transaction::{Operation, Transaction},
    CommitBuilder, Dataset, InsertBuilder, WriteMode, WriteParams,
};
use lance::index::DatasetIndexExt;
use lance_file::version::LanceFileVersion;
use lance_index::mem_wal::{MemWalIndexDetails, MergedGeneration, MEM_WAL_INDEX_NAME};
use lance_io::object_store::ObjectStore as LanceObjectStore;
use lance_table::format::{pb, Fragment, IndexMetadata, Manifest};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
};
use object_store::{path::Path, ObjectStore};
use uuid::Uuid;

use crate::{Binding, Entry, Error, Result, Sink};

pub const RUN_METADATA: &str = "lance-context.ingestion.run";
pub const SCHEMA_METADATA: &str = "lance-context.ingestion.schema";
const SHARD_NAMESPACE: Uuid = Uuid::from_u128(0x8370a74b_3aaf_4083_8ed0_610b9bd6e8da);

fn failure(error: impl std::fmt::Display) -> Error {
    Error::Stage(error.to_string())
}

/// Declare the run identity and the encoding policy before table creation.
/// All batches must use this same Arrow schema; this never alters turn IDs.
/// Zstd applies where Lance selects a compression codec. Lance 2.2 constant
/// pages use a scalar layout instead, even for a single large string.
pub fn table_schema(schema: &Schema, run: &str, identity_schema: &str) -> Schema {
    let fields = schema
        .fields
        .iter()
        .map(|field| {
            let mut metadata = field.metadata().clone();
            for (key, value) in [
                ("lance-encoding:compression", "zstd"),
                ("lance-encoding:compression-level", "3"),
                ("lance-encoding:dict-values-compression", "zstd"),
                ("lance-encoding:dict-values-compression-level", "3"),
            ] {
                metadata.insert(key.into(), value.into());
            }
            Arc::new(field.as_ref().clone().with_metadata(metadata))
        })
        .collect::<Vec<_>>();
    let mut metadata = schema.metadata.clone();
    metadata.insert(RUN_METADATA.into(), run.into());
    metadata.insert(SCHEMA_METADATA.into(), identity_schema.into());
    Schema::new_with_metadata(fields, metadata)
}

/// Encode already aligned output records. The WAL also persists the matching
/// state delta, so recovery does not run alignment or regenerate IDs.
pub fn encode_records(batches: &[RecordBatch]) -> Result<Vec<u8>> {
    let Some(first) = batches.first() else {
        return Ok(Vec::new());
    };
    let mut output = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut output, &first.schema()).map_err(failure)?;
        for batch in batches {
            writer.write(batch).map_err(failure)?;
        }
        writer.finish().map_err(failure)?;
    }
    Ok(output)
}

pub struct Staged {
    dataset_uri: String,
    schema: lance::datatypes::Schema,
    binding: Binding,
    first: u64,
    last: u64,
    fragments: Vec<Fragment>,
    rows: usize,
}

impl Staged {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn first_sequence(&self) -> u64 {
        self.first
    }
    pub fn last_sequence(&self) -> u64 {
        self.last
    }
}

fn validate_dataset(dataset: &Dataset, binding: &Binding) -> Result<()> {
    let schema = Schema::from(dataset.schema());
    if schema.metadata.get(RUN_METADATA) != Some(&binding.run)
        || schema.metadata.get(SCHEMA_METADATA) != Some(&binding.schema)
        || dataset.manifest().data_storage_format.version != "2.2"
    {
        return Err(Error::Invalid(
            "Lance run/schema/2.2 format mismatch".into(),
        ));
    }
    if schema.fields.iter().any(|field| {
        field
            .metadata()
            .get("lance-encoding:compression")
            .map(String::as_str)
            != Some("zstd")
    }) {
        return Err(Error::Invalid(
            "Lance schema must request Zstd encoding".into(),
        ));
    }
    Ok(())
}

fn shard(binding: &Binding) -> Result<Uuid> {
    Ok(Uuid::new_v5(
        &SHARD_NAMESPACE,
        &serde_json::to_vec(binding)?,
    ))
}

async fn watermarks(dataset: &Dataset) -> Result<HashMap<Uuid, u64>> {
    let indices = dataset.load_indices().await.map_err(failure)?;
    let Some(index) = indices
        .iter()
        .find(|index| index.name == MEM_WAL_INDEX_NAME)
    else {
        return Ok(HashMap::new());
    };
    let details = index
        .index_details
        .as_ref()
        .ok_or_else(|| Error::Invalid("missing Lance WAL watermark details".into()))?;
    let details = MemWalIndexDetails::try_from(
        details
            .to_msg::<pb::MemWalIndexDetails>()
            .map_err(failure)?,
    )
    .map_err(failure)?;
    Ok(details
        .merged_generations
        .into_iter()
        .map(|entry| (entry.shard_id, entry.generation))
        .collect())
}

/// Decode a bounded WAL range and prepare immutable Lance 2.2 files only. This
/// is safe to distribute across workers: no table manifest or WAL cursor changes.
/// `max_bytes` bounds decoded Arrow data retained by this stage; input WAL and
/// encoder working memory require separate reservations in the calling worker.
pub async fn stage(
    dataset: &Dataset,
    binding: &Binding,
    entries: &[Entry],
    max_bytes: usize,
) -> Result<Staged> {
    validate_dataset(dataset, binding)?;
    if entries.is_empty()
        || max_bytes == 0
        || entries[0].sequence == 0
        || entries
            .windows(2)
            .any(|pair| pair[0].sequence.checked_add(1) != Some(pair[1].sequence))
    {
        return Err(Error::Invalid("invalid Lance staging input range".into()));
    }
    let schema = Schema::from(dataset.schema());
    let mut batches = Vec::new();
    let mut bytes = 0_usize;
    for entry in entries {
        if entry.transition.records.is_empty() {
            continue;
        }
        let reader =
            StreamReader::try_new(Cursor::new(&entry.transition.records), None).map_err(failure)?;
        if reader.schema().as_ref() != &schema {
            return Err(Error::Invalid(
                "WAL Arrow schema differs from target table".into(),
            ));
        }
        for batch in reader {
            let batch = batch.map_err(failure)?;
            bytes = bytes
                .checked_add(batch.get_array_memory_size())
                .ok_or_else(|| Error::Invalid("Arrow byte count overflow".into()))?;
            if bytes > max_bytes {
                return Err(Error::Invalid(
                    "staging decoded Arrow budget exceeded".into(),
                ));
            }
            batches.push(batch);
        }
    }
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    let fragments = if rows == 0 {
        Vec::new()
    } else {
        let params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_bytes_per_file: max_bytes,
            ..Default::default()
        };
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&params)
            .execute_uncommitted(batches)
            .await
            .map_err(failure)?;
        let Operation::Append { fragments } = transaction.operation else {
            return Err(Error::Invalid("unexpected staging operation".into()));
        };
        fragments
    };
    Ok(Staged {
        dataset_uri: dataset.uri().to_owned(),
        schema: dataset.schema().clone(),
        binding: binding.clone(),
        first: entries[0].sequence,
        last: entries.last().unwrap().sequence,
        fragments,
        rows,
    })
}

/// A single manifest publisher. Supply the same guarded handler used by the
/// owner to open this dataset; the version pin wraps, never replaces, that guard.
/// After any uncertain commit, discard this sink, reopen the table and reconcile
/// its atomic watermarks. Do not retry a transaction against a newer base.
pub struct LanceTableSink {
    dataset: Dataset,
    handler: Arc<dyn CommitHandler>,
    max_stage_bytes: usize,
    poisoned: bool,
}

impl LanceTableSink {
    pub fn new(
        dataset: Dataset,
        handler: Arc<dyn CommitHandler>,
        max_stage_bytes: usize,
    ) -> Result<Self> {
        if max_stage_bytes == 0 {
            return Err(Error::Invalid("zero Lance staging budget".into()));
        }
        Ok(Self {
            dataset,
            handler,
            max_stage_bytes,
            poisoned: false,
        })
    }

    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    pub async fn covered_sequence(&mut self, binding: &Binding) -> Result<u64> {
        self.dataset.checkout_latest().await.map_err(failure)?;
        validate_dataset(&self.dataset, binding)?;
        Ok(watermarks(&self.dataset)
            .await?
            .get(&shard(binding)?)
            .copied()
            .unwrap_or(0))
    }

    pub async fn commit_staged(&mut self, staged: Vec<Staged>) -> Result<usize> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        self.dataset.checkout_latest().await.map_err(failure)?;
        let marks = watermarks(&self.dataset).await?;
        let mut seen = HashSet::new();
        let mut fragments = Vec::new();
        let mut merged = Vec::new();
        let mut rows = 0;
        for part in staged {
            validate_dataset(&self.dataset, &part.binding)?;
            let id = shard(&part.binding)?;
            if !seen.insert(id) || part.dataset_uri != self.dataset.uri() {
                return Err(Error::Invalid(
                    "duplicate partition or different staged dataset".into(),
                ));
            }
            let high = marks.get(&id).copied().unwrap_or(0);
            if part.last <= high {
                continue;
            }
            if high.checked_add(1) != Some(part.first) {
                return Err(Error::Invalid(
                    "staged range overlaps or skips merged WAL; restage from durable coverage"
                        .into(),
                ));
            }
            let options = lance::datatypes::SchemaCompareOptions {
                compare_metadata: true,
                compare_field_ids: true,
                ..Default::default()
            };
            if Schema::from(&part.schema) != Schema::from(self.dataset.schema())
                || part
                    .schema
                    .check_compatible(self.dataset.schema(), &options)
                    .is_err()
                || part
                    .fragments
                    .iter()
                    .map(|fragment| fragment.physical_rows.unwrap_or(0))
                    .sum::<usize>()
                    != part.rows
            {
                return Err(Error::Invalid("staged schema or row count changed".into()));
            }
            rows += part.rows;
            fragments.extend(part.fragments);
            merged.push(MergedGeneration::new(id, part.last));
        }
        if merged.is_empty() {
            return Ok(0);
        }
        let base = self.dataset.version().version;
        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments: Vec::new(),
            new_fragments: fragments,
            fields_modified: Vec::new(),
            merged_generations: merged,
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };
        let handler = Arc::new(PinnedCommit {
            delegate: self.handler.clone(),
            next_version: base
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("table version overflow".into()))?,
        });
        self.poisoned = true;
        self.dataset = CommitBuilder::new(Arc::new(self.dataset.clone()))
            .with_commit_handler(handler)
            .with_max_retries(0)
            .execute(Transaction::new(base, operation, None))
            .await
            .map_err(failure)?;
        self.poisoned = false;
        Ok(rows)
    }
}

#[async_trait]
impl Sink for LanceTableSink {
    async fn apply(&mut self, binding: &Binding, entries: &[Entry]) -> Result<()> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        if entries.is_empty() {
            return Ok(());
        }
        let high = self.covered_sequence(binding).await?;
        let remaining = entries.partition_point(|entry| entry.sequence <= high);
        if remaining == entries.len() {
            return Ok(());
        }
        let staged = stage(
            &self.dataset,
            binding,
            &entries[remaining..],
            self.max_stage_bytes,
        )
        .await?;
        self.commit_staged(vec![staged]).await?;
        Ok(())
    }
}

#[derive(Debug)]
struct PinnedCommit {
    delegate: Arc<dyn CommitHandler>,
    next_version: u64,
}

#[async_trait]
impl CommitHandler for PinnedCommit {
    async fn resolve_latest_location(
        &self,
        base: &Path,
        store: &LanceObjectStore,
    ) -> lance::Result<ManifestLocation> {
        self.delegate.resolve_latest_location(base, store).await
    }
    async fn resolve_version_location(
        &self,
        base: &Path,
        version: u64,
        store: &dyn ObjectStore,
    ) -> lance::Result<ManifestLocation> {
        self.delegate
            .resolve_version_location(base, version, store)
            .await
    }
    async fn version_exists(
        &self,
        base: &Path,
        version: u64,
        store: &dyn ObjectStore,
        scheme: ManifestNamingScheme,
    ) -> lance::Result<bool> {
        self.delegate
            .version_exists(base, version, store, scheme)
            .await
    }
    fn list_detached_manifest_locations<'a>(
        &self,
        base: &Path,
        store: &'a LanceObjectStore,
    ) -> BoxStream<'a, lance::Result<ManifestLocation>> {
        self.delegate.list_detached_manifest_locations(base, store)
    }
    fn list_manifest_locations<'a>(
        &self,
        base: &Path,
        store: &'a LanceObjectStore,
        descending: bool,
    ) -> BoxStream<'a, lance::Result<ManifestLocation>> {
        self.delegate
            .list_manifest_locations(base, store, descending)
    }
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base: &Path,
        store: &LanceObjectStore,
        writer: ManifestWriter,
        scheme: ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        if manifest.version != self.next_version {
            return Err(CommitError::OtherError(lance::Error::invalid_input(
                "ingestion commit base changed; reopen and reconcile WAL coverage",
            )));
        }
        self.delegate
            .commit(manifest, indices, base, store, writer, scheme, transaction)
            .await
    }
    async fn delete(&self, base: &Path) -> lance::Result<()> {
        self.delegate.delete(base).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatchIterator;
    use arrow_schema::{DataType, Field};

    #[tokio::test]
    async fn pinned_handler_rejects_initial_rebase_even_with_zero_retries() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let schema = Arc::new(table_schema(
            &Schema::new(vec![Field::new("id", DataType::Utf8, false)]),
            "run",
            "schema",
        ));
        let old = Dataset::write(
            RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema),
            uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let mut current = old.clone();
        current
            .update_metadata([("competing", "publication")])
            .await
            .unwrap();
        let version = current.version().version;
        let handler = Arc::new(PinnedCommit {
            delegate: lance_table::io::commit::commit_handler_from_url(uri, &None)
                .await
                .unwrap(),
            next_version: old.version().version + 1,
        });
        let transaction = Transaction::new(
            old.version().version,
            Operation::Append { fragments: vec![] },
            None,
        );
        assert!(CommitBuilder::new(Arc::new(old))
            .with_commit_handler(handler)
            .with_max_retries(0)
            .execute(transaction)
            .await
            .is_err());
        assert_eq!(Dataset::open(uri).await.unwrap().version().version, version);
    }
}
