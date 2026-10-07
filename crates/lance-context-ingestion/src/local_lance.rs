//! Local alignment state backed by a partition's append-only Lance log.
//! The caller owns session routing and the durable partition lease. Local files
//! are disposable: only a successful Lance commit establishes durability. Supply
//! the lease-checking commit handler when creating/opening the log and this API.
use std::{path::Path, sync::Arc};

use arrow_array::{
    new_null_array, Array, ArrayRef, LargeBinaryArray, RecordBatch, RecordBatchIterator,
    StringArray, StructArray, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::{dataset::WriteParams, Dataset};
use lance_file::version::LanceFileVersion;
use lance_table::io::commit::CommitHandler;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

use crate::{
    lance_sink::{
        encode_records, stage, table_schema, LanceTableSink, RUN_METADATA, SCHEMA_METADATA,
    },
    Binding, Entry, Error, Result, Transition,
};

const STATE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("state");
const RECEIPTS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("receipts");
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");
const PARTITION: &str = "lance-context.ingestion.partition";
const FORMAT: &str = "lance-context.ingestion.local-state-format";
const CHECKPOINT_FORMAT: &str = "lance-context.ingestion.checkpoint-format";
const CHECKPOINT_URI: &str = "lance-context.ingestion.checkpoint-wal-uri";
const CHECKPOINT_VERSION: &str = "lance-context.ingestion.checkpoint-wal-version";
const CHECKPOINT_SEQUENCE: &str = "lance-context.ingestion.checkpoint-sequence";
const PUBLISHED_CHECKPOINT_URI: &str = "lance-context.ingestion.checkpoint-uri";
const PUBLISHED_CHECKPOINT_VERSION: &str = "lance-context.ingestion.checkpoint-version";
const COLUMNS: [&str; 8] = [
    "sequence",
    "ordinal",
    "kind",
    "session",
    "receipt",
    "input_digest",
    "key",
    "value",
];
const RECEIPT: u8 = 0;
const PUT: u8 = 1;
const DELETE: u8 = 2;
const RECORD: u8 = 3;

fn fail(error: impl std::fmt::Display) -> Error {
    Error::Stage(error.to_string())
}

/// Values are individual binary state cells, not serialized session objects.
/// The adapter defines key/value encoding and must version it in Binding.schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mutation {
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>,
}

/// One already aligned call. Mutations, typed output rows, and the receipt are
/// one durable unit. The final receipt row seals this sequence during recovery.
pub struct AlignedCall {
    pub sequence: u64,
    pub session: String,
    pub receipt: String,
    pub input_digest: String,
    pub mutations: Vec<Mutation>,
    pub records: RecordBatch,
}

/// Binary tuple framing preserves arbitrary session/key bytes without delimiters
/// or changing the adapter's existing identity hashes.
fn local_key(session: &str, key: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(8 + session.len() + key.len());
    result.extend_from_slice(&(session.len() as u64).to_be_bytes());
    result.extend_from_slice(session.as_bytes());
    result.extend_from_slice(key);
    result
}

fn receipt_value(sequence: u64, input_digest: &str) -> Vec<u8> {
    let mut value = sequence.to_le_bytes().to_vec();
    value.extend_from_slice(input_digest.as_bytes());
    value
}

/// Typed record columns remain columnar inside the record struct. Recovery
/// projects only the eight state/receipt columns, excluding message bodies.
pub fn log_schema(records: &Schema, binding: &Binding) -> Schema {
    let mut fields = vec![
        Field::new("sequence", DataType::UInt64, false),
        Field::new("ordinal", DataType::UInt64, false),
        Field::new("kind", DataType::UInt8, false),
        Field::new("session", DataType::Utf8, false),
        Field::new("receipt", DataType::Utf8, false),
        Field::new("input_digest", DataType::Utf8, false),
        Field::new("key", DataType::LargeBinary, false),
        Field::new("value", DataType::LargeBinary, false),
    ];
    let records = table_schema(records, &binding.run, &binding.schema);
    let record_fields = records
        .fields
        .iter()
        .map(|field| {
            let mut metadata = field.metadata().clone();
            metadata.insert(
                "lance-context.ingestion.output-nullable".into(),
                field.is_nullable().to_string(),
            );
            Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_nullable(true)
                    .with_metadata(metadata),
            )
        })
        .collect::<Vec<_>>();
    fields.push(Field::new(
        "record",
        DataType::Struct(record_fields.into()),
        true,
    ));
    let mut schema = table_schema(&Schema::new(fields), &binding.run, &binding.schema);
    schema
        .metadata
        .insert(PARTITION.into(), binding.partition.to_string());
    schema.metadata.insert(FORMAT.into(), "1".into());
    schema
}

fn event_batch(
    schema: Arc<Schema>,
    call: &AlignedCall,
    first_ordinal: u64,
    kinds: Vec<u8>,
    keys: Vec<&[u8]>,
    values: Vec<&[u8]>,
    records: Option<RecordBatch>,
) -> Result<RecordBatch> {
    let n = kinds.len();
    let record = records.map_or_else(
        || new_null_array(schema.field(8).data_type(), n),
        |batch| Arc::new(StructArray::from(batch)) as ArrayRef,
    );
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![call.sequence; n])),
            Arc::new(UInt64Array::from_iter_values(
                first_ordinal..first_ordinal + n as u64,
            )),
            Arc::new(UInt8Array::from(kinds)),
            Arc::new(StringArray::from(vec![call.session.as_str(); n])),
            Arc::new(StringArray::from(vec![call.receipt.as_str(); n])),
            Arc::new(StringArray::from(vec![call.input_digest.as_str(); n])),
            Arc::new(LargeBinaryArray::from(keys)),
            Arc::new(LargeBinaryArray::from(values)),
            record,
        ],
    )
    .map_err(fail)
}

fn encode_call(schema: Arc<Schema>, call: &AlignedCall, batch_rows: usize) -> Result<Vec<u8>> {
    let mut batches = Vec::new();
    let mut ordinal = 0;
    for mutations in call.mutations.chunks(batch_rows) {
        batches.push(event_batch(
            schema.clone(),
            call,
            ordinal,
            mutations
                .iter()
                .map(|m| if m.value.is_some() { PUT } else { DELETE })
                .collect(),
            mutations.iter().map(|m| m.key.as_slice()).collect(),
            mutations
                .iter()
                .map(|m| m.value.as_deref().unwrap_or_default())
                .collect(),
            None,
        )?);
        ordinal += mutations.len() as u64;
    }
    for start in (0..call.records.num_rows()).step_by(batch_rows) {
        let n = batch_rows.min(call.records.num_rows() - start);
        // Match compression metadata on the durable child fields, not just types.
        let DataType::Struct(fields) = schema.field(8).data_type() else {
            unreachable!()
        };
        let records = RecordBatch::try_new(
            Arc::new(Schema::new(fields.clone())),
            call.records.slice(start, n).columns().to_vec(),
        )
        .map_err(fail)?;
        batches.push(event_batch(
            schema.clone(),
            call,
            ordinal,
            vec![RECORD; n],
            vec![&[]; n],
            vec![&[]; n],
            Some(records),
        )?);
        ordinal += n as u64;
    }
    batches.push(event_batch(
        schema,
        call,
        ordinal,
        vec![RECEIPT],
        vec![&[]],
        vec![&[]],
        None,
    )?);
    encode_records(&batches)
}

/// Open only on a blocking worker/runtime suitable for local database I/O.
/// The cache limit bounds redb's page cache, not Arrow or alignment allocations.
/// A failed/cancelled commit fences this instance, including uncertain success.
pub struct LocalLancePartition {
    db: Database,
    sink: LanceTableSink,
    binding: Binding,
    through: u64,
    version: u64,
    max_batch_bytes: usize,
    batch_rows: usize,
    poisoned: bool,
}

impl LocalLancePartition {
    /// Rebuild missing local state or replay its suffix from the supplied latest
    /// committed log. The caller must hold the partition lease; this never steals
    /// ownership or accepts a dataset whose history was compacted/deleted.
    pub async fn open(
        local_file: &Path,
        dataset: Dataset,
        binding: Binding,
        handler: Arc<dyn CommitHandler>,
        cache_bytes: usize,
        max_batch_bytes: usize,
        batch_rows: usize,
    ) -> Result<Self> {
        let schema = Schema::from(dataset.schema());
        if cache_bytes == 0
            || max_batch_bytes == 0
            || batch_rows == 0
            || schema.metadata.get(PARTITION) != Some(&binding.partition.to_string())
            || schema.metadata.get(FORMAT).map(String::as_str) != Some("1")
            || schema.fields.len() != 9
        {
            return Err(Error::Invalid(
                "invalid local Lance state configuration".into(),
            ));
        }
        let mut sink = LanceTableSink::new(dataset, handler, max_batch_bytes)?;
        let head = sink.covered_sequence(&binding).await?;
        if !local_file.exists() {
            let metadata = &sink.dataset().manifest().table_metadata;
            match (
                metadata.get(PUBLISHED_CHECKPOINT_URI),
                metadata.get(PUBLISHED_CHECKPOINT_VERSION),
            ) {
                (Some(uri), Some(version)) => {
                    let version: u64 = version.parse().map_err(fail)?;
                    let mut builder = lance::dataset::builder::DatasetBuilder::from_uri(uri)
                        .with_version(version);
                    if let Some(params) = sink.dataset().store_params() {
                        builder = builder.with_store_params(params.clone());
                    }
                    let checkpoint = builder.load().await.map_err(fail)?;
                    restore_checkpoint(
                        local_file,
                        &checkpoint,
                        sink.dataset(),
                        &binding,
                        cache_bytes,
                        batch_rows,
                    )
                    .await?;
                }
                (None, None) => {}
                _ => {
                    return Err(Error::Invalid(
                        "incomplete published checkpoint pointer".into(),
                    ))
                }
            }
        }
        let db = Database::builder()
            .set_cache_size(cache_bytes)
            .create(local_file)
            .map_err(fail)?;
        let tx = db.begin_write().map_err(fail)?;
        let mut metadata = tx.open_table(META).map_err(fail)?;
        for (key, value) in [
            ("run", binding.run.clone()),
            ("schema", binding.schema.clone()),
            ("partition", binding.partition.to_string()),
            ("uri", sink.dataset().uri().to_owned()),
        ] {
            let old = metadata.get(key).map_err(fail)?.map(|v| v.value().to_vec());
            if old.as_ref().is_some_and(|old| old != value.as_bytes()) {
                return Err(Error::Invalid("local state binding mismatch".into()));
            }
            metadata.insert(key, value.as_bytes()).map_err(fail)?;
        }
        let through = metadata
            .get("through")
            .map_err(fail)?
            .map(|value| {
                value
                    .value()
                    .try_into()
                    .map(u64::from_le_bytes)
                    .map_err(|_| Error::Invalid("invalid local state sequence".into()))
            })
            .transpose()?
            .unwrap_or(0);
        let version = metadata
            .get("version")
            .map_err(fail)?
            .map(|value| {
                value
                    .value()
                    .try_into()
                    .map(u64::from_le_bytes)
                    .map_err(|_| Error::Invalid("invalid local state version".into()))
            })
            .transpose()?
            .unwrap_or(0);
        if through > head || version > sink.dataset().version().version {
            return Err(Error::Invalid("local state ahead of supplied log".into()));
        }
        drop(metadata);
        tx.open_table(STATE).map_err(fail)?;
        tx.open_table(RECEIPTS).map_err(fail)?;
        tx.commit().map_err(fail)?;
        let mut value = Self {
            db,
            sink,
            binding,
            through,
            version,
            max_batch_bytes,
            batch_rows,
            poisoned: true,
        };
        value.replay(head).await?;
        value.poisoned = false;
        Ok(value)
    }

    pub fn through_sequence(&self) -> u64 {
        self.through
    }
    pub fn dataset(&self) -> &Dataset {
        self.sink.dataset()
    }

    /// Write an immutable full checkpoint in bounded Arrow batches. Invoke on a
    /// byte/time threshold, not after each call. The returned exact dataset must
    /// be published by the lease owner before relying on it for recovery. This
    /// method performs no WAL deletion or automatic retention change.
    pub async fn checkpoint(&self, uri: &str) -> Result<Dataset> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        let tx = self.db.begin_read().map_err(fail)?;
        let state = tx.open_table(STATE).map_err(fail)?;
        let receipts = tx.open_table(RECEIPTS).map_err(fail)?;
        let mut rows = state
            .range::<&[u8]>(..)
            .map_err(fail)?
            .map(|item| (PUT, item))
            .chain(
                receipts
                    .range::<&[u8]>(..)
                    .map_err(fail)?
                    .map(|item| (RECEIPT, item)),
            )
            .peekable();
        let mut schema = table_schema(
            &Schema::new(vec![
                Field::new("kind", DataType::UInt8, false),
                Field::new("key", DataType::LargeBinary, false),
                Field::new("value", DataType::LargeBinary, false),
            ]),
            &self.binding.run,
            &self.binding.schema,
        );
        for (key, value) in [
            (PARTITION, self.binding.partition.to_string()),
            (CHECKPOINT_FORMAT, "1".into()),
            (CHECKPOINT_URI, self.sink.dataset().uri().to_owned()),
            (CHECKPOINT_VERSION, self.version.to_string()),
            (CHECKPOINT_SEQUENCE, self.through.to_string()),
        ] {
            schema.metadata.insert(key.into(), value);
        }
        let schema = Arc::new(schema);
        let output_schema = schema.clone();
        let batch_rows = self.batch_rows;
        let batch_bytes = self.max_batch_bytes;
        let mut ended = false;
        let batches = std::iter::from_fn(move || {
            if ended {
                return None;
            }
            let mut kinds = Vec::new();
            let mut keys = Vec::new();
            let mut values = Vec::new();
            let mut bytes = 0usize;
            for _ in 0..batch_rows {
                if let Some((_, Ok((key, value)))) = rows.peek() {
                    let next_bytes = key.value().len().saturating_add(value.value().len());
                    if !kinds.is_empty() && bytes.saturating_add(next_bytes) > batch_bytes {
                        break;
                    }
                }
                let Some((kind, result)) = rows.next() else {
                    ended = true;
                    break;
                };
                let (key, value) = match result {
                    Ok(value) => value,
                    Err(error) => {
                        ended = true;
                        return Some(Err(arrow_schema::ArrowError::ExternalError(Box::new(
                            error,
                        ))));
                    }
                };
                bytes += key.value().len() + value.value().len();
                if bytes > batch_bytes {
                    ended = true;
                    return Some(Err(arrow_schema::ArrowError::InvalidArgumentError(
                        "checkpoint batch exceeds budget".into(),
                    )));
                }
                kinds.push(kind);
                keys.push(key.value().to_vec());
                values.push(value.value().to_vec());
            }
            if kinds.is_empty() {
                return None;
            }
            Some(RecordBatch::try_new(
                output_schema.clone(),
                vec![
                    Arc::new(UInt8Array::from(kinds)),
                    Arc::new(LargeBinaryArray::from_iter_values(keys)),
                    Arc::new(LargeBinaryArray::from_iter_values(values)),
                ],
            ))
        });
        Dataset::write(
            RecordBatchIterator::new(batches, schema),
            uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_bytes_per_file: self.max_batch_bytes,
                store_params: self.dataset().store_params().cloned(),
                ..Default::default()
            }),
        )
        .await
        .map_err(fail)
    }

    /// Make an exact completed checkpoint discoverable on cold restart. If the
    /// log advanced since capture, reject it; never point at a partial snapshot.
    pub async fn publish_checkpoint(&mut self, checkpoint: &Dataset) -> Result<()> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        let metadata = &checkpoint.schema().metadata;
        if metadata.get(CHECKPOINT_URI).map(String::as_str) != Some(self.dataset().uri())
            || metadata.get(RUN_METADATA) != Some(&self.binding.run)
            || metadata.get(SCHEMA_METADATA) != Some(&self.binding.schema)
            || metadata.get(CHECKPOINT_VERSION) != Some(&self.version.to_string())
            || metadata.get(CHECKPOINT_SEQUENCE) != Some(&self.through.to_string())
            || metadata.get(PARTITION) != Some(&self.binding.partition.to_string())
            || metadata.get(CHECKPOINT_FORMAT).map(String::as_str) != Some("1")
        {
            return Err(Error::Invalid(
                "checkpoint differs from current local state".into(),
            ));
        }
        self.poisoned = true;
        self.sink
            .update_metadata_at(
                std::collections::HashMap::from([
                    (PUBLISHED_CHECKPOINT_URI.into(), checkpoint.uri().to_owned()),
                    (
                        PUBLISHED_CHECKPOINT_VERSION.into(),
                        checkpoint.version().version.to_string(),
                    ),
                ]),
                self.version,
            )
            .await?;
        let tx = self.db.begin_write().map_err(fail)?;
        self.finish_local(tx, self.through)?;
        self.poisoned = false;
        Ok(())
    }

    pub fn get(&self, session: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        let tx = self.db.begin_read().map_err(fail)?;
        let table = tx.open_table(STATE).map_err(fail)?;
        let value = table
            .get(local_key(session, key).as_slice())
            .map_err(fail)?;
        Ok(value.map(|value| value.value().to_vec()))
    }

    /// Bounded local prefix scan. `after` is an exclusive state key, not an
    /// opaque remote cursor. This performs no network request or full DB scan.
    pub fn scan(
        &self,
        session: &str,
        prefix: &[u8],
        after: Option<&[u8]>,
        limit: usize,
    ) -> Result<Vec<Mutation>> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        if session.is_empty()
            || limit == 0
            || limit > self.batch_rows
            || after.is_some_and(|key| !key.starts_with(prefix))
        {
            return Err(Error::Invalid("invalid local state scan bounds".into()));
        }
        let tx = self.db.begin_read().map_err(fail)?;
        let table = tx.open_table(STATE).map_err(fail)?;
        let encoded_prefix = local_key(session, prefix);
        let start = local_key(session, after.unwrap_or(prefix));
        let mut output = Vec::new();
        for item in table.range(start.as_slice()..).map_err(fail)? {
            let (key, value) = item.map_err(fail)?;
            if !key.value().starts_with(&encoded_prefix) {
                break;
            }
            let suffix = &key.value()[8 + session.len()..];
            if after.is_some_and(|after| suffix <= after) {
                continue;
            }
            output.push(Mutation {
                key: suffix.to_vec(),
                value: Some(value.value().to_vec()),
            });
            if output.len() == limit {
                break;
            }
        }
        Ok(output)
    }

    /// Check before alignment. Same receipt with different input is an error,
    /// even when no output rows were produced by the original call.
    pub fn receipt(&self, session: &str, receipt: &str, input_digest: &str) -> Result<Option<u64>> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        let tx = self.db.begin_read().map_err(fail)?;
        let table = tx.open_table(RECEIPTS).map_err(fail)?;
        let value = table
            .get(local_key(session, receipt.as_bytes()).as_slice())
            .map_err(fail)?;
        let Some(value) = value else {
            return Ok(None);
        };
        let bytes = value.value();
        if bytes.len() < 8 || &bytes[8..] != input_digest.as_bytes() {
            return Err(Error::Invalid("changed committed receipt".into()));
        }
        Ok(Some(u64::from_le_bytes(bytes[..8].try_into().unwrap())))
    }

    /// Publish one coalesced batch, then apply its local transaction. Success
    /// means both output and state deltas are durable in the SAME Lance version.
    /// After uncertain success reopen and check receipts before realigning.
    pub async fn commit(&mut self, calls: &[AlignedCall]) -> Result<u64> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        if calls.is_empty() {
            return Ok(self.through);
        }
        let mut expected = self.through;
        let mut seen = std::collections::HashSet::new();
        let schema = Arc::new(Schema::from(self.sink.dataset().schema()));
        let mut entries = Vec::with_capacity(calls.len());
        let mut bytes = 0usize;
        let mut input_bytes = 0usize;
        for call in calls {
            expected = expected
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("sequence overflow".into()))?;
            if call.sequence != expected
                || call.session.is_empty()
                || call.receipt.is_empty()
                || call.input_digest.is_empty()
                || call.mutations.iter().any(|m| m.key.is_empty())
                || !seen.insert((&call.session, &call.receipt))
                || self
                    .receipt(&call.session, &call.receipt, &call.input_digest)?
                    .is_some()
            {
                return Err(Error::Invalid(
                    "invalid local alignment batch or committed retry".into(),
                ));
            }
            let expected_schema = log_schema(call.records.schema().as_ref(), &self.binding);
            if expected_schema != *schema
                || call
                    .records
                    .schema()
                    .fields
                    .iter()
                    .zip(call.records.columns())
                    .any(|(field, column)| {
                        field.data_type().is_nested()
                            || (!field.is_nullable() && column.null_count() != 0)
                    })
            {
                return Err(Error::Invalid(
                    "output schema differs from durable log".into(),
                ));
            }
            let row_count = call
                .mutations
                .len()
                .saturating_add(call.records.num_rows())
                .saturating_add(1);
            let row_overhead = call
                .session
                .len()
                .saturating_add(call.receipt.len())
                .saturating_add(call.input_digest.len())
                .saturating_add(64);
            input_bytes = input_bytes
                .saturating_add(row_count.saturating_mul(row_overhead))
                .saturating_add(call.records.get_array_memory_size());
            for mutation in &call.mutations {
                let cell_bytes = mutation
                    .key
                    .len()
                    .saturating_add(mutation.value.as_ref().map_or(0, Vec::len));
                if cell_bytes > 1 << 20 {
                    return Err(Error::Invalid(
                        "state cell exceeds 1 MiB; split the state into smaller keys".into(),
                    ));
                }
                input_bytes = input_bytes.saturating_add(cell_bytes);
            }
            if input_bytes > self.max_batch_bytes {
                return Err(Error::Invalid(
                    "local Lance input exceeds byte budget".into(),
                ));
            }
            let records = encode_call(schema.clone(), call, self.batch_rows)?;
            bytes = bytes
                .checked_add(records.len())
                .ok_or_else(|| Error::Invalid("batch size overflow".into()))?;
            if bytes > self.max_batch_bytes {
                return Err(Error::Invalid(
                    "local Lance batch exceeds byte budget".into(),
                ));
            }
            entries.push(Entry {
                sequence: call.sequence,
                session: call.session.clone(),
                receipt: call.receipt.clone(),
                input_digest: call.input_digest.clone(),
                transition: Transition {
                    delta: Vec::new(),
                    records,
                },
            });
        }
        let base = self.version;
        self.poisoned = true;
        let staged = stage(
            self.sink.dataset(),
            &self.binding,
            &entries,
            self.max_batch_bytes,
        )
        .await?;
        self.sink.commit_staged_at(vec![staged], base).await?;
        let tx = self.db.begin_write().map_err(fail)?;
        {
            let mut state = tx.open_table(STATE).map_err(fail)?;
            let mut receipts = tx.open_table(RECEIPTS).map_err(fail)?;
            for call in calls {
                for mutation in &call.mutations {
                    let key = local_key(&call.session, &mutation.key);
                    if let Some(value) = &mutation.value {
                        state
                            .insert(key.as_slice(), value.as_slice())
                            .map_err(fail)?;
                    } else {
                        state.remove(key.as_slice()).map_err(fail)?;
                    }
                }
                receipts
                    .insert(
                        local_key(&call.session, call.receipt.as_bytes()).as_slice(),
                        receipt_value(call.sequence, &call.input_digest).as_slice(),
                    )
                    .map_err(fail)?;
            }
        }
        self.finish_local(tx, expected)?;
        self.poisoned = false;
        Ok(expected)
    }

    fn finish_local(&mut self, tx: redb::WriteTransaction, sequence: u64) -> Result<()> {
        let version = self.sink.dataset().version().version;
        {
            let mut meta = tx.open_table(META).map_err(fail)?;
            meta.insert("through", sequence.to_le_bytes().as_slice())
                .map_err(fail)?;
            meta.insert("version", version.to_le_bytes().as_slice())
                .map_err(fail)?;
        }
        tx.commit().map_err(fail)?;
        self.through = sequence;
        self.version = version;
        Ok(())
    }

    async fn replay(&mut self, head: u64) -> Result<()> {
        // Old immutable fragments need not be opened to replay a new suffix.
        // Metadata equality is required; compaction/deletions are not an append.
        let mut fragments = self.sink.dataset().get_fragments();
        if self.version > 0 {
            let base = self
                .sink
                .dataset()
                .checkout_version(self.version)
                .await
                .map_err(fail)?;
            let previous = base.get_fragments();
            for old in &previous {
                if !fragments.iter().any(|new| new.metadata() == old.metadata()) {
                    return Err(Error::Invalid(
                        "state log is not an immutable append descendant".into(),
                    ));
                }
            }
            fragments.retain(|fragment| !previous.iter().any(|old| old.id() == fragment.id()));
        }
        if fragments.is_empty() {
            if self.through != head {
                return Err(Error::Invalid("state log watermark without data".into()));
            }
            let tx = self.db.begin_write().map_err(fail)?;
            return self.finish_local(tx, head);
        }
        let mut scan = self.sink.dataset().scan();
        scan.with_fragments(
            fragments
                .into_iter()
                .map(|fragment| fragment.metadata().clone())
                .collect(),
        );
        scan.project(&COLUMNS).map_err(fail)?;
        scan.filter(&format!("sequence > {}", self.through))
            .map_err(fail)?;
        scan.batch_size(self.batch_rows);
        scan.scan_in_order(true);
        let mut stream = scan.try_into_stream().await.map_err(fail)?;
        let tx = self.db.begin_write().map_err(fail)?;
        let mut state = tx.open_table(STATE).map_err(fail)?;
        let mut receipts = tx.open_table(RECEIPTS).map_err(fail)?;
        let mut through = self.through;
        let mut ordinal = 0;
        let mut identity: Option<(String, String, String)> = None;
        while let Some(batch) = stream.try_next().await.map_err(fail)? {
            let sequences = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| fail("invalid sequence column"))?;
            let ordinals = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| fail("invalid ordinal column"))?;
            let kinds = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| fail("invalid kind column"))?;
            let sessions = batch
                .column(3)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| fail("invalid session column"))?;
            let ids = batch
                .column(4)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| fail("invalid receipt column"))?;
            let digests = batch
                .column(5)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| fail("invalid digest column"))?;
            let keys = batch
                .column(6)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| fail("invalid key column"))?;
            let values = batch
                .column(7)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| fail("invalid value column"))?;
            if batch
                .columns()
                .iter()
                .any(|column| column.null_count() != 0)
            {
                return Err(Error::Invalid("null local state log cell".into()));
            }
            for i in 0..batch.num_rows() {
                let sequence = sequences.value(i);
                if through.checked_add(1) != Some(sequence)
                    || sequence > head
                    || ordinals.value(i) != ordinal
                {
                    return Err(Error::Invalid(
                        "local state log sequence or ordinal gap".into(),
                    ));
                }
                let session = sessions.value(i);
                let receipt = ids.value(i);
                let digest = digests.value(i);
                if session.is_empty() || receipt.is_empty() || digest.is_empty() {
                    return Err(Error::Invalid("empty local state log identity".into()));
                }
                let expected =
                    identity.get_or_insert_with(|| (session.into(), receipt.into(), digest.into()));
                if expected.0 != session || expected.1 != receipt || expected.2 != digest {
                    return Err(Error::Invalid(
                        "mixed identities in local state sequence".into(),
                    ));
                }
                let key = local_key(session, keys.value(i));
                match kinds.value(i) {
                    PUT if !keys.value(i).is_empty() => {
                        state
                            .insert(key.as_slice(), values.value(i))
                            .map_err(fail)?;
                    }
                    DELETE if !keys.value(i).is_empty() && values.value(i).is_empty() => {
                        state.remove(key.as_slice()).map_err(fail)?;
                    }
                    RECORD if keys.value(i).is_empty() && values.value(i).is_empty() => {}
                    RECEIPT if keys.value(i).is_empty() && values.value(i).is_empty() => {
                        let key = local_key(session, receipt.as_bytes());
                        if receipts.get(key.as_slice()).map_err(fail)?.is_some() {
                            return Err(Error::Invalid(
                                "duplicate receipt in committed log".into(),
                            ));
                        }
                        receipts
                            .insert(key.as_slice(), receipt_value(sequence, digest).as_slice())
                            .map_err(fail)?;
                        through = sequence;
                        ordinal = 0;
                        identity = None;
                        continue;
                    }
                    _ => return Err(Error::Invalid("invalid local state log event".into())),
                }
                ordinal += 1;
            }
        }
        if through != head || ordinal != 0 || identity.is_some() {
            return Err(Error::Invalid("incomplete local state log prefix".into()));
        }
        drop(state);
        drop(receipts);
        self.finish_local(tx, head)
    }
}

/// Seed a missing local cache from an owner-published exact checkpoint. Call
/// `LocalLancePartition::open` afterwards to verify/replay the committed suffix.
/// Existing local files are never overwritten. WAL/checkpoint objects remain
/// untouched, including when a read fails halfway through this transaction.
pub async fn restore_checkpoint(
    local_file: &Path,
    checkpoint: &Dataset,
    wal: &Dataset,
    binding: &Binding,
    cache_bytes: usize,
    batch_rows: usize,
) -> Result<()> {
    let schema = Schema::from(checkpoint.schema());
    let metadata = &schema.metadata;
    let number = |key: &str| -> Result<u64> {
        metadata
            .get(key)
            .ok_or_else(|| fail("missing checkpoint binding"))?
            .parse()
            .map_err(fail)
    };
    let sequence = number(CHECKPOINT_SEQUENCE)?;
    let version = number(CHECKPOINT_VERSION)?;
    let wal_schema = Schema::from(wal.schema());
    if local_file.exists()
        || cache_bytes == 0
        || batch_rows == 0
        || metadata.get(CHECKPOINT_FORMAT).map(String::as_str) != Some("1")
        || metadata.get(CHECKPOINT_URI).map(String::as_str) != Some(wal.uri())
        || version > wal.version().version
        || version == 0
        || [metadata, &wal_schema.metadata].iter().any(|meta| {
            meta.get(RUN_METADATA) != Some(&binding.run)
                || meta.get(SCHEMA_METADATA) != Some(&binding.schema)
                || meta.get(PARTITION) != Some(&binding.partition.to_string())
        })
        || checkpoint.manifest().data_storage_format.version != "2.2"
    {
        return Err(Error::Invalid(
            "checkpoint binding mismatch or existing local cache".into(),
        ));
    }
    // Ensure the captured ordinary version still belongs to this durable log.
    wal.checkout_version(version).await.map_err(fail)?;
    // Install only a complete, committed local database. A failed read or
    // cancelled future drops the temporary file; the next open can retry the
    // published checkpoint instead of accidentally falling back to old WAL.
    let temporary = tempfile::NamedTempFile::new_in(
        local_file
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )
    .map_err(fail)?;
    let db = Database::builder()
        .set_cache_size(cache_bytes)
        .create(temporary.path())
        .map_err(fail)?;
    let tx = db.begin_write().map_err(fail)?;
    let mut state = tx.open_table(STATE).map_err(fail)?;
    let mut receipts = tx.open_table(RECEIPTS).map_err(fail)?;
    let mut scan = checkpoint.scan();
    scan.project(&["kind", "key", "value"])
        .map_err(fail)?
        .batch_size(batch_rows);
    let mut stream = scan.try_into_stream().await.map_err(fail)?;
    while let Some(batch) = stream.try_next().await.map_err(fail)? {
        let kinds = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| fail("invalid checkpoint kind"))?;
        let keys = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| fail("invalid checkpoint key"))?;
        let values = batch
            .column(2)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| fail("invalid checkpoint value"))?;
        if batch
            .columns()
            .iter()
            .any(|column| column.null_count() != 0)
        {
            return Err(Error::Invalid("null checkpoint cell".into()));
        }
        for i in 0..batch.num_rows() {
            let key = keys.value(i);
            let value = values.value(i);
            if key.len() < 9 {
                return Err(Error::Invalid("invalid checkpoint state key".into()));
            }
            let session_len = u64::from_be_bytes(key[..8].try_into().unwrap());
            if session_len == 0 || session_len >= (key.len() - 8) as u64 {
                return Err(Error::Invalid("invalid checkpoint session framing".into()));
            }
            let table = match kinds.value(i) {
                PUT => &mut state,
                RECEIPT
                    if value.len() > 8
                        && u64::from_le_bytes(value[..8].try_into().unwrap()) > 0
                        && u64::from_le_bytes(value[..8].try_into().unwrap()) <= sequence =>
                {
                    &mut receipts
                }
                _ => return Err(Error::Invalid("invalid checkpoint state event".into())),
            };
            if table.get(key).map_err(fail)?.is_some() {
                return Err(Error::Invalid("duplicate checkpoint state key".into()));
            }
            table.insert(key, value).map_err(fail)?;
        }
    }
    drop(state);
    drop(receipts);
    {
        let mut meta = tx.open_table(META).map_err(fail)?;
        for (key, value) in [
            ("run", binding.run.clone()),
            ("schema", binding.schema.clone()),
            ("partition", binding.partition.to_string()),
            ("uri", wal.uri().to_owned()),
        ] {
            meta.insert(key, value.as_bytes()).map_err(fail)?;
        }
        meta.insert("through", sequence.to_le_bytes().as_slice())
            .map_err(fail)?;
        meta.insert("version", version.to_le_bytes().as_slice())
            .map_err(fail)?;
    }
    tx.commit().map_err(fail)?;
    drop(db);
    temporary.as_file().sync_all().map_err(fail)?;
    temporary.persist_noclobber(local_file).map_err(fail)?;
    Ok(())
}
