//! Version-pinned reads for an independent consumer of a local-state Lance log.
use crate::{lance_sink::covered_sequence_at, Binding, Error, Result};
use arrow_array::{Array, RecordBatch, StructArray, UInt64Array};
use futures::TryStreamExt;
use lance::Dataset;

fn fail(error: impl std::fmt::Display) -> Error {
    Error::Stage(error.to_string())
}

/// A complete immutable range. Versions identify manifests; sequences identify
/// entries, and neither is the downstream consumer's batch/generation number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogRange {
    pub base_version: u64,
    pub version: u64,
    pub first_sequence: u64,
    pub last_sequence: u64,
}

/// Validates both watermarks and full fragment ancestry before reading. The
/// physical byte count is the actual newly referenced files, not a descriptor's
/// small serialized size. State/receipt rows never appear in returned output.
pub struct OutputRange {
    dataset: Dataset,
    fragments: Vec<lance_table::format::Fragment>,
    pub physical_bytes: u64,
    range: LogRange,
}

impl OutputRange {
    pub async fn open(dataset: Dataset, binding: &Binding, range: LogRange) -> Result<Self> {
        if range.base_version == 0
            || range.version <= range.base_version
            || dataset.version().version != range.version
            || range.first_sequence == 0
            || range.last_sequence < range.first_sequence
        {
            return Err(Error::Invalid("invalid immutable Lance log range".into()));
        }
        let schema = &dataset.schema().metadata;
        if schema.get("lance-context.ingestion.partition") != Some(&binding.partition.to_string())
            || schema
                .get("lance-context.ingestion.local-state-format")
                .map(String::as_str)
                != Some("1")
        {
            return Err(Error::Invalid("unexpected local Lance log binding".into()));
        }
        let base = dataset
            .checkout_version(range.base_version)
            .await
            .map_err(fail)?;
        if covered_sequence_at(&base, binding).await?.checked_add(1) != Some(range.first_sequence)
            || covered_sequence_at(&dataset, binding).await? != range.last_sequence
            || base.schema() != dataset.schema()
        {
            return Err(Error::Invalid(
                "Lance log range watermark/schema mismatch".into(),
            ));
        }
        let previous = base.get_fragments();
        let current = dataset.get_fragments();
        let by_id = current
            .iter()
            .map(|f| (f.id(), f.metadata()))
            .collect::<std::collections::HashMap<_, _>>();
        if previous
            .iter()
            .any(|old| by_id.get(&old.id()).copied() != Some(old.metadata()))
        {
            return Err(Error::Invalid("Lance log range is not append-only".into()));
        }
        let previous_ids = previous
            .iter()
            .map(|f| f.id())
            .collect::<std::collections::HashSet<_>>();
        let fragments = current
            .into_iter()
            .filter(|new| !previous_ids.contains(&new.id()))
            .map(|f| f.metadata().clone())
            .collect::<Vec<_>>();
        let mut physical_bytes = 0u64;
        for fragment in &fragments {
            for file in &fragment.files {
                let size = file
                    .file_size_bytes
                    .get()
                    .ok_or_else(|| Error::Invalid("Lance log file size missing".into()))?
                    .get();
                physical_bytes = physical_bytes
                    .checked_add(size)
                    .ok_or_else(|| Error::Invalid("Lance log file sizes overflow".into()))?;
            }
        }
        if fragments.is_empty() {
            return Err(Error::Invalid(
                "Lance log range lacks sealed receipt rows".into(),
            ));
        }
        Ok(Self {
            dataset,
            fragments,
            physical_bytes,
            range,
        })
    }

    /// Bound both referenced physical data and retained decoded output. The
    /// caller separately budgets scanner working memory and its downstream
    /// representation. Errors return no partial success or advancement cursor.
    pub async fn read(
        self,
        max_physical_bytes: u64,
        max_decoded_bytes: usize,
        batch_rows: usize,
    ) -> Result<Vec<RecordBatch>> {
        self.read_projection(max_physical_bytes, max_decoded_bytes, batch_rows, None)
            .await
    }

    /// Read selected field paths within the output `record` struct. Projection
    /// first selects matching row IDs, then reads only the projected schema,
    /// preserving the record struct's validity before charging retained output.
    /// Physical-file, range and sequence checks are unchanged. Empty projections
    /// are rejected; omitted fields remain in the immutable log for recovery.
    pub async fn read_projected(
        self,
        max_physical_bytes: u64,
        max_decoded_bytes: usize,
        batch_rows: usize,
        fields: &[&str],
    ) -> Result<Vec<RecordBatch>> {
        if fields.is_empty() || fields.iter().any(|field| field.is_empty()) {
            return Err(Error::Invalid("empty Lance output projection".into()));
        }
        let columns = fields
            .iter()
            .map(|field| format!("record.{field}"))
            .collect::<Vec<_>>();
        if columns
            .iter()
            .any(|column| self.dataset.schema().field(column).is_none())
        {
            return Err(Error::Invalid(
                "unknown Lance output projection field".into(),
            ));
        }
        self.read_projection(
            max_physical_bytes,
            max_decoded_bytes,
            batch_rows,
            Some(&columns),
        )
        .await
    }

    async fn read_projection(
        self,
        max_physical_bytes: u64,
        max_decoded_bytes: usize,
        batch_rows: usize,
        columns: Option<&[String]>,
    ) -> Result<Vec<RecordBatch>> {
        if self.physical_bytes > max_physical_bytes || max_decoded_bytes == 0 || batch_rows == 0 {
            return Err(Error::Invalid("Lance log range exceeds read budget".into()));
        }
        let mut scan = self.dataset.scan();
        scan.with_fragments(self.fragments)
            .batch_size(batch_rows)
            .scan_in_order(true);
        let projection = columns
            .map(|columns| self.dataset.schema().project(columns))
            .transpose()
            .map_err(fail)?;
        if projection.is_some() {
            scan.empty_project().map_err(fail)?.with_row_id();
        } else {
            scan.project(&["record"]).map_err(fail)?;
        }
        scan.filter(&format!(
            "kind = 3 AND sequence >= {} AND sequence <= {}",
            self.range.first_sequence, self.range.last_sequence
        ))
        .map_err(fail)?;
        let mut stream = scan.try_into_stream().await.map_err(fail)?;
        let mut bytes = 0usize;
        let mut output = Vec::new();
        while let Some(batch) = stream.try_next().await.map_err(fail)? {
            let batch = if let Some(projection) = &projection {
                let row_ids = batch
                    .column_by_name("_rowid")
                    .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
                    .filter(|ids| ids.null_count() == 0)
                    .ok_or_else(|| Error::Invalid("invalid Lance output row IDs".into()))?;
                self.dataset
                    .take_rows(row_ids.values(), projection.clone())
                    .await
                    .map_err(fail)?
            } else {
                batch
            };
            bytes = bytes
                .checked_add(batch.get_array_memory_size())
                .ok_or_else(|| Error::Invalid("decoded Lance log size overflow".into()))?;
            if bytes > max_decoded_bytes {
                return Err(Error::Invalid(
                    "decoded Lance log exceeds read budget".into(),
                ));
            }
            let records = batch
                .column(0)
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| Error::Invalid("Lance log output is not typed records".into()))?;
            if records.null_count() != 0 {
                return Err(Error::Invalid("null Lance log output row".into()));
            }
            output.push(RecordBatch::from(records.clone()));
        }
        Ok(output)
    }
}

/// Small immutable object-store descriptor for a downstream batch publisher.
/// Application metadata is explicitly binary and bounded, e.g. native cumulative
/// counters and migration provenance. OutputRange independently checks the
/// referenced files; a descriptor is never accepted as proof of payload size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchReference {
    pub binding: Binding,
    pub uri: String,
    pub generation: u64,
    pub previous_generation: u64,
    pub range: LogRange,
    pub physical_bytes: u64,
    pub decoded_byte_limit: u64,
    pub output_rows: u64,
    pub application_metadata: Vec<u8>,
}

impl BatchReference {
    pub fn encode(&self) -> Result<Vec<u8>> {
        use sha2::{Digest, Sha256};
        if self.previous_generation.checked_add(1) != Some(self.generation)
            || self.range.base_version == 0
            || self.range.version <= self.range.base_version
            || self.range.first_sequence == 0
            || self.range.last_sequence < self.range.first_sequence
            || self.decoded_byte_limit == 0
            || self.physical_bytes == 0
            || [&self.binding.run, &self.binding.schema, &self.uri]
                .iter()
                .any(|s| s.is_empty() || s.len() > 4096)
            || self.application_metadata.len() > 64 * 1024
        {
            return Err(Error::Invalid(
                "invalid binary Lance batch reference".into(),
            ));
        }
        let mut bytes = b"LCBATCH1".to_vec();
        for n in [
            self.generation,
            self.previous_generation,
            self.range.base_version,
            self.range.version,
            self.range.first_sequence,
            self.range.last_sequence,
            self.physical_bytes,
            self.decoded_byte_limit,
            self.output_rows,
        ] {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(&self.binding.partition.to_le_bytes());
        for value in [
            self.binding.run.as_bytes(),
            self.binding.schema.as_bytes(),
            self.uri.as_bytes(),
            self.application_metadata.as_slice(),
        ] {
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value);
        }
        let digest = Sha256::digest(&bytes);
        bytes.extend_from_slice(&digest);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        use sha2::{Digest, Sha256};
        use std::io::{Cursor, Read};
        if bytes.len() < 8 + 9 * 8 + 4 + 4 * 4 + 32
            || bytes.len() > 128 * 1024
            || &bytes[..8] != b"LCBATCH1"
        {
            return Err(Error::Invalid(
                "invalid binary Lance batch reference size/format".into(),
            ));
        }
        let (payload, digest) = bytes.split_at(bytes.len() - 32);
        if Sha256::digest(payload).as_slice() != digest {
            return Err(Error::Invalid(
                "binary Lance batch reference checksum mismatch".into(),
            ));
        }
        let mut cursor = Cursor::new(&payload[8..]);
        let mut number = || -> Result<u64> {
            let mut b = [0u8; 8];
            cursor.read_exact(&mut b).map_err(fail)?;
            Ok(u64::from_le_bytes(b))
        };
        let generation = number()?;
        let previous_generation = number()?;
        let base_version = number()?;
        let version = number()?;
        let first_sequence = number()?;
        let last_sequence = number()?;
        let physical_bytes = number()?;
        let decoded_byte_limit = number()?;
        let output_rows = number()?;
        let mut p = [0u8; 4];
        cursor.read_exact(&mut p).map_err(fail)?;
        let partition = u32::from_le_bytes(p);
        let mut field = || -> Result<Vec<u8>> {
            let mut b = [0u8; 4];
            cursor.read_exact(&mut b).map_err(fail)?;
            let len = u32::from_le_bytes(b) as usize;
            if len > 64 * 1024
                || len
                    > cursor
                        .get_ref()
                        .len()
                        .saturating_sub(cursor.position() as usize)
            {
                return Err(Error::Invalid(
                    "binary Lance batch reference field size".into(),
                ));
            }
            let mut value = vec![0; len];
            cursor.read_exact(&mut value).map_err(fail)?;
            Ok(value)
        };
        let run = String::from_utf8(field()?).map_err(fail)?;
        let schema = String::from_utf8(field()?).map_err(fail)?;
        let uri = String::from_utf8(field()?).map_err(fail)?;
        let application_metadata = field()?;
        if cursor.position() as usize != payload.len() - 8 {
            return Err(Error::Invalid(
                "trailing binary Lance batch reference data".into(),
            ));
        }
        let value = Self {
            binding: Binding {
                run,
                schema,
                partition,
            },
            uri,
            generation,
            previous_generation,
            range: LogRange {
                base_version,
                version,
                first_sequence,
                last_sequence,
            },
            physical_bytes,
            decoded_byte_limit,
            output_rows,
            application_metadata,
        };
        if value.encode()?.as_slice() != bytes {
            return Err(Error::Invalid(
                "noncanonical binary Lance batch reference".into(),
            ));
        }
        Ok(value)
    }
}
