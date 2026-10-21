//! Immutable application images and bounded file transfer. Payloads stay opaque;
//! building and loading an image never decode Arrow or concatenate its bodies.
//! File I/O belongs on a bounded worker, outside the Raft owner. Only a later
//! durable Raft transition can install a prepared image. Files are retained on
//! errors; images omit retired bodies but physical collection is separate.

use super::*;
use bincode::Options;
use bytes::Bytes;
use raft_proto::eraftpb::{ConfState, Snapshot};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
};

fn format(limits: &Limits) -> u32 {
    if limits.producer_slots == 0 {
        9
    } else {
        10
    }
}
fn magic(limits: &Limits) -> &'static [u8; 8] {
    if limits.producer_slots == 0 {
        b"LCWALCP9"
    } else {
        b"LCWALCPA"
    }
}
pub const CHUNK_BYTES: usize = 1024 * 1024;

fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(4096)
        .reject_trailing_bytes()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Binding {
    cluster: Uuid,
    layout: [u8; 32],
    policy: [u8; 32],
    group: u64,
    index: u64,
    term: u64,
    voters: [u64; 3],
}

impl Binding {
    fn new(layout: &Layout, limits: &Limits, group: u64, index: u64, term: u64) -> Result<Self> {
        let layout_hash = layout.fingerprint()?;
        ensure!(
            index > 0 && index < u64::MAX && term > 0,
            "invalid image cut"
        );
        Ok(Self {
            cluster: layout.cluster,
            layout: layout_hash,
            policy: Sha256::digest(limits.identity()?).into(),
            group,
            index,
            term,
            voters: layout.groups.get(&group).context("unknown image group")?[..].try_into()?,
        })
    }

    fn validate(&self, layout: &Layout, limits: &Limits, group: u64) -> Result<()> {
        ensure!(
            *self == Self::new(layout, limits, group, self.index, self.term)?,
            "foreign application image scope, policy or membership"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Descriptor {
    format: u32,
    binding: Binding,
    id: Uuid,
    records: u64,
    producers: u64,
    tables: u64,
    pins: u64,
    closed: u64,
    retired: u64,
    metadata_bytes: u64,
    payload_bytes: u64,
    metadata_digest: [u8; 32],
    payload_digest: [u8; 32],
}

fn metadata_limit(limits: &Limits) -> u64 {
    // A framed record needs at most 8 + 4 + 256 bytes, plus a bounded header.
    // limits.identity() validates all policy maxima before this is used.
    (limits.max_records + limits.max_producers) * 268
        + limits.max_tables * (publication::MAX_BASE_TABLE_BYTES as u64 + 4)
        + limits.max_tables.min(limits.max_active_plans)
            * (128 * 1024 + limits.max_descriptor_bytes + 4096)
        + limits.max_tables.min(limits.max_active_plans) * (source::MAX_PIN_BYTES as u64 + 4)
        + limits.max_manifest_records * 2048
        + limits.max_tables * (256 + publication::advance_metadata_bytes())
        + limits.max_tables * (source::MAX_CLOSED_BYTES as u64 + 4)
        + limits.max_producers * (state::MAX_RETIRED_BYTES as u64 + 4)
        + u64::from(limits.producer_slots) * (producer::MAX_CELL_BYTES as u64 + 4)
        + 1024
}

impl Descriptor {
    fn validate(&self, layout: &Layout, limits: &Limits, group: u64) -> Result<()> {
        self.binding.validate(layout, limits, group)?;
        ensure!(
            self.format == format(limits)
                && !self.id.is_nil()
                && self.records <= limits.max_records
                && self.producers <= limits.max_producers
                && self.retired <= self.producers
                && self.tables <= limits.max_tables
                && self.pins <= limits.max_tables.min(limits.max_active_plans)
                && self.closed <= limits.max_tables
                && (group == publication::control_group(layout) || self.tables == 0)
                && self.metadata_bytes <= metadata_limit(limits)
                && self.payload_bytes <= self.records * limits.max_record_bytes,
            "application image format or bounds mismatch"
        );
        Ok(())
    }

    fn snapshot(&self) -> Result<Snapshot> {
        let mut snapshot = Snapshot::default();
        let metadata = snapshot.mut_metadata();
        metadata.index = self.binding.index;
        metadata.term = self.binding.term;
        metadata.set_conf_state(ConfState {
            voters: self.binding.voters.to_vec(),
            ..Default::default()
        });
        snapshot.data = codec().serialize(self)?.into();
        ensure!(
            snapshot.compute_size() <= 4096,
            "oversized image descriptor"
        );
        Ok(snapshot)
    }

    fn decode(snapshot: &Snapshot, layout: &Layout, limits: &Limits, group: u64) -> Result<Self> {
        ensure!(
            snapshot.compute_size() <= 4096,
            "oversized image descriptor"
        );
        let descriptor: Self = codec().deserialize(&snapshot.data)?;
        descriptor.validate(layout, limits, group)?;
        ensure!(
            descriptor.snapshot()? == *snapshot,
            "image descriptor and Raft cut disagree"
        );
        Ok(descriptor)
    }

    fn paths(&self, directory: &Path) -> [PathBuf; 2] {
        let name = format!("context-application-{}-{}", self.binding.group, self.id);
        [
            directory.join(format!("{name}.state")),
            directory.join(format!("{name}.pack")),
        ]
    }
}

pub(super) fn snapshot_sizes(
    snapshot: &Snapshot,
    layout: &Layout,
    limits: &Limits,
    group: u64,
) -> Result<[u64; 2]> {
    let descriptor = Descriptor::decode(snapshot, layout, limits, group)?;
    Ok([descriptor.metadata_bytes, descriptor.payload_bytes])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Metadata,
    Payload,
}
impl Part {
    fn slot(self) -> usize {
        match self {
            Self::Metadata => 0,
            Self::Payload => 1,
        }
    }
}

#[derive(Clone, Debug)]
struct Location {
    offset: u64,
    bytes: usize,
    digest: [u8; 32],
}

/// Open immutable files keep an outgoing transfer or reader attached to its
/// captured image. Collection retains its files while any image owner is alive.
pub struct Image {
    descriptor: Descriptor,
    files: [File; 2],
    owner: std::sync::OnceLock<Arc<super::images::Owner>>,
    #[cfg(not(unix))]
    paths: [PathBuf; 2],
    locations: BTreeMap<u64, Location>,
}
impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl Image {
    pub(super) fn identity(&self) -> (u64, Uuid) {
        (self.descriptor.binding.group, self.descriptor.id)
    }

    pub(super) fn keep_owner(&self, owner: &Arc<super::images::Owner>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(self.owner.get_or_init(|| owner.clone()), owner),
            "foreign application image owner"
        );
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        self.descriptor.snapshot()
    }

    pub fn bytes(&self, part: Part) -> u64 {
        match part {
            Part::Metadata => self.descriptor.metadata_bytes,
            Part::Payload => self.descriptor.payload_bytes,
        }
    }

    /// Return at most one MiB from a captured image. Transport authentication,
    /// deadlines and node-wide network/disk admission belong to the caller.
    pub fn chunk(&self, part: Part, offset: u64, bytes: usize) -> Result<Bytes> {
        ensure!(
            (1..=CHUNK_BYTES).contains(&bytes)
                && offset
                    .checked_add(bytes as u64)
                    .is_some_and(|end| end <= self.bytes(part)),
            "invalid image chunk range"
        );
        self.read_at(part, offset, bytes)
    }

    fn read_at(&self, part: Part, offset: u64, bytes: usize) -> Result<Bytes> {
        let slot = part.slot();
        ensure!(
            self.files[slot].metadata()?.len() == self.bytes(part),
            "image file size changed"
        );
        let mut body = vec![0; bytes];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.files[slot].read_exact_at(&mut body, offset)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Seek, SeekFrom};
            // Independently opened cursors avoid try_clone's shared seek offset.
            let mut file = File::open(&self.paths[slot])?;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut body)?;
        }
        Ok(body.into())
    }

    /// The build/read owner validates this body against its captured BatchMeta.
    /// Keep hashing at that common boundary, once for pack and log inputs alike.
    pub(super) fn read_for_validation(&self, index: u64) -> Result<Option<Bytes>> {
        let Some(location) = self.locations.get(&index) else {
            return Ok(None);
        };
        Ok(Some(self.read_at(
            Part::Payload,
            location.offset,
            location.bytes,
        )?))
    }
}

pub(super) struct Loaded {
    pub image: Arc<Image>,
    pub state: State,
}

pub(super) struct Cut {
    pub group: u64,
    pub index: u64,
    pub term: u64,
    pub id: Uuid,
}

struct Writer {
    file: BufWriter<File>,
    digest: Sha256,
    bytes: u64,
    limit: u64,
    failed: bool,
}
impl Writer {
    fn new(path: &Path, limit: u64) -> Result<Self> {
        Ok(Self {
            file: BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?),
            digest: Sha256::new(),
            bytes: 0,
            limit,
            failed: false,
        })
    }
    fn finish(mut self) -> Result<(u64, [u8; 32])> {
        ensure!(!self.failed, "image writer failed; abandon this candidate");
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        Ok((self.bytes, self.digest.finalize().into()))
    }
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.failed {
            return Err(std::io::Error::other("image writer failed"));
        }
        if (bytes.len() as u64) > self.limit - self.bytes {
            return Err(std::io::Error::other("image file exceeds bound"));
        }
        let n = match self.file.write(bytes) {
            Ok(n) => n,
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        self.bytes += n as u64;
        self.digest.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        let result = self.file.flush();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

struct Reader {
    file: BufReader<File>,
    digest: Sha256,
    bytes: u64,
}
impl Reader {
    fn new(path: &Path, expected: u64) -> Result<Self> {
        let file = File::open(path)?;
        ensure!(
            file.metadata()?.len() == expected,
            "image file length mismatch"
        );
        Ok(Self {
            file: BufReader::new(file),
            digest: Sha256::new(),
            bytes: 0,
        })
    }
    fn finish(mut self, bytes: u64, digest: [u8; 32]) -> Result<File> {
        let mut extra = [0];
        ensure!(
            self.read(&mut extra)? == 0 && self.bytes == bytes,
            "trailing image bytes"
        );
        ensure!(
            <[u8; 32]>::from(self.digest.finalize()) == digest,
            "image file checksum mismatch"
        );
        Ok(self.file.into_inner())
    }
}
impl Read for Reader {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read(bytes)?;
        self.bytes += n as u64;
        self.digest.update(&bytes[..n]);
        Ok(n)
    }
}

fn write_u64(out: &mut impl Write, value: u64) -> Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}
fn read_u64(input: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn write_frame(out: &mut impl Write, bytes: &[u8]) -> Result<()> {
    out.write_all(&u32::try_from(bytes.len())?.to_le_bytes())?;
    out.write_all(bytes)?;
    Ok(())
}
fn read_frame<const N: usize>(input: &mut impl Read) -> Result<([u8; N], usize)> {
    let mut length = [0; 4];
    input.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    ensure!(length > 0 && length <= N, "oversized image metadata frame");
    let mut bytes = [0; N];
    input.read_exact(&mut bytes[..length])?;
    Ok((bytes, length))
}
fn read_dynamic_frame(input: &mut impl Read, maximum: usize) -> Result<Vec<u8>> {
    let mut length = [0; 4];
    input.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    ensure!(
        length > 0 && length <= maximum,
        "oversized publisher image frame"
    );
    // Plan metadata is larger than a record header. Allocate its actual bounded
    // length on the heap rather than putting the maximum frame on the stack.
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn write_meta(out: &mut impl Write, index: u64, meta: &BatchMeta) -> Result<()> {
    write_u64(out, index)?;
    write_frame(out, &state::encode_meta(meta)?)
}
fn read_meta(input: &mut impl Read) -> Result<(u64, BatchMeta)> {
    let index = read_u64(input)?;
    let (bytes, n) = read_frame::<256>(input)?;
    Ok((index, state::decode_meta(&bytes[..n])?))
}

fn locations(state: &State) -> Result<(BTreeMap<u64, Location>, u64)> {
    let mut offset = 0u64;
    let mut locations = BTreeMap::new();
    for (index, meta) in &state.accepted {
        locations.insert(
            *index,
            Location {
                offset,
                bytes: usize::try_from(meta.body_bytes)?,
                digest: meta.digest,
            },
        );
        offset = offset
            .checked_add(meta.body_bytes)
            .context("image payload size overflow")?;
    }
    Ok((locations, offset))
}

/// Inputs must be a captured, immutable applied cut. The owner retains its log
/// and prior image while this worker reads one bounded body at a time.
pub(super) fn build(
    directory: &Path,
    layout: &Layout,
    limits: &Limits,
    cut: Cut,
    state: State,
    mut read: impl FnMut(u64, &BatchMeta) -> Result<Bytes>,
) -> Result<Loaded> {
    let Cut {
        group,
        index,
        term,
        id,
    } = cut;
    let binding = Binding::new(layout, limits, group, index, term)?;
    state.validate_checkpoint(layout, group, limits, binding.index)?;
    ensure!(!id.is_nil(), "nil application image id");
    let (locations, payload_bytes) = locations(&state)?;
    let mut descriptor = Descriptor {
        format: format(limits),
        binding,
        id,
        records: state.accepted.len() as u64,
        producers: state.producers.len() as u64,
        tables: state.catalog.tables.len() as u64,
        pins: state.pins.len() as u64,
        closed: state.closed.len() as u64,
        retired: state.retired.len() as u64,
        metadata_bytes: 0,
        payload_bytes,
        metadata_digest: [0; 32],
        payload_digest: [0; 32],
    };
    let paths = descriptor.paths(directory);
    let mut metadata = Writer::new(&paths[0], metadata_limit(limits))?;
    metadata.write_all(magic(limits))?;
    write_frame(&mut metadata, &codec().serialize(&descriptor.binding)?)?;
    write_u64(&mut metadata, descriptor.records)?;
    write_u64(&mut metadata, descriptor.producers)?;
    write_u64(&mut metadata, descriptor.tables)?;
    write_u64(&mut metadata, descriptor.pins)?;
    write_u64(&mut metadata, descriptor.closed)?;
    write_u64(&mut metadata, descriptor.retired)?;
    for (index, meta) in &state.accepted {
        write_meta(&mut metadata, *index, meta)?;
    }
    for (meta, index) in state.producers.values() {
        write_meta(&mut metadata, *index, meta)?;
    }
    for table in state.catalog.tables.values() {
        write_frame(&mut metadata, &publication::encode_table(table)?)?;
    }
    for pin in state.pins.values() {
        write_frame(&mut metadata, &source::encode_pin(pin)?)?;
    }
    for closed in state.closed.values() {
        write_frame(&mut metadata, &source::encode_closed(closed)?)?;
    }
    for (key, retired) in &state.retired {
        write_frame(&mut metadata, &state::encode_retired(*key, *retired)?)?;
    }
    // A flat, policy-sized cell table; decoding never trusts an allocation count
    // supplied by the file. Zero-slot groups retain the exact legacy layout.
    if limits.producer_slots > 0 {
        write_u64(&mut metadata, u64::from(limits.producer_slots))?;
        for cell in &state.producer_slots {
            write_frame(&mut metadata, &producer::encode_cell(cell)?)?;
        }
    }
    (descriptor.metadata_bytes, descriptor.metadata_digest) = metadata.finish()?;
    let mut pack = Writer::new(&paths[1], payload_bytes)?;
    for (index, meta) in &state.accepted {
        let body = read(*index, meta)?;
        meta.validate_payload(&body, limits.max_record_bytes as usize)?;
        pack.write_all(&body)?;
    }
    let (bytes, digest) = pack.finish()?;
    ensure!(bytes == payload_bytes, "image pack length mismatch");
    descriptor.payload_digest = digest;
    descriptor.validate(layout, limits, group)?;
    File::open(directory)?.sync_all()?;
    let files = [File::open(&paths[0])?, File::open(&paths[1])?];
    Ok(Loaded {
        image: Arc::new(Image {
            descriptor,
            files,
            owner: std::sync::OnceLock::new(),
            locations,
            #[cfg(not(unix))]
            paths,
        }),
        state,
    })
}

pub(super) fn load(
    directory: &Path,
    snapshot: &Snapshot,
    layout: &Layout,
    limits: &Limits,
    group: u64,
) -> Result<Loaded> {
    let descriptor = Descriptor::decode(snapshot, layout, limits, group)?;
    let paths = descriptor.paths(directory);
    let mut metadata = Reader::new(&paths[0], descriptor.metadata_bytes)?;
    let mut actual_magic = [0; 8];
    metadata.read_exact(&mut actual_magic)?;
    ensure!(
        &actual_magic == magic(limits),
        "unsupported application image format"
    );
    let (binding, n) = read_frame::<512>(&mut metadata)?;
    ensure!(
        codec().deserialize::<Binding>(&binding[..n])? == descriptor.binding,
        "image metadata binding mismatch"
    );
    ensure!(
        read_u64(&mut metadata)? == descriptor.records
            && read_u64(&mut metadata)? == descriptor.producers
            && read_u64(&mut metadata)? == descriptor.tables
            && read_u64(&mut metadata)? == descriptor.pins
            && read_u64(&mut metadata)? == descriptor.closed
            && read_u64(&mut metadata)? == descriptor.retired,
        "image metadata count mismatch"
    );
    let mut state = State {
        applied: descriptor.binding.index,
        ..Default::default()
    };
    let mut previous = 0;
    for _ in 0..descriptor.records {
        let (index, meta) = read_meta(&mut metadata)?;
        ensure!(
            index > previous,
            "unordered or duplicate accepted image index"
        );
        state.accepted.insert(index, meta);
        previous = index;
    }
    let mut previous = None;
    for _ in 0..descriptor.producers {
        let (index, meta) = read_meta(&mut metadata)?;
        let key = (meta.stream, meta.bucket, meta.producer);
        ensure!(
            previous.is_none_or(|old| old < key),
            "unordered or duplicate image producer"
        );
        state.producers.insert(key, (meta, index));
        previous = Some(key);
    }
    let mut previous = None;
    for _ in 0..descriptor.tables {
        let bytes = read_dynamic_frame(&mut metadata, publication::MAX_TABLE_BYTES)?;
        let table = publication::decode_table(&bytes)?;
        let id = table.authority.table;
        ensure!(
            previous.is_none_or(|old| old < id),
            "unordered or duplicate image table"
        );
        state.catalog.tables.insert(id, table);
        previous = Some(id);
    }
    let mut previous = None;
    for _ in 0..descriptor.pins {
        let bytes = read_dynamic_frame(&mut metadata, source::MAX_PIN_BYTES)?;
        let pin = source::decode_pin(&bytes)?;
        let table = pin.permit.selection().table;
        ensure!(
            previous.is_none_or(|old| old < table),
            "unordered or duplicate source pin"
        );
        state.pins.insert(table, Arc::new(pin));
        previous = Some(table);
    }
    let mut previous = None;
    for _ in 0..descriptor.closed {
        let bytes = read_dynamic_frame(&mut metadata, source::MAX_CLOSED_BYTES)?;
        let closed = source::decode_closed(&bytes)?;
        let table = closed.permit.closure().table;
        ensure!(
            previous.is_none_or(|old| old < table),
            "unordered or duplicate source closure"
        );
        state.closed.insert(table, Arc::new(closed));
        previous = Some(table);
    }
    let mut previous = None;
    for _ in 0..descriptor.retired {
        let (bytes, n) = read_frame::<{ state::MAX_RETIRED_BYTES }>(&mut metadata)?;
        let (key, retired) = state::decode_retired(&bytes[..n])?;
        ensure!(
            previous.is_none_or(|old| old < key),
            "unordered or duplicate retired boundary"
        );
        state.retired.insert(key, retired);
        previous = Some(key);
    }
    if limits.producer_slots > 0 {
        ensure!(
            read_u64(&mut metadata)? == u64::from(limits.producer_slots),
            "image producer slot count mismatch"
        );
        for _ in 0..limits.producer_slots {
            let (bytes, n) = read_frame::<{ producer::MAX_CELL_BYTES }>(&mut metadata)?;
            state
                .producer_slots
                .push(producer::decode_cell(&bytes[..n])?);
        }
    }
    let metadata_file = metadata.finish(descriptor.metadata_bytes, descriptor.metadata_digest)?;
    state.validate_checkpoint(layout, group, limits, descriptor.binding.index)?;
    let (locations, payload_bytes) = locations(&state)?;
    ensure!(
        payload_bytes == descriptor.payload_bytes,
        "image record lengths disagree with pack"
    );
    let mut pack = Reader::new(&paths[1], payload_bytes)?;
    let mut scratch = [0; 64 * 1024];
    for location in locations.values() {
        let mut remaining = location.bytes;
        let mut hash = Sha256::new();
        while remaining > 0 {
            let n = remaining.min(scratch.len());
            pack.read_exact(&mut scratch[..n])?;
            hash.update(&scratch[..n]);
            remaining -= n;
        }
        ensure!(
            <[u8; 32]>::from(hash.finalize()) == location.digest,
            "image record checksum mismatch"
        );
    }
    let pack_file = pack.finish(descriptor.payload_bytes, descriptor.payload_digest)?;
    Ok(Loaded {
        image: Arc::new(Image {
            descriptor,
            files: [metadata_file, pack_file],
            owner: std::sync::OnceLock::new(),
            locations,
            #[cfg(not(unix))]
            paths,
        }),
        state,
    })
}

/// One sequential receiver on a bounded blocking worker. No received bytes are
/// visible to Raft before finish validates and syncs both complete files.
pub(super) struct Receiver {
    directory: PathBuf,
    descriptor: Descriptor,
    layout: Arc<Layout>,
    limits: Limits,
    writers: [Writer; 2],
}
impl Receiver {
    pub fn new(
        directory: &Path,
        snapshot: &Snapshot,
        layout: Arc<Layout>,
        limits: Limits,
        group: u64,
        local_id: Uuid,
    ) -> Result<Self> {
        let mut descriptor = Descriptor::decode(snapshot, &layout, &limits, group)?;
        ensure!(
            !local_id.is_nil() && local_id != descriptor.id,
            "receiver needs a fresh local image id"
        );
        descriptor.id = local_id;
        let paths = descriptor.paths(directory);
        let writers = [
            Writer::new(&paths[0], descriptor.metadata_bytes)?,
            Writer::new(&paths[1], descriptor.payload_bytes)?,
        ];
        Ok(Self {
            directory: directory.to_path_buf(),
            descriptor,
            layout,
            limits,
            writers,
        })
    }

    pub fn write(&mut self, part: Part, offset: u64, bytes: &[u8]) -> Result<()> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= CHUNK_BYTES,
            "invalid received image chunk size"
        );
        let writer = &mut self.writers[part.slot()];
        ensure!(offset == writer.bytes, "noncontiguous received image chunk");
        writer.write_all(bytes)?;
        Ok(())
    }

    pub fn finish(self) -> Result<Loaded> {
        let expected = [
            (
                self.descriptor.metadata_bytes,
                self.descriptor.metadata_digest,
            ),
            (
                self.descriptor.payload_bytes,
                self.descriptor.payload_digest,
            ),
        ];
        for (writer, expected) in self.writers.into_iter().zip(expected) {
            ensure!(
                writer.finish()? == expected,
                "incomplete or corrupt received image"
            );
        }
        File::open(&self.directory)?.sync_all()?;
        load(
            &self.directory,
            &self.descriptor.snapshot()?,
            &self.layout,
            &self.limits,
            self.descriptor.binding.group,
        )
    }
}

#[cfg(test)]
mod tests;
