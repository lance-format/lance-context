use std::sync::Arc;
use std::time::Duration;

use object_store::{path::Path, ObjectStore, ObjectStoreExt, PutMode, PutOptions, UpdateVersion};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::consumer::validate_consumer_name;
use crate::{Error, Result};

/// Limit committed WAL segments not yet acknowledged by every named consumer.
/// This bounds outstanding payload bytes by `max_segments * max_segment_bytes`
/// for this journal; retained consumed history, orphan uploads and metadata are
/// not reclaimed or included. All scheduled publishers must use the same policy.
#[derive(Clone, Debug)]
pub struct BacklogPolicy {
    pub consumers: Vec<String>,
    pub max_segments: u64,
    pub poll_interval: Duration,
}

impl BacklogPolicy {
    fn validate(&self) -> Result<()> {
        if self.consumers.is_empty() || self.max_segments == 0 || self.poll_interval.is_zero() {
            return Err(Error::Invalid(
                "empty consumers or zero backlog limit".into(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        for name in &self.consumers {
            validate_consumer_name(name)?;
            if !names.insert(name) {
                return Err(Error::Invalid("duplicate backlog consumer".into()));
            }
        }
        Ok(())
    }
}

/// A namespace is permanently bound to one run, schema and virtual partition.
/// Worker count may change; these virtual partition identities must not.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub run: String,
    pub schema: String,
    pub partition: u32,
}

/// An opaque immutable segment reference. Sequence zero denotes the empty WAL.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub sequence: u64,
    pub generation: u64,
    pub segment: Option<String>,
}

/// Adapter-defined state delta and output records are committed together.
/// Deltas must replay without rerunning alignment or choosing new turn IDs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transition {
    pub delta: Vec<u8>,
    pub records: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub sequence: u64,
    pub session: String,
    pub receipt: String,
    pub input_digest: String,
    pub transition: Transition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Head {
    binding: Binding,
    epoch: u64,
    position: Position,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Segment {
    binding: Binding,
    predecessor: Position,
    entries: Vec<Entry>,
}

/// Small immutable skip links let lagging consumers seek a bounded page without
/// loading intervening record payloads or accumulating the whole WAL inventory.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Link {
    binding: Binding,
    position: Position,
    ancestors: Vec<Position>,
    payload_digest: String,
}

#[derive(Clone)]
pub struct Journal {
    store: Arc<dyn ObjectStore>,
    prefix: Path,
    binding: Binding,
    max_segment_bytes: usize,
    max_replay_segments: usize,
}

pub struct Writer {
    journal: Journal,
    head: Head,
    version: UpdateVersion,
    poisoned: bool,
    backlog: Option<BacklogPolicy>,
}

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl Journal {
    /// Storage must support atomic conditional puts. No process-local fallback is
    /// used. The replay limit bounds each page, not total recoverable backlog.
    /// This module deliberately performs no WAL garbage collection.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        prefix: Path,
        binding: Binding,
        max_segment_bytes: usize,
        max_replay_segments: usize,
    ) -> Result<Self> {
        if binding.run.is_empty()
            || binding.schema.is_empty()
            || max_segment_bytes == 0
            || max_replay_segments == 0
        {
            return Err(Error::Invalid("empty binding or zero journal limit".into()));
        }
        Ok(Self {
            store,
            prefix,
            binding,
            max_segment_bytes,
            max_replay_segments,
        })
    }

    pub fn binding(&self) -> &Binding {
        &self.binding
    }

    pub(crate) fn path(&self, child: &str) -> Path {
        child
            .split('/')
            .fold(self.prefix.clone(), |path, part| path.join(part))
    }

    pub(crate) async fn read<T: DeserializeOwned>(
        &self,
        path: &Path,
    ) -> Result<(T, UpdateVersion)> {
        let result = self.store.get(path).await?;
        let version = UpdateVersion {
            e_tag: result.meta.e_tag.clone(),
            version: result.meta.version.clone(),
        };
        if result.meta.size > self.max_segment_bytes as u64 {
            return Err(Error::Invalid("journal object exceeds read budget".into()));
        }
        let bytes = result.bytes().await?;
        Ok((serde_json::from_slice(&bytes)?, version))
    }

    pub(crate) async fn put<T: Serialize>(
        &self,
        path: &Path,
        value: &T,
        mode: PutMode,
    ) -> Result<UpdateVersion> {
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() > self.max_segment_bytes {
            return Err(Error::Invalid("journal object exceeds write budget".into()));
        }
        Ok(self
            .store
            .put_opts(path, bytes.into(), PutOptions::from(mode))
            .await?
            .into())
    }

    async fn head(&self) -> Result<(Head, UpdateVersion)> {
        let (head, version): (Head, _) = self.read(&self.path("head.json")).await?;
        if head.binding != self.binding {
            return Err(Error::Invalid("WAL run/schema/partition mismatch".into()));
        }
        validate_position(&head.position)?;
        Ok((head, version))
    }

    pub async fn position(&self) -> Result<Position> {
        Ok(self.head().await?.0.position)
    }

    /// Explicitly fence any previous publisher using a conditional head update.
    /// The caller must own scheduling for this partition. Old tasks can still
    /// upload orphan files, but cannot publish or ACK them after this CAS.
    /// Reopen after uncertain I/O; never resume speculative alignment state.
    pub async fn acquire(&self) -> Result<Writer> {
        let path = self.path("head.json");
        let (mut head, mode) = match self.head().await {
            Ok((head, version)) => (head, PutMode::Update(version)),
            Err(Error::Storage(object_store::Error::NotFound { .. })) => (
                Head {
                    binding: self.binding.clone(),
                    epoch: 0,
                    position: Position::default(),
                },
                PutMode::Create,
            ),
            Err(error) => return Err(error),
        };
        head.epoch = head
            .epoch
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("epoch overflow".into()))?;
        let version = self.put(&path, &head, mode).await?;
        Ok(Writer {
            journal: self.clone(),
            head,
            version,
            poisoned: false,
            backlog: None,
        })
    }

    async fn segment(&self, position: &Position) -> Result<Segment> {
        validate_position(position)?;
        let name = position
            .segment
            .as_ref()
            .ok_or_else(|| Error::Invalid("empty segment".into()))?;
        let (segment, _): (Segment, _) = self
            .read(&self.path(&format!("segments/{name}.json")))
            .await?;
        let link = self.link(position).await?;
        if digest(&serde_json::to_vec(&segment)?) != link.payload_digest
            || segment.predecessor != link.ancestors[0]
        {
            return Err(Error::Invalid("WAL payload integrity mismatch".into()));
        }
        if segment.binding != self.binding || segment.entries.is_empty() {
            return Err(Error::Invalid(
                "invalid WAL segment binding or empty entries".into(),
            ));
        }
        validate_position(&segment.predecessor)?;
        let mut expected = segment.predecessor.sequence;
        for entry in &segment.entries {
            expected = expected
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("sequence overflow".into()))?;
            if entry.sequence != expected {
                return Err(Error::Invalid("noncontiguous WAL sequence".into()));
            }
        }
        if expected != position.sequence {
            return Err(Error::Invalid("WAL segment/head mismatch".into()));
        }
        Ok(segment)
    }

    async fn link(&self, position: &Position) -> Result<Link> {
        validate_position(position)?;
        let name = position
            .segment
            .as_ref()
            .ok_or_else(|| Error::Invalid("empty link".into()))?;
        let (link, _): (Link, _) = self.read(&self.path(&format!("links/{name}.json"))).await?;
        let expected = u64::BITS - position.generation.leading_zeros();
        if link.binding != self.binding
            || link.position != *position
            || link.ancestors.len() != expected as usize
        {
            return Err(Error::Invalid("invalid WAL link".into()));
        }
        for (level, ancestor) in link.ancestors.iter().enumerate() {
            validate_position(ancestor)?;
            if ancestor.generation != position.generation - (1_u64 << level)
                || ancestor.sequence >= position.sequence
            {
                return Err(Error::Invalid("invalid WAL ancestor".into()));
            }
        }
        Ok(link)
    }

    /// Returns the next chronological page after `after`, bounded by
    /// max_replay_segments. Only the supplied committed chain is traversed;
    /// object listings and orphan uploads never establish durability.
    pub async fn pending(&self, after: &Position, through: &Position) -> Result<Vec<Position>> {
        validate_position(after)?;
        validate_position(through)?;
        if after.generation > through.generation {
            return Err(Error::Invalid("cursor ahead of durable WAL".into()));
        }
        let mut cursor = through.clone();
        let page_end = after
            .generation
            .saturating_add(self.max_replay_segments as u64)
            .min(through.generation);
        while cursor.generation > page_end {
            let link = self.link(&cursor).await?;
            cursor = link
                .ancestors
                .into_iter()
                .rev()
                .find(|ancestor| ancestor.generation >= page_end)
                .ok_or_else(|| Error::Invalid("missing WAL skip link".into()))?;
        }
        let mut segments = Vec::new();
        while cursor != *after {
            if cursor.sequence <= after.sequence {
                return Err(Error::Invalid(
                    "cursor is not on the committed WAL chain".into(),
                ));
            }
            let link = self.link(&cursor).await?;
            segments.push(cursor);
            cursor = link.ancestors[0].clone();
        }
        segments.reverse();
        Ok(segments)
    }

    async fn validate_ancestor(&self, ancestor: &Position, through: &Position) -> Result<()> {
        validate_position(ancestor)?;
        validate_position(through)?;
        let mut cursor = through.clone();
        while cursor.generation > ancestor.generation {
            cursor = self
                .link(&cursor)
                .await?
                .ancestors
                .into_iter()
                .rev()
                .find(|position| position.generation >= ancestor.generation)
                .ok_or_else(|| Error::Invalid("missing cursor skip link".into()))?;
        }
        if cursor != *ancestor {
            return Err(Error::Invalid(
                "consumer cursor is outside committed WAL".into(),
            ));
        }
        Ok(())
    }

    /// Read a segment previously obtained from `pending`; not an authorization
    /// to consume arbitrary uncommitted object paths.
    pub async fn entries(&self, position: &Position) -> Result<Vec<Entry>> {
        Ok(self.segment(position).await?.entries)
    }

    pub(crate) async fn receipt(&self, sequence: u64, through: &Position) -> Result<Entry> {
        let mut cursor = through.clone();
        while sequence > 0 && cursor.sequence >= sequence {
            let link = self.link(&cursor).await?;
            if sequence > link.ancestors[0].sequence {
                return self
                    .segment(&cursor)
                    .await?
                    .entries
                    .into_iter()
                    .find(|entry| entry.sequence == sequence)
                    .ok_or_else(|| Error::Invalid("receipt missing from WAL".into()));
            }
            cursor = link
                .ancestors
                .into_iter()
                .rev()
                .find(|ancestor| ancestor.sequence >= sequence)
                .ok_or_else(|| Error::Invalid("missing receipt skip link".into()))?;
        }
        Err(Error::Invalid("receipt is outside committed WAL".into()))
    }
}

fn validate_position(position: &Position) -> Result<()> {
    if (position.sequence == 0) != position.segment.is_none()
        || (position.generation == 0) != position.segment.is_none()
        || position.sequence < position.generation
        || position
            .segment
            .as_ref()
            .is_some_and(|id| Uuid::parse_str(id).is_err())
    {
        return Err(Error::Invalid("invalid WAL position".into()));
    }
    Ok(())
}

impl Writer {
    /// Apply backpressure before publishing another segment. Existing backlog
    /// above the limit is drained, never discarded. Required consumers must keep
    /// running while a pipeline shuts down; dropping a blocked append fences it.
    /// This policy is process configuration and must be reapplied after acquire.
    pub fn with_backlog(mut self, policy: BacklogPolicy) -> Result<Self> {
        policy.validate()?;
        self.backlog = Some(policy);
        Ok(self)
    }

    async fn wait_for_backlog(&self) -> Result<()> {
        let Some(policy) = &self.backlog else {
            return Ok(());
        };
        loop {
            let (head, _) = self.journal.head().await?;
            if head.epoch != self.head.epoch || head.position != self.head.position {
                return Err(Error::Fenced);
            }
            let mut full = false;
            for name in &policy.consumers {
                let cursor = self.journal.consumer_position(name).await?;
                self.journal
                    .validate_ancestor(&cursor, &head.position)
                    .await?;
                full |= head.position.generation - cursor.generation >= policy.max_segments;
            }
            if !full {
                return Ok(());
            }
            tokio::time::sleep(policy.poll_interval).await;
        }
    }

    pub fn position(&self) -> &Position {
        &self.head.position
    }

    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    /// Any storage failure poisons this writer, including ambiguous success.
    /// Recovery follows the head and verifies the original receipt before ACK.
    pub async fn append(&mut self, entries: Vec<Entry>) -> Result<Position> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        if entries.is_empty() {
            return Err(Error::Invalid("cannot append empty WAL batch".into()));
        }
        let mut sequence = self.head.position.sequence;
        for entry in &entries {
            sequence = sequence
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("sequence overflow".into()))?;
            if entry.sequence != sequence {
                return Err(Error::Invalid("append sequence gap or duplicate".into()));
            }
        }
        let position = Position {
            sequence,
            generation: self
                .head
                .position
                .generation
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("generation overflow".into()))?,
            segment: Some(Uuid::new_v4().to_string()),
        };
        let segment = Segment {
            binding: self.journal.binding.clone(),
            predecessor: self.head.position.clone(),
            entries,
        };
        // Set before the first await: cancelling this future also fences reuse.
        self.poisoned = true;
        self.wait_for_backlog().await?;
        let mut ancestors = vec![self.head.position.clone()];
        let mut level = 1;
        while let Some(ancestor) = ancestors.last().filter(|ancestor| ancestor.generation > 0) {
            let link = self.journal.link(ancestor).await?;
            match link.ancestors.get(level - 1) {
                Some(next) => ancestors.push(next.clone()),
                None => break,
            }
            level += 1;
        }
        let link = Link {
            binding: self.journal.binding.clone(),
            position: position.clone(),
            ancestors,
            payload_digest: digest(&serde_json::to_vec(&segment)?),
        };
        self.journal
            .put(
                &self.journal.path(&format!(
                    "segments/{}.json",
                    position.segment.as_ref().unwrap()
                )),
                &segment,
                PutMode::Create,
            )
            .await?;
        self.journal
            .put(
                &self.journal.path(&format!(
                    "links/{}.json",
                    position.segment.as_ref().unwrap()
                )),
                &link,
                PutMode::Create,
            )
            .await?;
        let mut next = self.head.clone();
        next.position = position.clone();
        let version = self
            .journal
            .put(
                &self.journal.path("head.json"),
                &next,
                PutMode::Update(self.version.clone()),
            )
            .await?;
        self.head = next;
        self.version = version;
        self.poisoned = false;
        Ok(position)
    }
}
