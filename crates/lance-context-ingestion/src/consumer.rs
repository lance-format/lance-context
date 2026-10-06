use async_trait::async_trait;
use object_store::{PutMode, UpdateVersion};
use serde::{Deserialize, Serialize};

use crate::{Binding, Entry, Error, Journal, Position, Result};

/// Apply ordered WAL entries idempotently using (binding, entry.sequence).
/// On restart or uncertain progress writes, a batch can repeat or be regrouped.
/// Return success only after outputs AND their coverage are durable together.
/// For Lance, staged fragments alone are insufficient: manifest publication must
/// atomically record the covered input range. This callback never owns WAL ACKs.
#[async_trait]
pub trait Sink: Send {
    async fn apply(&mut self, binding: &Binding, entries: &[Entry]) -> Result<()>;
}

#[derive(Serialize, Deserialize)]
struct Cursor {
    binding: Binding,
    position: Position,
}

/// Separate names (for example `checkpoint` and `table`) consume the same WAL
/// independently. Scheduler must assign a single active worker per consumer and
/// partition; cursor CAS detects ownership races but does not fence sink writes.
pub struct Consumer {
    journal: Journal,
    name: String,
    position: Position,
    version: UpdateVersion,
    poisoned: bool,
}

impl Consumer {
    pub async fn open(journal: Journal, name: &str) -> Result<Self> {
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(Error::Invalid("invalid consumer name".into()));
        }
        let path = journal.path(&format!("consumers/{name}.json"));
        let (cursor, version): (Cursor, _) = match journal.read(&path).await {
            Ok(pair) => pair,
            Err(Error::Storage(object_store::Error::NotFound { .. })) => {
                let cursor = Cursor {
                    binding: journal.binding().clone(),
                    position: Position::default(),
                };
                let version = journal.put(&path, &cursor, PutMode::Create).await?;
                (cursor, version)
            }
            Err(error) => return Err(error),
        };
        if &cursor.binding != journal.binding() {
            return Err(Error::Invalid(
                "consumer run/schema/partition mismatch".into(),
            ));
        }
        Ok(Self {
            journal,
            name: name.into(),
            position: cursor.position,
            version,
            poisoned: false,
        })
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Coalesce up to `max_segments` / `max_bytes` independently of producer batch
    /// sizes. No empty progress writes. A segment larger than the budget fails;
    /// it is never silently admitted. Cancellation requires reopening the cursor.
    pub async fn consume<S: Sink>(
        &mut self,
        sink: &mut S,
        max_segments: usize,
        max_bytes: usize,
    ) -> Result<usize> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        if max_segments == 0 || max_bytes == 0 {
            return Err(Error::Invalid("zero consumer limit".into()));
        }
        let through = self.journal.position().await?;
        let positions = self.journal.pending(&self.position, &through).await?;
        let mut entries = Vec::new();
        let mut bytes = 0;
        let mut last = self.position.clone();
        for position in positions.into_iter().take(max_segments) {
            let batch = self.journal.entries(&position).await?;
            let size = serde_json::to_vec(&batch)?.len();
            if size > max_bytes {
                return Err(Error::Invalid("WAL segment exceeds consumer budget".into()));
            }
            if size > max_bytes - bytes {
                break;
            }
            bytes += size;
            entries.extend(batch);
            last = position;
        }
        if entries.is_empty() {
            return Ok(0);
        }
        self.poisoned = true;
        sink.apply(self.journal.binding(), &entries).await?;
        let next = Cursor {
            binding: self.journal.binding().clone(),
            position: last.clone(),
        };
        let version = self
            .journal
            .put(
                &self.journal.path(&format!("consumers/{}.json", self.name)),
                &next,
                PutMode::Update(self.version.clone()),
            )
            .await?;
        self.position = last;
        self.version = version;
        self.poisoned = false;
        Ok(entries.len())
    }
}
