use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{stream, FutureExt, StreamExt, TryStreamExt};
use object_store::{PutMode, UpdateVersion};
use serde::{Deserialize, Serialize};

use crate::journal::digest;
use crate::{Binding, Entry, Error, Journal, Position, Result, Sink};

/// A per-session checkpoint may be ahead of the consumer's coherent prefix if a
/// previous multi-session batch failed halfway. Recovery must skip replay deltas
/// through this session's sequence, not apply them again. It must still replay
/// all other sessions through the committed WAL head before accepting requests.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionState {
    pub binding: Binding,
    pub session: String,
    pub through_sequence: u64,
    pub value: Vec<u8>,
}

/// One session reconstructed through an exact committed WAL position. The state
/// sequence is its last mutation, which can be older than `position.sequence`.
/// This does not include a publisher's uncommitted alignment suffix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveredSession {
    pub state: SessionState,
    pub position: Position,
}

/// Apply the already chosen alignment delta. This operation must be deterministic
/// and must not allocate new IDs or rerun history matching. Empty state denotes
/// a session with no checkpoint. Bound transient reducer memory in the adapter.
#[allow(
    clippy::double_must_use,
    reason = "async_trait adds must_use to boxed futures"
)]
#[async_trait]
pub trait Reducer: Send + Sync {
    async fn apply(&self, session: &str, state: &[u8], delta: &[u8]) -> Result<Vec<u8>>;

    /// Reduce one session's ordered, unapplied deltas. Adapters may decode state
    /// once and encode it once; the result must equal repeated `apply` calls.
    /// Preserve delta order and enforce `max_state_bytes` at every intermediate
    /// state, including in overrides. Errors publish no state for this session.
    async fn apply_batch(
        &self,
        session: &str,
        state: &[u8],
        deltas: &[&[u8]],
        max_state_bytes: usize,
    ) -> Result<Vec<u8>> {
        let mut result = None;
        for delta in deltas {
            let next = self
                .apply(session, result.as_deref().unwrap_or(state), delta)
                .await?;
            if next.len() > max_state_bytes {
                return Err(Error::Invalid(
                    "session checkpoint exceeds state budget".into(),
                ));
            }
            result = Some(next);
        }
        Ok(result.unwrap_or_else(|| state.to_vec()))
    }
}

#[derive(Clone)]
pub struct SessionCheckpoints {
    journal: Journal,
    max_state_bytes: usize,
}

impl SessionCheckpoints {
    pub fn new(journal: Journal, max_state_bytes: usize) -> Result<Self> {
        if max_state_bytes == 0 {
            return Err(Error::Invalid("zero session checkpoint budget".into()));
        }
        Ok(Self {
            journal,
            max_state_bytes,
        })
    }

    async fn read(&self, session: &str) -> Result<Option<(SessionState, UpdateVersion)>> {
        let path = self
            .journal
            .path(&format!("sessions/{}.json", digest(session.as_bytes())));
        match self.journal.read::<SessionState>(&path).await {
            Ok((state, version)) => {
                if state.binding != *self.journal.binding()
                    || state.session != session
                    || state.value.len() > self.max_state_bytes
                    || state.through_sequence == 0
                {
                    return Err(Error::Invalid("invalid session checkpoint".into()));
                }
                Ok(Some((state, version)))
            }
            Err(Error::Storage(object_store::Error::NotFound { .. })) => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub async fn load(&self, session: &str) -> Result<Option<SessionState>> {
        Ok(self.read(session).await?.map(|(state, _)| state))
    }

    /// Lazily recover one session after restart or cache eviction. `consumer`
    /// must name the consumer using this checkpoint store, never a table cursor.
    /// Read its coherent prefix before the session and the WAL head after it:
    /// a partially committed checkpoint batch can leave this session ahead of
    /// the prefix. Such deltas are skipped, not applied twice. No storage writes
    /// occur. Replay holds one WAL segment and one session state at a time;
    /// reducer transient allocations remain the adapter's responsibility.
    pub async fn recover<R: Reducer>(
        &self,
        consumer: &str,
        session: &str,
        reducer: &R,
    ) -> Result<RecoveredSession> {
        if session.is_empty() {
            return Err(Error::Invalid("empty recovery session".into()));
        }
        let mut after = self.journal.consumer_position(consumer).await?;
        let mut state = self.load(session).await?.unwrap_or_else(|| SessionState {
            binding: self.journal.binding().clone(),
            session: session.into(),
            through_sequence: 0,
            value: Vec::new(),
        });
        let position = self.journal.position().await?;
        if state.through_sequence > position.sequence {
            return Err(Error::Invalid(
                "session checkpoint is ahead of committed WAL".into(),
            ));
        }
        // pending also validates that the consumer's opaque position belongs to
        // this exact committed chain, including when there is nothing to replay.
        loop {
            let pending = self.journal.pending(&after, &position).await?;
            if pending.is_empty() {
                break;
            }
            for segment in pending {
                let entries = self.journal.entries(&segment).await?;
                let unapplied = entries
                    .iter()
                    .filter(|entry| {
                        entry.session == session && entry.sequence > state.through_sequence
                    })
                    .collect::<Vec<_>>();
                if let Some(last) = unapplied.last() {
                    let deltas = unapplied
                        .iter()
                        .map(|entry| entry.transition.delta.as_slice())
                        .collect::<Vec<_>>();
                    let value = reducer
                        .apply_batch(session, &state.value, &deltas, self.max_state_bytes)
                        .await?;
                    if value.len() > self.max_state_bytes {
                        return Err(Error::Invalid(
                            "recovered session exceeds state budget".into(),
                        ));
                    }
                    state.value = value;
                    state.through_sequence = last.sequence;
                }
                after = segment;
            }
        }
        Ok(RecoveredSession { state, position })
    }

    /// Each session is written once per consumer batch, even if it occurred in
    /// many producer WAL segments. Conditional writes never replace newer state.
    pub fn sink<R: Reducer>(&self, reducer: R, concurrency: usize) -> Result<CheckpointSink<R>> {
        if concurrency == 0 {
            return Err(Error::Invalid("zero checkpoint concurrency".into()));
        }
        Ok(CheckpointSink {
            store: self.clone(),
            reducer: Arc::new(reducer),
            concurrency,
        })
    }

    async fn apply<R: Reducer>(
        &self,
        reducer: &R,
        session: &str,
        entries: Vec<&Entry>,
    ) -> Result<()> {
        let (mut state, mode) = match self.read(session).await? {
            Some((state, version)) => (state, PutMode::Update(version)),
            None => (
                SessionState {
                    binding: self.journal.binding().clone(),
                    session: session.into(),
                    through_sequence: 0,
                    value: Vec::new(),
                },
                PutMode::Create,
            ),
        };
        let pending = entries
            .into_iter()
            .filter(|entry| entry.sequence > state.through_sequence)
            .collect::<Vec<_>>();
        if let Some(last) = pending.last() {
            let deltas = pending
                .iter()
                .map(|entry| entry.transition.delta.as_slice())
                .collect::<Vec<_>>();
            state.value = reducer
                .apply_batch(session, &state.value, &deltas, self.max_state_bytes)
                .await?;
            if state.value.len() > self.max_state_bytes {
                return Err(Error::Invalid(
                    "session checkpoint exceeds state budget".into(),
                ));
            }
            state.through_sequence = last.sequence;
            self.journal
                .put(
                    &self
                        .journal
                        .path(&format!("sessions/{}.json", digest(session.as_bytes()))),
                    &state,
                    mode,
                )
                .await?;
        }
        Ok(())
    }
}

pub struct CheckpointSink<R> {
    store: SessionCheckpoints,
    reducer: Arc<R>,
    concurrency: usize,
}

#[async_trait]
impl<R: Reducer> Sink for CheckpointSink<R> {
    async fn apply(&mut self, binding: &Binding, entries: &[Entry]) -> Result<()> {
        if binding != self.store.journal.binding()
            || entries
                .windows(2)
                .any(|pair| pair[0].sequence >= pair[1].sequence)
        {
            return Err(Error::Invalid(
                "checkpoint binding or sequence order mismatch".into(),
            ));
        }
        let mut sessions = BTreeMap::<&str, Vec<&Entry>>::new();
        for entry in entries {
            sessions.entry(&entry.session).or_default().push(entry);
        }
        let mut work = Vec::with_capacity(sessions.len());
        for (session, entries) in sessions {
            work.push(
                self.store
                    .apply(self.reducer.as_ref(), session, entries)
                    .boxed(),
            );
        }
        stream::iter(work)
            .buffer_unordered(self.concurrency)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }
}
