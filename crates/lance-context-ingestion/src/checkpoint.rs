use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{stream, FutureExt, StreamExt, TryStreamExt};
use object_store::{PutMode, UpdateVersion};
use serde::{Deserialize, Serialize};

use crate::journal::digest;
use crate::{Binding, Entry, Error, Journal, Result, Sink};

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

/// Apply the already chosen alignment delta. This operation must be deterministic
/// and must not allocate new IDs or rerun history matching. Empty state denotes
/// a session with no checkpoint. Bound transient reducer memory in the adapter.
#[async_trait]
pub trait Reducer: Send + Sync {
    async fn apply(&self, session: &str, state: &[u8], delta: &[u8]) -> Result<Vec<u8>>;
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
        let before = state.through_sequence;
        for entry in entries {
            if entry.sequence <= state.through_sequence {
                continue;
            }
            state.value = reducer
                .apply(session, &state.value, &entry.transition.delta)
                .await?;
            if state.value.len() > self.max_state_bytes {
                return Err(Error::Invalid(
                    "session checkpoint exceeds state budget".into(),
                ));
            }
            state.through_sequence = entry.sequence;
        }
        if state.through_sequence != before {
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
