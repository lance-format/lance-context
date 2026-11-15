use super::*;
use raft::{GetEntriesContext, RaftState};
use raft_proto::eraftpb::Snapshot;
use std::ops::Deref;

/// RawNode may read pending entries; application and status use durable_commit.
pub(super) enum LogView {
    Durable(GroupLog),
    #[cfg(feature = "runtime")]
    Buffered(Box<crate::raft_log::buffered::BufferedLog>),
}

impl Deref for LogView {
    type Target = GroupLog;
    fn deref(&self) -> &GroupLog {
        match self {
            Self::Durable(log) => log,
            #[cfg(feature = "runtime")]
            Self::Buffered(log) => log.durable_log(),
        }
    }
}

impl LogView {
    pub(super) fn durable_log(&self) -> &GroupLog {
        self
    }

    pub(super) fn durable_commit(&self) -> Result<u64> {
        match self {
            Self::Durable(log) => Ok(log.initial_state()?.hard_state.commit),
            #[cfg(feature = "runtime")]
            Self::Buffered(log) => Ok(log.durable_hard_state()?.commit),
        }
    }

    pub(super) fn checkpoint_index(&self) -> Result<u64> {
        match self {
            Self::Durable(log) => log.checkpoint_index(),
            #[cfg(feature = "runtime")]
            Self::Buffered(log) => log.checkpoint_index(),
        }
    }

    #[cfg(feature = "runtime")]
    pub(super) fn buffered(
        &mut self,
        window: crate::raft_log::buffered::Window,
    ) -> Result<&mut crate::raft_log::buffered::BufferedLog> {
        if let Self::Durable(log) = self {
            *self = Self::Buffered(Box::new(crate::raft_log::buffered::BufferedLog::new(
                log.clone(),
                window,
            )?));
        }
        let Self::Buffered(log) = self else {
            unreachable!()
        };
        Ok(log)
    }

    #[cfg(feature = "runtime")]
    pub(super) fn synchronous(&mut self) -> Result<()> {
        if let Self::Buffered(log) = self {
            ensure!(
                log.usage().readies == 0,
                "cannot switch a pending log to synchronous storage"
            );
            *self = Self::Durable(log.durable_log().clone());
        }
        Ok(())
    }
}

macro_rules! delegate {
    ($this:ident, $method:ident ($($arg:expr),*)) => {
        match $this {
            Self::Durable(log) => log.$method($($arg),*),
            #[cfg(feature = "runtime")]
            Self::Buffered(log) => log.$method($($arg),*),
        }
    };
}
impl Storage for LogView {
    fn initial_state(&self) -> raft::Result<RaftState> {
        delegate!(self, initial_state())
    }
    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        context: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        delegate!(self, entries(low, high, max_size.into(), context))
    }
    fn term(&self, index: u64) -> raft::Result<u64> {
        delegate!(self, term(index))
    }
    fn first_index(&self) -> raft::Result<u64> {
        delegate!(self, first_index())
    }
    fn last_index(&self) -> raft::Result<u64> {
        delegate!(self, last_index())
    }
    fn snapshot(&self, index: u64, to: u64) -> raft::Result<Snapshot> {
        delegate!(self, snapshot(index, to))
    }
}
