use super::*;

pub(super) fn validate(snapshot: &Snapshot, conf: &ConfState) -> Result<()> {
    let metadata = snapshot.get_metadata();
    ensure!(
        metadata.index > 0
            && metadata.index < u64::MAX
            && metadata.term > 0
            && metadata.get_conf_state() == conf
            && !snapshot.data.is_empty()
            && snapshot.compute_size() as usize <= MAX_SNAPSHOT_BYTES,
        "invalid checkpoint cut, static membership or bounded descriptor"
    );
    Ok(())
}

impl GroupLog {
    /// Atomically persist a received Raft Ready snapshot, suffix and HardState.
    ///
    /// The owner must validate and fsync the complete application snapshot and
    /// every referenced file/directory BEFORE passing it to Raft. This method
    /// supplies only the storage transition after Raft accepts that snapshot;
    /// it cannot authenticate a sender or validate opaque application state.
    /// Do not release snapshot/append responses or apply the suffix before it
    /// returns successfully. Membership must remain the bound static membership.
    ///
    /// A received snapshot advances beyond the previous durable commit and
    /// replaces ALL prior entries, including a divergent uncommitted suffix.
    /// Entries from the same Ready start immediately after its cut. HardState,
    /// application binding, snapshot descriptor and suffix cross one shared
    /// sync barrier. An uncertain write fences the physical engine; retain old
    /// and new application dependencies until recovery resolves the outcome.
    /// Repeated/stale installs are rejected; ordinary later Ready updates use
    /// `persist`, and a local committed cut uses `checkpoint` instead.
    pub fn install_snapshot(
        &self,
        snapshot: &Snapshot,
        entries: &[Entry],
        hard: &HardState,
    ) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| anyhow::anyhow!("group state lock poisoned"))?;
        self.node.check()?;
        validate(snapshot, &state.conf)?;
        let application = self
            .node
            .engine
            .get(self.group, b"application")
            .context("bind the application before receiving a snapshot")?;
        ensure!(
            !application.is_empty() && application.len() <= 4096,
            "invalid snapshot application binding"
        );
        let metadata = snapshot.get_metadata();
        ensure!(
            metadata.index > state.hard.commit
                && metadata.term
                    >= state
                        .term(state.hard.commit)
                        .context("missing committed term")?,
            "received snapshot does not advance the committed history"
        );
        let base = GroupState {
            hard: state.hard.clone(),
            conf: state.conf.clone(),
            terms: Vec::new(),
            snapshot: snapshot.clone(),
            truncated: (metadata.index, metadata.term),
        };
        let next = base.appended(entries, Some(hard))?;
        let mut batch = LogBatch::default();
        // Clean also removes this group's key/value metadata. Restore every
        // owned key in this SAME batch; the node/layout identity is in group 0.
        batch.add_command(self.group, raft_engine::Command::Clean);
        batch.put(self.group, b"application".to_vec(), application)?;
        batch.put_message(self.group, CONF.to_vec(), &next.conf)?;
        batch.put_message(self.group, HARD.to_vec(), &next.hard)?;
        batch.put_message(self.group, SNAPSHOT.to_vec(), &next.snapshot)?;
        batch.put(
            self.group,
            TRUNCATED.to_vec(),
            encode_boundary(next.truncated, metadata.index),
        )?;
        batch.put(self.group, TERMS.to_vec(), encode_terms(&next.terms)?)?;
        if !entries.is_empty() {
            batch.add_entries::<EntryIndex>(self.group, entries)?;
        }
        self.node.write(&mut batch)?;
        *state = next;
        Ok(())
    }

    /// Durably replace a committed prefix with an application checkpoint.
    ///
    /// The application MUST first validate the descriptor's scope and complete
    /// state at this exact applied index, fsync every referenced immutable file
    /// and its directory, and retain dependencies of both the old and new
    /// descriptors on an uncertain write. This layer treats Snapshot.data as
    /// opaque; it cannot establish application recovery or Lance publication.
    /// Its entire encoded Snapshot is limited to 4 KiB, for a small state or a
    /// descriptor rather than bulk payload transfer.
    ///
    /// This synchronously installs the descriptor, truncation boundary and term
    /// index in the same engine batch as logical compaction. It preserves the
    /// committed/uncommitted suffix and HardState. The same cut may only be
    /// retried with the identical descriptor. Remote snapshot installation and
    /// physical segment deletion are separate operations, not supplied here.
    /// replication::Node installs its validated application images here.
    pub fn checkpoint(&self, snapshot: &Snapshot) -> Result<()> {
        self.checkpoint_retaining(snapshot, 0)
    }

    /// Install a checkpoint while retaining a bounded log tail for replication.
    /// The allowance counts encoded entry bytes plus 128 bytes of framing per
    /// entry, and at most 1,048,576 entries. Selection reads only engine index
    /// metadata, without decoding or copying the payload. Existing node disk
    /// admission still bounds total physical allocation and rewrite headroom.
    ///
    /// Truncation index/term, checkpoint and term ranges cross one sync barrier.
    /// Existing directories without truncation metadata remain readable. Older
    /// binaries cannot open a directory with a tail preceding its checkpoint;
    /// checkpoint with a zero allowance before downgrading.
    pub fn checkpoint_retaining(&self, snapshot: &Snapshot, retained_bytes: u64) -> Result<()> {
        ensure!(
            retained_bytes <= 2 << 30,
            "checkpoint log tail exceeds maximum allowance"
        );
        let mut state = self
            .state
            .write()
            .map_err(|_| anyhow::anyhow!("group state lock poisoned"))?;
        self.node.check()?;
        validate(snapshot, &state.conf)?;
        ensure!(
            self.node.engine.get(self.group, b"application").is_some(),
            "bind the application before checkpointing"
        );
        let metadata = snapshot.get_metadata();
        ensure!(
            metadata.index >= state.snapshot.get_metadata().index
                && metadata.index <= state.hard.commit
                && state.term(metadata.index) == Some(metadata.term),
            "checkpoint is not a retained durable committed cut"
        );
        if metadata.index == state.snapshot.get_metadata().index {
            ensure!(
                *snapshot == state.snapshot,
                "checkpoint changed at the same cut"
            );
        }
        let first = if retained_bytes == 0 || metadata.index == state.boundary() {
            metadata.index + 1
        } else {
            self.node
                .engine
                .retained_entries_start(self.group, metadata.index, retained_bytes, 1 << 20)
                .context("checkpoint tail is missing from engine index")?
                .max(state.boundary() + 1)
        };
        let truncated = (
            first - 1,
            state.term(first - 1).context("missing truncation term")?,
        );
        if metadata.index == state.snapshot.get_metadata().index && truncated == state.truncated {
            return self.node.check();
        }
        let mut terms = state.terms.clone();
        terms.retain(|range| range.last > truncated.0);
        if let Some(first) = terms.first_mut() {
            first.first = truncated.0 + 1;
        }
        let next = GroupState {
            hard: state.hard.clone(),
            conf: state.conf.clone(),
            terms,
            snapshot: snapshot.clone(),
            truncated,
        };
        next.validate()?;
        let mut batch = LogBatch::default();
        batch.put_message(self.group, SNAPSHOT.to_vec(), snapshot)?;
        batch.put(
            self.group,
            TRUNCATED.to_vec(),
            encode_boundary(truncated, metadata.index),
        )?;
        batch.put(self.group, TERMS.to_vec(), encode_terms(&next.terms)?)?;
        // Engine::compact_to writes without sync and suppresses errors. Keep
        // the command and its recovery metadata under our shared failure fence.
        batch.add_command(self.group, raft_engine::Command::Compact { index: first });
        self.node.write(&mut batch)?;
        *state = next;
        Ok(())
    }

    /// Application recovery cut, independent of the retained replication log.
    pub fn checkpoint_index(&self) -> Result<u64> {
        Ok(self.read()?.snapshot.get_metadata().index)
    }
}

fn encode_boundary((index, term): (u64, u64), checkpoint: u64) -> Vec<u8> {
    [
        index.to_le_bytes(),
        term.to_le_bytes(),
        checkpoint.to_le_bytes(),
    ]
    .concat()
}

#[cfg(test)]
mod tests;
