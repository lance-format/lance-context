//! Bounded background consumption of a locally applied, committed source prefix.
//!
//! This is deliberately separate from a fresh quorum observation. An isolated
//! leader can return an older prefix; an empty page never proves that the source
//! is caught up. Point reads, acknowledgments, acquisition, publication and
//! reclamation must continue to establish their existing quorum requirements.
//! Deploy support on WAL servers before enabling this RPC in their consumers.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub table: Uuid,
    pub bucket: u32,
    pub after: u64,
    pub limit: usize,
}

impl Request {
    pub(super) fn validate(&self, layout: &Layout, group: u64) -> Result<()> {
        ensure!(
            !self.table.is_nil()
                && layout.group_for_bucket(self.bucket)? == group
                && self.after < u64::MAX
                && (1..=128).contains(&self.limit),
            "invalid committed source request"
        );
        Ok(())
    }
}

/// No term, quorum index or authorization is implied by this local observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub cluster: Uuid,
    pub layout: [u8; 32],
    pub group: u64,
    pub applied: u64,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Value {
    /// Both fields are captured from the same applied application state.
    Page {
        page: Page,
        closure: Option<Arc<source::Closed>>,
    },
    /// The caller's cursor is ahead of this replica, not an empty source.
    CursorAhead,
}

impl Node {
    /// Read metadata only from state applied after durable Raft persistence.
    /// Leader routing limits unnecessary lag; it does not prove that the local
    /// leader still has a quorum. This call creates no Raft entry or ReadIndex.
    pub fn read_source(&self, group: u64, request: &Request) -> Result<Observation> {
        self.healthy()?;
        request.validate(&self.layout, group)?;
        let replica = self.groups.get(&group).context("unhosted source group")?;
        ensure!(
            replica.raw.raft.state == StateRole::Leader,
            ProposalRejection::NotLeader
        );
        let state = &replica.state;
        let value = if request.after > state.applied {
            Value::CursorAhead
        } else {
            Value::Page {
                page: state.bucket_page(
                    request.table,
                    request.bucket,
                    request.after,
                    request.limit,
                )?,
                closure: state.closed.get(&request.table).cloned(),
            }
        };
        Ok(Observation {
            cluster: self.layout.cluster,
            layout: self.layout_hash,
            group,
            applied: state.applied,
            value,
        })
    }
}

#[cfg(test)]
mod tests;
