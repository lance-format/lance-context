use super::*;

/// Initial narrow state for an already admitted immutable turn. Existing IDs
/// keep their current annotation and grouping key, including staged updates.
/// This component neither persists nor verifies the corresponding turn body.
#[derive(Clone, Copy)]
pub struct Registration<'a> {
    pub turn: &'a str,
    pub rollout_id: &'a str,
    pub annotation: &'a TurnAnnotation,
}

#[derive(Clone)]
pub(super) struct PendingTurn {
    pub rollout_id: String,
    pub through: u64,
    pub row: u32,
}

/// First accepted registration for a turn, scoped to this staging shard.
/// The row ordinal is in the original registration batch, before duplicate
/// IDs are removed. Replay must preserve that input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NewTurnSource {
    pub record_index: u64,
    pub row: u32,
}

impl PendingTurn {
    pub(super) fn source(&self) -> NewTurnSource {
        NewTurnSource {
            record_index: self.through,
            row: self.row,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Batch<'a, 'b> {
    Patches(&'a [Mutation<'b>]),
    Registrations(&'a [Registration<'b>]),
}

#[derive(Clone, Copy)]
pub(super) enum Input<'a> {
    Patch {
        turn: &'a str,
        patch: &'a TurnAnnotationPatch,
    },
    Register(Registration<'a>),
}

impl<'a> Input<'a> {
    pub fn turn(self) -> &'a str {
        match self {
            Self::Patch { turn, .. } => turn,
            Self::Register(value) => value.turn,
        }
    }
}

impl<'a, 'b> Batch<'a, 'b> {
    pub fn len(self) -> usize {
        match self {
            Self::Patches(rows) => rows.len(),
            Self::Registrations(rows) => rows.len(),
        }
    }

    pub fn row(self, index: usize) -> Input<'b> {
        match self {
            Self::Patches(rows) => Input::Patch {
                turn: rows[index].turn,
                patch: rows[index].patch,
            },
            Self::Registrations(rows) => Input::Register(rows[index]),
        }
    }
}

impl AnnotationStaging {
    /// Apply one deduplicated, quorum-committed registration batch atomically.
    /// The caller must durably admit the complete turn and retain the original
    /// result/recovery evidence. This return value alone is not an ingestion ACK.
    /// Only separate narrow annotation tables support registration. Patch calls
    /// remain update-only and still reject missing turns.
    /// Keep the original source row order, including repeated turn IDs, so body
    /// materialization and reads can locate the first accepted body exactly.
    pub async fn register_committed(
        &self,
        after: u64,
        through: u64,
        rows: &[Registration<'_>],
    ) -> Result<ApplyOutcome> {
        self.apply_inner(after, through, Batch::Registrations(rows), true)
            .await
    }
}

pub(super) fn validate_target(base: &Dataset) -> Result<()> {
    if base.schema().fields.len() != 5
        || base
            .schema()
            .field("rollout_id")
            .is_none_or(|f| f.data_type() != DataType::Utf8 || f.nullable)
    {
        return Err(invalid(
            "turn registration requires a narrow annotation table with rollout_id",
        ));
    }
    Ok(())
}

pub(super) fn validate(
    value: Registration<'_>,
    limit: usize,
) -> std::result::Result<(), RejectionReason> {
    if value.turn.is_empty()
        || value.turn.len() > limit
        || value.rollout_id.is_empty()
        || value.rollout_id.len() > limit
    {
        return Err(RejectionReason::InvalidTurn);
    }
    if value.annotation.labels.iter().any(String::is_empty)
        || value.annotation.score.is_some_and(|s| !s.is_finite())
    {
        return Err(RejectionReason::InvalidPatch);
    }
    // Validate semantic lengths before cloning; caller allocation capacity is
    // not part of a replayable request's permanent result.
    let bytes = 512usize
        .checked_add(value.turn.len())
        .and_then(|n| n.checked_add(value.annotation.status.as_ref().map_or(0, String::len)))
        .and_then(|size| {
            value
                .annotation
                .labels
                .iter()
                .try_fold(size, |size, label| {
                    size.checked_add(512)?.checked_add(label.len())
                })
        })
        .ok_or(RejectionReason::ValueTooLarge)?;
    if bytes > limit {
        return Err(RejectionReason::ValueTooLarge);
    }
    Ok(())
}

pub(super) fn retained_charge(value_bytes: usize, pending: Option<&PendingTurn>) -> Result<usize> {
    match pending {
        None => Ok(value_bytes),
        Some(pending) => value_bytes
            .checked_add(pending.rollout_id.capacity())
            .and_then(|n| n.checked_add(std::mem::size_of::<PendingTurn>()))
            .ok_or_else(|| invalid("registration charge overflow")),
    }
}
