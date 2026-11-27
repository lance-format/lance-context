//! Explicit one-shot diagnostic for the actual post-staging historical key set.
//! No route, logging, native scan, publication or durable state is added here.
use super::*;
use crate::turn_ingestion::TurnIngestionSpec;
use lance::Dataset;
use lance_context_wal::replication::publication::Authority;
use std::{collections::BTreeMap, mem::size_of};
use tokio::sync::oneshot;

/// Input record interval for one lookup attempt. Positions belong to the named
/// source group/bucket; the interval may contain gaps and is not an applied cut.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Window {
    pub first_record: u64,
    pub last_record: u64,
    pub requests: usize,
    pub payload_bytes: u64,
    pub speculative: bool,
}

/// A complete input capture, before the native lookup starts. The lookup may
/// later fail, be cancelled or prove stale; this is neither a result nor an ACK.
/// No Dataset/reader lease is retained, so this does not protect files from GC.
/// Serialize on the diagnostic caller's task (prefer `serde_json::to_writer`).
#[derive(Serialize)]
pub struct Capture {
    #[serde(flatten)]
    data: Data,
    #[serde(skip)]
    _reservation: MergeReservation,
}

impl Capture {
    pub fn data(&self) -> &Data {
        &self.data
    }
}

/// Key-only capture contents. Access by reference so copied buffers cannot be
/// moved out of their admission owner.
#[derive(Serialize)]
pub struct Data {
    pub capture_id: Uuid,
    /// Actual publisher acquisition identity, not proof it remains current.
    pub owner: Authority,
    pub incarnation: Uuid,
    pub schema: [u8; 32],
    pub source_group: u64,
    pub bucket: u32,
    pub base_version: u64,
    /// All fragments in the captured base, not the fragments actually scanned.
    pub base_fragment_ids: Box<[u64]>,
    pub window: Window,
    pub requested_keys: usize,
    pub staged_keys: usize,
    pub primary_key_columns: Box<[Box<str>]>,
    /// Exact canonical JSON string arrays, in primary_key_columns order.
    /// Contains every distinct unresolved key; never truncates or samples.
    pub keys: Box<[Box<str>]>,
}

struct Request {
    id: Uuid,
    owner: Authority,
    budget: Arc<MergeMemoryBudget>,
    sender: oneshot::Sender<Result<Capture>>,
}

#[derive(Default)]
pub(super) struct State(Mutex<Option<Request>>);

/// Retain this handle before moving Runtime into its run loop. It refers only
/// to that local owner's diagnostic slot and does not keep its applier alive.
#[derive(Clone)]
pub struct Handle {
    state: Arc<State>,
    owner: Authority,
}

impl Handle {
    pub(super) fn new(state: Arc<State>, owner: Authority) -> Self {
        Self { state, owner }
    }

    /// Capture the next window needing a historical lookup. All-staged windows
    /// leave the request armed. Only one request may be armed per owner.
    /// Use a separate diagnostic budget; refusal only fails the capture, never
    /// the lookup or apply. The returned payload retains admission until drop.
    /// This accounts copied key/metadata buffers, not serialization or RSS.
    /// Apply a caller timeout: a quiet/stopped runtime need not produce a window.
    /// Dropping the receiver cancels; a later arm replaces a cancelled request.
    pub fn arm(
        &self,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<oneshot::Receiver<Result<Capture>>> {
        let mut slot = self.state.0.lock().unwrap();
        ensure!(
            slot.as_ref()
                .is_none_or(|request| request.sender.is_closed()),
            "body lookup capture already armed"
        );
        let (sender, receiver) = oneshot::channel();
        *slot = Some(Request {
            id: Uuid::new_v4(),
            owner: self.owner.clone(),
            budget,
            sender,
        });
        Ok(receiver)
    }
}

impl State {
    pub(super) fn record(
        &self,
        service: &Service,
        spec: &TurnIngestionSpec,
        base: &Dataset,
        keys: &BTreeMap<String, bool>,
        window: Window,
    ) {
        let request = {
            let mut slot = self.0.lock().unwrap();
            let Some(request) = slot.as_ref() else { return };
            if request.sender.is_closed() {
                *slot = None;
                return;
            }
            if keys.values().all(|present| *present) {
                return;
            }
            slot.take().unwrap()
        };
        // No await, external callback, log or disk I/O on the ingestion path.
        // A dropped receiver simply drops the charged diagnostic payload.
        let result = capture(&request, service, spec, base, keys, window);
        let _ = request.sender.send(result);
    }
}

fn capture(
    request: &Request,
    service: &Service,
    spec: &TurnIngestionSpec,
    base: &Dataset,
    keys: &BTreeMap<String, bool>,
    window: Window,
) -> Result<Capture> {
    let unresolved = keys.values().filter(|present| !**present).count();
    let fragments = base.fragments();
    let bytes = size_of::<Capture>()
        .checked_add(
            fragments
                .len()
                .checked_mul(size_of::<u64>())
                .context("capture size overflow")?,
        )
        .context("capture size overflow")?;
    let bytes = keys
        .iter()
        .filter(|(_, present)| !**present)
        .map(|(key, _)| key.as_str())
        .chain(spec.primary_key().iter().map(String::as_str))
        .try_fold(bytes, |bytes, value| {
            bytes
                .checked_add(size_of::<Box<str>>())?
                .checked_add(value.len())
        })
        .context("capture size overflow")?;
    // MergeMemoryBudget deliberately admits oversized merge operations alone.
    // Diagnostic snapshots must instead fail before allocating/truncating.
    ensure!(
        bytes <= request.budget.limit(),
        "body lookup capture exceeds diagnostic budget"
    );
    let reservation = request
        .budget
        .try_reserve(bytes)
        .context("body lookup capture budget unavailable")?;
    let mut copied = Vec::with_capacity(unresolved);
    for (key, present) in keys {
        if !*present {
            copied.push(Box::<str>::from(key.as_str()));
        }
    }
    let data = Data {
        capture_id: request.id,
        owner: request.owner.clone(),
        incarnation: service.ingress.scope.incarnation,
        schema: service.ingress.scope.contract.schema,
        source_group: service.group()?,
        bucket: service.bucket(),
        base_version: base.version_id(),
        base_fragment_ids: fragments.iter().map(|fragment| fragment.id).collect(),
        window,
        requested_keys: keys.len(),
        staged_keys: keys.len() - unresolved,
        primary_key_columns: spec
            .primary_key()
            .iter()
            .map(|key| Box::<str>::from(key.as_str()))
            .collect(),
        keys: copied.into_boxed_slice(),
    };
    Ok(Capture {
        data,
        _reservation: reservation,
    })
}
