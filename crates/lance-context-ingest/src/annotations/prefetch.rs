//! Read a small admitted prefix without advancing any application position.
use super::*;
use lance_context_wal::replication::runtime::client::payload_memory;

const MAX_REQUESTS: usize = 8;
// Match one core cold-lookup window. Bodies stay in their original requests;
// only IDs participate in the shared query. Bound cumulative wire bytes as
// well as requests/keys; decode reservations still charge the apply budget.
const MAX_ROWS: usize = 2048;
const MAX_PAYLOAD_BYTES: u64 = 32 << 20;
// Full-body windows: a history lookup is one scan of the whole base, so its
// cost is per scan, not per key. The window is bounded by the key workspace
// admitted against the reserved lookup budget (`PrefetchStop::KeyBound`), not
// by these sanity ceilings. Only the first `MAX_REQUESTS`/`MAX_PAYLOAD_BYTES`
// of decoded bodies stay resident; later records keep only their keys (and
// an apply reservation for those retained rows) and are re-read from the
// durable WAL when they become the ordered head.
//
// Capacity: `single_body::budget_parts` reserves about apply/4 for lookup and
// each window takes at most half of that (foreground plus one successor), so
// a window holds about apply/8 of key workspace. E.g. at ~1800 workspace
// bytes/key a 360 MiB lookup budget allows ~26k keys/window; 50k keys need an
// apply ceiling of at least ~687 MiB, before field-name charges and rounding.
// Per-owner bucket division of the apply ceiling divides this further.
const MAX_BODY_WINDOW_REQUESTS: usize = 1 << 16;
const MAX_BODY_WINDOW_PAYLOAD_BYTES: u64 = 1 << 40;

#[derive(Default)]
pub(super) struct Lookup {
    pub(super) registrations: Option<RegistrationLookup>,
    pub(super) patches: Option<PatchLookup>,
    pub(super) bodies: Option<single_body::Lookup>,
    pub(super) body_stop: Option<PrefetchStop>,
    body_local_end: bool,
}

/// Read committed metadata only. The caller retains the returned page; neither
/// this RPC nor its `through` cursor advances application progress.
pub(super) async fn source_page(
    service: &Service,
    after: u64,
    replay_cut: Option<u64>,
) -> Result<Option<lance_context_wal::replication::Page>> {
    Ok(read_page(service, after, replay_cut)
        .await?
        .map(|(page, _)| page))
}

async fn read_page(
    service: &Service,
    after: u64,
    replay_cut: Option<u64>,
) -> Result<Option<(lance_context_wal::replication::Page, bool)>> {
    let ingress = &service.ingress;
    let group = service.group()?;
    let response = service
        .shared
        .phases
        .measure(
            Phase::AnnotationObserve,
            ingress.client.request_routed(Request::ReadSource {
                group,
                request: source_read::Request {
                    table: ingress.scope.table,
                    bucket: service.bucket(),
                    after,
                    limit: 32,
                },
            }),
        )
        .await?;
    let Reply::Source(observed) = response.reply else {
        bail!("annotation source page unavailable");
    };
    ensure!(
        observed.cluster == ingress.scope.cluster
            && observed.layout == ingress.scope.layout
            && observed.group == group,
        "foreign committed annotation source"
    );
    let (mut page, closure) = match observed.value {
        source_read::Value::Page { page, closure } => (page, closure),
        source_read::Value::CursorAhead => {
            ensure!(observed.applied < after, "invalid lagging source cursor");
            if service.is_single_body() {
                let mut observed = service.shared.body_lookups.lock().unwrap();
                let count = &mut observed.prefetch.source_cursor_ahead;
                *count = count.saturating_add(1);
            }
            // A replica behind our cursor cannot prove catch-up.
            return Ok(None);
        }
    };
    // Page and closure share one durable applied state, which may be old.
    // Validate before fetching any payload or changing the retained cursor.
    service.check_source_closure(closure.as_deref())?;
    if let Some(cut) = replay_cut {
        page.through = page.through.min(cut);
        page.entries.retain(|(index, _)| *index <= cut);
    }
    ensure!(
        page.through >= after && page.through <= observed.applied,
        "invalid annotation source cursor"
    );
    ensure!(page.entries.len() <= 32, "oversized annotation source page");
    let mut previous = after;
    for (index, _) in &page.entries {
        ensure!(
            *index > previous && *index <= page.through,
            "unordered annotation source page"
        );
        previous = *index;
    }
    if service.is_single_body() && page.entries.is_empty() {
        // A bounded metadata page can be empty before the observed end because
        // it filtered other tables/buckets. Count validated responses, including
        // initial driver reads; these are not the prefetch-decision denominator.
        let mut lookup = service.shared.body_lookups.lock().unwrap();
        let count = if replay_cut.is_some_and(|cut| page.through >= cut) {
            &mut lookup.prefetch.source_empty_at_replay_cut
        } else if page.through < observed.applied {
            &mut lookup.prefetch.source_empty_with_remaining_metadata
        } else {
            &mut lookup.prefetch.source_empty_at_local_end
        };
        *count = count.saturating_add(1);
    }
    let local_end = page.entries.is_empty()
        && page.through == observed.applied
        && replay_cut.is_none_or(|cut| page.through < cut);
    Ok(Some((page, local_end)))
}

/// The WAL no longer holds this committed record: publication retired it.
#[derive(Debug)]
pub(super) struct PayloadRetired {
    pub index: u64,
}
impl std::fmt::Display for PayloadRetired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "committed annotation payload {} retired from the WAL",
            self.index
        )
    }
}
impl std::error::Error for PayloadRetired {}

pub(super) async fn fetch(
    service: &Service,
    index: u64,
    meta: BatchMeta,
) -> Result<Option<Pending>> {
    let ingress = &service.ingress;
    ensure!(
        meta.stream == ingress.scope.table
            && meta.bucket == service.bucket()
            && meta.kind == RecordKind::Data,
        "foreign annotation source"
    );
    let producer_slot = slot(&meta, ingress.client.producer_slots())?;
    let bytes = usize::try_from(meta.body_bytes)?;
    ensure!(
        bytes <= ingress.scope.contract.max_record_bytes as usize,
        "oversized annotation WAL record"
    );
    let Some(mut reservation) = service.apply_budget.try_reserve(decode_workspace(
        &ingress.scope.contract,
        &service.limits,
        bytes,
    )?) else {
        return Ok(None);
    };
    let (body, capacity) = match service.shared.local_payloads.body(producer_slot, &meta) {
        Some(body) => body,
        None => {
            let group = service.group()?;
            let response = service
                .shared
                .phases
                .measure(
                    Phase::AnnotationFetch,
                    ingress
                        .client
                        .request_routed(Request::Fetch { group, index }),
                )
                .await;
            let response = match response {
                Ok(response) => response,
                Err(error) if error.is::<payload_memory::Unavailable>() => return Ok(None),
                Err(error) => return Err(error),
            };
            if matches!(response.reply, Reply::Record(None)) {
                return Err(anyhow::Error::new(PayloadRetired { index }));
            }
            ensure!(
                matches!(response.reply, Reply::Record(Some(ref observed)) if observed == &meta),
                "committed annotation payload unavailable"
            );
            // RPC reception owns this complete payload allocation, unlike a
            // caller's Bytes slice. A configured pool follows its shared owner.
            let capacity = response.body_capacity();
            (response.body, Some(capacity))
        }
    };
    meta.check_body_length(body.len(), ingress.scope.contract.max_record_bytes as usize)?;
    if let Some(capacity) = capacity {
        let extra = capacity
            .checked_sub(bytes)
            .context("invalid local allocation size")?;
        let workspace = reservation
            .bytes()
            .checked_add(extra)
            .context("apply workspace overflow")?;
        if (extra > 0 && workspace > service.apply_budget.limit())
            || !reservation.try_grow_to(workspace)
        {
            return Ok(None);
        }
    }
    let input_allocation = capacity.map(|capacity| (body.as_ptr() as usize, capacity));
    let schema = ingress.schema.clone();
    let max_rows = ingress.scope.contract.max_rows;
    let value_bytes = service.limits.value_bytes;
    let mixed = ingress.scope.contract.policy == WritePolicy::TurnIngestion;
    let shard = service.shard;
    let layout = ingress.layout.clone();
    let scope = ingress.scope.clone();
    let spec = ingress.turn_spec.clone();
    Ok(Some(
        service
            .shared
            .phases
            .measure(Phase::AnnotationDecode, async {
                tokio::task::spawn_blocking(move || -> Result<Pending> {
                    let decoded = if single_body::is_schema(&schema) {
                        single_body::decode(body, schema, max_rows, value_bytes)?
                    } else if mixed {
                        mixed::wire::decode(body, schema, max_rows, value_bytes)?
                    } else {
                        mixed::Decoded::Patches(wire::decode(body, schema, max_rows, value_bytes)?)
                    };
                    decoded.validate_shard(shard, &layout, &scope.contract, spec.as_deref())?;
                    Ok(Pending {
                        index,
                        meta,
                        decoded,
                        input_allocation,
                        _reservation: reservation,
                        keys_only: false,
                    })
                })
                .await?
            })
            .await?,
    ))
}

/// Exact key count and conservative lookup-workspace charge of one record.
/// Summing per record over-counts the per-query field-name term; that only
/// makes the window bound stricter than `prepare_lookup` admission.
fn key_charge(service: &Service, pending: &Pending) -> Result<(usize, usize)> {
    let mixed::Decoded::Creates { rows, .. } = &pending.decoded else {
        bail!("immutable body cannot contain patches")
    };
    let spec = service
        .ingress
        .turn_spec
        .as_ref()
        .context("missing body spec")?;
    Ok((
        rows.len(),
        spec.lookup_workspace(rows.iter().map(|row| row.turn.len()))?,
    ))
}

/// Key workspace one full-body window may use: half the reserved lookup
/// budget, so the foreground window and one speculative successor both fit.
fn key_window(service: &Service) -> usize {
    let limit = service
        .lookup_budget
        .as_ref()
        .map_or(0, |budget| budget.limit() / 2);
    limit.min(test_key_window(service))
}

#[cfg(test)]
fn test_key_window(service: &Service) -> usize {
    match service
        .shared
        .key_window_for_test
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        0 => usize::MAX,
        cap => cap,
    }
}

#[cfg(not(test))]
fn test_key_window(_: &Service) -> usize {
    usize::MAX
}

/// The record alone needs more than one window's key workspace. It forms a
/// head-only window that may use the whole lookup budget, so no speculative
/// window may be started beside it, nor may it head a speculative window.
pub(super) fn exceeds_key_window(service: &Service, pending: &Pending) -> Result<bool> {
    if !service.is_single_body() {
        return Ok(false);
    }
    Ok(key_charge(service, pending)?.1 > key_window(service))
}

pub(super) async fn prepare(
    service: &Service,
    head: &Pending,
    page: &mut lance_context_wal::replication::Page,
    queued: &mut VecDeque<Pending>,
    replay_cut: Option<u64>,
) -> Result<Lookup> {
    if service.is_single_body() {
        // Accumulate whole admitted records, rather than repeatedly scanning
        // history for arbitrary 2048-key chunks. The apply budget bounds frames
        // and the subsequent shared key-only query independently.
        // A deferred speculative window may be topped up at promotion. Its
        // already-admitted tail still consumes the same wire-byte bound.
        let mut bytes = queued.iter().fold(head.meta.body_bytes, |bytes, pending| {
            bytes.saturating_add(pending.meta.body_bytes)
        });
        let mut resident = std::iter::once(head)
            .chain(queued.iter())
            .filter(|pending| !pending.keys_only)
            .fold((0usize, 0u64), |(n, b), pending| {
                (n + 1, b.saturating_add(pending.meta.body_bytes))
            });
        let key_limit = key_window(service);
        let mut key_workspace = 0usize;
        let mut keys = 0usize;
        for pending in std::iter::once(head).chain(queued.iter()) {
            let (n, w) = key_charge(service, pending)?;
            keys += n;
            key_workspace = key_workspace.saturating_add(w);
        }
        let mut local_end = false;
        let mut keys_only = 0usize;
        let stop = loop {
            if key_workspace >= key_limit {
                // Includes an oversize head: it is applied as a head-only
                // window (lookup admission decides whether it fits at all),
                // never extended past the per-window allowance.
                break PrefetchStop::KeyBound;
            }
            if queued.len() + 1 >= MAX_BODY_WINDOW_REQUESTS {
                break PrefetchStop::RequestBound;
            }
            if bytes >= MAX_BODY_WINDOW_PAYLOAD_BYTES {
                break PrefetchStop::ByteBound;
            }
            if page.entries.is_empty() {
                if replay_cut.is_some_and(|cut| page.through >= cut) {
                    break PrefetchStop::PageExhausted;
                }
                let Some((next, at_end)) = read_page(service, page.through, replay_cut).await?
                else {
                    break PrefetchStop::PageExhausted;
                };
                // Replace only after validation. On cancellation the old page
                // and every admitted input remain available to this driver.
                *page = next;
                local_end = at_end;
                // One empty response ends this window, even if it advances
                // past control/foreign records. Do not poll for future writes.
            }
            let Some((index, meta)) = page.entries.first().cloned() else {
                break PrefetchStop::PageExhausted;
            };
            if meta.body_bytes > MAX_BODY_WINDOW_PAYLOAD_BYTES - bytes {
                break PrefetchStop::ByteBound;
            }
            // Decode workspace is transient and admitted by `fetch`, one
            // record at a time; it is released below if the body is dropped.
            let Some(mut next) = fetch(service, index, meta).await? else {
                break PrefetchStop::Admission;
            };
            let (n, w) = key_charge(service, &next)?;
            if key_workspace.saturating_add(w) > key_limit {
                // The shared query would be refused by lookup admission.
                // Keep the metadata entry; this record starts a later window.
                break PrefetchStop::KeyBound;
            }
            // Owner batch mode keeps no bodies at all: survivors are
            // refetched from the WAL when the write batch is published.
            let retain = !service.owner_batch()
                && resident.0 < MAX_REQUESTS
                && next.meta.body_bytes <= MAX_PAYLOAD_BYTES.saturating_sub(resident.1);
            if retain {
                resident = (resident.0 + 1, resident.1 + next.meta.body_bytes);
            } else {
                next.drop_body();
                // The retained rows are charged to this window's key
                // workspace (see `Pending::drop_body`); never exceed it.
                ensure!(
                    next.retained_key_bytes() <= w,
                    "keys-only record exceeds its key workspace charge"
                );
                keys_only += 1;
            }
            keys += n;
            key_workspace += w;
            bytes += next.meta.body_bytes;
            queued.push_back(next);
            page.entries.remove(0);
        };
        {
            let mut observed = service.shared.body_lookups.lock().unwrap();
            observed
                .prefetch
                .record_keys(keys, key_workspace, keys_only);
        }
        service
            .shared
            .body_lookups
            .lock()
            .unwrap()
            .prefetch
            .record(queued.len() + 1, bytes, stop);
        // Driver refreshes this on stale-base detection as well as initially.
        return Ok(Lookup {
            body_stop: Some(stop),
            body_local_end: local_end,
            ..Lookup::default()
        });
    }
    if head.meta.body_bytes > MAX_PAYLOAD_BYTES
        || head.decoded.rows() > MAX_ROWS
        || !service.shared.window.lock().unwrap().admits(&head.meta)
    {
        return Ok(Lookup::default());
    }
    let mut rows = head.decoded.rows();
    let mut payload_bytes = head.meta.body_bytes;
    while queued.len() + 1 < MAX_REQUESTS && rows < MAX_ROWS {
        let Some((index, meta)) = page.entries.first().cloned() else {
            break;
        };
        if meta.body_bytes > MAX_PAYLOAD_BYTES - payload_bytes {
            break;
        }
        // Metadata was ordered and cut-bounded when the page was captured.
        // Never wait for the next reservation while this prefix can apply.
        let Some(next) = fetch(service, index, meta).await? else {
            break;
        };
        rows += next.decoded.rows();
        payload_bytes += next.meta.body_bytes;
        queued.push_back(next);
        page.entries.remove(0);
    }
    if queued.is_empty() || rows > MAX_ROWS {
        return Ok(Lookup::default());
    }
    let mut registrations = 0;
    let mut patches = 0;
    let mut turns = Vec::new();
    let mut patch_turns = Vec::new();
    for pending in std::iter::once(head).chain(queued.iter()) {
        match &pending.decoded {
            mixed::Decoded::Patches(rows) => {
                patches += 1;
                patch_turns.extend(rows.iter().map(|row| row.turn.as_str()));
            }
            mixed::Decoded::Creates { rows, .. } => {
                registrations += 1;
                turns.extend(rows.iter().map(|row| row.turn.as_str()));
            }
        }
    }
    // Keep registration sharing even when the head is a patch. The two optional
    // lookups share the existing workspace/read admission and total key bound.
    // Each captures a protected base without applying any request.
    // Ordered application rechecks active/frozen values and base identity, so
    // interleaved creates, later patches and publication cannot reuse stale values.
    let registration_lookup = async {
        if registrations < 2 {
            return Ok(None);
        }
        service
            .shared
            .phases
            .measure(
                Phase::AnnotationRegistrationLookup,
                service
                    .shared
                    .staging
                    .prepare_registration_lookup(&turns, &service.apply_budget),
            )
            .await
    };
    let patch_lookup = async {
        if patches < 2 {
            return Ok(None);
        }
        service
            .shared
            .phases
            .measure(
                Phase::AnnotationPatchLookup,
                service
                    .shared
                    .staging
                    .prepare_patch_lookup(&patch_turns, &service.apply_budget),
            )
            .await
    };
    let (registrations, patches) =
        futures::future::try_join(registration_lookup, patch_lookup).await?;
    Ok(Lookup {
        registrations,
        patches,
        bodies: None,
        body_stop: None,
        body_local_end: false,
    })
}

/// Give released producer lanes a bounded opportunity to commit their next
/// request before sealing a foreground query. A local end is only an observed
/// gap, not proof that the upstream producer has no queued input. The caller
/// retains the absolute deadline across cancellation; it never resets on input.
pub(super) async fn coalesce(
    service: &Service,
    head: &Pending,
    page: &mut lance_context_wal::replication::Page,
    queued: &mut VecDeque<Pending>,
    replay_cut: Option<u64>,
    deadline: tokio::time::Instant,
) -> Result<Lookup> {
    let mut lookup = prepare(service, head, page, queued, replay_cut).await?;
    while lookup.body_local_end
        && matches!(lookup.body_stop, Some(PrefetchStop::PageExhausted))
        && service.apply_backpressure().is_none()
        && tokio::time::Instant::now() < deadline
    {
        // Use the driver's existing idle cadence. Do not spin on an empty
        // replica or hold the ordered apply/publication gate while waiting.
        tokio::time::sleep_until(
            deadline.min(tokio::time::Instant::now() + Duration::from_millis(5)),
        )
        .await;
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        match tokio::time::timeout_at(deadline, prepare(service, head, page, queued, replay_cut))
            .await
        {
            Ok(result) => lookup = result?,
            // prepare retains validated metadata and decoded bodies before
            // advancing the page. A cancelled refill cannot skip a record.
            Err(_) => break,
        }
    }
    Ok(lookup)
}
