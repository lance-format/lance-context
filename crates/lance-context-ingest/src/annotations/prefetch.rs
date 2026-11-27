//! Read a small admitted prefix without advancing any application position.
use super::*;
use lance_context_wal::replication::runtime::client::payload_memory;

const MAX_REQUESTS: usize = 8;
// Match one core cold-lookup window. Bodies stay in their original requests;
// only IDs participate in the shared query. Bound cumulative wire bytes as
// well as requests/keys; decode reservations still charge the apply budget.
const MAX_ROWS: usize = 2048;
const MAX_PAYLOAD_BYTES: u64 = 32 << 20;

#[derive(Default)]
pub(super) struct Lookup {
    pub(super) registrations: Option<RegistrationLookup>,
    pub(super) patches: Option<PatchLookup>,
    pub(super) bodies: Option<single_body::Lookup>,
}

/// Read committed metadata only. The caller retains the returned page; neither
/// this RPC nor its `through` cursor advances application progress.
pub(super) async fn source_page(
    service: &Service,
    after: u64,
    replay_cut: Option<u64>,
) -> Result<Option<lance_context_wal::replication::Page>> {
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
    Ok(Some(page))
}

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
                    })
                })
                .await?
            })
            .await?,
    ))
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
        let mut bytes = head.meta.body_bytes;
        let stop = loop {
            if queued.len() + 1 >= MAX_REQUESTS {
                break PrefetchStop::RequestBound;
            }
            if bytes >= MAX_PAYLOAD_BYTES {
                break PrefetchStop::ByteBound;
            }
            if page.entries.is_empty() {
                if replay_cut.is_some_and(|cut| page.through >= cut) {
                    break PrefetchStop::PageExhausted;
                }
                let Some(next) = source_page(service, page.through, replay_cut).await? else {
                    break PrefetchStop::PageExhausted;
                };
                // Replace only after validation. On cancellation the old page
                // and every admitted input remain available to this driver.
                *page = next;
                // One empty response ends this window, even if it advances
                // past control/foreign records. Do not poll for future writes.
            }
            let Some((index, meta)) = page.entries.first().cloned() else {
                break PrefetchStop::PageExhausted;
            };
            if meta.body_bytes > MAX_PAYLOAD_BYTES - bytes {
                break PrefetchStop::ByteBound;
            }
            let Some(next) = fetch(service, index, meta).await? else {
                break PrefetchStop::Admission;
            };
            bytes += next.meta.body_bytes;
            queued.push_back(next);
            page.entries.remove(0);
        };
        service
            .shared
            .body_lookups
            .lock()
            .unwrap()
            .prefetch
            .record(queued.len() + 1, bytes, stop);
        // Driver refreshes this on stale-base detection as well as initially.
        return Ok(Lookup::default());
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
    })
}
