//! Read a small admitted prefix without advancing any application position.
use super::*;

const MAX_REQUESTS: usize = 8;
const MAX_ROWS: usize = 128;
const MAX_PAYLOAD_BYTES: u64 = 256 << 10;

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
    let Some(reservation) = service.apply_budget.try_reserve(decode_workspace(
        &ingress.scope.contract,
        &service.limits,
        bytes,
    )?) else {
        return Ok(None);
    };
    let body = match service.shared.local_payloads.body(producer_slot, &meta) {
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
                .await?;
            ensure!(
                matches!(response.reply, Reply::Record(Some(ref observed)) if observed == &meta),
                "committed annotation payload unavailable"
            );
            response.body
        }
    };
    meta.check_body_length(body.len(), ingress.scope.contract.max_record_bytes as usize)?;
    let schema = ingress.schema.clone();
    let max_rows = ingress.scope.contract.max_rows;
    let value_bytes = service.limits.value_bytes;
    let mixed = ingress.scope.contract.policy == WritePolicy::TurnIngestion;
    let shard = service.shard;
    let layout = ingress.layout.clone();
    let scope = ingress.scope.clone();
    Ok(Some(
        service
            .shared
            .phases
            .measure(Phase::AnnotationDecode, async {
                tokio::task::spawn_blocking(move || -> Result<Pending> {
                    let decoded = if mixed {
                        mixed::wire::decode(body, schema, max_rows, value_bytes)?
                    } else {
                        mixed::Decoded::Patches(wire::decode(body, schema, max_rows, value_bytes)?)
                    };
                    decoded.validate_shard(shard, &layout, &scope.contract)?;
                    Ok(Pending {
                        index,
                        meta,
                        decoded,
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
) -> Result<Option<RegistrationLookup>> {
    if head.meta.body_bytes > MAX_PAYLOAD_BYTES
        || head.decoded.rows() > MAX_ROWS
        || !matches!(head.decoded, mixed::Decoded::Creates { .. })
        || !service.shared.window.lock().unwrap().admits(&head.meta)
    {
        return Ok(None);
    }
    let mut rows = head.decoded.rows();
    while queued.len() + 1 < MAX_REQUESTS && rows < MAX_ROWS {
        let Some((index, meta)) = page.entries.first().cloned() else {
            break;
        };
        if meta.body_bytes > MAX_PAYLOAD_BYTES {
            break;
        }
        // Metadata was ordered and cut-bounded when the page was captured.
        // Never wait for the next reservation while this prefix can apply.
        let Some(next) = fetch(service, index, meta).await? else {
            break;
        };
        rows += next.decoded.rows();
        queued.push_back(next);
        page.entries.remove(0);
    }
    if queued.is_empty() || rows > MAX_ROWS {
        return Ok(None);
    }
    let mut requests = 0;
    let mut turns = Vec::new();
    for pending in std::iter::once(head).chain(queued.iter()) {
        if let mixed::Decoded::Creates { rows, .. } = &pending.decoded {
            requests += 1;
            turns.extend(rows.iter().map(|row| row.turn.as_str()));
        }
    }
    if requests < 2 {
        return Ok(None);
    }
    Ok(service
        .shared
        .phases
        .measure(
            Phase::AnnotationRegistrationLookup,
            service
                .shared
                .staging
                .prepare_registration_lookup(&turns, &service.apply_budget),
        )
        .await?)
}
