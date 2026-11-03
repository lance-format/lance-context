# Durable publication plans

Refs #378.

A publisher takeover must recover the same transaction and source selection that
the previous owner was preparing. The optional replicated WAL now accepts an
immutable `PlanRequest` through its catalog group. Only a durable quorum completion
returns the plan's original log index. Exact retries use consensus; an isolated
leader cannot answer them from its local catalog.

A plan contains a transaction UUID, an existing Lance read version, the fixed
target `read_version + 1`, and an ordered vector of source groups. Each source has
its own `(after, through]` range and selected record metadata. Independent group
positions are never compared as a global index. The catalog checks the current
publisher authority at apply, the table's per-group published cursors and the
ingestion record-size limit. A source range in the catalog group must precede the
plan entry itself. Other groups may have larger numeric positions.

Takeover changes authority while retaining the immutable pending plan and progress.
The new owner can retry that exact plan and recover its original index. Replacing
it with a different transaction, source selection or target is rejected. One plan
is retained per table; the configured global active-plan limit does not prevent
an existing plan's retry or an ownership takeover. Cloned table/checkpoint views
share the immutable plan with `Arc`; payload bodies do not live in the catalog.

The bounds are separate: acquisition commands retain their 8 KiB limit, plan
commands have a 512 KiB limit and at most 2048 selected records, and the policy
bounds their aggregate encoded source bytes before payload fetch. Defaults allow
128 active plans and 64 MiB of selected source bytes per plan. These limits bound
metadata and selected input. Each registered ingestion contract's maximum record
must fit the publication source-byte budget so an accepted record can be selected.
The budget may be smaller than the global WAL record ceiling for smaller contracts.
These limits do not reserve encoder memory or transport
queues. Checkpoint metadata capacity accounts separately for table bindings and
active plans. Large metadata frames allocate only their bounded actual length on
the heap. Decoders reject noncanonical maps instead of silently accepting duplicate
keys normalized by the serializer.

Windows above 512 records are opt-in. Upgrade every WAL voter, publisher and
recovery reader before configuring a larger window: older binaries cannot read
the larger plans or source pins. The encoding is unchanged, but these expanded
bounds are not safe to enable during a mixed-version rollout. The annotation
driver's default remains 512 records, and the materializer's default remains 128.
Source-byte and application memory limits still apply. In particular, active and
frozen generations must fit their existing budgets; a larger record limit does
not reserve more payload memory or disable backpressure. Source-pin encoding and
checkpoint metadata budgets follow the plan's byte limit.

Source pins subsequently advance the optional application identity and checkpoint
format to v9. Existing v1/v2/v3/v4/v5/v6/v7/v8 directories and images are rejected; there is no in-place migration
or production rollout in this increment. Catalog-only and mixed data/catalog
snapshots preserve plans, with semantic validation after checksum verification.

A stored plan is not a source pin, proof of complete source selection or permission
to commit a Lance manifest. Source groups validate their full selected prefixes
through the [source pin protocol](wal-source-pins.md). Descriptor CAS, exact-target Lance
outcome recovery, resolution and cleanup commands remain subsequent work; this
increment intentionally retains pending plans and advances no published cursor.
No source payload can be retired on the strength of this plan alone.

Validation includes real TCP replicas that persist a source append and a catalog
plan, lose leaders, transfer a catalog-only snapshot, take over ownership and
restart every process. The restarted owner retries the original plan index and
the source payload remains exact. Mixed snapshots exercise plans larger than the
old acquisition limit and reject corrupt inner plan state even when outer digests
are recomputed. Tests also cover shared view ownership, metadata/source-byte and
active-plan bounds, malformed maps and stale publisher authority.
