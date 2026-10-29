# Single-shard annotation publication runtime

`annotations::runtime::Runtime` owns one ordered applier and one publisher. The
application supplies the existing scheduler's durable acquisition identity. A
service handle submits exact requests and reads current narrow annotations while
the runtime drains its bounded active/frozen generations in the background.

A full source window, apply pressure, or the configured batch timer triggers a
flush. The publisher freezes values, outcomes and the complete source selection
together, records its fixed plan, obtains the existing verified source pin, and
prepares a native narrow-column update. It records the original descriptors,
resolves the selected manifest, installs the compatible base/staging view, and
uses the existing confirmation, source closure and checkpoint advancement flow.
Result-only batches follow the same lifecycle without writing data files.

After confirmed publication and installation of the compatible read view, the
publisher releases its frozen/prepared handles before awaiting source retirement.
Their dirty-value and workspace permits must not keep the next update waiting
on a slow closure RPC. Other held readers or handles keep their own charges.
WAL reclamation still requires the existing durable closure authorization and
advancement proofs. An unknown retirement result still stops the owner; recovery
uses the selected value/result checkpoint and retained WAL suffix.

The upload, apply and publication budgets are independent. Publication admission
must accommodate a fully occupied dirty-value generation and result metadata;
waiting uploads cannot consume its workspace. There is at most one publishing
generation. Further source/value pressure pauses application rather than creating
additional frozen generations or acknowledging an unapplied mutation.

Startup handles the durable states explicitly:

- **Plan without preparation:** replay only to the original cut, freeze exactly
  that source selection, then resume the suffix while preparing the old generation.
- **Preparation without selected manifest:** publish the original durable native
  transaction and file paths before opening the new live service view.
- **Selected manifest with unfinished completion:** verify the selected native
  transaction against the retained descriptors, confirm it and finish closure.
- **Completed publication:** restore its exact value/result checkpoint and replay
  the remaining WAL suffix.

A publication/protocol error, including an unknown RPC result, stops this runtime
and its read barriers. A fenced restart reconciles the retained operation. The runtime
does not generate a replacement mutation identity, guess a publication result, or
blindly retry application errors. Acquisition retries retain the caller's existing
durable scheduler identity as required by the catalog protocol.

Exclusive physical collection treats a failed DELETE separately: after a confirmed
quorum seal, it retains the unresolved immutable paths in one bounded page and
keeps the page's memory reservation. New publication and compaction pause until
that page clears. The applier keeps its existing source/staging limits, so an
extended storage failure backpressures updates while owned reads still serve
previously applied values. Each later collection interval checks ownership and
reissues only unresolved DELETE requests from that original page. Successful paths are
removed; an absent path resolves a lost reply. An unknown whole-batch response
retains all paths in that batch. Paths are never added after sealing.

This behavior covers any DELETE error, including a permission failure: it does
not guess whether an opaque SDK error is transient. A persistent failure keeps
publication paused, exposes pending object/byte and failure counters, and requires
the storage problem to be corrected. There is no application-mutation retry.
Unknown seals still issue no deletion and require fenced recovery. After a crash,
the successor uses the existing fresh listing/protection/sealing protocol; the
process-local page is not a new durable journal.

Data listing uses a time slice bounded by the collection interval, capped at one
second. If the next listing item is pending, the runtime retains that exact
stream and closes any candidates already collected into the usual finite page.
Publication can proceed between slices. Each subsequent pass captures current
protected bases and quorum history before protecting/sealing its candidates;
the stream's continuation token or pending response is not discarded. Actual
listing errors still stop the runtime, and an unresolved sealed DELETE still
pauses publication. Reference checks, protection reads and DELETE are not covered
by this data-listing slice. Prolonged listing delays can temporarily increase
obsolete blob files; bounded working memory is not a bound on total blob garbage.

Protection reads reuse selected immutable manifests in the existing weighted
Lance session metadata cache. Entries are scoped to one collector's fixed table
and storage binding and include the immutable catalog identity, path and size;
a version number alone is not a cache key. Independent collectors use separate
namespaces even when they share a Session and equal destination URI strings.
Each page still captures fresh reader roots and quorum history and obtains its
own deletion fence. Cached metadata neither pins files nor authorizes deletion.
Cache eviction, clearing or zero capacity falls back to the same selected
manifest read, under the existing publication workspace check.

The Rust tests exercise plan-only recovery, prepared-file reuse, lost selected
manifest replies, automatic timed result-only publication, and repeated concurrent
status/score updates with complete drain between cycles. A separate Rust child
process runs all three WAL replicas and the runtime, publishes a prefix, acknowledges
an unmaterialized suffix, and is killed with SIGKILL. Its parent reopens the real WAL
and Lance state, checks the original response and current values, then drains the
suffix. This process-crash test does not simulate loss of the backing storage.

The [application roles](annotation-application.md) supply authenticated HTTP,
etcd ownership and routing through a configured owner address. Deployments must
follow their explicit table handoff and reclamation requirements. Tiny correctness
fixtures do not qualify annotation throughput: sustained-load results need their
workload, latency, refusal and backlog measurements. Mixed ingestion, filter
indexing and multi-shard scale are later work.
