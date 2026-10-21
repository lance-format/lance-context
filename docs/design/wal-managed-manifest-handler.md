# Lance manifest publication through the WAL catalog

`replication::publication::lance::ManagedManifest`, behind `lance-integration`,
is a reusable Lance 13 `CommitHandler`. It moves the actual reader/writer adapter
out of the process-test fixture. The handler uploads a unique manifest object and
returns a location only after the catalog commits its exact reservation. Readers
resolve both current and specific versions from quorum queries to that catalog.
No change to Lance OSS or the replicated storage format is required.

A materializer supplies the immutable `ManifestCommit`: table authority, dataset
incarnation, transaction, plan/preparation positions, descriptor digest, exact
predecessor identity and target version. Construction validates scope, but current
ownership and predecessor checks occur in the replicated reservation. The caller
must construct the actual append transaction from the validated prepared fragment
descriptors; this adapter checks the transaction UUID and fixed target, not the
contents of Arrow records or the complete semantics of Lance fragments.

The `ManifestCatalog` implementation is the integration boundary for the network
service. It must return correlated, authenticated quorum reads and durable command
outcomes, with bounded admission and deadlines. The application must bind the
handler and Lance object store to the same authorized storage namespace and dataset.
Checking the object path alone does not identify a bucket/account or authorize a
client. This change does not supply production transport, HTTP routing, bootstrap,
or migration from the existing writer.

Use the handler for every managed reader and writer, including `DatasetBuilder`
and `CommitBuilder`. Raw URI readers cannot find unique catalog-selected objects.
Build publication transactions from the captured prepared descriptors, retain their
UUID/read version, and disable Lance automatic cleanup using
`with_skip_auto_cleanup(true)`. The catalog, rather than object listing, decides
which staged path is visible.

## Retry and failure behavior

- A changed target or transaction UUID fails before uploading a manifest.
- A canceled call, unknown outcome, rejected authority, or lost reservation reply
  leaves its staging object intact. Recover the original plan and query its outcome;
  do not manufacture a new producer identity or publication target.
- An exact retry may return the previously selected object instead of the object
  just uploaded. Validate its incarnation-derived identity, transaction, target,
  preparation positions and descriptor digest before returning success.
- A foreign quorum scope, invalid record identity, retired version or unknown read
  outcome returns an error. Retired and unknown results are not reported as absence.
- Lance's conservative defaults for ambiguous commit errors and non-definitive
  absence remain in force. The adapter does not delete files on those outcomes.

Lance conflict resolution asks for versions newer than a read version. The adapter
walks only catalog-selected versions, in descending order, with at most 128 versions
per request. A missing/retired intervening version is an error. Separate queries
are not an atomic multi-version snapshot; concurrent retirement can fail the walk
and requires reconciliation. The bounded walk is for conflict resolution, not a
claim of optimized bulk catalog enumeration.

General/detached manifest listing, history deletion and dataset deletion are
unsupported until coordinated reader/GC integration exists. Falling back to an
object list here would expose unselected staging objects. WAL metadata retirement
remains separate from permission to delete their underlying objects.

## Qualification

The six-process WAL/Lance tests use the library adapter through the fixture's
catalog transport. They cover full process restart, lost reservation replies,
old-owner suspension after manifest upload, retained manifest history beyond its
cap, and producer-slot reuse, with actual Lance payload verification. Focused
adapter tests inject foreign read scopes, invalid identities, unknown/retired
results and history gaps, and verify that ambiguous reservations retain an actual
in-memory object-store manifest. The tests do not establish cloud throughput or
loss of a quorum's persistent volumes.

Tracking: [#378](https://github.com/lance-format/lance-context/issues/378).

A reader can use `ManagedManifest::reader` with the authorized table generation,
incarnation, layout and dataset path. It needs no publisher owner, preparation or
pending transaction. This supports a bootstrapped table before its first append
and readers after publication has advanced. The reader rejects `commit` before
uploading an object or sending a reservation; it still uses the same catalog
scope and incarnation identity checks. `ManagedManifest::new` retains the full
fixed-target writer contract.
