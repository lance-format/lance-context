# Inspect installed fragment sizes

`Service::observation_with_manifest()` returns the existing `(Observation, bool)`
and fills `observation.installed_base.fragment_summary`. The summary implements
`Serialize`, so an application's existing diagnostic response can include it
without listing individual files. Ordinary `observation()` leaves the field
`None` and does not traverse fragment metadata.

The diagnostic captures the installed base together with its reported version
and materialized position, then releases the staging lock before traversing the
loaded manifest. It performs no object listing, data/deletion reads, index loads
or remote calls. It adds no cache. Work is linear in loaded files/fragments and
the result has four fixed size bands. The installed base may lag a selected
publication; these values are neither a publication acknowledgment nor GC proof.

The summary reports:

- `fragments`: total installed fragment count.
- `known_size_fragments` and `known_size_bytes`: coverage and subtotal for
  fragments with nonempty file lists and known, representable total file sizes.
- `known_physical_rows_fragments` and `known_physical_rows`: coverage and
  physical-row subtotal for nonempty fragments with known counts from a
  versioned writer. These include deleted rows, unlike `installed_base.rows`.
- `fragments_with_deletions`: fragments naming a deletion file.
- `small_fragments`: cumulative candidates at target sizes 16, 32, 64 and
  128 MiB. Each reports `target_file_bytes`, `fragments`, `bytes` and
  `physical_rows`. Candidates need known nonzero size strictly below half the
  target, trustworthy nonzero rows and no deletion file.

Unknown size/row metadata does not mean an empty file. Subtotals exclude unknown
fragments; compare coverage counts with `fragments`. A null subtotal means
arithmetic overflow. Size bands overlap; do not sum them.

Size candidates are not a compaction plan. Native index coverage must still be
uniform within each rewrite group. Minimum fragments, per-task row/byte bounds
and workspace admission also apply. This diagnostic intentionally does not load
index metadata or declare a rewrite executable.
