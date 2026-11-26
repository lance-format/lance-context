# Lance file 14.0.2 read recovery

This directory starts from the official published `lance-file 14.0.2` crate.
Archive SHA256: `23f909cbffa69e3287fe8bbcbb2b004e3b9c3a8f013b4d0ad0ae751a38128ab8`.
The original manifest and license are retained. Cargo.toml selects the existing
canonical sibling overlays; the archive's nested Cargo.lock is omitted.

FileReaderOptions accepts an optional task-owned recovery callback. The shared
DecodeEngine releases failed decoder state before waiting, retries an assigned
task's exact physical selection, and reopens only the unassigned suffix. Typed
error context retains the stream identity and physical row assignment across
error wrapping. Both full and projection-scoped readers use this path.

Encoded-filter streams keep their existing error path because output row counts
cannot identify physical progress. Fragment readers also keep that path for
adaptive reads combining multiple files, whose independently reopened batch
boundaries cannot safely be merged. Fragment open and metadata preparation retry
before row assignment. Cancellation and retry deadlines belong to the callback's
owning task; there is no scanner restart or built-in retry budget.
