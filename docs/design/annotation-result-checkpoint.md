# Annotation results at the materialization cut

A values-only Lance snapshot cannot recover whether a retained request succeeded
or was rejected. Annotation publication now freezes its request-result snapshot
under the same ordered apply gate as its values. Both travel in the exact native
transaction and its existing managed publication descriptors.

The independent WAL already retains the latest producer receipt after source
payload retirement. The annotation response contains Applied or Rejected(row,
reason), rather than a historical value. Consequently, the native checkpoint only
needs the latest rejection per producer slot. For an exact request which the WAL
has confirmed is accepted at or before the proven cut, a matching rejection returns
that rejection; otherwise the request was applied. A bare caller-provided position
never authorizes this inference. Changed bytes, superseded sequences and fenced
generations are rejected by the existing WAL protocol before result lookup.

The result snapshot stores bounded tuples of slot, accepted index, row and reason.
All 16,384 maximum producer slots fit below 640 KiB even at maximum integer widths;
the native descriptor retains its existing 1 MiB total cap. No second copy of all
successful BatchMeta receipts or permanent request/turn history is introduced.
The tuple snapshot and the scoped cut are properties of the same native transaction.

When all mutations in a batch are rejected, there may be no dirty annotation
values. Such a batch still needs a publication and source closure to prevent
unbounded retention. It uses a native metadata-only UpdateConfig transaction with
no configuration/schema/field changes and no data files. The publication adapter
accepts this exception only with validated result metadata and the exact source cut.
Ordinary annotation publication continues to require RewriteColumns and protected
body-file references.

Results and values freeze together; later requests continue in active staging and
the latest result slots. Installing the original publication replaces only the
frozen value generation and its base result snapshot. Later application outcomes
remain. A separate bounded metadata reserve prevents full dirty-value admission
from consuming the space needed to freeze its own publication. Retired metadata
handles keep their charges until released.

Recovery opens the exact manifest version captured from the quorum catalog. For
a selected publication whose confirmation/closure is unfinished, it verifies the
original descriptors and selected transaction through the existing adapter. For
a completed publication, the native cut must match the durable catalog checkpoint.
Only then does application start at that cut and replay the WAL suffix. Missing
result metadata cannot be treated as an all-success checkpoint.

This is the checkpoint/publication integration, not automatic scheduling. The
caller still owns plan selection, source pins, publication result resolution,
confirmation and closure. An independently running view which falls behind source
closure fails and must reconnect to the durable checkpoint; production owner routing
and continuous publisher recovery remain to be wired. SIGKILL and sustained mixed
pressure tests remain required before enabling the application path.
