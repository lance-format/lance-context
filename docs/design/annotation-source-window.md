# Bounded source prefixes for annotation publication

Repeated mutations of one turn can coalesce into one dirty value while their WAL
records keep growing. Bounding only the number and size of dirty values does not
ensure their publication fits the catalog's existing input and source-byte caps.

The ordered annotation applier now bounds the records and source bytes represented
by its active generation. At the cap it retains the pending decoded batch and
pauses application. It neither rejects nor acknowledges that unapplied request.
Freezing atomically captures the complete source metadata with the value/result
cut and opens the next bounded active window. The frozen prefix still requires a
verified source pin before publication and confirmed coverage before closure.

There is at most one frozen generation. Repeated pressure cannot create additional
frozen generations; the publisher must finish and release its current one. Source
metadata has a fixed bound of one active window and two retained frozen handles,
each of at most 512 entries, independently of the dirty-value budget. Retaining
old handles consumes those slots until they are released. The pending decoded
batch retains its existing apply budget.
Each frozen handle exposes the captured selection so publication need not rediscover
it by scanning the WAL. The normal point-read path is unchanged.

Real TLS/RF3/Lance tests separately force the record cap and byte cap with repeated
updates of one turn. They hold an unapplied request under pressure, publish the old
prefix while the suffix applies, confirm the exact retry result, then publish the
suffix. Three cycles verify that value and metadata retention return to zero.
This supplies bounded prefixes; automatic publisher scheduling is separate work.
