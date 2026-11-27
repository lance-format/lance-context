# Empty committed-source pages

`BodyLookupObservation.prefetch` includes counters for validated `ReadSource`
responses used by a full-body owner:

| Counter | Meaning |
| --- | --- |
| `source_empty_with_remaining_metadata` | The page has no matching records and its cursor is below the response's locally applied position. More committed metadata exists, but it may belong to another table or bucket. |
| `source_empty_at_local_end` | The empty page reaches that replica's observed applied position. It does not prove fresh quorum catch-up. |
| `source_empty_at_replay_cut` | The empty page reaches the consumer's fixed publication-recovery cut, after clipping to that cut. |
| `source_cursor_ahead` | The replica is behind the requested cursor; no empty page or catch-up is implied. |

The WAL bounds each bucket-page scan to 4,096 accepted metadata entries. An empty
filtered page can therefore precede more metadata. The first counter makes that
case distinguishable from a locally exhausted source without changing the read
loop, adding polling, or changing apply/ACK progress.

These count responses, including initial driver reads. They do not partition
`page_exhausted`, which counts completed prefetch decisions: a decision can stop
at an already-retained recovery cut without issuing another read, and a driver
can receive an empty page before it has any request to prepare. Invalid replies
and failed RPCs are excluded. Compare deltas from the same owner lifetime; absent
fields in an older runtime are unavailable, not measured zero.

Related: #378.
