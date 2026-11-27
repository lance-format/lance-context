# Native statistics for completed immutable lookups

An immutable-key lookup stops as soon as every requested key is found. The scanner's completion callback runs only when the stream is polled to EOF. A successful lookup can therefore read data and return its answer without contributing a native summary.

The lookup captures the execution plan built by the existing scanner and executes that same plan through Lance's public execution API. It preserves the scanner's execution options, batch size and dataset stream adapter. After dropping the result stream, the successful lookup takes one metrics snapshot from its plan, for either all-hit early exit or exhausted scan. It does not consume extra batches or wait for cancelled background work merely to collect statistics. Admission refusals, errors and cancellation do not become successful lookup observations.

These are plan metrics observed when the lookup finishes, not a complete account of subsequent cleanup or all object-store traffic. Outstanding work can finish later; index/planning I/O can have different instrumentation. Owner-wide object-store counters remain a separate measurement, and cumulative task times may overlap. No callback lifecycle outside this lookup is changed.

The change stays in lance-context, using existing dependencies and APIs. It does not change schema, business identity, durable sequencing, acknowledgement, read concurrency or publication.
