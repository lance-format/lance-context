# Caller-owned native index allocations

Based on published `lance-index 14.0.2`.
Archive SHA256: `df56fde7e5e34b1d6cf67de22fdfed3a9f6f2875fc5a5d452ae8ab5a1cef150b`.
Upstream commit: `40f15ed1b71b66950961b175829fdf17a763f108`.

Optional caller memory ownership follows index reads and decoded backing allocations.
Cache weights remain eviction estimates, independent of allocation admission.

`scalar/inverted/rewrite.rs` restores query-metadata readiness and conversion
for existing partitioned FST indexes. It converts dictionaries one partition at
a time and streams document lengths in bounded batches, preserving postings,
row IDs, deletions and tokenizer metadata. Conversion requires a fresh empty
destination; it does not publish an index or change source files. The caller
owns memory admission and cancellation. The existing paged dictionary format
is reused.
