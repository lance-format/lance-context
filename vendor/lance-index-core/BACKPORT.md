# Caller-owned native index allocations

Based on published `lance-index-core 14.0.2`.
Archive SHA256: `cc6ab34b812d49209a8772809e24897d43c713115e11dfd2bb6d48a81beab39c`.
Upstream commit: `40f15ed1b71b66950961b175829fdf17a763f108`.

Optional caller memory ownership follows index reads and decoded backing allocations.
Cache weights remain eviction estimates, independent of allocation admission.

Rebased to `lance-index-core 14.0.3`: upstream 14.0.2→14.0.3 changes only the package and
sibling Lance dependency versions in `Cargo.toml`; sources are unchanged.
