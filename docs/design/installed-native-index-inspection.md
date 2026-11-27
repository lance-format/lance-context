# Inspect native indices on the installed base

Call `Service::inspect_installed_indices().await?` explicitly on the owner.
The serializable result includes the staging scope, installed Lance version,
materialized WAL cut, current fragment count and query-visible native indices.
It uses the same dataset object that serves historical lookups. The installed
version may lag catalog selection; this result is not a latest-version proof.

The call takes one existing reader permit and pins the captured base for local
GC. It releases the staging state lock before loading native index metadata.
Lance may read manifest/index metadata, including fragment-reuse coverage.
It does not scan body rows, create indices or change publication. Ordinary
observations and `observation_with_manifest` retain their existing behavior.

Each group reports the native index name, keyed field IDs and schema paths,
exact native type URL, segment count and current covered/uncovered fragment
counts. Carried fields are not advertised as lookup keys. Coverage is the union
of same-name/type/key segments intersected with this manifest's fragments;
overlaps and retired fragment references cannot inflate it. A missing segment
bitmap makes both coverage counts unknown (`null`). A missing type URL also
remains `null`.

This is Lance's query-visible listing, after its native compatibility/coverage
handling, not an inventory of hidden or unsupported index files. Coverage says
which current fragments an index addresses. It does not show whether a query
chose that index or how many rows a zone map/Bloom filter can prune. Compare it
with successful lookup-window metrics on the same owner/version and workload.

Validation uses a real B-tree followed by an unindexed append, then removes all
body data files before inspecting metadata. It also checks overlapping segment
coverage and unknown coverage. The RF3 full-body compaction test checks the
service API before and after the installed base changes from three fragments to
one, while its materialized cut remains fixed.

Related: #378.
