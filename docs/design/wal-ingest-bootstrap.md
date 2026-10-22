# Provisioning a managed rollout generation

`bootstrap::provision` creates a new empty rollout generation and selects its
genesis manifest through the durable WAL catalog. The application must reserve
and persist the authorized table binding, dataset URI, publisher owner, acquisition
request and manifest request UUIDs before calling it. A timeout is retried with
that same configuration. This is not an existing-table importer or a legacy WAL
migration.

`Storage::resolve` uses Lance's own object-store registry and URI/credential
handling to derive the manifest base path. Credentials stay in application storage
options and never enter replicated metadata. Readers and publishers can share
the same Lance session and object-store cache.

Initialization acquires the initial WAL publisher epoch, creates an empty Lance
dataset in create-only mode, and writes the complete binding and operation identity
into the genesis transaction properties. If the dataset already exists, it opens
version 1 and requires the exact marker, rollout schema and zero rows before it
can select that manifest. An unrelated empty dataset is also rejected. Other
write failures remain errors; there is no overwrite or automatic file deletion.

After catalog initialization, retries use only the catalog-selected manifest.
They do not read a canonical genesis path to discover visibility, and they do not
take ownership from a running publisher. A different table contract, incarnation
or manifest request is rejected. The operation does not generate replacement IDs
on restart or silently migrate an existing namespace binding.

The application still needs to publish the completed binding into its authorized
namespace and provide it to gateways and materializer workers. This initializer
alone is not a dynamic namespace registry or a deployment executable.

Focused qualification passed seven ordinary integration tests, the two real-etcd
worker tests and all three checked feature Clippy configurations. The bootstrap
tests use actual TLS WAL replicas and Lance datasets, including all-original-directory restart before
genesis selection, exact original genesis transaction reuse, completed-operation
retries, and rejection of a foreign empty genesis. The existing HTTP/materializer
suite also uses this initializer when its materializer feature is enabled.

Tracking: #378.
