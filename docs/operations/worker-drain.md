# Worker shutdown request inventory

On SIGTERM, workers close HTTP admission and join accepted handlers, including
handlers whose clients have disconnected. The shutdown log reports the exact
active count and up to 128 outstanding requests. Each entry has a process-local
`drain_id` / `request_id`, HTTP method, matched route template, and a validated
store name or context ID when available. Query strings, record IDs, credentials,
and request bodies are excluded. Unknown targets remain `None`.

`inventory_truncated=true` means the emitted list is incomplete. All handlers
remain tracked and joined even if they were omitted from that initial log.
Subsequent `accepted HTTP handler joined during shutdown` records identify each
completed request and the remaining count. Logging happens outside the admission
lock, so a completion can appear before its initial inventory entry: correlate
IDs rather than assuming log order. Bind the logs to the Pod UID and container
identity; request IDs alone are not durable execution identities.

This inventory attributes outstanding HTTP work. A join means that the handler
returned or panicked; it does not prove a storage commit succeeded or that every
detached background task joined. These logs do not authorize cancellation,
fencing bypass, or replacement of a writer. Shutdown still waits for accepted
handlers, and response connections retain their existing progress-based idle
handling. No deadline, retry, storage read, etcd request, or Kubernetes action is
added.

Already-running older binaries cannot produce this inventory retroactively.
An old count-only shutdown message remains insufficient to identify the active
tables or distinguish reads from writes.
