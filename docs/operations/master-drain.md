# Drain one master executor

New masters expose `GET /api/v1/executor` and
`POST /api/v1/executor/drain` on the admin API. Use the exact Pod endpoint,
read its `executor_id`, and include that identity in the POST JSON. A request
for a different process returns HTTP 409. Draining is monotonic for that
process; restarting creates a fresh identity and opens local admission.

After POST, poll that same Pod and identity until `accepting` is false and
`active_operations` is zero. Only then terminate the Pod. Operations are
reserved before ownership RPCs, so an in-flight accepted claim is included.
Existing tasks, scanner passes (including cold retirement) and catch-up
controller operations finish normally. New scheduler claims, scanner passes
and catch-up provisioning on that process are refused. Queued tasks remain
available to other masters; table owners and fences are not reset.

SIGTERM/SIGINT closes the same gate and waits for admitted work before HTTP
shutdown and runtime exit. Existing progress watchdogs remain responsible for
stalled operations; drain adds no elapsed-time writer cancellation. Kubernetes
can still SIGKILL after its termination grace period: explicitly drain through
the API before deletion instead of relying on a short grace period.

This interface is process-local. Enqueue/read requests can still be served,
and other masters and already launched dedicated Jobs continue. Dedicated
one-table executor mode has no admin listener; use its supervisor's joined
pass lifecycle. The endpoint is part of the existing administrative API and
requires the same network access controls.

Older images do not implement this protocol. Observing zero tasks and then
deleting an old Pod is not an atomic drain, and installing this release does
not retroactively add the endpoint to running old processes. Do not claim an
old executor is drained merely from an idle metrics sample or SIGTERM receipt.
