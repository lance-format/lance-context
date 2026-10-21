# Snapshot transfer over mutual TLS

A lagging replica whose prefix was compacted needs the corresponding application
image before Raft can resume log replication. `peer::snapshot::Transport` serves
that image on a separate listener with its own ALPN, certificate pins, connection
limits and whole-transfer deadline. Each destination can retain only one queued
or active image in addition to the shared node limit. Attach its route to the ordinary transport
before enabling compaction. Both services must belong to the same runtime.

The sender retains its captured image and outgoing runtime credit. Two file
parts stream in chunks of at most one MiB without Arrow decoding or whole-image
concatenation. The receiver validates the authenticated route and bounded image
descriptor, obtains exact file lengths from the owner, and reserves disk space
before creating receive files. Complete file validation and durable runtime
installation precede the success response. A socket write is not success.

One dedicated file thread processes a bounded queue for reads, writes, validation
and admission. Shutdown stops network tasks, drains file work and joins that
thread before releasing its storage owners. A file-thread panic stops the image
service and surfaces through its joined shutdown. Cancellation cannot terminate an
already running filesystem operation. Dropping a transport requests shutdown;
call `shutdown().await` before reopening storage. A stalled image connection does
not occupy an ordinary Raft connection.

The embedding service must supply `Admission`; there is no permissive default.
Its accounting must cover existing installed images, orphaned/partial files,
concurrent log growth and maintenance headroom, and recover after restart. A
reservation owner spans pending file work. Releasing the owner must not release
charges for files that still exist. Collection, rather than network completion,
reclaims those charges. `max_image_bytes` and connection counts do not impose a
physical node disk quota. The shared production disk-admission implementation,
automatic maintenance and public ingestion integration remain follow-up work.

Tests exercise actual mutual TLS and files, including two compacted groups,
reopening all original engines with complete payload and exact retries, stalled
snapshot isolation from healthy quorum commits, admission rejection before
files, interrupted reception without successful acknowledgment, and retained
orphan charges. The test admission provider is a conservative image-only budget;
it does not implement the missing production-wide disk policy. A three-process qualification also pauses after a real payload chunk is written,
kills that receiver, rejoins its original files and endpoints, kills the leader,
and restarts all original engines. Every payload and exact retry index must
survive. The parent carries only test control requests; TLS carries both Raft
traffic and snapshots. Throughput measurements remain separate work.

Tracking: #378.
