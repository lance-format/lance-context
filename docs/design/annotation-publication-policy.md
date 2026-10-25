# Annotation publication policy

Annotation patches require an explicit `TurnAnnotations` ingestion binding.
An immutable append binding continues to use `ImmutableUniqueIds`; changing a
publisher's policy cannot reinterpret its existing source records. The existing
full-contract comparison rejects such a takeover.

The catalog accepts sourced annotation preparation as inline descriptor format 4
or a destination-store metadata reference in format 5. It rejects both formats
for immutable append tables and for source-free maintenance,
and rejects append descriptors for annotation sources. The same checks run when
restoring catalog checkpoints. Formats 2 and 3 remain source-free maintenance.
Recording preparation still does not advance source coverage or authorize GC.

This extends the existing prepared metadata envelope, not the WAL record file
format. Record payloads stay opaque to replicas. The annotation materializer must
validate the native narrow-column update, exact base and source cut before
publishing through the existing managed manifest protocol. A descriptor digest
binds the recorded file selection; it does not prove patch application.

Upgrade every voter and application worker before using the new policy or
descriptor format. Old binaries reject the new values. Existing append policy
bytes and descriptor formats are unchanged; no annotation path is enabled by
default. This change alone provides neither a mutation endpoint nor an ACK,
read-visibility or recovery guarantee for annotation values.
