//! Independent Rust WAL experiments owned by lance-context.
//!
//! The optional `lab` feature builds Arrow IPC and raft-engine component
//! benchmarks. It is not connected to production ingestion and makes no
//! distributed durability or end-to-end Lance throughput claim.
//! `placement` exposes immutable session routing and scoped receipt types.

#[cfg(feature = "placement")]
pub mod placement;

#[cfg(feature = "raft-log")]
pub mod raft_log;

#[cfg(feature = "replication")]
pub mod replication;

#[cfg(feature = "service")]
pub mod service;

#[cfg(feature = "admission")]
pub mod admission;

#[cfg(feature = "ipc")]
pub mod ipc;

#[cfg(feature = "record")]
pub mod record;

#[cfg(feature = "contract")]
pub mod ingest_contract;

#[cfg(feature = "vectored-io")]
pub mod transport_io;

/// Local recoverable client state with bounded shared durable barriers.
#[cfg(feature = "client-journal")]
pub mod client_journal;
