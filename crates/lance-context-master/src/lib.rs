//! Control-plane (master) library surface.

pub mod admission;
pub mod catchup;
pub mod config;
pub mod discovery;
pub mod error;
mod maintenance_execution;
mod merge_execution;
mod resident_recovery;
pub mod rollout_append;
pub mod routes;
pub mod scanner;
pub mod scheduler;
pub mod state;
pub mod stats_store;
pub mod task_store;
pub mod wal_tail;
