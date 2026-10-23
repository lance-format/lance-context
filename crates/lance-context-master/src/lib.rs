//! Control-plane (master) library surface.
#![recursion_limit = "256"]

pub mod admission;
pub mod catchup;
pub mod config;
pub mod demand_publish;
pub mod planner;
pub use lance_context_merge::demand;
pub mod discovery;
pub mod eligibility;
pub mod error;
pub mod executors;
pub mod legacy_retirement;
pub mod local_merge;
mod maintenance_execution;
mod merge_capacity;
mod merge_execution;
mod resident_recovery;
pub mod rollout_append;
pub mod routes;
pub mod scanner;
pub mod scheduler;
mod scheduler_wakeup;
pub mod scoring;
pub mod state;
pub mod stats_store;
pub mod task_store;
pub mod wal_tail;
