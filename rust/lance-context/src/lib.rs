//! Core types for the lance-context storage layer.

mod context;
mod record;
mod store;

pub use context::{Context, ContextEntry, Snapshot};
pub use record::{ContextRecord, StateMetadata};
pub use store::ContextStore;
