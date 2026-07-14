//! Identifier helpers for context and rollout records.
//!
//! Records are keyed by string ids that callers frequently generate on the
//! write path. Using a **time-ordered** UUID (rather than a fully random
//! [`Uuid::new_v4`]) means the lexical/byte order of ids matches creation
//! order, which keeps appends clustered at the tail of an index instead of
//! scattering them randomly — friendlier to Lance's on-write indexing and to
//! any downstream B-tree/LSM key space.
//!
//! UUIDv7 (RFC 9562) lays out a 48-bit Unix-millisecond timestamp in the high
//! bits followed by random bits, so ids are sortable by time. Ordering across
//! the same millisecond (and across processes/machines) is only approximate:
//! the sub-millisecond bits are random, so two ids minted in the same
//! millisecond have no guaranteed relative order.

use uuid::Uuid;

/// Generate a new time-ordered [`UUIDv7`](Uuid::now_v7) as a hyphenated string.
///
/// The result sorts (lexically) roughly in creation order, making it a good
/// default primary key for append-heavy tables such as the rollout store.
///
/// ```
/// let id = lance_context_core::generate_id();
/// assert_eq!(id.len(), 36); // 32 hex digits + 4 hyphens
/// ```
#[must_use]
pub fn generate_id() -> String {
    new_uuid().to_string()
}

/// Generate a new time-ordered [`UUIDv7`](Uuid::now_v7).
///
/// Prefer [`generate_id`] when a `String` id is what you ultimately need; this
/// variant is for callers that want to keep the raw [`Uuid`] (e.g. to compare
/// or to encode differently).
#[must_use]
pub fn new_uuid() -> Uuid {
    Uuid::now_v7()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_id_is_a_hyphenated_uuid() {
        let id = generate_id();
        // Round-trips as a valid UUID and is the canonical hyphenated form.
        let parsed = Uuid::parse_str(&id).expect("valid uuid string");
        assert_eq!(parsed.to_string(), id);
        assert_eq!(parsed.get_version_num(), 7);
    }

    #[test]
    fn ids_are_unique() {
        let a = generate_id();
        let b = generate_id();
        assert_ne!(a, b);
    }

    #[test]
    fn ids_sort_in_creation_order_across_milliseconds() {
        // Two ids minted a millisecond apart must sort in creation order,
        // because UUIDv7 places the millisecond timestamp in the high bits.
        let earlier = new_uuid();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let later = new_uuid();
        assert!(earlier < later, "{earlier} should sort before {later}");
        assert!(earlier.to_string() < later.to_string());
    }
}
