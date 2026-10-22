use serde::{Deserialize, Serialize};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Payloads may be read from any configured replica. The selected record still
/// has to match the durable source pin and pass payload/schema validation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceReadPolicy {
    #[default]
    FirstVoter,
    /// Spread initial requests by immutable group/index, without hashing or
    /// copying the payload. This is a read preference, not durable sharding;
    /// changing the hash implementation cannot change the publication plan.
    Spread,
}

impl SourceReadPolicy {
    pub(super) fn candidates(
        self,
        voters: &[u64],
        group: u64,
        index: u64,
    ) -> impl Iterator<Item = u64> + '_ {
        let start = if self == Self::Spread && voters.len() > 1 {
            // Raw index modulo replica count can concentrate reads when data
            // entries interleave with control entries at a regular stride.
            let mut hash = DefaultHasher::new();
            (group, index).hash(&mut hash);
            (hash.finish() % voters.len() as u64) as usize
        } else {
            0
        };
        voters
            .iter()
            .cycle()
            .skip(start)
            .take(voters.len())
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_indices_spread_first_requests_and_keep_each_fallback_once() {
        let voters = [7, 19, 43];
        let mut counts = [0usize; 3];
        for index in (3..=9000).step_by(3) {
            let order = SourceReadPolicy::Spread
                .candidates(&voters, 10, index)
                .collect::<Vec<_>>();
            counts[voters.iter().position(|n| *n == order[0]).unwrap()] += 1;
            let mut set = order;
            set.sort_unstable();
            assert_eq!(set, voters, "a fallback was repeated or omitted");
        }
        // Deliberately broad: a regression to index % 3 sends all 3000 reads
        // to one voter. We do not promise exact balance or a fixed hash mapping.
        assert!(counts.into_iter().all(|n| (500..1500).contains(&n)));
    }

    #[test]
    fn old_configuration_keeps_original_preference() {
        let options: super::super::Options = serde_json::from_str("{} ").unwrap();
        assert_eq!(options.source_read_policy, SourceReadPolicy::FirstVoter);
        assert_eq!(
            options
                .source_read_policy
                .candidates(&[7, 19, 43], 10, u64::MAX - 1)
                .collect::<Vec<_>>(),
            [7, 19, 43]
        );
    }
}
