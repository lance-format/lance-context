//! Reuse already-published narrow values within the existing session cache.
use super::*;
use lance::cache::{CacheKey, LanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use std::borrow::Cow;

#[derive(Clone)]
pub(super) struct PublishedValues {
    cache: Option<LanceCache>,
    // Serialize current-value insertion with publication warming, without
    // retaining a staging view or holding a lock during native reads.
    current_cut: Arc<Mutex<u64>>,
}

pub(super) struct Cached {
    pub value: TurnAnnotation,
    through: u64,
    charge: usize,
}

impl DeepSizeOf for Cached {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        // Includes conservative string, key and collection-node overhead.
        self.charge
    }
}

struct Key<'a>(&'a str);

impl CacheKey for Key<'_> {
    type ValueType = Cached;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.0)
    }

    fn type_name() -> &'static str {
        "lance-context.published-annotation"
    }
}

struct SnapshotKey<'a> {
    version: u64,
    turn: &'a str,
}

impl CacheKey for SnapshotKey<'_> {
    type ValueType = SnapshotValue;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{}/{}", self.version, self.turn))
    }

    fn type_name() -> &'static str {
        "lance-context.snapshot-annotation"
    }
}

struct SnapshotValue {
    value: Option<TurnAnnotation>,
    charge: usize,
}

impl DeepSizeOf for SnapshotValue {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.charge
    }
}

impl PublishedValues {
    pub fn new(base: &Dataset, materialized: u64) -> Self {
        let session = base.session();
        let cache = session.file_metadata_cache();
        // A custom backend without a reported byte bound is not a suitable
        // value cache. Do not allocate a new budget or guess its capacity.
        let cache = cache.capacity_bytes().filter(|bytes| *bytes > 0).map(|_| {
            // Separate owners/views can share a Session and even a URI. Only
            // this staging instance may update these current-base values.
            cache.with_key_prefix(&format!("annotation-values/{}", uuid::Uuid::new_v4()))
        });
        Self {
            cache,
            current_cut: Arc::new(Mutex::new(materialized)),
        }
    }

    pub async fn get(&self, turn: &str) -> Option<Arc<Cached>> {
        match &self.cache {
            Some(cache) => cache.get_with_key(&Key(turn)).await,
            None => None,
        }
    }

    pub async fn get_at(&self, turn: &str, materialized: u64) -> Option<Arc<Cached>> {
        let value = self.get(turn).await?;
        // Warming precedes the base switch, and an older read may resolve after
        // later publications. Never substitute a future value for its base.
        (value.through <= materialized).then_some(value)
    }

    /// Exact-snapshot entries coalesce cold reads, including absent values.
    /// Promote a present value only while its materialized cut is still current.
    /// Publication replaces modified keys before retiring their frozen values;
    /// unrelated publications can therefore keep reusing this known value.
    /// Both caches share the existing byte bound and own no base/reader pins.
    pub async fn read_base(
        &self,
        base: &Dataset,
        turn: &str,
        materialized: u64,
        value_bytes: usize,
    ) -> Result<Option<TurnAnnotation>> {
        let Some(cache) = &self.cache else {
            return super::read_base(base, turn, value_bytes).await;
        };
        let value = cache
            .get_or_insert_with_key(
                SnapshotKey {
                    version: base.version_id(),
                    turn,
                },
                || async {
                    let value = super::read_base(base, turn, value_bytes).await?;
                    let charge = match &value {
                        Some(value) => charge(turn, value)?,
                        None => 512 + turn.len(),
                    };
                    Ok(SnapshotValue { value, charge })
                },
            )
            .await?;
        // Cache insertion must never queue a read behind publication. If the
        // cut changed during native IO, retain only the exact-snapshot entry.
        // An absent result also stays snapshot-scoped so later creates remain
        // visible. The guard excludes interleaving with publication warming.
        if let Some(current) = &value.value {
            if let Ok(cut) = self.current_cut.try_lock() {
                if *cut == materialized {
                    cache
                        .insert_with_key(
                            &Key(turn),
                            Arc::new(Cached {
                                charge: charge(turn, current)?,
                                value: current.clone(),
                                through: materialized,
                            }),
                        )
                        .await;
                }
            }
        }
        Ok(value.value.clone())
    }

    pub async fn remember(&self, values: &Values, through: u64) -> Result<()> {
        let Some(cache) = &self.cache else {
            return Ok(());
        };
        let mut cut = self.current_cut.lock().await;
        // Fence old cold reads before the first insertion, even if warming is
        // cancelled. Frozen/active values mask partial warming until the same
        // publication resumes and completes its original base handoff.
        *cut = through;
        for (turn, stored) in values {
            cache
                .insert_with_key(
                    &Key(turn),
                    Arc::new(Cached {
                        charge: charge(turn, &stored.value)?,
                        value: stored.value.clone(),
                        through,
                    }),
                )
                .await;
        }
        Ok(())
    }
}
