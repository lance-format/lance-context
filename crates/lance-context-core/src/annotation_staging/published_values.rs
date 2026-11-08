//! Reuse already-published narrow values within the existing session cache.
use super::*;
use lance::cache::{CacheKey, LanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use std::borrow::Cow;

#[derive(Clone)]
pub(super) struct PublishedValues(Option<LanceCache>);

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
    pub fn new(base: &Dataset) -> Self {
        let session = base.session();
        let cache = session.file_metadata_cache();
        // A custom backend without a reported byte bound is not a suitable
        // value cache. Do not allocate a new budget or guess its capacity.
        Self(cache.capacity_bytes().filter(|bytes| *bytes > 0).map(|_| {
            // Separate owners/views can share a Session and even a URI. Only
            // this staging instance may update these current-base values.
            cache.with_key_prefix(&format!("annotation-values/{}", uuid::Uuid::new_v4()))
        }))
    }

    pub async fn get(&self, turn: &str) -> Option<Arc<Cached>> {
        match &self.0 {
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

    /// Cache cold point results only for their exact immutable base. This uses
    /// the same bounded session cache and staging-instance prefix as published
    /// values, but a separate key type: a delayed old read cannot overwrite the
    /// current value or hide a later insertion. Entries own no base/reader pins.
    pub async fn read_base(
        &self,
        base: &Dataset,
        turn: &str,
        value_bytes: usize,
    ) -> Result<Option<TurnAnnotation>> {
        let Some(cache) = &self.0 else {
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
        Ok(value.value.clone())
    }

    pub async fn remember(&self, values: &Values, through: u64) -> Result<()> {
        let Some(cache) = &self.0 else {
            return Ok(());
        };
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
