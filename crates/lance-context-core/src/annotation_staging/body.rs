//! An immutable body reference bound to the exact narrow preparation.
use super::*;

pub const BODY_SNAPSHOT_PROPERTY: &str = "lance-context.annotation-body";
const MAX_ENCODED_BYTES: usize = 32 << 10;

/// Locate the body using the same view as the current narrow annotation.
pub enum TurnBodySource<'a> {
    /// The caller retains the decoded source frame under its source-window
    /// gate. This locator by itself neither pins WAL nor prevents source GC.
    Unmaterialized(NewTurnSource),
    /// Follow the body snapshot reference in this exact narrow base. Retain the
    /// TurnRead until the response owns its body data and annotation values.
    Materialized(&'a Dataset),
}

pub(super) enum CapturedBody {
    Staged(NewTurnSource),
    Base(Arc<views::Base>),
}

/// A full-turn capture consumes one of the existing reader permits. A staged
/// annotation can still need a materialized body; that case pins its compatible
/// base too. Annotation-only PointRead keeps its existing narrower lifetime.
pub struct TurnRead {
    pub(super) annotations: PointRead,
    pub(super) source: CapturedBody,
}

impl TurnRead {
    pub fn annotations(&self) -> &PointRead {
        &self.annotations
    }

    pub fn body_source(&self) -> TurnBodySource<'_> {
        match &self.source {
            CapturedBody::Staged(source) => TurnBodySource::Unmaterialized(*source),
            CapturedBody::Base(base) => TurnBodySource::Materialized(base),
        }
    }
}

/// Native body snapshot selected together with a narrow annotation cut.
///
/// The publisher supplies the authorized destination and logical table
/// incarnation, verifies source coverage and retains both datasets' files.
/// This identity alone is neither a durability receipt nor a source-GC proof.
/// Never follow a URI from untrusted input without checking its table binding.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodySnapshot {
    pub table_instance: String,
    pub uri: String,
    pub version: u64,
    pub transaction: String,
}

impl BodySnapshot {
    /// Capture identity without scanning or copying body rows. Both mainline
    /// bootstrap snapshots and native detached snapshots are supported.
    pub async fn capture(table_instance: String, dataset: &Dataset) -> Result<Self> {
        let value = Self {
            table_instance,
            uri: dataset.uri().to_string(),
            version: dataset.version_id(),
            transaction: dataset
                .read_transaction()
                .await?
                .ok_or_else(|| invalid("body snapshot has no native transaction"))?
                .uuid,
        };
        value.validate()?;
        Ok(value)
    }

    /// Read the reference from this exact narrow snapshot. This does not open
    /// its body URI or establish authority over that destination.
    pub async fn read(narrow: &Dataset) -> Result<Option<Self>> {
        let Some(transaction) = narrow.read_transaction().await? else {
            return Ok(None);
        };
        let properties = transaction.transaction_properties.as_ref();
        let Some(encoded) = properties.and_then(|p| p.get(BODY_SNAPSHOT_PROPERTY)) else {
            return Ok(None);
        };
        if encoded.len() > MAX_ENCODED_BYTES {
            return Err(invalid("body snapshot metadata exceeds its bound"));
        }
        let value: Self = serde_json::from_str(encoded).map_err(|e| invalid(e.to_string()))?;
        value.validate()?;
        let cut = properties
            .and_then(|p| p.get("lance-context.annotation-cut"))
            .ok_or_else(|| invalid("body reference has no annotation cut"))?;
        let (scope, through): (Scope, u64) =
            serde_json::from_str(cut).map_err(|e| invalid(e.to_string()))?;
        // A trusted bootstrap may bind its initial pair before any WAL data.
        if value.table_instance != scope.table_instance || through == u64::MAX {
            return Err(invalid("body reference differs from its annotation scope"));
        }
        Ok(Some(value))
    }

    pub(super) fn validate(&self) -> Result<()> {
        let transaction = uuid::Uuid::parse_str(&self.transaction).ok();
        if self.table_instance.is_empty()
            || self.table_instance.len() > 1024
            || self.uri.is_empty()
            || self.uri.len() > 4096
            || self.version == 0
            || self.transaction.len() != 36
            || transaction.is_none_or(|id| id.is_nil())
        {
            return Err(invalid("invalid body snapshot identity"));
        }
        Ok(())
    }

    pub(super) fn retained_bytes(&self) -> Result<usize> {
        self.table_instance
            .capacity()
            .checked_add(self.uri.capacity())
            .and_then(|n| n.checked_add(self.transaction.capacity()))
            .and_then(|n| n.checked_add(512))
            .ok_or_else(|| invalid("body reference allocation overflow"))
    }

    pub(super) fn encode(&self) -> Result<String> {
        self.validate()?;
        let value = serde_json::to_string(self).map_err(|e| invalid(e.to_string()))?;
        if value.len() > MAX_ENCODED_BYTES {
            return Err(invalid("body snapshot metadata exceeds its bound"));
        }
        Ok(value)
    }
}
