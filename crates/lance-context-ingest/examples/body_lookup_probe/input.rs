use anyhow::{ensure, Context, Result};
use lance_context_core::annotation_staging::Limits;
use lance_context_ingest::{
    annotations::Shard, application::config::WalConnection, turn_ingestion::TurnIngestionSpec,
    TableBinding,
};
use lance_context_wal::replication::{publication::Authority, query::Cut};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::Read, path::Path};
use uuid::Uuid;

/// Deployment-authorized input, not a discovery mechanism or owner config.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub wal: WalConnection,
    pub binding: TableBinding,
    pub shard: Shard,
    pub staging: Limits,
    pub spec: TurnIngestionSpec,
    pub uri: String,
    #[serde(default)]
    pub storage_options: HashMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    pub capture_id: Uuid,
    pub owner: Authority,
    pub incarnation: Uuid,
    pub schema: [u8; 32],
    pub source_group: u64,
    pub bucket: u32,
    pub base_version: u64,
    pub base_fragment_ids: Vec<u64>,
    pub window: Window,
    pub requested_keys: usize,
    pub staged_keys: usize,
    pub primary_key_columns: Vec<String>,
    pub keys: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Window {
    pub first_record: u64,
    pub last_record: u64,
    pub requests: usize,
    pub payload_bytes: u64,
    pub speculative: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub capture_id: Uuid,
    pub cluster: Uuid,
    pub layout: [u8; 32],
    pub table: Uuid,
    pub incarnation: Uuid,
    pub schema: [u8; 32],
    pub source_group: u64,
    pub bucket: u32,
    pub observed: Cut,
    pub manifest: lance_context_wal::replication::publication::ManifestRecord,
}

pub fn read<T: DeserializeOwned>(path: &Path, max: u64) -> Result<T> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= max,
        "invalid input size/type"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "input must be private (0600)"
        );
    }
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= max, "input grew past limit");
    // Do not leak keys or credential values in serde error messages.
    serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("invalid input JSON"))
}

impl Config {
    pub fn validate_capture(&self, capture: &Capture) -> Result<()> {
        self.wal.layout.validate()?;
        self.staging.validate()?;
        let layout = &self.wal.layout;
        let a = &capture.owner;
        ensure!(
            a.cluster == layout.cluster
                && a.layout == layout.fingerprint()?
                && a.group == *layout.groups.first_key_value().context("empty layout")?.0
                && a.table == self.binding.table
                && !a.owner.is_nil()
                && a.epoch > 0
                && a.epoch <= a.grant_index
                && a.grant_index < u64::MAX
                && capture.incarnation == self.binding.incarnation
                && !capture.incarnation.is_nil()
                && !capture.capture_id.is_nil()
                && capture.bucket == self.shard.bucket
                && capture.source_group == layout.group_for_bucket(self.shard.bucket)?
                && capture.schema
                    == lance_context_ingest::annotations::single_body::contract(
                        &self.binding,
                        &self.spec,
                        &self.staging,
                        self.shard
                    )?
                    .schema
                && capture.primary_key_columns == self.spec.primary_key()
                && capture.base_version > 0
                && capture.base_version < u64::MAX,
            "capture does not match authorized binding/spec"
        );
        ensure!(
            !capture.keys.is_empty()
                && capture.staged_keys.checked_add(capture.keys.len())
                    == Some(capture.requested_keys)
                && capture.keys.windows(2).all(|p| p[0] < p[1])
                && capture
                    .base_fragment_ids
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == capture.base_fragment_ids.len()
                && capture.window.first_record > 0
                && capture.window.first_record <= capture.window.last_record
                && capture.window.requests > 0
                && capture.window.payload_bytes > 0,
            "incomplete or noncanonical capture"
        );
        let routing_index = self
            .spec
            .primary_key()
            .iter()
            .position(|key| key == self.spec.routing_column())
            .context("routing key missing")?;
        for key in &capture.keys {
            ensure!(key.len() <= 64 << 10, "oversized captured key");
            let values: Vec<&str> =
                serde_json::from_str(key).map_err(|_| anyhow::anyhow!("invalid captured key"))?;
            ensure!(
                self.spec.business_key(&values, 64 << 10)? == *key,
                "noncanonical captured key"
            );
            ensure!(
                Shard::for_session(layout, &self.binding, values[routing_index])? == self.shard,
                "captured key belongs to another shard"
            );
        }
        Ok(())
    }
}

impl Record {
    pub fn validate(&self, config: &Config, capture: &Capture) -> Result<()> {
        config.validate_capture(capture)?;
        ensure!(
            self.capture_id == capture.capture_id
                && self.cluster == capture.owner.cluster
                && self.layout == capture.owner.layout
                && self.table == capture.owner.table
                && self.incarnation == capture.incarnation
                && self.schema == capture.schema
                && self.source_group == capture.source_group
                && self.bucket == capture.bucket
                && self.manifest.version == capture.base_version
                && self.observed.cluster == self.cluster
                && self.observed.layout == self.layout
                && self.observed.group == capture.owner.group
                && self.observed.term > 0
                && self.observed.quorum_index > 0
                && self.observed.applied >= self.observed.quorum_index
                && self.manifest.index <= self.observed.applied,
            "foreign manifest record or observation"
        );
        // Validate relative object path and binding without opening a dataset.
        lance_context_wal::replication::publication::lance::PinnedManifest::new(
            object_store::path::Path::from("probe-validation"),
            self.cluster,
            self.layout,
            self.table,
            self.incarnation,
            &self.manifest,
        )?;
        Ok(())
    }
}
