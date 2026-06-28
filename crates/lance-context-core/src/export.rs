//! Curate stored [`ContextRecord`]s into trainable datasets and export them as
//! SFT / preference / RL-rollout JSONL plus a reproducible manifest.
//!
//! This is the downstream half of the post-training pipeline: raw logs are
//! ingested into faithful `ContextRecord`s, then *curated* (lifecycle-correct
//! filtering, semantic dedup, decontamination, reward thresholding), grouped
//! into conversations/trajectories, and *exported* into the shape a trainer
//! expects.
//!
//! ## Field conventions
//!
//! Reward / preference signals live in each record's free-form `metadata`
//! object:
//! - `metadata.reward` (number) — scalar reward / verifier score.
//! - `metadata.reward_source` (string) — e.g. `"verifier"`, `"tests"`,
//!   `"judge"`.
//! - `metadata.group_id` (string) — links the N samples generated for one
//!   prompt (GRPO/RLVR groups, candidate sets).
//! - `metadata.label` (string `"chosen"`/`"rejected"`) — unpaired (KTO-style)
//!   preference label.
//! - `metadata.rank` (integer, 1 = best) — N-way ranking position.
//!
//! Messages are built from each record's `role` + `text_payload`.
//!
//! ## Curation lifecycle
//!
//! Curation is *selection only* — it never deletes or tombstones records. By
//! default it keeps only records visible under default lifecycle (drops
//! tombstoned/expired/retired/superseded) and additionally drops
//! `contradicted` records.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufWriter, Write};

use chrono::{DateTime, Utc};
use lance::{Error as LanceError, Result as LanceResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::record::{ContextRecord, LifecycleQueryOptions, RecordFilters, LIFECYCLE_CONTRADICTED};
use crate::store::ContextStore;

/// Schema version stamped into every export manifest.
pub const EXPORT_SCHEMA_VERSION: &str = "1";

/// Which training shape to export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ExportTask {
    /// Supervised fine-tuning (also the rejection-sampling / Best-of-N target).
    #[default]
    Sft,
    /// Preference optimization (DPO/SimPO/ORPO paired, KTO unpaired, judge N-way).
    Preference,
    /// RL rollout (PPO/GRPO/RLVR): prompt + response group(s) with rewards.
    Rollout,
}

impl ExportTask {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sft => "sft",
            Self::Preference => "preference",
            Self::Rollout => "rollout",
        }
    }
}

/// Shape of a preference export (selected by the caller to match their labels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PreferenceForm {
    /// Paired `chosen`/`rejected` (DPO/SimPO/ORPO).
    #[default]
    Paired,
    /// One completion with an unpaired binary label (KTO).
    Unpaired,
    /// N-way ranked candidate list (LLM-judge / RULER).
    Ranked,
}

/// How records are grouped into one training example.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum GroupBy {
    /// One example per record.
    None,
    #[default]
    SessionId,
    RunId,
    Tenant,
    Source,
    BotId,
    /// Group by the `external_id` prefix up to the first occurrence of the
    /// delimiter (e.g. `"doc-7#chunk-1"` -> `"doc-7"` with delimiter `"#"`).
    ExternalIdPrefix(String),
}

impl GroupBy {
    fn label(&self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::SessionId => "session_id".to_string(),
            Self::RunId => "run_id".to_string(),
            Self::Tenant => "tenant".to_string(),
            Self::Source => "source".to_string(),
            Self::BotId => "bot_id".to_string(),
            Self::ExternalIdPrefix(delim) => format!("external_id_prefix:{delim}"),
        }
    }

    /// Group key for a record; `None` records become their own singleton group.
    fn key(&self, record: &ContextRecord) -> String {
        let value = match self {
            Self::None => None,
            Self::SessionId => record.session_id.clone(),
            Self::RunId => Some(record.run_id.clone()),
            Self::Tenant => record.tenant.clone(),
            Self::Source => record.source.clone(),
            Self::BotId => record.bot_id.clone(),
            Self::ExternalIdPrefix(delim) => record.external_id.as_ref().map(|external_id| {
                external_id
                    .split_once(delim.as_str())
                    .map_or(external_id.as_str(), |(prefix, _)| prefix)
                    .to_string()
            }),
        };
        value.unwrap_or_else(|| format!("__rec__{}", record.id))
    }
}

/// Reproducible train/eval split configuration.
///
/// Each record is assigned to train or eval by a stable hash of its `by`
/// grouping key plus `seed`, so no group (e.g. session) spans both sides and
/// the same `seed` reproduces the identical partition.
#[derive(Clone)]
pub struct SplitConfig {
    /// Fraction of groups assigned to the eval side, in `0.0..=1.0`.
    pub eval_fraction: f64,
    /// Grouping key the split is disjoint on (defaults to `session_id`).
    pub by: GroupBy,
    pub seed: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            eval_fraction: 0.1,
            by: GroupBy::SessionId,
            seed: 0,
        }
    }
}

/// Curation + export configuration.
#[derive(Clone)]
pub struct ExportConfig {
    pub task: ExportTask,
    pub group_by: GroupBy,
    pub preference_form: PreferenceForm,
    /// Metadata/quality filters applied before lifecycle curation.
    pub filters: Option<RecordFilters>,
    /// Lifecycle visibility; defaults to visible-only.
    pub lifecycle: LifecycleQueryOptions,
    /// Collapse near-duplicates whose cosine distance is `<=` this threshold.
    pub dedup_threshold: Option<f32>,
    /// Holdout embeddings to decontaminate against.
    pub decontaminate_against: Vec<Vec<f32>>,
    /// Drop records within this cosine distance of any holdout embedding.
    pub decontaminate_threshold: Option<f32>,
    /// Drop records whose `metadata.reward` is present and below this value
    /// (rejection sampling / Best-of-N / quality threshold).
    pub min_reward: Option<f64>,
    /// Pin the export to a dataset version (time-travel); restored afterward.
    pub version: Option<u64>,
    /// Optional JSON summary of the applied selectors, recorded in the manifest.
    pub filters_summary: Option<Value>,
    /// When set, write group-disjoint `train` / `eval` outputs instead of one.
    pub split: Option<SplitConfig>,
    /// When `true`, also write a `<output_path>.stats.json` dataset report.
    pub emit_stats: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            task: ExportTask::Sft,
            group_by: GroupBy::SessionId,
            preference_form: PreferenceForm::Paired,
            filters: None,
            lifecycle: LifecycleQueryOptions::default(),
            dedup_threshold: None,
            decontaminate_against: Vec::new(),
            decontaminate_threshold: None,
            min_reward: None,
            version: None,
            filters_summary: None,
            split: None,
            emit_stats: false,
        }
    }
}

/// A single chat message in a training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Where an exported example came from, for auditability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub context_uri: String,
    pub version: u64,
    pub record_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub external_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tenant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bot_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub run_id: Option<String>,
    pub created_at_start: DateTime<Utc>,
    pub created_at_end: DateTime<Utc>,
}

/// SFT example: an ordered message list (doubles as the rejection-sampling /
/// Best-of-N target when filtered by `min_reward`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftExample {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reward: Option<f64>,
    pub provenance: Provenance,
}

/// One ranked candidate in an N-way preference example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub messages: Vec<Message>,
    pub rank: i64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reward: Option<f64>,
}

/// Preference example in one of three forms, tagged by `form`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "lowercase")]
pub enum PreferenceExample {
    Paired {
        prompt: Vec<Message>,
        chosen: Vec<Message>,
        rejected: Vec<Message>,
        provenance: Provenance,
    },
    Unpaired {
        prompt: Vec<Message>,
        completion: Vec<Message>,
        /// `true` = chosen, `false` = rejected.
        label: bool,
        provenance: Provenance,
    },
    Ranked {
        prompt: Vec<Message>,
        candidates: Vec<RankedCandidate>,
        provenance: Provenance,
    },
}

/// One response within a rollout example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutResponse {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reward: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reward_source: Option<String>,
}

/// RL rollout example: a prompt plus one or more rewarded responses, with an
/// optional `group_id` linking the N samples for one prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutExample {
    pub prompt: Vec<Message>,
    pub responses: Vec<RolloutResponse>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub group_id: Option<String>,
    pub provenance: Provenance,
}

/// Record counts at each curation stage.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ExportCounts {
    pub input_records: usize,
    pub after_lifecycle: usize,
    pub after_dedup: usize,
    pub after_decontaminate: usize,
    pub after_reward_filter: usize,
    pub examples: usize,
}

/// Records which side of a train/eval split an output is, and the parameters
/// needed to reproduce the partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitManifest {
    /// `"train"` or `"eval"`.
    pub side: String,
    pub eval_fraction: f64,
    pub by: String,
    pub seed: u64,
    /// Path of the complementary (other side's) output.
    pub complement_path: String,
}

/// Reproducible description of an export, written next to the JSONL output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportManifest {
    pub context_uri: String,
    pub version: u64,
    pub task: String,
    pub group_by: String,
    pub schema_version: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub preference_form: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub filters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dedup_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub decontaminate_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub min_reward: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub split: Option<SplitManifest>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created_at_start: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created_at_end: Option<DateTime<Utc>>,
    pub source_record_ids: Vec<String>,
    pub counts: ExportCounts,
}

/// Numeric distribution summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Distribution {
    pub count: usize,
    pub min: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
    pub mean: f64,
}

impl Distribution {
    fn from_sorted(mut values: Vec<f64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let percentile = |p: f64| {
            let idx = ((p * (count - 1) as f64).round() as usize).min(count - 1);
            values[idx]
        };
        Some(Self {
            count,
            min: values[0],
            median: percentile(0.5),
            p95: percentile(0.95),
            max: values[count - 1],
            mean: sum / count as f64,
        })
    }
}

/// Token-length statistics and which source produced them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStats {
    #[serde(flatten)]
    pub distribution: Distribution,
    /// `"tokens_used"`, `"length_proxy"`, or `"mixed"`.
    pub source: String,
}

/// Source records excluded during curation, by reason.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ExcludedCounts {
    pub lifecycle: usize,
    pub reward_threshold: usize,
    pub dedup: usize,
    pub decontaminate: usize,
}

/// Auditable dataset statistics for one export output, written to
/// `<output_path>.stats.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStats {
    pub task: String,
    pub examples: usize,
    pub num_groups: usize,
    /// Record counts by `role`.
    pub by_role: BTreeMap<String, usize>,
    /// Record counts by `source` (`"__none__"` when absent).
    pub by_source: BTreeMap<String, usize>,
    /// Record counts by `tenant` (`"__none__"` when absent).
    pub by_tenant: BTreeMap<String, usize>,
    pub records_per_group: Distribution,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tokens: Option<TokenStats>,
    pub excluded: ExcludedCounts,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub preference_form: Option<String>,
    /// Reward distribution over records carrying `metadata.reward`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reward: Option<Distribution>,
    /// Counts by `metadata.reward_source`.
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub reward_sources: BTreeMap<String, usize>,
}

// ----- metadata helpers -----------------------------------------------------

fn metadata_field<'a>(record: &'a ContextRecord, key: &str) -> Option<&'a Value> {
    record.metadata.as_ref()?.get(key)
}

fn record_reward(record: &ContextRecord) -> Option<f64> {
    metadata_field(record, "reward")?.as_f64()
}

fn record_reward_source(record: &ContextRecord) -> Option<String> {
    Some(
        metadata_field(record, "reward_source")?
            .as_str()?
            .to_string(),
    )
}

fn record_group_id(record: &ContextRecord) -> Option<String> {
    Some(metadata_field(record, "group_id")?.as_str()?.to_string())
}

fn record_label(record: &ContextRecord) -> Option<String> {
    Some(metadata_field(record, "label")?.as_str()?.to_string())
}

fn record_rank(record: &ContextRecord) -> Option<i64> {
    metadata_field(record, "rank")?.as_i64()
}

fn message_of(record: &ContextRecord) -> Message {
    Message {
        role: record.role.clone(),
        content: record.text_payload.clone().unwrap_or_default(),
    }
}

fn is_assistant(record: &ContextRecord) -> bool {
    record.role.eq_ignore_ascii_case("assistant")
}

/// Cosine distance in `0.0..=2.0`; `0` = identical direction. Returns `None`
/// for zero-magnitude or mismatched-length vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(1.0 - dot / (na.sqrt() * nb.sqrt()))
}

// ----- grouping -------------------------------------------------------------

/// Records grouped for one example, already ordered by `(created_at, id)`.
struct Group {
    key: String,
    records: Vec<ContextRecord>,
}

fn group_records(records: Vec<ContextRecord>, group_by: &GroupBy) -> Vec<Group> {
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<ContextRecord>> = HashMap::new();
    for record in records {
        let key = group_by.key(&record);
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(record);
    }

    let mut result: Vec<Group> = order
        .into_iter()
        .map(|key| {
            let mut records = groups.remove(&key).unwrap_or_default();
            records.sort_by(|a, b| {
                a.created_at
                    .cmp(&b.created_at)
                    .then_with(|| a.id.cmp(&b.id))
            });
            Group { key, records }
        })
        .collect();

    // Deterministic group order: by each group's earliest (created_at, id).
    result.sort_by(|a, b| {
        let left = a.records.first();
        let right = b.records.first();
        match (left, right) {
            (Some(l), Some(r)) => l
                .created_at
                .cmp(&r.created_at)
                .then_with(|| l.id.cmp(&r.id)),
            _ => a.key.cmp(&b.key),
        }
    });
    result
}

fn provenance_for(records: &[ContextRecord], context_uri: &str, version: u64) -> Provenance {
    let first = records.first();
    let created_at_start = records
        .iter()
        .map(|r| r.created_at)
        .min()
        .unwrap_or_else(Utc::now);
    let created_at_end = records
        .iter()
        .map(|r| r.created_at)
        .max()
        .unwrap_or(created_at_start);
    Provenance {
        context_uri: context_uri.to_string(),
        version,
        record_ids: records.iter().map(|r| r.id.clone()).collect(),
        external_ids: records
            .iter()
            .filter_map(|r| r.external_id.clone())
            .collect(),
        tenant: first.and_then(|r| r.tenant.clone()),
        source: first.and_then(|r| r.source.clone()),
        bot_id: first.and_then(|r| r.bot_id.clone()),
        session_id: first.and_then(|r| r.session_id.clone()),
        run_id: first.map(|r| r.run_id.clone()),
        created_at_start,
        created_at_end,
    }
}

/// Split a group into leading non-assistant prompt messages and the assistant
/// candidate records that follow.
fn split_prompt_candidates(records: &[ContextRecord]) -> (Vec<Message>, Vec<&ContextRecord>) {
    let mut prompt = Vec::new();
    let mut candidates = Vec::new();
    for record in records {
        if is_assistant(record) {
            candidates.push(record);
        } else if candidates.is_empty() {
            prompt.push(message_of(record));
        } else {
            // Non-assistant turn after a candidate (e.g. tool result) — keep it
            // attached to the conversation prompt context.
            prompt.push(message_of(record));
        }
    }
    (prompt, candidates)
}

fn write_line<T: Serialize>(writer: &mut impl Write, value: &T) -> LanceResult<()> {
    let line = serde_json::to_string(value).map_err(|err| LanceError::io(err.to_string()))?;
    writeln!(writer, "{line}")?;
    Ok(())
}

// ----- curation -------------------------------------------------------------

/// Greedily collapse near-duplicates: keep a record only if its embedding is
/// not within `threshold` cosine distance of an already-kept record. Records
/// without an embedding are always kept.
fn dedup(records: Vec<ContextRecord>, threshold: f32) -> Vec<ContextRecord> {
    let mut kept: Vec<ContextRecord> = Vec::new();
    for record in records {
        let is_dup = record.embedding.as_ref().is_some_and(|embedding| {
            kept.iter().any(|other| {
                other
                    .embedding
                    .as_ref()
                    .and_then(|other_embedding| cosine_distance(embedding, other_embedding))
                    .is_some_and(|distance| distance <= threshold)
            })
        });
        if !is_dup {
            kept.push(record);
        }
    }
    kept
}

/// Drop records whose embedding is within `threshold` cosine distance of any
/// holdout embedding. Records without an embedding are kept.
fn decontaminate(
    records: Vec<ContextRecord>,
    holdout: &[Vec<f32>],
    threshold: f32,
) -> Vec<ContextRecord> {
    records
        .into_iter()
        .filter(|record| match &record.embedding {
            Some(embedding) => !holdout.iter().any(|held| {
                cosine_distance(embedding, held).is_some_and(|distance| distance <= threshold)
            }),
            None => true,
        })
        .collect()
}

/// Apply the curation pipeline (lifecycle/contradicted, reward threshold,
/// dedup, decontamination) and return the curated records plus stage counts.
fn curate(
    records: Vec<ContextRecord>,
    config: &ExportConfig,
) -> (Vec<ContextRecord>, ExportCounts) {
    let mut counts = ExportCounts {
        input_records: records.len(),
        ..ExportCounts::default()
    };

    // `list` already dropped tombstoned/expired/retired/superseded; also drop
    // `contradicted`, which is visible by default.
    let lifecycle: Vec<ContextRecord> = records
        .into_iter()
        .filter(|record| record.lifecycle_status != LIFECYCLE_CONTRADICTED)
        .collect();
    counts.after_lifecycle = lifecycle.len();

    let rewarded: Vec<ContextRecord> = match config.min_reward {
        Some(min) => lifecycle
            .into_iter()
            .filter(|record| record_reward(record).is_none_or(|reward| reward >= min))
            .collect(),
        None => lifecycle,
    };
    counts.after_reward_filter = rewarded.len();

    let deduped = match config.dedup_threshold {
        Some(threshold) => dedup(rewarded, threshold),
        None => rewarded,
    };
    counts.after_dedup = deduped.len();

    let clean = match config.decontaminate_threshold {
        Some(threshold) if !config.decontaminate_against.is_empty() => {
            decontaminate(deduped, &config.decontaminate_against, threshold)
        }
        _ => deduped,
    };
    counts.after_decontaminate = clean.len();

    (clean, counts)
}

// ----- exporters ------------------------------------------------------------

fn write_sft(
    groups: &[Group],
    writer: &mut impl Write,
    context_uri: &str,
    version: u64,
) -> LanceResult<usize> {
    let mut written = 0;
    for group in groups {
        if group.records.is_empty() {
            continue;
        }
        let example = SftExample {
            messages: group.records.iter().map(message_of).collect(),
            reward: group
                .records
                .iter()
                .filter_map(record_reward)
                .reduce(f64::max),
            provenance: provenance_for(&group.records, context_uri, version),
        };
        write_line(writer, &example)?;
        written += 1;
    }
    Ok(written)
}

/// Preference score used to order candidates: an explicit label dominates a
/// numeric reward, which dominates "unscored".
fn preference_score(record: &ContextRecord) -> f64 {
    match record_label(record).as_deref() {
        Some("chosen") => f64::INFINITY,
        Some("rejected") => f64::NEG_INFINITY,
        _ => record_reward(record).unwrap_or(0.0),
    }
}

fn unpaired_label(record: &ContextRecord, min_reward: Option<f64>) -> Option<bool> {
    match record_label(record).as_deref() {
        Some("chosen") => Some(true),
        Some("rejected") => Some(false),
        _ => match (record_reward(record), min_reward) {
            (Some(reward), Some(min)) => Some(reward >= min),
            _ => None,
        },
    }
}

fn write_preference(
    groups: &[Group],
    writer: &mut impl Write,
    form: PreferenceForm,
    min_reward: Option<f64>,
    context_uri: &str,
    version: u64,
) -> LanceResult<usize> {
    let mut written = 0;
    for group in groups {
        let (prompt, candidates) = split_prompt_candidates(&group.records);
        if candidates.is_empty() {
            continue;
        }
        let provenance = provenance_for(&group.records, context_uri, version);

        match form {
            PreferenceForm::Paired => {
                if candidates.len() < 2 {
                    continue;
                }
                let mut best = candidates[0];
                let mut worst = candidates[0];
                for candidate in &candidates {
                    if preference_score(candidate) > preference_score(best) {
                        best = candidate;
                    }
                    if preference_score(candidate) < preference_score(worst) {
                        worst = candidate;
                    }
                }
                if best.id == worst.id {
                    continue; // no signal to separate chosen from rejected
                }
                let example = PreferenceExample::Paired {
                    prompt,
                    chosen: vec![message_of(best)],
                    rejected: vec![message_of(worst)],
                    provenance,
                };
                write_line(writer, &example)?;
                written += 1;
            }
            PreferenceForm::Unpaired => {
                for candidate in &candidates {
                    let Some(label) = unpaired_label(candidate, min_reward) else {
                        continue;
                    };
                    let example = PreferenceExample::Unpaired {
                        prompt: prompt.clone(),
                        completion: vec![message_of(candidate)],
                        label,
                        provenance: provenance.clone(),
                    };
                    write_line(writer, &example)?;
                    written += 1;
                }
            }
            PreferenceForm::Ranked => {
                let mut ordered: Vec<&ContextRecord> = candidates.clone();
                ordered.sort_by(|a, b| {
                    let rank_a = record_rank(a);
                    let rank_b = record_rank(b);
                    match (rank_a, rank_b) {
                        (Some(ra), Some(rb)) => ra.cmp(&rb),
                        _ => preference_score(b).total_cmp(&preference_score(a)),
                    }
                });
                let candidates: Vec<RankedCandidate> = ordered
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| RankedCandidate {
                        messages: vec![message_of(candidate)],
                        rank: record_rank(candidate).unwrap_or((index + 1) as i64),
                        reward: record_reward(candidate),
                    })
                    .collect();
                let example = PreferenceExample::Ranked {
                    prompt,
                    candidates,
                    provenance,
                };
                write_line(writer, &example)?;
                written += 1;
            }
        }
    }
    Ok(written)
}

fn write_rollout(
    groups: &[Group],
    writer: &mut impl Write,
    context_uri: &str,
    version: u64,
) -> LanceResult<usize> {
    let mut written = 0;
    for group in groups {
        let (prompt, candidates) = split_prompt_candidates(&group.records);
        if candidates.is_empty() {
            continue;
        }
        let responses: Vec<RolloutResponse> = candidates
            .iter()
            .map(|candidate| RolloutResponse {
                messages: vec![message_of(candidate)],
                reward: record_reward(candidate),
                reward_source: record_reward_source(candidate),
            })
            .collect();
        let group_id = candidates
            .iter()
            .find_map(|candidate| record_group_id(candidate));
        let example = RolloutExample {
            prompt,
            responses,
            group_id,
            provenance: provenance_for(&group.records, context_uri, version),
        };
        write_line(writer, &example)?;
        written += 1;
    }
    Ok(written)
}

fn summarize_groups(
    groups: &[Group],
) -> (Option<DateTime<Utc>>, Option<DateTime<Utc>>, Vec<String>) {
    let mut start: Option<DateTime<Utc>> = None;
    let mut end: Option<DateTime<Utc>> = None;
    let mut ids = Vec::new();
    for group in groups {
        for record in &group.records {
            start = Some(start.map_or(record.created_at, |s| s.min(record.created_at)));
            end = Some(end.map_or(record.created_at, |e| e.max(record.created_at)));
            ids.push(record.id.clone());
        }
    }
    (start, end, ids)
}

/// Stable 64-bit FNV-1a hash of `seed` + `key`, used for reproducible,
/// platform-independent split assignment.
fn stable_hash(seed: u64, key: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mix = |hash: &mut u64, byte: u8| {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for byte in seed.to_le_bytes() {
        mix(&mut hash, byte);
    }
    for byte in key.as_bytes() {
        mix(&mut hash, *byte);
    }
    hash
}

/// Map a group key to a stable fraction in `0.0..1.0`.
fn split_fraction(seed: u64, key: &str) -> f64 {
    // FNV-1a alone has biased high bits; run the MurmurHash3 fmix64 finalizer
    // for good avalanche before taking the top 53 bits.
    let mut hash = stable_hash(seed, key);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^= hash >> 33;
    (hash >> 11) as f64 / (1u64 << 53) as f64
}

/// Derive `train` / `eval` output paths by inserting the side before the final
/// extension (e.g. `cut.jsonl` -> `cut.train.jsonl`).
fn split_paths(output_path: &str) -> (String, String) {
    let slash = output_path.rfind('/').map_or(0, |i| i + 1);
    match output_path[slash..].rfind('.') {
        Some(rel_dot) => {
            let dot = slash + rel_dot;
            let (stem, ext) = output_path.split_at(dot);
            (format!("{stem}.train{ext}"), format!("{stem}.eval{ext}"))
        }
        None => (
            format!("{output_path}.train"),
            format!("{output_path}.eval"),
        ),
    }
}

/// Group `records`, write the task-shaped JSONL to `output_path`, write the
/// sibling manifest, and return it.
#[allow(clippy::too_many_arguments)]
fn emit_export(
    records: Vec<ContextRecord>,
    config: &ExportConfig,
    context_uri: &str,
    version: u64,
    mut counts: ExportCounts,
    output_path: &str,
    split: Option<SplitManifest>,
) -> LanceResult<ExportManifest> {
    let groups = group_records(records, &config.group_by);
    let (created_at_start, created_at_end, source_record_ids) = summarize_groups(&groups);

    let file = std::fs::File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let examples = match config.task {
        ExportTask::Sft => write_sft(&groups, &mut writer, context_uri, version)?,
        ExportTask::Preference => write_preference(
            &groups,
            &mut writer,
            config.preference_form,
            config.min_reward,
            context_uri,
            version,
        )?,
        ExportTask::Rollout => write_rollout(&groups, &mut writer, context_uri, version)?,
    };
    writer.flush()?;
    counts.examples = examples;

    let manifest = ExportManifest {
        context_uri: context_uri.to_string(),
        version,
        task: config.task.as_str().to_string(),
        group_by: config.group_by.label(),
        schema_version: EXPORT_SCHEMA_VERSION.to_string(),
        preference_form: matches!(config.task, ExportTask::Preference).then(|| {
            match config.preference_form {
                PreferenceForm::Paired => "paired",
                PreferenceForm::Unpaired => "unpaired",
                PreferenceForm::Ranked => "ranked",
            }
            .to_string()
        }),
        filters: config.filters_summary.clone(),
        dedup_threshold: config.dedup_threshold,
        decontaminate_threshold: config.decontaminate_threshold,
        min_reward: config.min_reward,
        split,
        created_at_start,
        created_at_end,
        source_record_ids,
        counts,
    };

    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|err| LanceError::io(err.to_string()))?;
    std::fs::write(format!("{output_path}.manifest.json"), manifest_json)?;

    if config.emit_stats {
        let stats = compute_stats(&groups, &counts, examples, config);
        let stats_json =
            serde_json::to_string_pretty(&stats).map_err(|err| LanceError::io(err.to_string()))?;
        std::fs::write(format!("{output_path}.stats.json"), stats_json)?;
    }

    Ok(manifest)
}

/// Compute the dataset statistics report for one export output.
fn compute_stats(
    groups: &[Group],
    counts: &ExportCounts,
    examples: usize,
    config: &ExportConfig,
) -> ExportStats {
    let mut by_role: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_source: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_tenant: BTreeMap<String, usize> = BTreeMap::new();
    let mut token_values: Vec<f64> = Vec::new();
    let mut used_tokens_used = false;
    let mut used_fallback = false;
    let mut reward_values: Vec<f64> = Vec::new();
    let mut reward_sources: BTreeMap<String, usize> = BTreeMap::new();
    let mut records_per_group: Vec<f64> = Vec::new();

    for group in groups {
        records_per_group.push(group.records.len() as f64);
        for record in &group.records {
            *by_role.entry(record.role.clone()).or_insert(0) += 1;
            *by_source
                .entry(
                    record
                        .source
                        .clone()
                        .unwrap_or_else(|| "__none__".to_string()),
                )
                .or_insert(0) += 1;
            *by_tenant
                .entry(
                    record
                        .tenant
                        .clone()
                        .unwrap_or_else(|| "__none__".to_string()),
                )
                .or_insert(0) += 1;

            match record.state_metadata.as_ref().and_then(|m| m.tokens_used) {
                Some(tokens) if tokens >= 0 => {
                    token_values.push(f64::from(tokens));
                    used_tokens_used = true;
                }
                _ => {
                    // Fallback proxy: whitespace-delimited word count of content.
                    let proxy = record
                        .text_payload
                        .as_deref()
                        .map_or(0, |text| text.split_whitespace().count());
                    token_values.push(proxy as f64);
                    used_fallback = true;
                }
            }

            if let Some(reward) = record_reward(record) {
                reward_values.push(reward);
            }
            if let Some(source) = record_reward_source(record) {
                *reward_sources.entry(source).or_insert(0) += 1;
            }
        }
    }

    let tokens = Distribution::from_sorted(token_values).map(|distribution| TokenStats {
        distribution,
        source: match (used_tokens_used, used_fallback) {
            (true, true) => "mixed",
            (true, false) => "tokens_used",
            _ => "length_proxy",
        }
        .to_string(),
    });

    let excluded = ExcludedCounts {
        lifecycle: counts.input_records.saturating_sub(counts.after_lifecycle),
        reward_threshold: counts
            .after_lifecycle
            .saturating_sub(counts.after_reward_filter),
        dedup: counts
            .after_reward_filter
            .saturating_sub(counts.after_dedup),
        decontaminate: counts
            .after_dedup
            .saturating_sub(counts.after_decontaminate),
    };

    ExportStats {
        task: config.task.as_str().to_string(),
        examples,
        num_groups: groups.len(),
        by_role,
        by_source,
        by_tenant,
        records_per_group: Distribution::from_sorted(records_per_group).unwrap_or_default(),
        tokens,
        excluded,
        preference_form: matches!(config.task, ExportTask::Preference).then(|| {
            match config.preference_form {
                PreferenceForm::Paired => "paired",
                PreferenceForm::Unpaired => "unpaired",
                PreferenceForm::Ranked => "ranked",
            }
            .to_string()
        }),
        reward: Distribution::from_sorted(reward_values),
        reward_sources,
    }
}

impl ContextStore {
    /// Curate stored records and export them as task-shaped JSONL plus a
    /// sibling `<output_path>.manifest.json`, returning the manifest.
    ///
    /// If `config.version` is set the export is pinned to that dataset version
    /// (time-travel) and the store is restored to its current version
    /// afterward. Output is written line-by-line, so the full JSONL is never
    /// held in memory at once.
    pub async fn export_training(
        &mut self,
        config: &ExportConfig,
        output_path: &str,
    ) -> LanceResult<ExportManifest> {
        let restore = match config.version {
            Some(target) => {
                let original = self.version();
                self.checkout(target).await?;
                Some(original)
            }
            None => None,
        };

        let result = self.export_inner(config, output_path).await;

        if let Some(original) = restore {
            self.checkout(original).await?;
        }
        result
    }

    async fn export_inner(
        &self,
        config: &ExportConfig,
        output_path: &str,
    ) -> LanceResult<ExportManifest> {
        let context_uri = self.uri().to_string();
        let version = self.version();

        let records = self
            .list_filtered_with_options(
                None,
                None,
                config.filters.as_ref(),
                config.lifecycle.clone(),
            )
            .await?;
        let (curated, counts) = curate(records, config);

        let Some(split) = &config.split else {
            return emit_export(
                curated,
                config,
                &context_uri,
                version,
                counts,
                output_path,
                None,
            );
        };

        // Group-disjoint, reproducible partition: a record's side is decided by
        // a stable hash of its `split.by` key, so no group spans both sides.
        let (train_path, eval_path) = split_paths(output_path);
        let mut train_records = Vec::new();
        let mut eval_records = Vec::new();
        for record in curated {
            let key = split.by.key(&record);
            if split_fraction(split.seed, &key) < split.eval_fraction {
                eval_records.push(record);
            } else {
                train_records.push(record);
            }
        }

        let train_manifest = emit_export(
            train_records,
            config,
            &context_uri,
            version,
            counts,
            &train_path,
            Some(SplitManifest {
                side: "train".to_string(),
                eval_fraction: split.eval_fraction,
                by: split.by.label(),
                seed: split.seed,
                complement_path: eval_path.clone(),
            }),
        )?;
        emit_export(
            eval_records,
            config,
            &context_uri,
            version,
            counts,
            &eval_path,
            Some(SplitManifest {
                side: "eval".to_string(),
                eval_fraction: split.eval_fraction,
                by: split.by.label(),
                seed: split.seed,
                complement_path: train_path.clone(),
            }),
        )?;
        Ok(train_manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::LIFECYCLE_ACTIVE;
    use crate::store::{ContextStore, ContextStoreOptions};
    use chrono::TimeZone;
    use serde_json::json;
    use tempfile::TempDir;

    const DIM: i32 = 4;

    async fn open_store(dir: &TempDir) -> ContextStore {
        let uri = dir.path().join("ctx.lance").to_string_lossy().to_string();
        ContextStore::open_with_options(
            &uri,
            ContextStoreOptions {
                embedding_dim: Some(DIM),
                ..Default::default()
            },
        )
        .await
        .unwrap()
    }

    fn rec(id: &str, role: &str, text: &str, secs: i64) -> ContextRecord {
        ContextRecord {
            id: id.to_string(),
            external_id: None,
            run_id: "run".to_string(),
            bot_id: None,
            session_id: Some("s1".to_string()),
            tenant: None,
            source: None,
            created_at: Utc.timestamp_opt(1_700_000_000 + secs, 0).unwrap(),
            role: role.to_string(),
            state_metadata: None,
            metadata: None,
            relationships: Vec::new(),
            expires_at: None,
            retention_policy: None,
            lifecycle_status: LIFECYCLE_ACTIVE.to_string(),
            retired_at: None,
            retired_reason: None,
            supersedes_id: None,
            superseded_by_id: None,
            content_type: "text/plain".to_string(),
            text_payload: Some(text.to_string()),
            binary_payload: None,
            payload_uri: None,
            payload_size: None,
            payload_checksum: None,
            embedding: None,
        }
    }

    fn emb(lead: &[f32]) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM as usize];
        for (i, x) in lead.iter().enumerate() {
            v[i] = *x;
        }
        v
    }

    fn read_lines(path: &str) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn read_manifest(path: &str) -> ExportManifest {
        let raw = std::fs::read_to_string(format!("{path}.manifest.json")).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    fn out_path(dir: &TempDir) -> String {
        dir.path().join("out.jsonl").to_string_lossy().to_string()
    }

    #[test]
    fn sft_groups_session_into_ordered_conversation() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            // Added out of order; export must order by (created_at, id).
            store
                .add(&[
                    rec("r2", "assistant", "hi there", 2),
                    rec("r1", "user", "hello", 1),
                    rec("r3", "user", "bye", 3),
                ])
                .await
                .unwrap();

            let out = out_path(&dir);
            let manifest = store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Sft,
                        group_by: GroupBy::SessionId,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let lines = read_lines(&out);
            assert_eq!(lines.len(), 1);
            let messages = lines[0]["messages"].as_array().unwrap();
            let contents: Vec<&str> = messages
                .iter()
                .map(|m| m["content"].as_str().unwrap())
                .collect();
            assert_eq!(contents, ["hello", "hi there", "bye"]);
            assert_eq!(manifest.task, "sft");
            assert_eq!(manifest.counts.examples, 1);
            assert_eq!(lines[0]["provenance"]["version"], json!(manifest.version));

            // The sibling manifest file is written and matches the return value.
            let manifest_file = read_manifest(&out);
            assert_eq!(manifest_file.task, "sft");
            assert_eq!(manifest_file.counts.examples, 1);
            assert_eq!(manifest_file.schema_version, EXPORT_SCHEMA_VERSION);
            assert_eq!(manifest_file.source_record_ids.len(), 3);
        });
    }

    #[test]
    fn sft_rejection_sampling_filters_low_reward() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut good = rec("g", "assistant", "good", 1);
            good.session_id = Some("a".to_string());
            good.metadata = Some(json!({"reward": 0.9}));
            let mut bad = rec("b", "assistant", "bad", 2);
            bad.session_id = Some("b".to_string());
            bad.metadata = Some(json!({"reward": 0.1}));
            store.add(&[good, bad]).await.unwrap();

            let out = out_path(&dir);
            let manifest = store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Sft,
                        group_by: GroupBy::SessionId,
                        min_reward: Some(0.5),
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let lines = read_lines(&out);
            assert_eq!(lines.len(), 1, "only the high-reward record survives");
            assert_eq!(lines[0]["messages"][0]["content"], "good");
            assert_eq!(manifest.counts.after_reward_filter, 1);
            assert_eq!(manifest.min_reward, Some(0.5));
        });
    }

    #[test]
    fn preference_paired_uses_reward_for_chosen_rejected() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let prompt = rec("p", "user", "question", 1);
            let mut hi = rec("hi", "assistant", "great answer", 2);
            hi.metadata = Some(json!({"reward": 0.9}));
            let mut lo = rec("lo", "assistant", "poor answer", 3);
            lo.metadata = Some(json!({"reward": 0.2}));
            store.add(&[prompt, hi, lo]).await.unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Preference,
                        preference_form: PreferenceForm::Paired,
                        group_by: GroupBy::SessionId,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let lines = read_lines(&out);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0]["form"], "paired");
            assert_eq!(lines[0]["prompt"][0]["content"], "question");
            assert_eq!(lines[0]["chosen"][0]["content"], "great answer");
            assert_eq!(lines[0]["rejected"][0]["content"], "poor answer");
        });
    }

    #[test]
    fn preference_unpaired_uses_kto_labels() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let prompt = rec("p", "user", "q", 1);
            let mut a = rec("a", "assistant", "yes", 2);
            a.metadata = Some(json!({"label": "chosen"}));
            let mut b = rec("b", "assistant", "no", 3);
            b.metadata = Some(json!({"label": "rejected"}));
            store.add(&[prompt, a, b]).await.unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Preference,
                        preference_form: PreferenceForm::Unpaired,
                        group_by: GroupBy::SessionId,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let lines = read_lines(&out);
            assert_eq!(lines.len(), 2);
            assert!(lines.iter().all(|l| l["form"] == "unpaired"));
            let chosen = lines
                .iter()
                .find(|l| l["completion"][0]["content"] == "yes")
                .unwrap();
            assert_eq!(chosen["label"], json!(true));
            let rejected = lines
                .iter()
                .find(|l| l["completion"][0]["content"] == "no")
                .unwrap();
            assert_eq!(rejected["label"], json!(false));
        });
    }

    #[test]
    fn preference_ranked_orders_by_rank() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let prompt = rec("p", "user", "q", 1);
            let mut second = rec("c2", "assistant", "second", 2);
            second.metadata = Some(json!({"rank": 2}));
            let mut first = rec("c1", "assistant", "first", 3);
            first.metadata = Some(json!({"rank": 1}));
            store.add(&[prompt, second, first]).await.unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Preference,
                        preference_form: PreferenceForm::Ranked,
                        group_by: GroupBy::SessionId,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let lines = read_lines(&out);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0]["form"], "ranked");
            let cands = lines[0]["candidates"].as_array().unwrap();
            assert_eq!(cands[0]["messages"][0]["content"], "first");
            assert_eq!(cands[0]["rank"], json!(1));
            assert_eq!(cands[1]["messages"][0]["content"], "second");
        });
    }

    #[test]
    fn rollout_groups_responses_with_rewards() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let prompt = rec("p", "user", "solve x", 1);
            let mut r1 = rec("r1", "assistant", "ans1", 2);
            r1.metadata =
                Some(json!({"reward": 1.0, "reward_source": "verifier", "group_id": "g1"}));
            let mut r2 = rec("r2", "assistant", "ans2", 3);
            r2.metadata =
                Some(json!({"reward": 0.0, "reward_source": "verifier", "group_id": "g1"}));
            store.add(&[prompt, r1, r2]).await.unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Rollout,
                        group_by: GroupBy::SessionId,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let lines = read_lines(&out);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0]["group_id"], "g1");
            assert_eq!(lines[0]["prompt"][0]["content"], "solve x");
            let responses = lines[0]["responses"].as_array().unwrap();
            assert_eq!(responses.len(), 2);
            assert_eq!(responses[0]["reward"], json!(1.0));
            assert_eq!(responses[0]["reward_source"], "verifier");
        });
    }

    #[test]
    fn dedup_collapses_near_duplicates() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut a = rec("a", "user", "dup one", 1);
            a.session_id = Some("a".to_string());
            a.embedding = Some(emb(&[1.0, 0.0]));
            let mut b = rec("b", "user", "dup two", 2);
            b.session_id = Some("b".to_string());
            b.embedding = Some(emb(&[1.0, 0.0])); // identical direction -> distance 0
            store.add(&[a, b]).await.unwrap();

            let out = out_path(&dir);
            let manifest = store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Sft,
                        group_by: GroupBy::SessionId,
                        dedup_threshold: Some(0.01),
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            assert_eq!(
                manifest.counts.after_dedup, 1,
                "one near-duplicate collapsed"
            );
            assert_eq!(read_lines(&out).len(), 1);
        });
    }

    #[test]
    fn decontaminate_drops_holdout_matches() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut keep = rec("k", "user", "keep", 1);
            keep.session_id = Some("k".to_string());
            keep.embedding = Some(emb(&[0.0, 1.0]));
            let mut leak = rec("l", "user", "leak", 2);
            leak.session_id = Some("l".to_string());
            leak.embedding = Some(emb(&[1.0, 0.0]));
            store.add(&[keep, leak]).await.unwrap();

            let out = out_path(&dir);
            let manifest = store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Sft,
                        group_by: GroupBy::SessionId,
                        decontaminate_against: vec![emb(&[1.0, 0.0])], // matches "leak"
                        decontaminate_threshold: Some(0.01),
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            assert_eq!(manifest.counts.after_decontaminate, 1);
            let lines = read_lines(&out);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0]["messages"][0]["content"], "keep");
        });
    }

    #[test]
    fn curation_drops_contradicted_records() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let good = rec("g", "user", "valid", 1);
            let mut bad = rec("c", "user", "contradicted", 2);
            bad.session_id = Some("other".to_string());
            bad.lifecycle_status = LIFECYCLE_CONTRADICTED.to_string();
            store.add(&[good, bad]).await.unwrap();

            let out = out_path(&dir);
            let manifest = store
                .export_training(&ExportConfig::default(), &out)
                .await
                .unwrap();

            assert_eq!(manifest.counts.after_lifecycle, 1);
            let lines = read_lines(&out);
            assert!(lines
                .iter()
                .all(|l| l["messages"][0]["content"] != "contradicted"));
        });
    }

    #[test]
    fn version_pinning_exports_old_state_and_restores() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            store.add(&[rec("r1", "user", "first", 1)]).await.unwrap();
            store.compact(None).await.unwrap();
            let pinned = store.version();

            store.add(&[rec("r2", "user", "second", 2)]).await.unwrap();
            store.compact(None).await.unwrap();
            let latest = store.version();

            let out = out_path(&dir);
            let manifest = store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Sft,
                        group_by: GroupBy::None,
                        version: Some(pinned),
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            assert_eq!(manifest.version, pinned);
            assert_eq!(
                store.version(),
                latest,
                "store restored after pinned export"
            );
        });
    }

    #[test]
    fn export_is_reproducible() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            store
                .add(&[
                    rec("r1", "user", "a", 1),
                    rec("r2", "assistant", "b", 2),
                    rec("r3", "user", "c", 3),
                ])
                .await
                .unwrap();

            let config = ExportConfig {
                task: ExportTask::Sft,
                group_by: GroupBy::SessionId,
                ..Default::default()
            };
            let first = out_path(&dir);
            let second = dir.path().join("out2.jsonl").to_string_lossy().to_string();
            store.export_training(&config, &first).await.unwrap();
            store.export_training(&config, &second).await.unwrap();

            assert_eq!(
                std::fs::read_to_string(&first).unwrap(),
                std::fs::read_to_string(&second).unwrap(),
                "same version + config produces identical output"
            );
        });
    }

    #[test]
    fn external_id_prefix_grouping() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut a = rec("a", "user", "doc7 turn1", 1);
            a.external_id = Some("doc-7#chunk-1".to_string());
            let mut b = rec("b", "assistant", "doc7 turn2", 2);
            b.external_id = Some("doc-7#chunk-2".to_string());
            let mut c = rec("c", "user", "doc8 turn1", 3);
            c.external_id = Some("doc-8#chunk-1".to_string());
            store.add(&[a, b, c]).await.unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Sft,
                        group_by: GroupBy::ExternalIdPrefix("#".to_string()),
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let lines = read_lines(&out);
            // doc-7 (2 turns) + doc-8 (1 turn) => 2 examples.
            assert_eq!(lines.len(), 2);
            assert_eq!(lines[0]["messages"].as_array().unwrap().len(), 2);
        });
    }

    fn session_ids(path: &str) -> std::collections::HashSet<String> {
        read_lines(path)
            .iter()
            .filter_map(|line| {
                line["provenance"]["session_id"]
                    .as_str()
                    .map(str::to_string)
            })
            .collect()
    }

    #[test]
    fn split_is_deterministic_and_group_disjoint() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut records = Vec::new();
            for s in 0..10 {
                let mut user = rec(&format!("u{s}"), "user", "q", s * 2);
                user.session_id = Some(format!("s{s}"));
                let mut asst = rec(&format!("a{s}"), "assistant", "r", s * 2 + 1);
                asst.session_id = Some(format!("s{s}"));
                records.push(user);
                records.push(asst);
            }
            store.add(&records).await.unwrap();

            let config = ExportConfig {
                task: ExportTask::Sft,
                group_by: GroupBy::SessionId,
                split: Some(SplitConfig {
                    eval_fraction: 0.5,
                    by: GroupBy::SessionId,
                    seed: 42,
                }),
                ..Default::default()
            };

            let base1 = dir.path().join("a.jsonl").to_string_lossy().to_string();
            let base2 = dir.path().join("b.jsonl").to_string_lossy().to_string();
            store.export_training(&config, &base1).await.unwrap();
            store.export_training(&config, &base2).await.unwrap();

            let (train1, eval1) = split_paths(&base1);
            let (train2, eval2) = split_paths(&base2);

            // Determinism: identical partition across runs with the same seed.
            assert_eq!(
                std::fs::read_to_string(&train1).unwrap(),
                std::fs::read_to_string(&train2).unwrap()
            );
            assert_eq!(
                std::fs::read_to_string(&eval1).unwrap(),
                std::fs::read_to_string(&eval2).unwrap()
            );

            // Group-disjoint: no session appears on both sides.
            let train_sessions = session_ids(&train1);
            let eval_sessions = session_ids(&eval1);
            assert!(!train_sessions.is_empty() && !eval_sessions.is_empty());
            assert!(train_sessions.is_disjoint(&eval_sessions));
            assert_eq!(train_sessions.len() + eval_sessions.len(), 10);
        });
    }

    #[test]
    fn split_fraction_is_approximately_respected() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut records = Vec::new();
            for s in 0..200 {
                let mut r = rec(&format!("r{s}"), "user", "q", s);
                r.session_id = Some(format!("s{s}"));
                records.push(r);
            }
            store.add(&records).await.unwrap();

            let config = ExportConfig {
                task: ExportTask::Sft,
                group_by: GroupBy::SessionId,
                split: Some(SplitConfig {
                    eval_fraction: 0.25,
                    by: GroupBy::SessionId,
                    seed: 7,
                }),
                ..Default::default()
            };
            let base = dir.path().join("c.jsonl").to_string_lossy().to_string();
            store.export_training(&config, &base).await.unwrap();

            let (train, eval) = split_paths(&base);
            let eval_count = read_lines(&eval).len();
            let train_count = read_lines(&train).len();
            assert_eq!(train_count + eval_count, 200);
            let fraction = eval_count as f64 / 200.0;
            assert!(
                (fraction - 0.25).abs() < 0.1,
                "eval fraction {fraction} too far from 0.25"
            );
        });
    }

    #[test]
    fn split_manifests_record_params_and_complement() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut a = rec("a", "user", "x", 1);
            a.session_id = Some("s1".to_string());
            let mut b = rec("b", "user", "y", 2);
            b.session_id = Some("s2".to_string());
            store.add(&[a, b]).await.unwrap();

            let config = ExportConfig {
                task: ExportTask::Sft,
                group_by: GroupBy::SessionId,
                split: Some(SplitConfig {
                    eval_fraction: 0.5,
                    by: GroupBy::SessionId,
                    seed: 99,
                }),
                ..Default::default()
            };
            let base = dir.path().join("d.jsonl").to_string_lossy().to_string();
            store.export_training(&config, &base).await.unwrap();

            let (train, eval) = split_paths(&base);
            assert!(std::fs::metadata(&train).is_ok());
            assert!(std::fs::metadata(&eval).is_ok());

            let train_manifest = read_manifest(&train);
            let split = train_manifest.split.unwrap();
            assert_eq!(split.side, "train");
            assert_eq!(split.seed, 99);
            assert_eq!(split.eval_fraction, 0.5);
            assert_eq!(split.by, "session_id");
            assert_eq!(split.complement_path, eval);

            let eval_manifest = read_manifest(&eval);
            assert_eq!(eval_manifest.split.unwrap().side, "eval");
        });
    }

    fn read_stats(path: &str) -> ExportStats {
        let raw = std::fs::read_to_string(format!("{path}.stats.json")).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    #[test]
    fn stats_report_counts_roles_tokens_and_exclusions() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let mut user = rec("u", "user", "hello there", 1);
            user.source = Some("memory".to_string());
            user.tenant = Some("acme".to_string());
            user.state_metadata = Some(crate::record::StateMetadata {
                tokens_used: Some(5),
                ..Default::default()
            });
            let mut asst = rec("a", "assistant", "hi", 2);
            asst.source = Some("memory".to_string());
            asst.tenant = Some("acme".to_string());
            asst.state_metadata = Some(crate::record::StateMetadata {
                tokens_used: Some(11),
                ..Default::default()
            });
            // a contradicted record that curation excludes by lifecycle
            let mut dropped = rec("d", "user", "nope", 3);
            dropped.session_id = Some("other".to_string());
            dropped.lifecycle_status = LIFECYCLE_CONTRADICTED.to_string();
            store.add(&[user, asst, dropped]).await.unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Sft,
                        group_by: GroupBy::SessionId,
                        emit_stats: true,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let stats = read_stats(&out);
            assert_eq!(stats.task, "sft");
            assert_eq!(stats.examples, 1);
            assert_eq!(stats.num_groups, 1);
            assert_eq!(stats.by_role.get("user"), Some(&1));
            assert_eq!(stats.by_role.get("assistant"), Some(&1));
            assert_eq!(stats.by_source.get("memory"), Some(&2));
            assert_eq!(stats.by_tenant.get("acme"), Some(&2));
            assert_eq!(stats.excluded.lifecycle, 1, "contradicted record excluded");

            let tokens = stats.tokens.unwrap();
            assert_eq!(tokens.source, "tokens_used");
            assert_eq!(tokens.distribution.count, 2);
            assert_eq!(tokens.distribution.min, 5.0);
            assert_eq!(tokens.distribution.max, 11.0);
            assert_eq!(tokens.distribution.mean, 8.0);
        });
    }

    #[test]
    fn stats_token_fallback_uses_length_proxy() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            // no state_metadata -> fallback to whitespace word count
            store
                .add(&[rec("u", "user", "one two three four", 1)])
                .await
                .unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        emit_stats: true,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let tokens = read_stats(&out).tokens.unwrap();
            assert_eq!(tokens.source, "length_proxy");
            assert_eq!(tokens.distribution.max, 4.0);
        });
    }

    #[test]
    fn stats_report_reward_distribution_for_rollout() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            let prompt = rec("p", "user", "solve", 1);
            let mut r1 = rec("r1", "assistant", "a1", 2);
            r1.metadata = Some(json!({"reward": 1.0, "reward_source": "verifier"}));
            let mut r2 = rec("r2", "assistant", "a2", 3);
            r2.metadata = Some(json!({"reward": 0.0, "reward_source": "verifier"}));
            store.add(&[prompt, r1, r2]).await.unwrap();

            let out = out_path(&dir);
            store
                .export_training(
                    &ExportConfig {
                        task: ExportTask::Rollout,
                        group_by: GroupBy::SessionId,
                        emit_stats: true,
                        ..Default::default()
                    },
                    &out,
                )
                .await
                .unwrap();

            let stats = read_stats(&out);
            let reward = stats.reward.unwrap();
            assert_eq!(reward.count, 2);
            assert_eq!(reward.min, 0.0);
            assert_eq!(reward.max, 1.0);
            assert_eq!(stats.reward_sources.get("verifier"), Some(&2));
        });
    }

    #[test]
    fn stats_not_written_without_flag() {
        let dir = TempDir::new().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = open_store(&dir).await;
            store.add(&[rec("u", "user", "hi", 1)]).await.unwrap();
            let out = out_path(&dir);
            store
                .export_training(&ExportConfig::default(), &out)
                .await
                .unwrap();
            assert!(std::fs::metadata(format!("{out}.stats.json")).is_err());
        });
    }
}
