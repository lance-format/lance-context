use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::record::Relationship;

/// Well-known values for the [`RolloutRecord::role`] dictionary column.
pub const ROLE_ASSISTANT: &str = "assistant";
pub const ROLE_TOOL: &str = "tool";
pub const ROLE_GRADE: &str = "grade";
pub const ROLE_ARTIFACT: &str = "artifact";

/// Exact-match filters supported by rollout list queries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RolloutFilters {
    pub rollout_id: Option<String>,
    pub problem_id: Option<String>,
    pub policy_version: Option<String>,
    pub role: Option<String>,
    pub include_in_training: Option<bool>,
    pub artifact_type: Option<String>,
}

impl RolloutFilters {
    pub fn from_json_value(value: Value) -> Result<Self, String> {
        let Value::Object(object) = value else {
            return Err("rollout filters must be a JSON object".to_string());
        };

        let mut filters = Self::default();
        for (key, value) in object {
            match key.as_str() {
                "rollout_id" => filters.rollout_id = Some(filter_string(&key, value)?),
                "problem_id" => filters.problem_id = Some(filter_string(&key, value)?),
                "policy_version" => {
                    filters.policy_version = Some(filter_string(&key, value)?);
                }
                "role" => filters.role = Some(filter_string(&key, value)?),
                "include_in_training" => {
                    filters.include_in_training = Some(value.as_bool().ok_or_else(|| {
                        "rollout filter 'include_in_training' must be a boolean".to_string()
                    })?);
                }
                "artifact_type" => filters.artifact_type = Some(filter_string(&key, value)?),
                _ => return Err(format!("unsupported rollout filter '{key}'")),
            }
        }
        Ok(filters)
    }

    pub(crate) fn predicate(&self) -> Option<String> {
        let mut parts = Vec::new();
        push_string_predicate(&mut parts, "rollout_id", self.rollout_id.as_deref());
        push_string_predicate(&mut parts, "problem_id", self.problem_id.as_deref());
        push_string_predicate(&mut parts, "policy_version", self.policy_version.as_deref());
        push_string_predicate(&mut parts, "role", self.role.as_deref());
        if let Some(value) = self.include_in_training {
            parts.push(format!("include_in_training = {value}"));
        }
        push_string_predicate(&mut parts, "artifact_type", self.artifact_type.as_deref());
        (!parts.is_empty()).then(|| parts.join(" AND "))
    }
}

fn filter_string(name: &str, value: Value) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("rollout filter '{name}' must be a string"))
}

fn push_string_predicate(parts: &mut Vec<String>, column: &str, value: Option<&str>) {
    if let Some(value) = value {
        parts.push(format!("{column} = '{}'", value.replace('\'', "''")));
    }
}

/// One row of a reinforcement-learning rollout dataset.
///
/// A row is one message in a trajectory — an assistant turn, a tool call, a
/// grade, or an artifact. A whole trajectory is many rows sharing
/// [`Self::rollout_id`]; the N GRPO samples of one prompt share
/// [`Self::problem_id`]. This is a second, independent record type alongside
/// [`crate::ContextRecord`]; the two schemas share infrastructure (versioning,
/// blob offload, the relationship graph) but no columns.
///
/// Every token and training-signal column is nullable: a grade row carries a
/// reward but no tokens; an assistant row carries tokens but no score. Trainers
/// project only the columns they read.
#[derive(Debug, Clone)]
pub struct RolloutRecord {
    // Identity & grouping.
    pub id: String,
    /// The trajectory this row belongs to.
    pub rollout_id: String,
    /// Prompt / GRPO group key linking the N samples of one prompt. For
    /// non-grouped rollouts, set equal to `rollout_id`; keeping this column
    /// dense (never null) makes group-by scans cheap.
    pub problem_id: String,
    /// Source dataset name, for provenance.
    pub dataset: Option<String>,
    /// Explicit intra-rollout ordering; `created_at` is not a reliable total
    /// order across concurrently-appended rows.
    pub sequence_order: i32,
    /// `assistant` / `tool` / `grade` / `artifact` / … (see the `ROLE_*`
    /// constants). Stored as a dictionary column.
    pub role: String,
    pub created_at: DateTime<Utc>,

    // Message content.
    pub content: Option<String>,
    pub content_type: String,

    // Tokens.
    pub input_tokens: Option<Vec<i32>>,
    pub output_tokens: Option<Vec<i32>>,
    pub num_input_tokens: Option<i32>,
    pub num_output_tokens: Option<i32>,

    // Training signals — variable-length arrays aligned to tokens.
    /// Generation-time (old) logprobs — the PPO/GRPO ratio numerator.
    pub output_logprobs: Option<Vec<f32>>,
    pub input_logprobs: Option<Vec<f32>>,
    /// Reference-model logprobs — the KL term. May instead be re-annotated in
    /// the companion learner-annotations dataset.
    pub ref_logprobs: Option<Vec<f32>>,
    /// Gradient only on model-generated tokens (multi-turn / tool use).
    pub loss_mask: Option<Vec<i8>>,
    /// Group-normalized advantage. Scalar today; per-token GAE can graduate to
    /// a `List<Float32>` later.
    pub advantage: Option<f32>,

    // Reward.
    pub reward: Option<f32>,
    pub raw_reward: Option<f32>,
    pub grader_id: Option<String>,
    pub score: Option<f32>,

    // Training control & provenance.
    pub include_in_training: Option<bool>,
    pub exclude_reason: Option<String>,
    /// Checkpoint that generated this trajectory.
    pub policy_version: Option<String>,

    // Graph, artifacts, escape hatch.
    pub relationships: Vec<Relationship>,
    /// Artifact bytes, physically offloaded via blob v2 so column scans skip
    /// them (see spec §6). `payload_size` / `payload_checksum` carry size and
    /// checksum.
    pub binary_payload: Option<Vec<u8>>,
    pub payload_size: Option<i64>,
    pub payload_checksum: Option<String>,
    /// User-defined semantic category of an artifact, e.g.
    /// `"excel_grade_screenshot"`. Orthogonal to `content_type`, which is the
    /// transport/media type (e.g. `"image/png"`): `content_type` says how to
    /// decode the bytes, `artifact_type` says what the artifact *means*. A
    /// first-class column so it can be filtered / grouped-by / projected
    /// without materializing the free-form `metadata` JSON.
    pub artifact_type: Option<String>,
    /// Harness metadata — the open-ended escape hatch, for genuinely
    /// unstructured fields only (e.g. an artifact's `filename`). Semantic
    /// categories that you filter/group-by belong in `artifact_type` instead.
    pub metadata: Option<Value>,
}

impl RolloutRecord {
    /// Whether this row stores an artifact (see spec §6).
    #[must_use]
    pub fn is_artifact(&self) -> bool {
        self.role == ROLE_ARTIFACT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rollout_filters_parse_supported_fields() {
        let filters = RolloutFilters::from_json_value(json!({
            "rollout_id": "traj-1",
            "problem_id": "problem-7",
            "policy_version": "ckpt-42",
            "role": "assistant",
            "include_in_training": false,
            "artifact_type": "screenshot"
        }))
        .unwrap();

        assert_eq!(filters.rollout_id.as_deref(), Some("traj-1"));
        assert_eq!(filters.problem_id.as_deref(), Some("problem-7"));
        assert_eq!(filters.policy_version.as_deref(), Some("ckpt-42"));
        assert_eq!(filters.role.as_deref(), Some("assistant"));
        assert_eq!(filters.include_in_training, Some(false));
        assert_eq!(filters.artifact_type.as_deref(), Some("screenshot"));
    }

    #[test]
    fn rollout_filters_reject_unknown_and_wrong_types() {
        assert!(RolloutFilters::from_json_value(json!({"reward": 1.0})).is_err());
        assert!(RolloutFilters::from_json_value(json!({"policy_version": 42})).is_err());
        assert!(RolloutFilters::from_json_value(json!({"include_in_training": "yes"})).is_err());
        assert!(RolloutFilters::from_json_value(json!([])).is_err());
    }

    #[test]
    fn rollout_filter_predicate_escapes_strings_and_fields() {
        let filters = RolloutFilters {
            policy_version: Some("worker's-ckpt".to_string()),
            role: Some("assistant".to_string()),
            include_in_training: Some(true),
            ..Default::default()
        };

        assert_eq!(
            filters.predicate().as_deref(),
            Some(
                "policy_version = 'worker''s-ckpt' AND role = 'assistant' AND include_in_training = true"
            )
        );
    }
}
