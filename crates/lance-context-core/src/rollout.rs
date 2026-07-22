use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::record::Relationship;

/// Well-known values for the [`RolloutRecord::role`] dictionary column.
pub const ROLE_ASSISTANT: &str = "assistant";
pub const ROLE_TOOL: &str = "tool";
pub const ROLE_GRADE: &str = "grade";
pub const ROLE_ARTIFACT: &str = "artifact";

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

    // Oversized message fields offloaded via the claim-check write path. Each is
    // its own nullable, individually-projectable column (not packed into
    // `content`/`binary_payload`) so a reader can select one without
    // materializing the rest.
    pub model_input_string: Option<String>,
    pub model_output_string: Option<String>,
    pub rationale: Option<String>,
    pub problem_text: Option<String>,
    pub user_metadata: Option<String>,

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
