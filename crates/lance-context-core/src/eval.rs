//! Retrieval-quality evaluation harness.
//!
//! Measures retrieval quality (recall@k / precision@k / MRR / nDCG@k /
//! hit-rate) of [`ContextStore::search_filtered_with_options`] (vector) and
//! [`ContextStore::retrieve_filtered_with_options`] (hybrid) against a labeled
//! query set, and compares quality across dataset versions.
//!
//! Eval targets are referenced by stable `external_id`, so a query set stays
//! valid across the append-only supersession that changes internal `id`s.

use std::collections::HashMap;

use lance::{Error as LanceError, Result as LanceResult};
use serde::{Deserialize, Serialize};

use crate::record::{LifecycleQueryOptions, RecordFilters};
use crate::store::ContextStore;

fn default_grade() -> f32 {
    1.0
}

/// A single relevance judgment: a relevant `external_id` and its grade.
///
/// `grade` defaults to `1.0` (binary relevance). Graded relevance (e.g. `2.0`
/// for "highly relevant") is used by nDCG; any grade `> 0` counts as relevant
/// for recall/precision/MRR/hit-rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceLabel {
    pub external_id: String,
    #[serde(default = "default_grade")]
    pub grade: f32,
}

/// One labeled query: a vector and/or text channel plus its relevant records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
    pub query_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(default)]
    pub relevant: Vec<RelevanceLabel>,
}

impl EvalQuery {
    fn relevance_map(&self) -> HashMap<&str, f32> {
        self.relevant
            .iter()
            .map(|label| (label.external_id.as_str(), label.grade))
            .collect()
    }
}

/// A labeled query set, referenced by a stable `id` for reproducible reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuerySet {
    pub id: String,
    pub queries: Vec<EvalQuery>,
}

impl EvalQuerySet {
    #[must_use]
    pub fn new(id: impl Into<String>, queries: Vec<EvalQuery>) -> Self {
        Self {
            id: id.into(),
            queries,
        }
    }

    /// Parse a query set from JSONL, one [`EvalQuery`] object per line. Blank
    /// lines are ignored. The set `id` is supplied by the caller (e.g. derived
    /// from the file name) so the same labels can be reused across runs.
    pub fn from_jsonl(id: impl Into<String>, contents: &str) -> LanceResult<Self> {
        let mut queries = Vec::new();
        for (index, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let query: EvalQuery = serde_json::from_str(line).map_err(|err| {
                LanceError::invalid_input(format!(
                    "invalid eval query on line {}: {err}",
                    index + 1
                ))
            })?;
            queries.push(query);
        }
        Ok(Self::new(id, queries))
    }

    /// Serialize the query set to JSONL (one query per line).
    pub fn to_jsonl(&self) -> LanceResult<String> {
        let mut out = String::new();
        for query in &self.queries {
            let line = serde_json::to_string(query)
                .map_err(|err| LanceError::invalid_input(err.to_string()))?;
            out.push_str(&line);
            out.push('\n');
        }
        Ok(out)
    }
}

/// Which retrieval API to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RetrievalMode {
    /// Pure vector search ([`ContextStore::search_filtered_with_options`]).
    #[default]
    Vector,
    /// Hybrid text + vector retrieval
    /// ([`ContextStore::retrieve_filtered_with_options`]).
    Hybrid,
}

impl RetrievalMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Hybrid => "hybrid",
        }
    }
}

/// Runtime configuration for an evaluation run.
#[derive(Clone)]
pub struct EvalConfig {
    /// Rank cutoff `k` for all @k metrics and the retrieval limit.
    pub k: usize,
    pub mode: RetrievalMode,
    pub filters: Option<RecordFilters>,
    pub lifecycle: LifecycleQueryOptions,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            k: 10,
            mode: RetrievalMode::Vector,
            filters: None,
            lifecycle: LifecycleQueryOptions::default(),
        }
    }
}

/// Aggregate or per-query retrieval-quality metrics, all in `0.0..=1.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct MetricScores {
    pub recall: f64,
    pub precision: f64,
    pub mrr: f64,
    pub ndcg: f64,
    pub hit_rate: f64,
}

impl MetricScores {
    /// Per-metric difference `self - baseline`, for A/B deltas.
    #[must_use]
    pub fn delta(&self, baseline: &MetricScores) -> MetricScores {
        MetricScores {
            recall: self.recall - baseline.recall,
            precision: self.precision - baseline.precision,
            mrr: self.mrr - baseline.mrr,
            ndcg: self.ndcg - baseline.ndcg,
            hit_rate: self.hit_rate - baseline.hit_rate,
        }
    }
}

/// Metrics and retrieved ids for a single query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEval {
    pub query_id: String,
    /// External ids retrieved in rank order (top-k). Records retrieved without
    /// an `external_id` appear as an empty string and never match a label.
    pub retrieved: Vec<String>,
    pub scores: MetricScores,
}

/// A reproducible evaluation report: a manifest (query-set id, version, config)
/// plus aggregate and per-query scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub query_set_id: String,
    /// Dataset version the run was pinned to.
    pub version: u64,
    pub k: usize,
    pub mode: String,
    /// Distance metric the context is configured with (part of the manifest).
    pub distance_metric: String,
    pub num_queries: usize,
    pub aggregate: MetricScores,
    pub per_query: Vec<QueryEval>,
}

/// Result of comparing the same query set across two dataset versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbReport {
    pub query_set_id: String,
    pub baseline: EvalReport,
    pub candidate: EvalReport,
    /// `candidate.aggregate - baseline.aggregate`, per metric.
    pub deltas: MetricScores,
}

/// Compute @k metrics for one ranked result list against a relevance map.
fn compute_scores(retrieved: &[String], relevant: &HashMap<&str, f32>, k: usize) -> MetricScores {
    let k = k.max(1);
    let num_relevant = relevant.values().filter(|grade| **grade > 0.0).count();

    let mut hits = 0usize;
    let mut first_relevant_rank: Option<usize> = None;
    let mut dcg = 0.0f64;
    for (index, external_id) in retrieved.iter().take(k).enumerate() {
        let grade = relevant.get(external_id.as_str()).copied().unwrap_or(0.0);
        if grade > 0.0 {
            hits += 1;
            if first_relevant_rank.is_none() {
                first_relevant_rank = Some(index + 1);
            }
            // rank = index + 1, discount = log2(rank + 1) = log2(index + 2)
            dcg += f64::from(grade) / ((index + 2) as f64).log2();
        }
    }

    // Ideal DCG: best achievable ordering of the graded labels, truncated at k.
    let mut ideal_grades: Vec<f64> = relevant
        .values()
        .filter(|grade| **grade > 0.0)
        .map(|grade| f64::from(*grade))
        .collect();
    ideal_grades.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal_grades
        .iter()
        .take(k)
        .enumerate()
        .map(|(index, grade)| grade / ((index + 2) as f64).log2())
        .sum();

    MetricScores {
        recall: if num_relevant > 0 {
            hits as f64 / num_relevant as f64
        } else {
            0.0
        },
        precision: hits as f64 / k as f64,
        mrr: first_relevant_rank.map_or(0.0, |rank| 1.0 / rank as f64),
        ndcg: if idcg > 0.0 { dcg / idcg } else { 0.0 },
        hit_rate: if hits > 0 { 1.0 } else { 0.0 },
    }
}

fn mean_scores(per_query: &[QueryEval]) -> MetricScores {
    let n = per_query.len();
    if n == 0 {
        return MetricScores::default();
    }
    let mut agg = MetricScores::default();
    for query in per_query {
        agg.recall += query.scores.recall;
        agg.precision += query.scores.precision;
        agg.mrr += query.scores.mrr;
        agg.ndcg += query.scores.ndcg;
        agg.hit_rate += query.scores.hit_rate;
    }
    let n = n as f64;
    MetricScores {
        recall: agg.recall / n,
        precision: agg.precision / n,
        mrr: agg.mrr / n,
        ndcg: agg.ndcg / n,
        hit_rate: agg.hit_rate / n,
    }
}

impl ContextStore {
    /// Run a labeled query set against this context at its current version and
    /// return a reproducible [`EvalReport`].
    ///
    /// Each query is retrieved with `config.mode` (vector or hybrid) at the
    /// `config.k` cutoff, with `config.filters` / `config.lifecycle` applied,
    /// then scored against its `relevant` labels by `external_id`.
    pub async fn evaluate(
        &self,
        query_set: &EvalQuerySet,
        config: &EvalConfig,
    ) -> LanceResult<EvalReport> {
        let mut per_query = Vec::with_capacity(query_set.queries.len());
        for query in &query_set.queries {
            let retrieved = self.run_eval_query(query, config).await?;
            let relevant = query.relevance_map();
            let scores = compute_scores(&retrieved, &relevant, config.k);
            per_query.push(QueryEval {
                query_id: query.query_id.clone(),
                retrieved,
                scores,
            });
        }

        Ok(EvalReport {
            query_set_id: query_set.id.clone(),
            version: self.version(),
            k: config.k,
            mode: config.mode.as_str().to_string(),
            distance_metric: self.distance_metric().as_str().to_string(),
            num_queries: per_query.len(),
            aggregate: mean_scores(&per_query),
            per_query,
        })
    }

    /// A/B the same query set across two dataset versions and report per-metric
    /// deltas (`candidate - baseline`). The store is restored to its current
    /// version before returning.
    pub async fn evaluate_versions(
        &mut self,
        query_set: &EvalQuerySet,
        config: &EvalConfig,
        baseline_version: u64,
        candidate_version: u64,
    ) -> LanceResult<AbReport> {
        let original_version = self.version();

        self.checkout(baseline_version).await?;
        let baseline = self.evaluate(query_set, config).await?;
        self.checkout(candidate_version).await?;
        let candidate = self.evaluate(query_set, config).await?;
        self.checkout(original_version).await?;

        let deltas = candidate.aggregate.delta(&baseline.aggregate);
        Ok(AbReport {
            query_set_id: query_set.id.clone(),
            baseline,
            candidate,
            deltas,
        })
    }

    /// Retrieve the top-k `external_id`s for one query under `config`.
    async fn run_eval_query(
        &self,
        query: &EvalQuery,
        config: &EvalConfig,
    ) -> LanceResult<Vec<String>> {
        let limit = Some(config.k);
        let records = match config.mode {
            RetrievalMode::Vector => {
                let vector = query.vector.as_deref().ok_or_else(|| {
                    LanceError::invalid_input(format!(
                        "query '{}' has no vector for vector-mode eval",
                        query.query_id
                    ))
                })?;
                self.search_filtered_with_options(
                    vector,
                    limit,
                    config.filters.as_ref(),
                    config.lifecycle.clone(),
                )
                .await?
                .into_iter()
                .map(|hit| hit.record)
                .collect::<Vec<_>>()
            }
            RetrievalMode::Hybrid => self
                .retrieve_filtered_with_options(
                    query.text.as_deref(),
                    query.vector.as_deref(),
                    limit,
                    config.filters.as_ref(),
                    config.lifecycle.clone(),
                )
                .await?
                .into_iter()
                .map(|hit| hit.record)
                .collect::<Vec<_>>(),
        };

        Ok(records
            .into_iter()
            .map(|record| record.external_id.unwrap_or_default())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{ContextRecord, LIFECYCLE_ACTIVE};
    use crate::store::ContextStore;
    use chrono::Utc;
    use serde_json::json;
    use tempfile::TempDir;
    use uuid::Uuid;

    // ----- metric fixtures (pure) -------------------------------------------

    fn scores(retrieved: &[&str], relevant: &[(&str, f32)], k: usize) -> MetricScores {
        let retrieved: Vec<String> = retrieved.iter().map(|s| s.to_string()).collect();
        let relevant: HashMap<&str, f32> = relevant.iter().copied().collect();
        compute_scores(&retrieved, &relevant, k)
    }

    fn approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn metrics_perfect_ranking() {
        let s = scores(&["a", "b"], &[("a", 1.0), ("b", 1.0)], 2);
        approx(s.recall, 1.0);
        approx(s.precision, 1.0);
        approx(s.mrr, 1.0);
        approx(s.ndcg, 1.0);
        approx(s.hit_rate, 1.0);
    }

    #[test]
    fn metrics_single_relevant_at_rank_two() {
        // retrieved [x, a]; only `a` relevant; k=2.
        let s = scores(&["x", "a"], &[("a", 1.0)], 2);
        approx(s.recall, 1.0); // found the 1 relevant
        approx(s.precision, 0.5); // 1 hit / k=2
        approx(s.mrr, 0.5); // first relevant at rank 2
        approx(s.hit_rate, 1.0);
        // dcg = 1/log2(3); idcg = 1/log2(2) = 1.0
        approx(s.ndcg, 1.0 / 3.0_f64.log2());
    }

    #[test]
    fn metrics_no_relevant_in_topk() {
        let s = scores(&["x", "y"], &[("a", 1.0)], 2);
        approx(s.recall, 0.0);
        approx(s.precision, 0.0);
        approx(s.mrr, 0.0);
        approx(s.ndcg, 0.0);
        approx(s.hit_rate, 0.0);
    }

    #[test]
    fn metrics_graded_ndcg() {
        // retrieved [a(grade1), b(grade3)]; ideal order is [b, a].
        let s = scores(&["a", "b"], &[("a", 1.0), ("b", 3.0)], 2);
        let dcg = 1.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2();
        let idcg = 3.0 / 2.0_f64.log2() + 1.0 / 3.0_f64.log2();
        approx(s.ndcg, dcg / idcg);
        approx(s.recall, 1.0);
    }

    #[test]
    fn metrics_precision_is_over_k() {
        // Only one item retrieved but k=2 -> precision = 1/2.
        let s = scores(&["a"], &[("a", 1.0)], 2);
        approx(s.precision, 0.5);
        approx(s.recall, 1.0);
        approx(s.hit_rate, 1.0);
    }

    #[test]
    fn query_set_jsonl_round_trip() {
        let jsonl = concat!(
            "{\"query_id\":\"q1\",\"vector\":[1.0,0.0],\"relevant\":[{\"external_id\":\"a\"}]}\n",
            "\n",
            "{\"query_id\":\"q2\",\"text\":\"hi\",\"relevant\":[{\"external_id\":\"b\",\"grade\":2.0}]}\n",
        );
        let set = EvalQuerySet::from_jsonl("set-1", jsonl).unwrap();
        assert_eq!(set.queries.len(), 2);
        assert_eq!(set.queries[0].query_id, "q1");
        assert_eq!(set.queries[1].relevant[0].grade, 2.0);
        // default grade applies when omitted
        assert_eq!(set.queries[0].relevant[0].grade, 1.0);

        let reparsed = EvalQuerySet::from_jsonl("set-1", &set.to_jsonl().unwrap()).unwrap();
        assert_eq!(reparsed.queries.len(), 2);
        assert_eq!(reparsed.queries[1].relevant[0].external_id, "b");
    }

    // ----- runner integration -----------------------------------------------

    fn embedding(store: &ContextStore, lead: &[f32]) -> Vec<f32> {
        let dim = store.embedding_dim() as usize;
        let mut v = vec![0.0f32; dim];
        for (i, x) in lead.iter().enumerate() {
            v[i] = *x;
        }
        v
    }

    fn record(external_id: &str, text: &str, embedding: Vec<f32>) -> ContextRecord {
        ContextRecord {
            id: Uuid::new_v4().to_string(),
            external_id: Some(external_id.to_string()),
            run_id: "run".to_string(),
            bot_id: None,
            session_id: None,
            tenant: None,
            source: None,
            created_at: Utc::now(),
            role: "user".to_string(),
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
            embedding: Some(embedding),
        }
    }

    #[test]
    fn evaluate_vector_mode_scores_query_set() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let a = embedding(&store, &[1.0]);
            let b = embedding(&store, &[0.5]);
            let c = embedding(&store, &[0.0, 1.0]);
            store
                .add(&[
                    record("doc-a", "alpha", a.clone()),
                    record("doc-b", "beta", b),
                    record("doc-c", "gamma", c),
                ])
                .await
                .unwrap();

            // query closest to doc-a (then doc-b), only doc-a relevant.
            let query_set = EvalQuerySet::new(
                "qs",
                vec![EvalQuery {
                    query_id: "q1".to_string(),
                    text: None,
                    vector: Some(a),
                    relevant: vec![RelevanceLabel {
                        external_id: "doc-a".to_string(),
                        grade: 1.0,
                    }],
                }],
            );
            let config = EvalConfig {
                k: 2,
                mode: RetrievalMode::Vector,
                ..Default::default()
            };
            let report = store.evaluate(&query_set, &config).await.unwrap();

            assert_eq!(report.num_queries, 1);
            assert_eq!(report.mode, "vector");
            assert_eq!(report.k, 2);
            assert_eq!(report.per_query[0].retrieved.first().unwrap(), "doc-a");
            approx(report.aggregate.recall, 1.0);
            approx(report.aggregate.precision, 0.5);
            approx(report.aggregate.mrr, 1.0);
            approx(report.aggregate.hit_rate, 1.0);
        });
    }

    #[test]
    fn evaluate_respects_lifecycle_visibility() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let q = embedding(&store, &[1.0]);
            // the only relevant doc is retired -> hidden by default.
            let mut retired = record("doc-a", "alpha", q.clone());
            retired.retired_at = Some(Utc::now());
            store.add(&[retired]).await.unwrap();

            let query_set = EvalQuerySet::new(
                "qs",
                vec![EvalQuery {
                    query_id: "q1".to_string(),
                    text: None,
                    vector: Some(q),
                    relevant: vec![RelevanceLabel {
                        external_id: "doc-a".to_string(),
                        grade: 1.0,
                    }],
                }],
            );

            let default_cfg = EvalConfig {
                k: 5,
                mode: RetrievalMode::Vector,
                ..Default::default()
            };
            let hidden = store.evaluate(&query_set, &default_cfg).await.unwrap();
            approx(hidden.aggregate.recall, 0.0); // retired doc excluded

            let include_retired = EvalConfig {
                k: 5,
                mode: RetrievalMode::Vector,
                lifecycle: LifecycleQueryOptions::new(true, true),
                ..Default::default()
            };
            let visible = store.evaluate(&query_set, &include_retired).await.unwrap();
            approx(visible.aggregate.recall, 1.0); // surfaced with include_retired
        });
    }

    #[test]
    fn evaluate_respects_filters() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let shared = embedding(&store, &[1.0]);
            let mut a = record("doc-a", "alpha", shared.clone());
            a.tenant = Some("x".to_string());
            let mut b = record("doc-b", "beta", shared.clone());
            b.tenant = Some("y".to_string());
            store.add(&[a, b]).await.unwrap();

            // doc-b is relevant but filtered out by tenant=x.
            let query_set = EvalQuerySet::new(
                "qs",
                vec![EvalQuery {
                    query_id: "q1".to_string(),
                    text: None,
                    vector: Some(shared),
                    relevant: vec![RelevanceLabel {
                        external_id: "doc-b".to_string(),
                        grade: 1.0,
                    }],
                }],
            );
            let config = EvalConfig {
                k: 5,
                mode: RetrievalMode::Vector,
                filters: Some(RecordFilters::from_json_value(json!({"tenant": "x"})).unwrap()),
                ..Default::default()
            };
            let report = store.evaluate(&query_set, &config).await.unwrap();
            approx(report.aggregate.recall, 0.0); // doc-b excluded by filter
        });
    }

    #[test]
    fn evaluate_hybrid_mode_finds_relevant() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let a = embedding(&store, &[1.0]);
            let b = embedding(&store, &[0.0, 1.0]);
            store
                .add(&[
                    record("doc-a", "alpha unique", a.clone()),
                    record("doc-b", "beta other", b),
                ])
                .await
                .unwrap();

            let query_set = EvalQuerySet::new(
                "qs",
                vec![EvalQuery {
                    query_id: "q1".to_string(),
                    text: Some("alpha".to_string()),
                    vector: Some(a),
                    relevant: vec![RelevanceLabel {
                        external_id: "doc-a".to_string(),
                        grade: 1.0,
                    }],
                }],
            );
            let config = EvalConfig {
                k: 2,
                mode: RetrievalMode::Hybrid,
                ..Default::default()
            };
            let report = store.evaluate(&query_set, &config).await.unwrap();
            approx(report.aggregate.hit_rate, 1.0);
        });
    }

    #[test]
    fn config_ab_delta_detects_k_sensitivity() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let a = embedding(&store, &[1.0]);
            let b = embedding(&store, &[0.5]);
            store
                .add(&[
                    record("doc-a", "alpha", a.clone()),
                    record("doc-b", "beta", b),
                ])
                .await
                .unwrap();

            // relevant doc-b ranks second behind doc-a.
            let query_set = EvalQuerySet::new(
                "qs",
                vec![EvalQuery {
                    query_id: "q1".to_string(),
                    text: None,
                    vector: Some(a),
                    relevant: vec![RelevanceLabel {
                        external_id: "doc-b".to_string(),
                        grade: 1.0,
                    }],
                }],
            );
            let k1 = EvalConfig {
                k: 1,
                mode: RetrievalMode::Vector,
                ..Default::default()
            };
            let k2 = EvalConfig {
                k: 2,
                mode: RetrievalMode::Vector,
                ..Default::default()
            };
            let at_1 = store.evaluate(&query_set, &k1).await.unwrap();
            let at_2 = store.evaluate(&query_set, &k2).await.unwrap();
            approx(at_1.aggregate.recall, 0.0); // doc-b not in top-1
            approx(at_2.aggregate.recall, 1.0); // doc-b in top-2
            let delta = at_2.aggregate.delta(&at_1.aggregate);
            approx(delta.recall, 1.0);
        });
    }

    #[test]
    fn evaluate_versions_same_version_is_zero_delta_and_restores() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let a = embedding(&store, &[1.0]);
            store
                .add(&[record("doc-a", "alpha", a.clone())])
                .await
                .unwrap();
            let version = store.version();

            let query_set = EvalQuerySet::new(
                "qs",
                vec![EvalQuery {
                    query_id: "q1".to_string(),
                    text: None,
                    vector: Some(a),
                    relevant: vec![RelevanceLabel {
                        external_id: "doc-a".to_string(),
                        grade: 1.0,
                    }],
                }],
            );
            let config = EvalConfig {
                k: 1,
                mode: RetrievalMode::Vector,
                ..Default::default()
            };
            let ab = store
                .evaluate_versions(&query_set, &config, version, version)
                .await
                .unwrap();

            approx(ab.deltas.recall, 0.0);
            approx(ab.deltas.ndcg, 0.0);
            assert_eq!(ab.baseline.version, version);
            assert_eq!(ab.candidate.version, version);
            assert_eq!(
                store.version(),
                version,
                "store restored to original version"
            );
        });
    }
}
