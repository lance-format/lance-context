"""Run a retrieval-quality evaluation against a labeled query set.

This is a worked, fully offline example of lance-context's built-in eval harness
(``Context.evaluate``). It:

  1. builds a small knowledge base with a deterministic local embedder,
  2. loads a labeled query set from ``queries.jsonl``,
  3. scores retrieval quality in ``vector`` and ``hybrid`` modes, and
  4. A/B-compares two embedders on the same corpus to show the metric deltas —
     the everyday "did my change improve retrieval?" workflow.

No API keys, no network, no model downloads — re-running it prints the same
numbers every time, so it doubles as a regression check.

Run it:

    uv run eval-quality            # from this directory
    # or: python -m eval_quality.main
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from lance_context import Context

from .corpus import CORPUS
from .embedder import HashingEmbedder

K = 5
QUERIES_PATH = Path(__file__).resolve().parent.parent.parent / "queries.jsonl"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / ".artifacts"


def load_query_set(path: Path) -> list[dict]:
    """Load a JSONL query set. One JSON object per line:

    {"query_id": "...", "text": "...", "relevant": [{"external_id": "...",
     "grade": 2.0}]}

    ``grade`` defaults to 1.0 (binary relevance); grades > 0 all count as
    relevant for recall/precision/MRR/hit-rate, while nDCG rewards ranking
    higher-graded records first.
    """
    queries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            queries.append(json.loads(line))
    return queries


def build_store(embedder: HashingEmbedder) -> Context:
    """Create a fresh context store and load the corpus, auto-embedded."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    dataset_path = ARTIFACTS_DIR / f"eval_kb_{uuid4().hex}.lance"
    ctx = Context.create(
        dataset_path.as_posix(),
        embedding_dim=embedder.dims(),
        distance_metric="cosine",
        embedding_provider=embedder,
    )
    for external_id, role, text in CORPUS:
        # With a provider configured, `add` embeds the text for us.
        ctx.add(role, text, external_id=external_id)
    return ctx


def embed_queries(queries: list[dict], embedder: HashingEmbedder) -> list[dict]:
    """Attach a `vector` to each query using the corpus embedder.

    The eval harness runs inside the store and cannot call back into a Python
    embedding provider, so the caller embeds queries. `queries.jsonl` stays as
    human-authored text + labels; the vector channel is derived here. (Hybrid
    mode additionally uses the `text` field for the lexical channel.)
    """
    return [
        {**query, "vector": embedder.embed_texts([query["text"]])[0]}
        for query in queries
    ]


def print_report(report: dict) -> None:
    agg = report["aggregate"]
    print(
        f"  aggregate @k={report['k']} ({report['mode']}, "
        f"{report['distance_metric']}): "
        f"recall={agg['recall']:.3f} precision={agg['precision']:.3f} "
        f"mrr={agg['mrr']:.3f} ndcg={agg['ndcg']:.3f} "
        f"hit_rate={agg['hit_rate']:.3f}"
    )
    for q in report["per_query"]:
        s = q["scores"]
        # A misfire is easy to spot: recall < 1 means at least one labeled
        # record fell outside the top-k retrieved ids shown here.
        flag = "" if s["recall"] >= 1.0 else "  <-- missed a relevant record"
        print(
            f"    {q['query_id']:<22} recall={s['recall']:.2f} "
            f"ndcg={s['ndcg']:.2f}  top-{report['k']}={q['retrieved']}{flag}"
        )


def main() -> None:
    queries = load_query_set(QUERIES_PATH)
    print(f"Loaded {len(queries)} labeled queries from {QUERIES_PATH.name}")

    # --- Baseline embedder: score vector vs. hybrid ---------------------
    # "vector" runs pure embedding search; "hybrid" fuses lexical + vector
    # recall (reciprocal-rank fusion). Comparing the two tells you whether
    # adding lexical matching would help on your data.
    strong = HashingEmbedder(dims=64)
    ctx = build_store(strong)
    print(f"Built KB: {ctx.entries()} records, embedder dims={strong.dims()}\n")

    strong_queries = embed_queries(queries, strong)
    print("Vector search:")
    strong_vec = ctx.evaluate(strong_queries, k=K, mode="vector")
    print_report(strong_vec)
    print("\nHybrid retrieval (lexical + vector):")
    print_report(ctx.evaluate(strong_queries, k=K, mode="hybrid"))

    # --- A/B two embedders on the same corpus and queries ---------------
    # The everyday eval loop: change something (here, the embedding
    # dimensionality — fewer dims => more hash collisions => weaker signal) and
    # check whether retrieval quality moved before you ship it.
    weak = HashingEmbedder(dims=12)
    weak_ctx = build_store(weak)
    weak_vec = weak_ctx.evaluate(embed_queries(queries, weak), k=K, mode="vector")

    base, cand = weak_vec["aggregate"], strong_vec["aggregate"]
    print(f"\nA/B (candidate dims=64  -  baseline dims=12), vector @k={K}:")
    for metric in ("recall", "precision", "mrr", "ndcg", "hit_rate"):
        delta = cand[metric] - base[metric]
        print(
            f"  {metric:<9} baseline={base[metric]:.3f}  "
            f"candidate={cand[metric]:.3f}  Δ={delta:+.3f}"
        )

    print(
        "\nThis is how you'd guard retrieval quality in CI: keep queries.jsonl "
        "next to your data, assert on aggregate metrics, and A/B a change "
        "before promoting a new embedder or index. To compare two *persisted "
        "dataset versions* by time-travel instead, use "
        "ctx.evaluate_versions(queries, baseline_version, candidate_version)."
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
