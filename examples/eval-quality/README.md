# Measuring retrieval quality

This example shows how to answer the question every RAG/memory system eventually
faces: **is my retrieval actually any good, and did my last change help or hurt?**

`lance-context` ships a built-in evaluation harness — `Context.evaluate` and
`Context.evaluate_versions` — that scores retrieval against a *labeled query
set*: queries paired with the records that *should* come back. This example is
fully offline and deterministic (no API keys, no model downloads), so you can
run it as-is and get the same numbers every time.

## What it does

1. Builds a small "Acme Cloud" support KB (`corpus.py`) with a stable
   `external_id` on every record.
2. Loads a labeled query set from [`queries.jsonl`](./queries.jsonl).
3. Scores retrieval in **vector** and **hybrid** modes.
4. **A/B-compares two embedders** on the same corpus and queries — the everyday
   "did my change improve retrieval?" loop.

It uses a tiny dependency-free `HashingEmbedder` (`embedder.py`) so related text
lands nearby without any model. It's not a production embedder — it's just enough
to make the metrics meaningful and reproducible.

## Run it

```bash
uv run eval-quality
# or, without installing:
PYTHONPATH=src python -m eval_quality.main
```

## The query set format

`queries.jsonl` is one JSON object per line — human-authored text plus the
labels. You embed the query text at run time with the *same* embedder you used
for the corpus (the harness runs inside the store and can't call a Python
embedder for you):

```json
{"query_id": "q-auth-token", "text": "my api token expired and requests return 401",
 "relevant": [{"external_id": "kb-auth-token", "grade": 2.0}]}
```

- `relevant` lists the records that should be retrieved, each with an optional
  `grade` (defaults to `1.0`).
- Any `grade > 0` counts as relevant for **recall/precision/MRR/hit-rate**.
- **nDCG** additionally rewards ranking higher-graded records first, so use
  `2.0` for "exactly this" and `1.0` for "also acceptable."

## Reading the metrics

All metrics are in `0.0..=1.0`, reported both as an `aggregate` and `per_query`:

| Metric      | Answers |
|-------------|---------|
| `recall`    | Of the relevant records, how many made the top-k? |
| `precision` | Of the top-k returned, how many were relevant? |
| `mrr`       | How high did the *first* relevant record rank? |
| `ndcg`      | Are the most-relevant records ranked first? (grade-aware) |
| `hit_rate`  | Did *any* relevant record show up at all? |

## What the output shows

**Hybrid beats pure vector** on this corpus — fusing lexical + vector recall
pushes the first relevant hit to the top for more queries:

```
Vector search:
  aggregate @k=5 (vector, cosine): recall=0.896 ... mrr=0.900 ndcg=0.863 ...
Hybrid retrieval (lexical + vector):
  aggregate @k=5 (hybrid, cosine): recall=0.938 ... mrr=1.000 ndcg=0.968 ...
```

**A/B-ing two embedders** shows a change moving the numbers — here a
higher-dimensional embedder (fewer hash collisions) wins across the board:

```
A/B (candidate dims=64  -  baseline dims=12), vector @k=5:
  recall    baseline=0.771  candidate=0.896  Δ=+0.125
  mrr       baseline=0.667  candidate=0.900  Δ=+0.233
  ndcg      baseline=0.605  candidate=0.863  Δ=+0.259
  ...
```

That delta is the whole point: wire this into CI, assert on the aggregate
metrics, and you'll catch a regression before an embedder or index change ships.

## Comparing two dataset versions

To A/B the *same* query set across two **persisted dataset versions** (rather
than two embedders) — e.g. before and after a re-index — use the time-travel
variant, which checks out each version and restores the store afterward:

```python
report = ctx.evaluate_versions(
    queries, baseline_version, candidate_version, k=5, mode="vector"
)
print(report["deltas"])  # candidate - baseline, per metric
```
