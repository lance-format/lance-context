# Rollouts (RolloutDB)

Store training trajectories — one row per step (an assistant turn, a tool call,
a grade, or an artifact). Only `id` and `rollout_id` are required; everything
else (tokens, logprobs, rewards, advantages) is optional and filled in as your
pipeline computes it.

We call this shape a **RolloutDB**: a versioned, columnar store purpose-built
for RL rollout data, the same way a vector DB is built for embeddings.

## Basic use

```python
from lance_context import RolloutStore

store = RolloutStore.open("rollouts.lance")

# One step of a trajectory. `id` is auto-generated if you omit it.
store.add({
    "rollout_id": "traj-1",   # the trajectory this row belongs to
    "problem_id": "prompt-7", # groups the N samples of one prompt (for GRPO)
    "role": "assistant",
    "content": "The answer is 42.",
    "reward": 1.0,
    "policy_version": "ckpt-100",
})

for row in store.list():
    print(row["rollout_id"], row["reward"])

# Exact-match filters are combined with AND and applied before pagination.
training_rows = store.list(filters={
    "policy_version": "ckpt-100",
    "include_in_training": True,
    "role": "assistant",
})
```

## Distributed training

In a real training run, generation workers and the learner talk to a shared
store over HTTP. Use `AsyncRolloutStore` from async code so writes don't block
your event loop:

```python
from lance_context import AsyncRolloutStore

store = await AsyncRolloutStore.connect_or_create("http://localhost:8080", "rl-run-1")
await store.add({"rollout_id": "traj-1", "role": "assistant", "reward": 1.0})
```

## Further reading

- [Rollout schema design](../specs/rollout-schema-design.md) — the full column
  set and why it looks the way it does
- [Rollout deployment](../specs/rollout-deployment.md) — distributed ingest with
  server-ID sharding
- [Blob streaming](../specs/rollout-blob-streaming.md) — memory behavior of the
  artifact/blob path
