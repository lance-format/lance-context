"""A tiny, dependency-free embedding provider for offline, reproducible evals.

lance-context bundles no models on purpose — you plug in your own provider
(OpenAI, sentence-transformers, a CLIP model, ...). For an *example* we want
something that (a) runs with no network, no API key, and no heavy deps, and
(b) still produces embeddings where lexically related text lands nearby, so the
retrieval metrics are meaningful rather than random.

``HashingEmbedder`` is a classic hashing bag-of-words: tokenize, hash each token
into one of ``dims`` buckets with a signed contribution, then L2-normalize. Two
texts that share vocabulary end up with a high cosine similarity. It is *not* a
good production embedder — it has no semantics beyond exact word overlap — but it
is perfect for demonstrating the eval harness deterministically.

It satisfies the ``EmbeddingProvider`` protocol (``dims`` + ``embed_texts``), so
``Context`` will auto-embed text on ``add`` and auto-embed string queries on
``search`` / ``evaluate``.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic, offline hashing bag-of-words text embedder."""

    def __init__(self, dims: int = 64) -> None:
        self._dims = dims

    # --- EmbeddingProvider protocol -------------------------------------
    def dims(self) -> int:
        return self._dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    # --- internals ------------------------------------------------------
    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for token in _TOKEN.findall(text.lower()):
            # Stable 8-byte digest -> bucket index + sign. Hashing (rather than a
            # learned vocabulary) keeps this stateless and fixed-dimension.
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            bucket = h % self._dims
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(component * component for component in vec))
        if norm == 0.0:
            return vec
        return [component / norm for component in vec]
