from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["EmbeddingProvider", "OpenAIProvider", "SentenceTransformersProvider"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface for pluggable embedding providers.

    Implement this protocol to plug in any embedding backend. Providers
    receive lists so :meth:`Context.add_many` can embed in a single call.
    """

    @property
    def dims(self) -> int:
        """Dimensionality of the vectors produced by this provider."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


def _build_provider(config: dict[str, Any]) -> EmbeddingProvider:
    """Instantiate a built-in provider from a config dict.

    Config keys:
        provider: "openai" | "sentence-transformers"
        model: model name / ID (optional, uses provider default)
        **kwargs: forwarded to the provider constructor
    """
    name = config.get("provider")
    if not name:
        raise ValueError("embedding config must include a 'provider' key")
    kwargs = {k: v for k, v in config.items() if k != "provider"}
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding provider {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None
    return cls(**kwargs)


class OpenAIProvider:
    """Embedding provider backed by the OpenAI embeddings API.

    Requires ``pip install lance-context[openai]``.
    """

    def __init__(
        self, model: str = "text-embedding-3-small", **client_kwargs: Any
    ) -> None:
        try:
            from openai import OpenAI  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "openai is required for the OpenAI embedding provider. "
                "Install it with: pip install lance-context[openai]"
            ) from exc
        self._client = OpenAI(**client_kwargs)
        self._model = model
        self._dims: int | None = None

    @property
    def dims(self) -> int:
        if self._dims is None:
            # Probe dims with a dummy call the first time they're requested.
            result = self._client.embeddings.create(input=["probe"], model=self._model)
            self._dims = len(result.data[0].embedding)
        return self._dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embeddings.create(input=texts, model=self._model)
        ordered = sorted(result.data, key=lambda d: d.index)
        vecs = [item.embedding for item in ordered]
        if self._dims is None and vecs:
            self._dims = len(vecs[0])
        return vecs


class SentenceTransformersProvider:
    """Embedding provider backed by sentence-transformers (local / offline).

    Requires ``pip install lance-context[sentence-transformers]``.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", **model_kwargs: Any) -> None:
        try:
            from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                SentenceTransformer,
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the "
                "SentenceTransformers provider. "
                "Install it with: pip install lance-context[sentence-transformers]"
            ) from exc
        self._model = SentenceTransformer(model, **model_kwargs)

    @property
    def dims(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


_REGISTRY: dict[str, type] = {
    "openai": OpenAIProvider,
    "sentence-transformers": SentenceTransformersProvider,
}
