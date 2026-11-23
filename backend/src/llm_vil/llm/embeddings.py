"""Embedders: fast offline ``DemoEmbedder``, default local ``LocalFastEmbedder`` (fastembed
ONNX, multilingual e5-small, 384-d), optional OpenAI-compatible ``/embeddings``.

Every vector is tagged ``(embed_model, dim)``; cache keys are ``sha256(model + "\\0" + text)``.
The persistent cache goes through a ``VectorCache`` (implemented by the engine's
single-writer ``SaveStore``), so this module never touches SQLite itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

import httpx

from .protocol import HashingEmbedder

DemoEmbedder = HashingEmbedder

DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CACHE_DIR = Path(os.environ.get("LLM_VIL_FASTEMBED_CACHE",
                                        Path.home() / ".cache" / "llm-vil" / "fastembed"))
# Models fastembed doesn't ship but that exist as ONNX on the Hugging Face Hub.
CUSTOM_MODELS: dict[str, dict[str, Any]] = {
    "intfloat/multilingual-e5-small": {"dim": 384, "model_file": "onnx/model.onnx",
                                       "pooling": "MEAN", "normalization": True},
}


_REGISTRY_LOCK = threading.Lock()  # fastembed's custom-model registry is process-global


class EmbedderUnavailable(RuntimeError):
    pass


class VectorCache(Protocol):
    def get_vector(self, cache_key: str, embed_model: str) -> list[float] | None: ...

    def put_vector(self, cache_key: str, embed_model: str, dim: int,
                   vector: list[float]) -> None: ...


def cache_key(embed_model: str, text: str) -> str:
    return hashlib.sha256(f"{embed_model}\0{text}".encode()).hexdigest()


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


class CachingEmbedder:
    """Two-level cache (in-memory LRU, then persistent ``VectorCache``) around ``_compute``."""

    provider = "base"
    blocking = False  # True when _compute is slow (ONNX / HTTP): callers must use aembed()

    def __init__(self, model: str, dim: int, cache: VectorCache | None = None,
                 memory_size: int = 4096) -> None:
        self.model = model
        self.dim = dim
        self.cache = cache
        self._mem: OrderedDict[str, list[float]] = OrderedDict()
        self._mem_size = memory_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._active = 0  # worker threads currently inside _compute (via aembed)
        self._active_lock = threading.Lock()

    def _compute(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        out, todo = self._lookup(texts)
        if todo:
            self._store(out, todo, self._compute([t for _, _, t in todo]))
        return [v for v in out if v is not None]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Cache lookups/writes stay on the caller's thread (SQLite is single-writer);
        only model inference runs in a worker thread."""
        out, todo = self._lookup(texts)
        if todo:
            vecs = await asyncio.to_thread(self._compute_tracked, [t for _, _, t in todo])
            self._store(out, todo, vecs)
        return [v for v in out if v is not None]

    def _lookup(self, texts: list[str]
                ) -> tuple[list[list[float] | None], list[tuple[list[int], str, str]]]:
        """Cache lookup. Misses are de-duplicated: each distinct text is computed once."""
        out: list[list[float] | None] = []
        todo: list[tuple[list[int], str, str]] = []
        by_key: dict[str, list[int]] = {}
        with self._lock:
            for i, t in enumerate(texts):
                key = cache_key(self.model, t)
                if key in by_key:  # duplicate miss within this batch
                    by_key[key].append(i)
                    out.append(None)
                    continue
                vec = self._mem.get(key)
                if vec is None and self.cache is not None:
                    vec = self.cache.get_vector(key, self.model)
                if vec is not None and len(vec) == self.dim:
                    self.hits += 1
                    self._remember(key, vec)
                    out.append(vec)
                else:
                    out.append(None)
                    by_key[key] = [i]
                    todo.append((by_key[key], key, t))
        return out, todo

    def _store(self, out: list[list[float] | None], todo: list[tuple[list[int], str, str]],
               vecs: list[list[float]]) -> None:
        with self._lock:
            for (indices, key, _), raw in zip(todo, vecs, strict=True):
                vec = _normalize([float(x) for x in raw])
                if len(vec) != self.dim:
                    raise EmbedderUnavailable(
                        f"{self.model} returned dim {len(vec)}, expected {self.dim}")
                self.misses += 1
                self._remember(key, vec)
                if self.cache is not None:
                    self.cache.put_vector(key, self.model, self.dim, vec)
                for i in indices:
                    out[i] = vec

    def _compute_tracked(self, texts: list[str]) -> list[list[float]]:
        with self._active_lock:
            self._active += 1
        try:
            return self._compute(texts)
        finally:
            with self._active_lock:
                self._active -= 1

    async def drain(self, timeout: float = 30.0) -> bool:
        """Wait until no worker thread is inside ``_compute``. A cancelled ``aembed`` task
        returns at once, but its thread keeps running; close() must wait for it."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self._active:
            if asyncio.get_running_loop().time() > deadline:
                return False
            await asyncio.sleep(0.01)
        return True

    async def warmup(self) -> None:
        """Load model weights off the event loop."""

    def close(self) -> None:
        """Release resources (HTTP clients); safe to call more than once."""

    def _remember(self, key: str, vec: list[float]) -> None:
        self._mem[key] = vec
        self._mem.move_to_end(key)
        if len(self._mem) > self._mem_size:
            self._mem.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def describe(self) -> dict[str, Any]:
        return {"embed_provider": self.provider, "embed_model": self.model,
                "embed_dim": self.dim, "embed_cache_hits": self.hits,
                "embed_cache_misses": self.misses}


class LocalFastEmbedder(CachingEmbedder):
    """fastembed ONNX embedder. The model is lazy-loaded (``warmup`` runs it in a thread)."""

    provider = "local"
    blocking = True

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, dim: int = 384,
                 cache: VectorCache | None = None, cache_dir: str | Path | None = None,
                 threads: int | None = None) -> None:
        super().__init__(model, dim, cache)
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.threads = threads
        self._model: Any = None
        self._load_lock = threading.Lock()
        # e5 models are trained with instruction prefixes; "query: " suits symmetric use
        self.prefix = "query: " if "e5" in model.lower() else ""

    def _load(self) -> Any:
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from fastembed import TextEmbedding
                from fastembed.common.model_description import ModelSource, PoolingType
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise EmbedderUnavailable(
                    "fastembed is not installed; run `uv add fastembed` or use the 'demo' "
                    "embeddings profile") from exc
            custom = CUSTOM_MODELS.get(self.model)
            if custom:
                with _REGISTRY_LOCK:  # two embedders must not register the same model twice
                    if self.model not in {
                        m["model"] for m in TextEmbedding.list_supported_models()
                    }:
                        TextEmbedding.add_custom_model(
                            model=self.model, pooling=getattr(PoolingType, custom["pooling"]),
                            normalization=custom["normalization"],
                            sources=ModelSource(hf=self.model), dim=custom["dim"],
                            model_file=custom["model_file"],
                        )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self.model, cache_dir=str(self.cache_dir),
                                        threads=self.threads)
            return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def _compute(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [list(v) for v in model.embed([self.prefix + t for t in texts])]


class OpenAICompatEmbedder(CachingEmbedder):
    """``POST {base_url}/embeddings`` (sync client; only used when explicitly configured)."""

    provider = "openai_compat"
    blocking = True

    def __init__(self, *, base_url: str, model: str, dim: int, api_key: str | None = None,
                 extra_headers: dict[str, str] | None = None, cache: VectorCache | None = None,
                 timeout_s: float = 30.0, transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(model, dim, cache)
        headers = dict(extra_headers or {})
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                    timeout=timeout_s, transport=transport)

    def __repr__(self) -> str:
        return f"OpenAICompatEmbedder(model={self.model!r})"

    def close(self) -> None:
        self._client.close()

    def _compute(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self._client.post("/embeddings", json={"model": self.model, "input": texts})
        except httpx.HTTPError as exc:
            # never echo the exception text: it can carry request headers / URLs
            raise EmbedderUnavailable(
                f"embeddings transport error: {type(exc).__name__}") from None
        if resp.status_code >= 400:
            raise EmbedderUnavailable(f"embeddings HTTP {resp.status_code}")
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


class DemoCachedEmbedder(CachingEmbedder):
    """Demo hashing vectors behind the same cache/stats interface."""

    provider = "demo"

    def __init__(self, dim: int = 64, cache: VectorCache | None = None) -> None:
        self._hash = HashingEmbedder(dim=dim)
        super().__init__(self._hash.model, dim, cache)

    def _compute(self, texts: list[str]) -> list[list[float]]:
        return [self._hash.embed(t) for t in texts]


def reembed_missing(store: Any, embedder: Any, batch: int = 64) -> int:
    """Backfill archived memories whose vectors came from another embedding model/dim.

    Keyset-paginated (each row is visited once) and one transaction per batch.
    """
    total = 0
    after = 0
    while True:
        rows = store.archive_needing_embedding(embedder.model, embedder.dim, limit=batch,
                                               after_rowid=after)
        if not rows:
            return total
        vecs = embedder.embed_many([text for _, _, _, text in rows])
        store.update_archive_embeddings(
            embedder.model,
            [(agent_id, memory_id, vec)
             for (_, agent_id, memory_id, _), vec in zip(rows, vecs, strict=True)],
        )
        after = rows[-1][0]
        total += len(rows)
