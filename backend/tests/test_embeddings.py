"""Embedding cache keying, model-tag guard, backfill, and real local Korean embeddings."""

from __future__ import annotations

import importlib.util
import os

import pytest

from llm_vil.agent.memory import MemoryStream
from llm_vil.engine import SaveStore
from llm_vil.llm.embeddings import (
    CachingEmbedder,
    DemoCachedEmbedder,
    LocalFastEmbedder,
    cache_key,
    reembed_missing,
)
from llm_vil.llm.protocol import HashingEmbedder, cosine
from llm_vil.schemas.config import MemoryConfig


class CountingEmbedder(CachingEmbedder):
    provider = "test"

    def __init__(self, model: str, cache=None, dim: int = 8):
        super().__init__(model, dim, cache)
        self.computed: list[str] = []

    def _compute(self, texts):
        self.computed.extend(texts)
        return [[float(len(t) + i) for i in range(self.dim)] for t in texts]


def test_cache_key_is_model_and_text_specific():
    assert cache_key("m1", "안녕") != cache_key("m2", "안녕")
    assert cache_key("m1", "안녕") != cache_key("m1", "안녕!")
    assert len(cache_key("m1", "x")) == 64


def test_persistent_cache_hit_miss_by_model_and_text(tmp_path):
    store = SaveStore(tmp_path / "e.sqlite")
    a = CountingEmbedder("model-a", store)
    v1 = a.embed("준호와 장작")
    a.embed("준호와 장작")
    assert (a.hits, a.misses) == (1, 1) and a.computed == ["준호와 장작"]
    fresh = CountingEmbedder("model-a", store)  # new process: memory cold, SQLite warm
    assert fresh.embed("준호와 장작") == v1 and fresh.computed == [] and fresh.hits == 1
    other = CountingEmbedder("model-b", store)  # same text, other model -> miss
    other.embed("준호와 장작")
    assert other.computed == ["준호와 장작"]
    assert store.count("embed_cache") == 2


async def test_aembed_keeps_cache_on_caller_thread(tmp_path):
    store = SaveStore(tmp_path / "a.sqlite")  # sqlite3 would raise if used cross-thread
    emb = CountingEmbedder("model-a", store)
    vecs = await emb.aembed(["하나", "둘", "하나"])
    assert len(vecs) == 3 and vecs[0] == vecs[2]
    assert store.count("embed_cache") == 2


def test_retrieve_never_mixes_embedding_models():
    m = MemoryStream("mina", HashingEmbedder(), MemoryConfig())
    own = m.add(0, "observation", "장작 이야기", 3)
    foreign = own.model_copy(update={"id": "x", "embed_model": "other-model"})
    alien_dim = own.model_copy(update={"id": "y", "embedding": own.embedding[:10]})
    top = m.retrieve(10, "장작 이야기", extra=[foreign, alien_dim], k=3)
    assert top[0].id == own.id  # only the same-model, same-dim vector gets relevance


def test_reembed_missing_backfills_archive_and_hot_memory(tmp_path):
    store = SaveStore(tmp_path / "r.sqlite")
    old = MemoryStream("mina", HashingEmbedder(), MemoryConfig())
    recs = [old.add(i, "observation", f"기억 {i}", 3, subjects=["junho"]) for i in range(5)]
    store.write_batch(archive=[(r, 10) for r in recs])
    new = DemoCachedEmbedder(dim=32)
    new.model = "demo-32"
    assert reembed_missing(store, new, batch=2) == 5
    assert reembed_missing(store, new) == 0  # idempotent
    for r in store.search_archive("mina", subject="junho", limit=5):
        assert r.embed_model == "demo-32" and len(r.embedding) == 32
    mem = MemoryStream("mina", new, MemoryConfig())
    mem.records = list(recs)
    assert mem.reembed() == 5 and all(r.embed_model == "demo-32" for r in mem.records)


fastembed_available = importlib.util.find_spec("fastembed") is not None


@pytest.mark.skipif(not fastembed_available or os.environ.get("LLM_VIL_SKIP_FASTEMBED") == "1",
                    reason="fastembed not installed")
async def test_local_fastembed_korean_semantic_sanity(tmp_path):
    store = SaveStore(tmp_path / "f.sqlite")
    emb = LocalFastEmbedder(cache=store)
    try:
        await emb.warmup()  # downloads/loads ONNX weights in a worker thread
    except Exception as exc:  # offline CI without a cached model
        pytest.skip(f"local model unavailable: {exc}")
    assert emb.loaded and emb.dim == 384
    q, near, far = await emb.aembed(["준호와 장작 이야기를 했다", "준호에게 장작을 나눠 주었다",
                                     "연못에 비가 내린다"])
    assert len(q) == 384
    assert cosine(q, near) > cosine(q, far)
    assert store.count("embed_cache") == 3
    again = LocalFastEmbedder(cache=store)  # cache hit: no model load needed
    assert again.embed("준호와 장작 이야기를 했다") == q and not again.loaded
