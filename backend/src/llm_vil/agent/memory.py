"""Three-tier memory: episodic hot window, reflections/facts/goals, archived episodes.

The stream never touches SQLite: archived records leave through ``drain_pending`` and the
engine's single writer indexes them (subject + FTS5) for two-stage retrieval.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .. import PROMPT_VERSION
from ..llm.protocol import Embedder, cosine
from ..schemas.agent import MemoryKind, MemoryRecord
from ..schemas.config import MemoryConfig

PERMANENT_KINDS = frozenset({"reflection", "fact", "goal", "summary"})
HEARSAY_DECAY = 0.7


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


RUMOR_MIN_CONFIDENCE = 0.35


def rumor_root(record: MemoryRecord) -> str:
    """Id of the first-hand record (or God rumor) a hearsay chain started from."""
    if record.kind == "hearsay" and record.evidence_ids:
        return record.evidence_ids[0]
    return record.id


def rumor_text(record: MemoryRecord) -> str:
    """The rumor itself, without nested "X에게 들은 이야기:" prefixes."""
    text = record.text
    while "에게 들은 이야기: " in text:
        text = text.split("에게 들은 이야기: ", 1)[1]
    return text.removeprefix("소문: ")


class MemoryStream:
    def __init__(self, owner_id: str, embedder: Embedder, config: MemoryConfig,
                 ticks_per_day: int = 1440) -> None:
        self.owner_id = owner_id
        self.embedder = embedder
        self.config = config
        self.ticks_per_day = ticks_per_day
        self.records: list[MemoryRecord] = []
        self.next_seq = 1
        self.importance_accum = 0
        self.cooldowns: dict[str, int] = {}
        self.pending_archive: list[MemoryRecord] = []
        self.archived_count = 0
        # Blocking embedders (ONNX, HTTP) never run on the event loop: new records are
        # stored without a vector and embedded in batches via ``aembed`` (worker thread).
        self.deferred = bool(getattr(embedder, "blocking", False))
        self._pending_ids: list[str] = []

    # ------------------------------------------------------------ writes
    def add(self, tick: int, kind: MemoryKind, text: str, importance: int, *,
            valence: float = 0.0, subjects: Iterable[str] = (), source_actor_id: str | None = None,
            confidence: float = 1.0, evidence_ids: Iterable[str] = ()) -> MemoryRecord:
        rec = MemoryRecord(
            id=f"{self.owner_id}-m{self.next_seq:05d}", owner_id=self.owner_id, tick=tick,
            kind=kind, text=text, importance=max(1, min(10, int(importance))),
            valence=max(-1.0, min(1.0, valence)), subjects=sorted(set(subjects)),
            source_actor_id=source_actor_id, confidence=round(max(0.0, min(1.0, confidence)), 4),
            evidence_ids=list(evidence_ids), prompt_version=PROMPT_VERSION,
            embed_model="" if self.deferred else self.embedder.model,
            embedding=[] if self.deferred else self.embedder.embed(text),
        )
        self.next_seq += 1
        self.records.append(rec)
        if self.deferred:
            self._pending_ids.append(rec.id)
        if kind not in PERMANENT_KINDS:  # reflections/facts/goals/summaries never self-trigger
            self.importance_accum += rec.importance
        return rec

    def perceive(self, tick: int, key: str, text: str, importance: int, *,
                 subjects: Iterable[str] = (), valence: float = 0.0) -> MemoryRecord | None:
        """Change-only perception with per-subject cooldown (salient events bypass it)."""
        last = self.cooldowns.get(key)
        if last is not None and tick - last < self.config.perception_cooldown and importance < 7:
            return None
        self.cooldowns[key] = tick
        return self.add(tick, "observation", text, importance, subjects=subjects,
                        valence=valence)

    def add_hearsay(self, tick: int, original: MemoryRecord, speaker_id: str,
                    speaker_name: str | None = None) -> MemoryRecord:
        """Store a rumor heard from ``speaker_id``: confidence decays x0.7 per hop and
        ``evidence_ids[0]`` always names the *root* record, so retellings stay one rumor."""
        base = original.confidence if original.kind == "hearsay" else 1.0
        return self.add(
            tick, "hearsay", f"{speaker_name or speaker_id}에게 들은 이야기: "
            f"{rumor_text(original)}",
            max(1, original.importance - 1), valence=original.valence,
            # subjects stay what the rumor is *about*; who told it is source_actor_id
            # (adding the messenger would make listeners distrust whoever passed it on)
            subjects=list(original.subjects),
            source_actor_id=speaker_id,
            confidence=base * HEARSAY_DECAY, evidence_ids=[rumor_root(original)],
        )

    def knows_rumor(self, root_id: str) -> bool:
        return any(r.id == root_id or (r.kind == "hearsay" and r.evidence_ids[:1] == [root_id])
                   for r in self.records)

    # ------------------------------------------------------------ reads
    def get(self, memory_id: str) -> MemoryRecord | None:
        return next((r for r in self.records if r.id == memory_id), None)

    def is_hot(self, r: MemoryRecord, now: int) -> bool:
        return (
            r.kind in PERMANENT_KINDS
            or r.importance >= self.config.salient_importance
            or now - r.tick <= self.config.hot_days * self.ticks_per_day
        )

    # ------------------------------------------------------------ deferred embedding
    def pending_embeddings(self) -> list[tuple[str, str]]:
        """(record id, text) for records still waiting for a vector."""
        if not self._pending_ids:
            return []
        wanted = set(self._pending_ids)
        return [(r.id, r.text) for r in self.records if r.id in wanted]

    def apply_embeddings(self, items: list[tuple[str, list[float]]]) -> None:
        vecs = dict(items)
        for i, r in enumerate(self.records):
            vec = vecs.get(r.id)
            if vec is not None:
                self.records[i] = r.model_copy(update={"embed_model": self.embedder.model,
                                                        "embedding": vec})
        self._pending_ids = [p for p in self._pending_ids if p not in vecs]

    async def ensure_embedded(self) -> None:
        pending = self.pending_embeddings()
        if pending:
            vecs = await self.embedder.aembed([t for _, t in pending])  # type: ignore[attr-defined]
            self.apply_embeddings([(rid, v) for (rid, _), v in zip(pending, vecs, strict=True)])

    async def aretrieve(self, now: int, query: str, subjects: Iterable[str] = (),
                        k: int | None = None,
                        extra: Iterable[MemoryRecord] = ()) -> list[MemoryRecord]:
        """Async retrieval: pending vectors and the query vector are computed off-loop."""
        if not self.deferred:
            return self.retrieve(now, query, subjects, k, extra)
        await self.ensure_embedded()
        qv = (await self.embedder.aembed([query]))[0] if query else []  # type: ignore[attr-defined]
        return self._rank(now, qv, subjects, k, extra)

    def retrieve(self, now: int, query: str, subjects: Iterable[str] = (), k: int | None = None,
                 extra: Iterable[MemoryRecord] = ()) -> list[MemoryRecord]:
        """Min-max normalized recency + importance + relevance + subject boost."""
        qv = self.embedder.embed(query) if query else []
        return self._rank(now, qv, subjects, k, extra)

    def _rank(self, now: int, qv: list[float], subjects: Iterable[str], k: int | None,
              extra: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        k = k or self.config.retrieve_k
        cands = [r for r in self.records if self.is_hot(r, now)]
        cands.extend(extra)
        if not cands:
            return []
        subj = set(subjects)
        rec = _minmax([0.99 ** ((now - r.tick) / 60.0) for r in cands])
        imp = _minmax([float(r.importance) for r in cands])
        # never mix vectors from different embedding models / dims in one cosine; records
        # without a comparable vector get the lowest comparable relevance (never a free 0.0
        # that would outrank genuinely dissimilar, negative-cosine memories)
        raw: list[float | None] = [
            cosine(qv, r.embedding)
            if qv and r.embed_model == self.embedder.model and len(r.embedding) == len(qv)
            else None
            for r in cands
        ]
        valid = [v for v in raw if v is not None]
        floor = min(valid) if valid else 0.0
        rel = _minmax([floor if v is None else v for v in raw])
        scored = []
        for i, r in enumerate(cands):
            boost = 1.0 if subj.intersection(r.subjects) else 0.0
            scored.append((rec[i] + imp[i] + rel[i] + boost, r.tick, r.id, r))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return [t[3] for t in scored[:k]]

    def since(self, tick: int) -> list[MemoryRecord]:
        return [r for r in self.records if r.tick >= tick]

    def of_kind(self, kind: str) -> list[MemoryRecord]:
        return [r for r in self.records if r.kind == kind]

    def needs_reflection(self) -> bool:
        return self.importance_accum >= self.config.reflection_threshold

    # ------------------------------------------------------------ archive
    def archive_old(self, now: int) -> list[MemoryRecord]:
        """Move cold episodic records out of the hot stream (into ``pending_archive``).

        Low-importance routine observations are first condensed into one permanent summary.
        """
        cold = [r for r in self.records if not self.is_hot(r, now)]
        if not cold:
            return []
        cold_ids = {r.id for r in cold}
        self.records = [r for r in self.records if r.id not in cold_ids]
        routine = [r for r in cold if r.kind == "observation" and r.importance <= 3]
        if routine:
            subjects = sorted({s for r in routine for s in r.subjects})[:8]
            day = routine[0].tick // self.ticks_per_day
            self.add(now, "summary",
                     f"{day}일차 무렵의 평범한 일상 관찰 {len(routine)}건 요약: "
                     + ", ".join(subjects[:5]), 3,
                     subjects=subjects, evidence_ids=[r.id for r in routine[:10]])
        self.pending_archive.extend(cold)
        self.archived_count += len(cold)
        return cold

    def set_embedder(self, embedder: Embedder) -> int:
        """Switch embedding model live. Records with other vectors are re-embedded: in one
        sync batch for fast embedders, or queued for off-loop batching for blocking ones.
        Returns how many records need new vectors."""
        self.embedder = embedder
        self.deferred = bool(getattr(embedder, "blocking", False))
        if not self.deferred:
            return self.reembed()
        self._pending_ids = [self.records[i].id for i in self._stale()]
        return len(self._pending_ids)

    def _stale(self) -> list[int]:
        dim = getattr(self.embedder, "dim", None)
        return [i for i, r in enumerate(self.records)
                if r.embed_model != self.embedder.model
                or (dim is not None and len(r.embedding) != dim)]

    def reembed(self) -> int:
        """Re-embed hot records whose vectors came from another model or dim (one batch)."""
        stale = self._stale()
        if not stale:
            return 0
        texts = [self.records[i].text for i in stale]
        many = getattr(self.embedder, "embed_many", None)
        vecs = many(texts) if many else [self.embedder.embed(t) for t in texts]
        for i, vec in zip(stale, vecs, strict=True):
            self.records[i] = self.records[i].model_copy(
                update={"embed_model": self.embedder.model, "embedding": vec})
        self._pending_ids = []
        return len(stale)

    def drain_pending(self) -> list[MemoryRecord]:
        out, self.pending_archive = self.pending_archive, []
        return out

    # ------------------------------------------------------------ persistence
    def snapshot(self) -> dict[str, Any]:
        return {
            "records": [r.model_dump(mode="json") for r in self.records],
            "next_seq": self.next_seq,
            "importance_accum": self.importance_accum,
            "cooldowns": dict(sorted(self.cooldowns.items())),
            "pending_archive": [r.model_dump(mode="json") for r in self.pending_archive],
            "archived_count": self.archived_count,
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.records = [MemoryRecord.model_validate(r) for r in data["records"]]
        self.next_seq = data["next_seq"]
        self.importance_accum = data["importance_accum"]
        self.cooldowns = dict(data["cooldowns"])
        self.pending_archive = [MemoryRecord.model_validate(r) for r in data["pending_archive"]]
        self.archived_count = data["archived_count"]
        # vectors from another model/dim (e.g. saved before an embedder hot-swap) are
        # re-queued too, not only records that never had one
        self._pending_ids = [self.records[i].id for i in self._stale()] if self.deferred else []
        if not self.deferred:
            self.reembed()
