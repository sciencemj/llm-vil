"""Single-writer SQLite store. The ONLY module in llm_vil that opens SQLite.

WAL + busy_timeout, forward ``schema_version`` migrations, checkpoints, append-only event
log, subject-indexed FTS5 memory archive, and the LLM ``tape``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
from array import array
from pathlib import Path
from typing import Any

from ..llm.protocol import TapeEntry, dumps
from ..schemas.agent import MemoryRecord
from ..schemas.config import StorageConfig
from ..schemas.world import WorldEvent

SCHEMA_VERSION = 7
AUTO_VACUUM_INCREMENTAL = 2
CONVERT_MAX_BYTES = 256 * 1024 * 1024  # larger old saves are never rewritten automatically
VACUUM_PAGES = 200         # pages released per incremental_vacuum step
VACUUM_MAX_STEPS = 50      # bound the work done by one maintenance pass
WAL_LIMIT_BYTES = 64 * 1024 * 1024

log = logging.getLogger(__name__)


def _migrate_v3(c: sqlite3.Connection) -> None:
    """Normalized subject index + tape error column; backfills existing archive rows."""
    c.execute("""CREATE TABLE archive_subjects (
        agent_id TEXT NOT NULL,
        memory_id TEXT NOT NULL,
        subject TEXT NOT NULL,
        archived_tick INTEGER NOT NULL,
        PRIMARY KEY (agent_id, subject, memory_id))""")  # PK doubles as the subject index
    c.execute("CREATE INDEX idx_archive_subjects_tick ON archive_subjects(archived_tick)")
    c.execute("ALTER TABLE tape ADD COLUMN error TEXT NOT NULL DEFAULT ''")
    rows = c.execute(
        "SELECT agent_id, memory_id, subjects, archived_tick FROM memory_archive").fetchall()
    c.executemany(
        "INSERT OR IGNORE INTO archive_subjects VALUES (?,?,?,?)",
        [(a, m, subj, t) for a, m, subjects, t in rows for subj in subjects.split("|") if subj],
    )


def _migrate_v5(c: sqlite3.Connection) -> None:
    """Indexed (embed_model, embed_dim) columns so re-embedding never parses every row."""
    c.execute("ALTER TABLE memory_archive ADD COLUMN embed_model TEXT NOT NULL DEFAULT ''")
    c.execute("ALTER TABLE memory_archive ADD COLUMN embed_dim INTEGER NOT NULL DEFAULT 0")
    rows = c.execute("SELECT rowid, data FROM memory_archive").fetchall()
    c.executemany(
        "UPDATE memory_archive SET embed_model = ?, embed_dim = ? WHERE rowid = ?",
        [(d.get("embed_model") or "", len(d.get("embedding") or []), rid)
         for rid, d in ((rid, json.loads(data)) for rid, data in rows)],
    )
    c.execute("CREATE INDEX idx_archive_embed ON memory_archive(embed_model, embed_dim)")


MIGRATIONS: dict[int, list[Any]] = {
    1: [
        """CREATE TABLE checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            data TEXT NOT NULL)""",
        "CREATE INDEX idx_checkpoints_tick ON checkpoints(tick)",
        """CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            kind TEXT NOT NULL,
            actor_id TEXT,
            target_id TEXT,
            data TEXT NOT NULL)""",
        "CREATE INDEX idx_events_tick ON events(tick)",
        """CREATE TABLE memory_archive (
            agent_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            tick INTEGER NOT NULL,
            archived_tick INTEGER NOT NULL,
            kind TEXT NOT NULL,
            subjects TEXT NOT NULL,
            importance INTEGER NOT NULL,
            text TEXT NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (agent_id, memory_id))""",
        "CREATE INDEX idx_archive_subject ON memory_archive(agent_id, subjects)",
        """CREATE VIRTUAL TABLE memory_fts USING fts5(
            text, subjects, agent_id UNINDEXED, memory_id UNINDEXED, archived_tick UNINDEXED,
            tokenize = 'unicode61')""",
    ],
    2: [
        """CREATE TABLE tape (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            purpose TEXT NOT NULL,
            model TEXT NOT NULL,
            request TEXT NOT NULL,
            response TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            ok INTEGER NOT NULL,
            attempt INTEGER NOT NULL)""",
        "CREATE INDEX idx_tape_tick ON tape(tick)",
    ],
    3: [_migrate_v3],
    4: [
        """CREATE TABLE embed_cache (
            cache_key TEXT PRIMARY KEY,
            embed_model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL)""",
        "CREATE INDEX idx_embed_cache_model ON embed_cache(embed_model)",
    ],
    5: [_migrate_v5],
    6: ["ALTER TABLE checkpoints ADD COLUMN label TEXT NOT NULL DEFAULT ''"],
    7: ["""CREATE TABLE replay_frames (
            tick INTEGER PRIMARY KEY,
            day INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            frame_json TEXT NOT NULL)"""],
}

_WORD = re.compile(r"[\w가-힣]+", re.UNICODE)


class SaveStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.execute("PRAGMA busy_timeout = 5000")
        # must precede the first write (incl. the WAL switch): a no-op on an existing file
        self.conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute(f"PRAGMA journal_size_limit = {WAL_LIMIT_BYTES}")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._migrate()
        self._ensure_auto_vacuum()

    # ------------------------------------------------------------ schema
    def _migrate(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = c.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] or 0
        for version in range(current + 1, SCHEMA_VERSION + 1):
            c.execute("BEGIN")
            try:
                for stmt in MIGRATIONS[version]:
                    if callable(stmt):
                        stmt(c)
                    else:
                        c.execute(stmt)
                c.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def _pragma(self, name: str) -> int:
        return int(self.conn.execute(f"PRAGMA {name}").fetchone()[0])

    def _ensure_auto_vacuum(self) -> None:
        """Saves created before auto_vacuum need one full VACUUM to switch. Only done for
        small files with enough free disk; otherwise the save keeps working (freed pages are
        reused, the file just does not shrink) and the manual command is logged."""
        if self.path == ":memory:" or self._pragma("auto_vacuum") == AUTO_VACUUM_INCREMENTAL:
            return
        size = self._pragma("page_count") * self._pragma("page_size")
        try:
            free = shutil.disk_usage(Path(self.path).resolve().parent).free
            if size <= CONVERT_MAX_BYTES and free > size * 3:
                self.conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                self.conn.execute("VACUUM")
                return
        except (OSError, sqlite3.OperationalError) as exc:  # e.g. another open connection
            log.warning("could not enable auto_vacuum on %s: %s", self.path, exc)
            return
        log.warning("%s (%d MB) predates auto_vacuum and will not shrink; run once while the "
                    "server is stopped: sqlite3 %s 'PRAGMA auto_vacuum=INCREMENTAL; VACUUM;'",
                    self.path, size // 1_000_000, self.path)

    # ------------------------------------------------------------ bounded storage
    def prune_auto_checkpoints(self, keep: int) -> int:
        """Keep the newest ``keep`` unlabeled saves; labeled (manual) saves are never touched."""
        ids = [r[0] for r in self.conn.execute(
            "SELECT id FROM checkpoints WHERE label = '' ORDER BY id DESC LIMIT -1 OFFSET ?",
            (max(1, keep),))]
        for cid in ids:  # one transaction per (possibly huge) row keeps the WAL small
            self.conn.execute("DELETE FROM checkpoints WHERE id = ? AND label = ''", (cid,))
        return len(ids)

    def prune_rows(self, *, max_tape: int, max_events: int, max_embed_cache: int) -> None:
        """Cap the append-only logs. Events newer than the newest checkpoint always stay."""
        c = self.conn
        c.execute("DELETE FROM tape WHERE id <= COALESCE((SELECT id FROM tape"
                  " ORDER BY id DESC LIMIT 1 OFFSET ?), -1)", (max_tape,))
        c.execute("DELETE FROM events WHERE id <= COALESCE((SELECT id FROM events"
                  " ORDER BY id DESC LIMIT 1 OFFSET ?), -1)"
                  " AND tick <= COALESCE((SELECT tick FROM checkpoints"
                  " ORDER BY id DESC LIMIT 1), -1)", (max_events,))
        c.execute("DELETE FROM embed_cache WHERE rowid <= COALESCE((SELECT rowid FROM embed_cache"
                  " ORDER BY rowid DESC LIMIT 1 OFFSET ?), -1)", (max_embed_cache,))

    def incremental_vacuum(self) -> int:
        """Return free pages to the filesystem in bounded steps; returns pages still free."""
        free = self._pragma("freelist_count")
        if self._pragma("auto_vacuum") != AUTO_VACUUM_INCREMENTAL:
            return free
        for _ in range(VACUUM_MAX_STEPS):
            if not free:
                break
            # the pragma frees one page per fetched row: it must be drained
            self.conn.execute(f"PRAGMA incremental_vacuum({VACUUM_PAGES})").fetchall()
            before, free = free, self._pragma("freelist_count")
            if free >= before:
                break
        return free

    def maintain(self, limits: StorageConfig) -> None:
        """Daily / per-checkpoint housekeeping: prune, then shrink the file."""
        self.prune_auto_checkpoints(limits.max_auto_checkpoints)
        self.prune_rows(max_tape=limits.max_tape_rows, max_events=limits.max_events_rows,
                        max_embed_cache=limits.max_embed_cache_rows)
        self.incremental_vacuum()

    def schema_version(self) -> int:
        return self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]

    # ------------------------------------------------------------ batched writes
    def write_batch(self, *, events: list[WorldEvent] = (),  # type: ignore[assignment]
                    archive: list[tuple[MemoryRecord, int]] = (),  # type: ignore[assignment]
                    tape: list[TapeEntry] = ()) -> None:  # type: ignore[assignment]
        if not (events or archive or tape):
            return
        c = self.conn
        c.execute("BEGIN")
        try:
            c.executemany(
                "INSERT INTO events(tick, kind, actor_id, target_id, data) VALUES (?,?,?,?,?)",
                [(e.tick, e.kind, e.actor_id, e.target_id,
                  dumps({"pos": e.pos, "data": e.data, "importance": e.importance,
                         "witnesses": e.witnesses})) for e in events],
            )
            for rec, archived_tick in archive:
                subjects = "|" + "|".join(rec.subjects) + "|"
                cur = c.execute(
                    "INSERT OR IGNORE INTO memory_archive(agent_id, memory_id, tick, archived_tick,"
                    " kind, subjects, importance, text, embed_model, embed_dim, data)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (rec.owner_id, rec.id, rec.tick, archived_tick, rec.kind, subjects,
                     rec.importance, rec.text, rec.embed_model, len(rec.embedding),
                     dumps(rec.model_dump(mode="json"))),  # keeps embedding + embed_model
                )
                if cur.rowcount:
                    c.executemany(
                        "INSERT OR IGNORE INTO archive_subjects VALUES (?,?,?,?)",
                        [(rec.owner_id, rec.id, subj, archived_tick) for subj in rec.subjects],
                    )
                    c.execute(
                        "INSERT INTO memory_fts(text, subjects, agent_id, memory_id, archived_tick)"
                        " VALUES (?,?,?,?,?)",
                        (rec.text, " ".join(rec.subjects), rec.owner_id, rec.id, archived_tick),
                    )
            c.executemany(
                "INSERT INTO tape(tick, agent_id, purpose, model, request, response, latency_ms,"
                " tokens_in, tokens_out, ok, attempt, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(t.tick, t.agent_id, t.purpose, t.model, t.request, t.response, t.latency_ms,
                  t.tokens_in, t.tokens_out, int(t.ok), t.attempt, t.error) for t in tape],
            )
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, tick: int, data: str, label: str = "") -> int:
        cur = self.conn.execute("INSERT INTO checkpoints(tick, data, label) VALUES (?, ?, ?)",
                                (tick, data, label[:80]))
        return int(cur.lastrowid or 0)

    def list_checkpoints(self, limit: int = 50,
                         ticks_per_day: int = 1440) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, tick, label, created_at FROM checkpoints ORDER BY tick DESC, id DESC"
            " LIMIT ?", (limit,),
        ).fetchall()
        return [{"id": i, "tick": t, "day": t // ticks_per_day, "minute": t % ticks_per_day,
                 "label": label or "", "created_at": created} for i, t, label, created in rows]

    def checkpoint_meta(self, checkpoint_id: int,
                        ticks_per_day: int = 1440) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, tick, label, created_at FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            return None
        i, t, label, created = row
        return {"id": i, "tick": t, "day": t // ticks_per_day, "minute": t % ticks_per_day,
                "label": label or "", "created_at": created}

    def has_checkpoints(self) -> bool:
        return self.conn.execute("SELECT 1 FROM checkpoints LIMIT 1").fetchone() is not None

    def reset_all(self) -> None:
        """Forget the whole save: checkpoints, event log, memory archive and LLM tape."""
        c = self.conn
        c.execute("BEGIN")
        try:
            for table in ("checkpoints", "events", "memory_archive", "archive_subjects",
                          "replay_frames",
                          "memory_fts", "tape"):
                c.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        self.incremental_vacuum()

    def load_checkpoint_id(self, checkpoint_id: int) -> tuple[int, dict[str, Any]]:
        row = self.conn.execute("SELECT tick, data FROM checkpoints WHERE id = ?",
                                (checkpoint_id,)).fetchone()
        if row is None:
            raise LookupError(f"no checkpoint #{checkpoint_id}")
        return int(row[0]), json.loads(row[1])

    def load_checkpoint(self, at_or_before: int | None = None) -> tuple[int, dict[str, Any]]:
        if at_or_before is None:
            row = self.conn.execute(
                "SELECT tick, data FROM checkpoints ORDER BY tick DESC, id DESC LIMIT 1"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT tick, data FROM checkpoints WHERE tick <= ? ORDER BY tick DESC, id DESC"
                " LIMIT 1", (at_or_before,),
            ).fetchone()
        if row is None:
            raise LookupError("no checkpoint found")
        return int(row[0]), json.loads(row[1])

    def truncate_after(self, tick: int, *, keep_checkpoints: bool = False) -> None:
        """Drop log rows newer than a restored checkpoint (they will be re-simulated).

        ``keep_checkpoints`` keeps later save points so a rewind can be undone (branching).
        """
        c = self.conn
        c.execute("BEGIN")
        try:
            c.execute("DELETE FROM events WHERE tick > ?", (tick,))
            c.execute("DELETE FROM memory_archive WHERE archived_tick > ?", (tick,))
            c.execute("DELETE FROM archive_subjects WHERE archived_tick > ?", (tick,))
            c.execute("DELETE FROM memory_fts WHERE archived_tick > ?", (tick,))
            c.execute("DELETE FROM tape WHERE tick > ?", (tick,))
            c.execute("DELETE FROM replay_frames WHERE tick > ?", (tick,))
            if not keep_checkpoints:
                c.execute("DELETE FROM checkpoints WHERE tick > ?", (tick,))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------ replay frames
    def write_frames(self, frames: list[tuple[int, int, int, str]], cap: int) -> None:
        """Batch-insert replay frames and keep only the newest ``cap`` ticks."""
        if not frames:
            return
        c = self.conn
        c.execute("BEGIN")
        try:
            c.executemany("INSERT OR REPLACE INTO replay_frames(tick, day, minute, frame_json)"
                          " VALUES (?,?,?,?)", frames)
            newest = max(f[0] for f in frames)
            c.execute("DELETE FROM replay_frames WHERE tick <= ?", (newest - cap,))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def frame_ticks(self) -> list[int]:
        return [r[0] for r in self.conn.execute("SELECT tick FROM replay_frames ORDER BY tick")]

    def frame_at(self, tick: int) -> tuple[int, str] | None:
        """Exact frame at ``tick`` or the nearest one before it."""
        row = self.conn.execute(
            "SELECT tick, frame_json FROM replay_frames WHERE tick <= ? ORDER BY tick DESC"
            " LIMIT 1", (tick,)).fetchone()
        return (int(row[0]), row[1]) if row else None

    def frames_between(self, lo: int, hi: int) -> list[tuple[int, str]]:
        return [(int(t), j) for t, j in self.conn.execute(
            "SELECT tick, frame_json FROM replay_frames WHERE tick BETWEEN ? AND ? ORDER BY tick",
            (lo, hi))]

    # ------------------------------------------------------------ reads
    def search_archive(self, agent_id: str, *, subject: str | None = None,
                       query: str | None = None, limit: int = 3) -> list[MemoryRecord]:
        """Two-stage archive retrieval: subject index first, then FTS5 relevance."""
        rows: list[tuple[str]] = []
        if subject:
            rows = self.conn.execute(
                "SELECT a.data FROM archive_subjects s JOIN memory_archive a"
                " ON a.agent_id = s.agent_id AND a.memory_id = s.memory_id"
                " WHERE s.agent_id = ? AND s.subject = ?"
                " ORDER BY a.importance DESC, a.tick DESC, a.memory_id LIMIT ?",
                (agent_id, subject, limit),
            ).fetchall()
        if len(rows) < limit and query:
            words = _WORD.findall(query)
            if words:
                # Prefix terms: Korean particles stay attached to tokens ("스튜가").
                match = " OR ".join('"' + w.replace('"', "") + '"*' for w in words[:8])
                seen = {json.loads(r[0])["id"] for r in rows}
                fts = self.conn.execute(
                    "SELECT a.data FROM memory_fts f JOIN memory_archive a"
                    " ON a.agent_id = f.agent_id AND a.memory_id = f.memory_id"
                    " WHERE memory_fts MATCH ? AND f.agent_id = ? ORDER BY rank, a.memory_id"
                    " LIMIT ?", (match, agent_id, limit),
                ).fetchall()
                rows.extend(r for r in fts if json.loads(r[0])["id"] not in seen)
        return [MemoryRecord.model_validate(json.loads(r[0])) for r in rows[:limit]]

    # ------------------------------------------------------------ embedding cache
    def get_vector(self, cache_key: str, embed_model: str) -> list[float] | None:
        row = self.conn.execute(
            "SELECT dim, vector FROM embed_cache WHERE cache_key = ? AND embed_model = ?",
            (cache_key, embed_model),
        ).fetchone()
        if row is None:
            return None
        vec = array("f")
        vec.frombytes(row[1])
        return [round(v, 6) for v in vec] if len(vec) == row[0] else None

    def put_vector(self, cache_key: str, embed_model: str, dim: int,
                   vector: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embed_cache(cache_key, embed_model, dim, vector)"
            " VALUES (?,?,?,?)", (cache_key, embed_model, dim, array("f", vector).tobytes()),
        )

    def archive_needing_embedding(self, embed_model: str, dim: int, limit: int = 64,
                                  after_rowid: int = 0) -> list[tuple[int, str, str, str]]:
        """Keyset page of archived rows whose vector is from another model/dim.

        Uses the indexed ``embed_model`` / ``embed_dim`` columns: no JSON parsing, no rescans.
        Returns ``(rowid, agent_id, memory_id, text)``.
        """
        return [tuple(r) for r in self.conn.execute(
            "SELECT rowid, agent_id, memory_id, text FROM memory_archive"
            " WHERE rowid > ? AND (embed_model != ? OR embed_dim != ?)"
            " ORDER BY rowid LIMIT ?", (after_rowid, embed_model, dim, limit),
        ).fetchall()]

    def update_archive_embeddings(self, embed_model: str,
                                  items: list[tuple[str, str, list[float]]]) -> None:
        """Apply one batch of ``(agent_id, memory_id, vector)`` in a single transaction."""
        c = self.conn
        c.execute("BEGIN")
        try:
            for agent_id, memory_id, vector in items:
                row = c.execute(
                    "SELECT data FROM memory_archive WHERE agent_id = ? AND memory_id = ?",
                    (agent_id, memory_id),
                ).fetchone()
                if row is None:
                    continue
                d = json.loads(row[0])
                d["embed_model"] = embed_model
                d["embedding"] = vector
                c.execute(
                    "UPDATE memory_archive SET data = ?, embed_model = ?, embed_dim = ?"
                    " WHERE agent_id = ? AND memory_id = ?",
                    (dumps(d), embed_model, len(vector), agent_id, memory_id),
                )
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def count(self, table: str) -> int:
        if table not in {"events", "memory_archive", "archive_subjects", "tape", "checkpoints",
                         "embed_cache", "replay_frames"}:
            raise ValueError(table)
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
