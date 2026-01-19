"""SQLite cache store for ScopePack.

Tables:
- content_cache: Compressed text by content hash
- file_summaries: Per-file summaries by path and file hash
- session_state: Session state packs for cross-session persistence
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

# Default cache directory
DEFAULT_CACHE_DIR = Path(os.environ.get("SCOPE_CACHE_DIR", "~/.cache/scopepack")).expanduser()
DB_PATH = DEFAULT_CACHE_DIR / "scopepack.db"


def get_db_path() -> Path:
    """Get the database path, ensuring the directory exists."""
    db_path = Path(os.environ.get("SCOPE_DB_PATH", str(DB_PATH))).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def content_hash(text: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: str) -> str:
    """Compute SHA256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


SCHEMA = """
-- Compressed content cache
-- Key: content_hash + query_hash + budget_tokens + model_version
CREATE TABLE IF NOT EXISTS content_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL,
    query_hash TEXT,
    budget_tokens INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    compressed_text TEXT NOT NULL,
    original_tokens INTEGER,
    compressed_tokens INTEGER,
    compression_ratio REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
    access_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(content_hash, query_hash, budget_tokens, model_version)
);

CREATE INDEX IF NOT EXISTS idx_content_cache_hash ON content_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_content_cache_accessed ON content_cache(last_accessed_at);

-- File summaries for quick lookup
CREATE TABLE IF NOT EXISTS file_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    summary TEXT NOT NULL,
    summary_tokens INTEGER,
    file_size INTEGER,
    file_type TEXT,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
    access_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(file_path, file_hash, model_version)
);

CREATE INDEX IF NOT EXISTS idx_file_summaries_path ON file_summaries(file_path);
CREATE INDEX IF NOT EXISTS idx_file_summaries_hash ON file_summaries(file_hash);

-- Session state for cross-session persistence
CREATE TABLE IF NOT EXISTS session_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    project_dir TEXT NOT NULL,
    state_pack TEXT NOT NULL,  -- JSON blob
    git_branch TEXT,
    git_commit TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id)
);

CREATE INDEX IF NOT EXISTS idx_session_state_project ON session_state(project_dir);
CREATE INDEX IF NOT EXISTS idx_session_state_updated ON session_state(updated_at);

-- Hot files tracking (frequently accessed/edited)
CREATE TABLE IF NOT EXISTS hot_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_dir TEXT NOT NULL,
    file_path TEXT NOT NULL,
    read_count INTEGER NOT NULL DEFAULT 0,
    edit_count INTEGER NOT NULL DEFAULT 0,
    last_read_at TEXT,
    last_edit_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_dir, file_path)
);

CREATE INDEX IF NOT EXISTS idx_hot_files_project ON hot_files(project_dir);

-- Cache statistics for monitoring
CREATE TABLE IF NOT EXISTS cache_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stat_date TEXT NOT NULL,
    cache_hits INTEGER NOT NULL DEFAULT 0,
    cache_misses INTEGER NOT NULL DEFAULT 0,
    total_tokens_saved INTEGER NOT NULL DEFAULT 0,
    total_compressions INTEGER NOT NULL DEFAULT 0,
    avg_compression_ratio REAL,
    UNIQUE(stat_date)
);
"""


async def init_db() -> None:
    """Initialize the database with schema."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()
    print(f"Database initialized at {db_path}")


def init_db_sync() -> None:
    """Synchronous version of init_db for scripts."""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        db.commit()
    print(f"Database initialized at {db_path}")


@dataclass
class CacheEntry:
    """Cached compressed content."""

    content_hash: str
    query_hash: str | None
    budget_tokens: int
    model_version: str
    compressed_text: str
    original_tokens: int | None
    compressed_tokens: int | None
    compression_ratio: float | None


@dataclass
class FileSummary:
    """Cached file summary."""

    file_path: str
    file_hash: str
    summary: str
    summary_tokens: int | None
    file_size: int | None
    file_type: str | None
    model_version: str


@dataclass
class SessionState:
    """Session state pack."""

    session_id: str
    project_dir: str
    state_pack: dict[str, Any]
    git_branch: str | None
    git_commit: str | None


class CacheDB:
    """Async database interface for ScopePack cache."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_db_path()

    async def get_compressed(
        self,
        content_hash: str,
        query_hash: str | None,
        budget_tokens: int,
        model_version: str,
    ) -> CacheEntry | None:
        """Get cached compressed content."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM content_cache
                WHERE content_hash = ? AND query_hash IS ? AND budget_tokens = ? AND model_version = ?
                """,
                (content_hash, query_hash, budget_tokens, model_version),
            )
            row = await cursor.fetchone()
            if row:
                # Update access stats
                await db.execute(
                    """
                    UPDATE content_cache
                    SET last_accessed_at = datetime('now'), access_count = access_count + 1
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                await db.commit()
                return CacheEntry(
                    content_hash=row["content_hash"],
                    query_hash=row["query_hash"],
                    budget_tokens=row["budget_tokens"],
                    model_version=row["model_version"],
                    compressed_text=row["compressed_text"],
                    original_tokens=row["original_tokens"],
                    compressed_tokens=row["compressed_tokens"],
                    compression_ratio=row["compression_ratio"],
                )
            return None

    async def put_compressed(self, entry: CacheEntry) -> None:
        """Store compressed content in cache."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO content_cache
                (content_hash, query_hash, budget_tokens, model_version, compressed_text,
                 original_tokens, compressed_tokens, compression_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.content_hash,
                    entry.query_hash,
                    entry.budget_tokens,
                    entry.model_version,
                    entry.compressed_text,
                    entry.original_tokens,
                    entry.compressed_tokens,
                    entry.compression_ratio,
                ),
            )
            await db.commit()

    async def get_file_summary(
        self, file_path: str, file_hash: str, model_version: str
    ) -> FileSummary | None:
        """Get cached file summary."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM file_summaries
                WHERE file_path = ? AND file_hash = ? AND model_version = ?
                """,
                (file_path, file_hash, model_version),
            )
            row = await cursor.fetchone()
            if row:
                await db.execute(
                    """
                    UPDATE file_summaries
                    SET last_accessed_at = datetime('now'), access_count = access_count + 1
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                await db.commit()
                return FileSummary(
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    summary=row["summary"],
                    summary_tokens=row["summary_tokens"],
                    file_size=row["file_size"],
                    file_type=row["file_type"],
                    model_version=row["model_version"],
                )
            return None

    async def put_file_summary(self, summary: FileSummary) -> None:
        """Store file summary in cache."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO file_summaries
                (file_path, file_hash, summary, summary_tokens, file_size, file_type, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.file_path,
                    summary.file_hash,
                    summary.summary,
                    summary.summary_tokens,
                    summary.file_size,
                    summary.file_type,
                    summary.model_version,
                ),
            )
            await db.commit()

    async def get_session_state(self, session_id: str) -> SessionState | None:
        """Get session state pack."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM session_state WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row:
                return SessionState(
                    session_id=row["session_id"],
                    project_dir=row["project_dir"],
                    state_pack=json.loads(row["state_pack"]),
                    git_branch=row["git_branch"],
                    git_commit=row["git_commit"],
                )
            return None

    async def get_latest_session_for_project(self, project_dir: str) -> SessionState | None:
        """Get the most recent session state for a project."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM session_state
                WHERE project_dir = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (project_dir,),
            )
            row = await cursor.fetchone()
            if row:
                return SessionState(
                    session_id=row["session_id"],
                    project_dir=row["project_dir"],
                    state_pack=json.loads(row["state_pack"]),
                    git_branch=row["git_branch"],
                    git_commit=row["git_commit"],
                )
            return None

    async def put_session_state(self, state: SessionState) -> None:
        """Store session state pack."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO session_state
                (session_id, project_dir, state_pack, git_branch, git_commit, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    state.session_id,
                    state.project_dir,
                    json.dumps(state.state_pack),
                    state.git_branch,
                    state.git_commit,
                ),
            )
            await db.commit()

    async def record_file_access(
        self, project_dir: str, file_path: str, is_edit: bool = False
    ) -> None:
        """Record file access for hot file tracking."""
        async with aiosqlite.connect(self.db_path) as db:
            if is_edit:
                await db.execute(
                    """
                    INSERT INTO hot_files (project_dir, file_path, edit_count, last_edit_at)
                    VALUES (?, ?, 1, datetime('now'))
                    ON CONFLICT(project_dir, file_path) DO UPDATE SET
                        edit_count = edit_count + 1,
                        last_edit_at = datetime('now')
                    """,
                    (project_dir, file_path),
                )
            else:
                await db.execute(
                    """
                    INSERT INTO hot_files (project_dir, file_path, read_count, last_read_at)
                    VALUES (?, ?, 1, datetime('now'))
                    ON CONFLICT(project_dir, file_path) DO UPDATE SET
                        read_count = read_count + 1,
                        last_read_at = datetime('now')
                    """,
                    (project_dir, file_path),
                )
            await db.commit()

    async def get_hot_files(self, project_dir: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get most frequently accessed files for a project."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT file_path, read_count, edit_count, last_read_at, last_edit_at
                FROM hot_files
                WHERE project_dir = ?
                ORDER BY (read_count + edit_count * 2) DESC
                LIMIT ?
                """,
                (project_dir, limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def record_stats(self, hit: bool, tokens_saved: int = 0) -> None:
        """Record cache statistics."""
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            if hit:
                await db.execute(
                    """
                    INSERT INTO cache_stats (stat_date, cache_hits, total_tokens_saved)
                    VALUES (?, 1, ?)
                    ON CONFLICT(stat_date) DO UPDATE SET
                        cache_hits = cache_hits + 1,
                        total_tokens_saved = total_tokens_saved + ?
                    """,
                    (today, tokens_saved, tokens_saved),
                )
            else:
                await db.execute(
                    """
                    INSERT INTO cache_stats (stat_date, cache_misses, total_compressions)
                    VALUES (?, 1, 1)
                    ON CONFLICT(stat_date) DO UPDATE SET
                        cache_misses = cache_misses + 1,
                        total_compressions = total_compressions + 1
                    """,
                    (today,),
                )
            await db.commit()

    async def cleanup_old_entries(self, days: int = 30) -> int:
        """Remove cache entries older than specified days. Returns count deleted."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM content_cache
                WHERE last_accessed_at < datetime('now', ? || ' days')
                """,
                (f"-{days}",),
            )
            content_deleted = cursor.rowcount

            cursor = await db.execute(
                """
                DELETE FROM file_summaries
                WHERE last_accessed_at < datetime('now', ? || ' days')
                """,
                (f"-{days}",),
            )
            summary_deleted = cursor.rowcount

            await db.commit()
            return content_deleted + summary_deleted


if __name__ == "__main__":
    # Initialize database when run directly
    init_db_sync()
