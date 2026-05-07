"""Session persistence -- JSON snapshots of ConversationContext.

Sessions live at `~/.miniclaudecode/sessions/{id}.json`. Each save fully
overwrites the file (atomic via tmp + rename) so an interrupted save can
never produce a half-written corrupted snapshot.

Snapshot shape:
    {
      "id": "20260503-153045-abcd",
      "created_at": "2026-05-03T15:30:45.123456",
      "updated_at": "2026-05-03T15:42:11.987654",
      "model": "claude-sonnet-4-5",
      "provider": "anthropic",
      "profile": "anthropic",
      "system_prompt": "...",
      "messages": [...],            // ConversationContext.messages verbatim
      "compactions": 2,
      "depth": 0,
      "todos": [...],               // serialized TodoStore items (optional)
      "telemetry": {...}            // optional summary, see _telemetry_dump
    }

The CLI exposes `/resume <id>` which calls `load_session` and reconstructs the
context. Tool registries / hooks / skills are NOT serialized -- those are
re-derived from the current settings.json on resume.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..agent_loop import AgentLoop


SESSION_DIR = Path.home() / ".miniclaudecode" / "sessions"


class SessionStore:
    """Owns one session's id and writes snapshots through `record(...)`.

    AgentLoop holds at most one. Calling `record(agent)` after each user
    turn keeps the on-disk snapshot fresh. Construction is cheap (no IO);
    the directory is created lazily on first save.
    """

    def __init__(self, session_id: str | None = None, base_dir: Path | None = None) -> None:
        self.id = session_id or _new_session_id()
        self.base_dir = base_dir or SESSION_DIR
        self.created_at = _utcnow_iso()

    @property
    def path(self) -> Path:
        return self.base_dir / f"{self.id}.json"

    def record(self, agent: "AgentLoop") -> Path:
        """Atomically write a snapshot of `agent` to disk."""
        snapshot = _serialize(agent, self.id, self.created_at)
        return _write_atomic(self.path, snapshot)


def save_session(agent: "AgentLoop", store: SessionStore | None = None) -> SessionStore:
    """One-shot save: build a fresh store if needed, write the snapshot, return it."""
    store = store or SessionStore()
    store.record(agent)
    return store


def load_session(session_id: str, base_dir: Path | None = None) -> dict[str, Any]:
    """Read a snapshot dict from disk. Raises FileNotFoundError if absent."""
    base_dir = base_dir or SESSION_DIR
    path = base_dir / f"{session_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions(base_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return summaries (id, model, created_at, updated_at, message_count)
    for every snapshot, sorted newest-first by updated_at.
    """
    base_dir = base_dir or SESSION_DIR
    if not base_dir.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for path in base_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "id": data.get("id", path.stem),
            "model": data.get("model"),
            "provider": data.get("provider"),
            "profile": data.get("profile"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": len(data.get("messages") or []),
        })
    out.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
    return out


# ---------- serialization ----------

def _serialize(agent: "AgentLoop", session_id: str, created_at: str) -> dict[str, Any]:
    cfg = agent.config
    todos: list[dict[str, Any]] = []
    if getattr(agent, "todo_store", None) is not None:
        for t in agent.todo_store.todos:
            todos.append(asdict(t))

    return {
        "id": session_id,
        "created_at": created_at,
        "updated_at": _utcnow_iso(),
        "model": cfg.model,
        "provider": cfg.provider.value,
        "profile": cfg.profile_name,
        "permission_mode": cfg.permission_mode.value,
        "system_prompt": agent.context.system_prompt,
        "messages": agent.context.messages,
        "compactions": agent.context.compactions,
        "depth": agent.context.depth,
        "todos": todos,
        "telemetry": _telemetry_dump(agent),
    }


def _telemetry_dump(agent: "AgentLoop") -> dict[str, Any]:
    tele = getattr(agent, "telemetry", None)
    if tele is None:
        return {}
    cum = tele.cumulative
    return {
        "input_tokens": cum.input_tokens,
        "output_tokens": cum.output_tokens,
        "cost_usd": cum.cost_usd,
        "calls": len(tele.turns),
    }


def restore_into(agent: "AgentLoop", snapshot: dict[str, Any]) -> None:
    """Apply a loaded snapshot to an AgentLoop in place.

    Only context state is restored. Tool registry, hooks, and skills come
    from the *current* config (so resuming with a tweaked settings.json
    picks up the change), not the snapshot.
    """
    agent.context.messages = list(snapshot.get("messages") or [])
    agent.context.set_system_prompt(snapshot.get("system_prompt") or "")
    agent.context.compactions = int(snapshot.get("compactions") or 0)
    agent.context.depth = int(snapshot.get("depth") or 0)

    # Best-effort todo restore: skip if shape doesn't match so a stale snapshot
    # never blocks resume.
    if hasattr(agent, "todo_store"):
        from ..tools.todo_write import Todo
        agent.todo_store.todos = []
        for entry in snapshot.get("todos") or []:
            try:
                agent.todo_store.todos.append(Todo(
                    content=entry["content"],
                    status=entry.get("status", "pending"),
                    active_form=entry.get("active_form", ""),
                ))
            except (KeyError, TypeError):
                continue


# ---------- helpers ----------

def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _new_session_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def _write_atomic(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    # Write to a sibling tmp file then rename so a kill mid-write can't leave
    # a half-baked JSON file behind.
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.stem}-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path
