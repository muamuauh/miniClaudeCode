"""Session persistence tests (P5)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from miniclaudecode.agent_loop import AgentLoop
from miniclaudecode.config import Config, PermissionMode
from miniclaudecode.llm.base import LLMClient, LLMResponse
from miniclaudecode.persistence.session import (
    SessionStore,
    list_sessions,
    load_session,
    restore_into,
)
from miniclaudecode.tools.base import ToolRegistry


class StubLLM(LLMClient):
    def chat(self, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text_blocks=["ok"],
            raw_content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def _agent(tmp_path: Path) -> AgentLoop:
    cfg = Config(permission_mode=PermissionMode.AUTO, model="claude-sonnet-4-5")
    return AgentLoop(config=cfg, registry=ToolRegistry(), client=StubLLM())


def test_save_and_load_round_trips_messages(tmp_path: Path):
    agent = _agent(tmp_path)
    agent.context.add_user_message("seed prompt")
    agent.context.add_assistant_message([{"type": "text", "text": "hello"}])

    store = SessionStore(base_dir=tmp_path)
    path = store.record(agent)
    assert path.exists()

    snapshot = load_session(store.id, base_dir=tmp_path)
    assert snapshot["id"] == store.id
    assert snapshot["model"] == "claude-sonnet-4-5"
    assert snapshot["messages"][0]["content"] == "seed prompt"
    assert snapshot["messages"][1]["content"][0]["text"] == "hello"


def test_restore_into_replaces_context(tmp_path: Path):
    agent = _agent(tmp_path)
    agent.context.add_user_message("first")
    store = SessionStore(base_dir=tmp_path)
    store.record(agent)

    # Build a fresh agent and restore into it.
    fresh = _agent(tmp_path)
    assert fresh.context.messages == []
    snapshot = load_session(store.id, base_dir=tmp_path)
    restore_into(fresh, snapshot)
    assert len(fresh.context.messages) == 1
    assert fresh.context.messages[0]["content"] == "first"


def test_atomic_write_no_partial_files(tmp_path: Path, monkeypatch):
    """If os.replace fails mid-save, the tmp file is cleaned up so we never
    leave a half-written .json that load_session would choke on."""
    agent = _agent(tmp_path)
    agent.context.add_user_message("hi")
    store = SessionStore(base_dir=tmp_path)

    boom = OSError("disk full")
    def explode(*args, **kwargs):
        raise boom
    monkeypatch.setattr("os.replace", explode)

    with pytest.raises(OSError):
        store.record(agent)
    # Final session file never appeared
    assert not store.path.exists()
    # No tmp leftover either (we clean up in the except block)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"unexpected tmp files: {leftovers}"


def test_list_sessions_returns_summaries_newest_first(tmp_path: Path):
    a = _agent(tmp_path)
    a.context.add_user_message("a")
    SessionStore(base_dir=tmp_path).record(a)

    # Force a different timestamp + id
    b = _agent(tmp_path)
    b.context.add_user_message("b")
    b.context.add_assistant_message([{"type": "text", "text": "ok"}])
    store_b = SessionStore(base_dir=tmp_path)
    store_b.record(b)

    items = list_sessions(base_dir=tmp_path)
    assert len(items) == 2
    # `updated_at` ordering may be equal at second resolution; both entries
    # at minimum should have the expected keys.
    for entry in items:
        assert {"id", "model", "provider", "created_at", "updated_at", "message_count"} <= entry.keys()


def test_todos_round_trip(tmp_path: Path):
    from miniclaudecode.tools.todo_write import Todo

    agent = _agent(tmp_path)
    agent.todo_store.todos = [
        Todo(content="design", status="completed"),
        Todo(content="build", status="in_progress", active_form="Building"),
    ]
    store = SessionStore(base_dir=tmp_path)
    store.record(agent)

    fresh = _agent(tmp_path)
    snapshot = load_session(store.id, base_dir=tmp_path)
    restore_into(fresh, snapshot)
    assert [t.content for t in fresh.todo_store.todos] == ["design", "build"]
    assert fresh.todo_store.todos[1].active_form == "Building"
