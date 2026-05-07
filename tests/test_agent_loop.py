"""AgentLoop tests with a stub LLMClient (P1).

Drives the loop end-to-end without hitting a real API: the stub returns a
scripted sequence of LLMResponse objects.
"""
from __future__ import annotations

from typing import Any

from miniclaudecode.agent_loop import AgentLoop
from miniclaudecode.config import Config, PermissionMode
from miniclaudecode.llm.base import LLMClient, LLMResponse, ToolCall
from miniclaudecode.tools.base import Tool, ToolRegistry, ToolResult


class ScriptedClient(LLMClient):
    """Returns the next LLMResponse from a scripted list each call."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.script:
            return LLMResponse(text_blocks=["(end)"], stop_reason="end_turn",
                               raw_content=[{"type": "text", "text": "(end)"}])
        return self.script.pop(0)


class EchoTool(Tool):
    """Trivial deterministic tool for end-to-end loop tests."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo input back."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}

    def execute(self, params: dict[str, Any]) -> ToolResult:
        return ToolResult(output=f"echo:{params.get('msg', '')}")


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        text_blocks=[text],
        raw_content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
    )


def _tool_response(tool_calls: list[ToolCall]) -> LLMResponse:
    raw = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tool_calls]
    return LLMResponse(text_blocks=[], tool_calls=tool_calls, raw_content=raw, stop_reason="tool_use")


def _make_agent(client: ScriptedClient) -> AgentLoop:
    cfg = Config(permission_mode=PermissionMode.AUTO)
    reg = ToolRegistry()
    reg.register(EchoTool())
    return AgentLoop(config=cfg, registry=reg, client=client)


def test_single_turn_no_tool():
    client = ScriptedClient([_text_response("hello")])
    agent = _make_agent(client)
    out = agent.run("hi")
    assert out == "hello"
    # Context should contain user + assistant
    assert len(agent.context.messages) == 2


def test_tool_use_then_text():
    client = ScriptedClient([
        _tool_response([ToolCall(id="t1", name="echo", input={"msg": "ping"})]),
        _text_response("done"),
    ])
    agent = _make_agent(client)
    out = agent.run("call echo")
    assert out == "done"
    # Verify tool_result block was appended after the assistant tool_use turn.
    last_user = [m for m in agent.context.messages if m["role"] == "user"][-1]
    assert isinstance(last_user["content"], list)
    assert last_user["content"][0]["tool_use_id"] == "t1"
    assert "echo:ping" in last_user["content"][0]["content"]


def test_unknown_tool_returns_error_result():
    client = ScriptedClient([
        _tool_response([ToolCall(id="t1", name="nosuch", input={})]),
        _text_response("recovered"),
    ])
    agent = _make_agent(client)
    out = agent.run("call missing")
    assert out == "recovered"
    last_user = [m for m in agent.context.messages if m["role"] == "user"][-1]
    block = last_user["content"][0]
    assert block["is_error"] is True
    assert "unknown" in block["content"].lower() or "nosuch" in block["content"].lower()


def test_max_turns_safety():
    # Always returns tool_use -> would loop forever without max_turns.
    perpetual = LLMResponse(
        tool_calls=[ToolCall(id="x", name="echo", input={"msg": "loop"})],
        raw_content=[{"type": "tool_use", "id": "x", "name": "echo", "input": {"msg": "loop"}}],
        stop_reason="tool_use",
    )
    client = ScriptedClient([perpetual] * 10)
    cfg = Config(permission_mode=PermissionMode.AUTO, max_turns=3)
    reg = ToolRegistry()
    reg.register(EchoTool())
    agent = AgentLoop(config=cfg, registry=reg, client=client)
    out = agent.run("never ends")
    assert "max turns" in out.lower()
    # Loop body called exactly max_turns times.
    assert len(client.calls) == 3
