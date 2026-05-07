"""OpenAI-compatible LLM client (P5).

Works with any service that exposes the OpenAI Chat Completions API at a
configurable base_url. Verified shapes:
    - OpenAI official      (base_url=https://api.openai.com/v1)
    - DeepSeek             (base_url=https://api.deepseek.com/v1)
    - OpenRouter           (base_url=https://openrouter.ai/api/v1)
    - SiliconFlow / Moonshot / Zhipu (most claim "OpenAI-compatible")
    - Local Ollama         (base_url=http://localhost:11434/v1)
    - Generic LiteLLM / vLLM proxies

The agent loop only ever sees Anthropic-shaped internal messages; this client
translates in both directions:

    Internal (Anthropic-shape) -> OpenAI Chat Completions
        - assistant content is split into text + tool_calls
        - tool_use blocks become OpenAI `tool_calls` with stringified arguments
        - tool_result blocks become separate `role: "tool"` messages
        - top-level `system` becomes a leading `system` role message

    OpenAI response -> Internal LLMResponse
        - choices[0].message.content -> text block
        - choices[0].message.tool_calls -> tool_use blocks (arguments parsed)
        - usage.prompt_tokens / completion_tokens -> input_tokens / output_tokens

Provider quirks handled:
    - assistant.content may be `None` when only tool_calls are present (OpenAI
      requires `None`, not empty string, when tool_calls is the only payload)
    - tool_calls.function.arguments is a JSON string; we parse defensively
      since some providers occasionally emit malformed JSON for tiny tools
    - finish_reason "tool_calls" / "stop" / "length" mapped to a stable string
"""
from __future__ import annotations

import json
from typing import Any

from .base import LLMClient, LLMResponse, ToolCall


class OpenAICompatClient(LLMClient):
    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover -- handled in factory
            raise RuntimeError(
                "openai package not installed; run `pip install openai`"
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        oa_messages: list[dict[str, Any]] = []
        if system:
            oa_messages.append({"role": "system", "content": system})
        oa_messages.extend(self._to_openai_messages(messages))

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oa_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)

        response = self._client.chat.completions.create(**kwargs)
        return self._to_internal_response(response)

    # ---------- inbound translation ----------

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    @classmethod
    def _to_openai_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                continue

            if role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        })
                merged_text = "\n".join(t for t in text_parts if t)
                # OpenAI requires content=None (not "") when only tool_calls are present.
                msg_out: dict[str, Any] = {
                    "role": "assistant",
                    "content": merged_text or None,
                }
                if tool_calls:
                    msg_out["tool_calls"] = tool_calls
                out.append(msg_out)
            else:
                # role == "user" with list content: either tool_results or text blocks.
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if tool_results:
                    # Each tool_result becomes its own `tool` role message in OpenAI.
                    for tr in tool_results:
                        body = tr.get("content")
                        if isinstance(body, list):
                            body = "".join(
                                b.get("text", "") for b in body
                                if isinstance(b, dict)
                            )
                        if tr.get("is_error"):
                            body = f"[ERROR] {body}"
                        out.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": str(body if body is not None else ""),
                        })
                else:
                    text = "".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    out.append({"role": "user", "content": text})
        return out

    # ---------- outbound translation ----------

    @staticmethod
    def _to_internal_response(response: Any) -> LLMResponse:
        text_blocks: list[str] = []
        tool_calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = []

        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None) if choice else None

        if message is not None:
            content = getattr(message, "content", None)
            if content:
                text_blocks.append(content)
                raw.append({"type": "text", "text": content})

            for tc in (getattr(message, "tool_calls", None) or []):
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args_raw = getattr(fn, "arguments", "") or ""
                try:
                    parsed = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
                tc_id = getattr(tc, "id", "") or ""
                tc_name = getattr(fn, "name", "") or ""
                tool_calls.append(ToolCall(id=tc_id, name=tc_name, input=parsed))
                raw.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": tc_name,
                    "input": parsed,
                })

        usage_obj = getattr(response, "usage", None)
        usage: dict[str, int] = {}
        if usage_obj is not None:
            usage = {
                "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            }

        finish = getattr(choice, "finish_reason", "") if choice else ""
        # Map OpenAI finish reasons onto Anthropic-shaped names so callers can
        # treat both providers uniformly.
        stop_reason = {
            "tool_calls": "tool_use",
            "stop": "end_turn",
            "length": "max_tokens",
        }.get(finish or "", finish or "")

        return LLMResponse(
            text_blocks=text_blocks,
            tool_calls=tool_calls,
            raw_content=raw,
            stop_reason=stop_reason,
            usage=usage,
        )
