"""OpenAI content-shape and tool-calling compatibility helpers.

The OmniRogue upstream (POST /api/llm/chat) accepts only a list of
{role, content} messages where content is a plain string. It has no native
function-calling support and no notion of a "tool" role.

These helpers translate in both directions so that standard OpenAI clients and
agent frameworks (which send `tools`, `tool_choice`, assistant `tool_calls`, and
`role: "tool"` results) work unchanged:

  request  -> tool catalog is injected as a system instruction, and prior tool
              calls/results are rendered back into plain text turns
  response -> a JSON tool-call envelope emitted by the model is parsed into
              OpenAI `tool_calls` with finish_reason="tool_calls"

This module deliberately imports nothing from FastAPI/pydantic so it stays
unit-testable on its own.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

# Roles that carry tool results back to the model.
TOOL_ROLES = {"tool", "function"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# ── Content normalization ─────────────────────────────────────────────

def flatten_content(content: Any) -> str:
    """Normalize OpenAI/Anthropic message content into a plain string.

    Clients may send `content` either as a string or as a list of typed blocks
    like [{"type": "text", "text": "..."}] when attaching files or images.
    Upstream requires a string, so block arrays are joined and non-text parts
    (e.g. images, which the upstream cannot render) are dropped.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    # Numbers, booleans, or unexpected objects: coerce rather than reject.
    return str(content)


# ── Request translation ───────────────────────────────────────────────

def tool_specs(tools: Any, tool_choice: Any = None) -> list[dict]:
    """Normalize an OpenAI `tools` (or legacy `functions`) payload.

    Returns [] when tool calling is disabled or nothing usable was supplied,
    which callers use as the signal to take the plain chat path.
    """
    if tool_choice == "none" or not isinstance(tools, list):
        return []
    specs: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        specs.append(
            {
                "name": name,
                "description": function.get("description") or "",
                "parameters": parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}},
            }
        )
    return specs


def forced_tool_name(tool_choice: Any) -> str | None:
    """Extract the tool name from tool_choice={"type":"function",...}, if any."""
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
        if isinstance(tool_choice.get("name"), str):
            return tool_choice["name"]
    return None


def tool_system_prompt(specs: list[dict], tool_choice: Any = None) -> str:
    """Build the system instruction that teaches the JSON tool-call protocol."""
    catalog = json.dumps(specs, indent=2, ensure_ascii=False)
    forced = forced_tool_name(tool_choice)

    lines = [
        "You can call tools. The available tools, with JSON Schema for their arguments, are:",
        "",
        catalog,
        "",
        "To call one or more tools, reply with ONLY a single JSON object in exactly this shape "
        "and no other text, no explanation, and no markdown fence:",
        '{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments matching that tool\'s schema>}}]}',
        "",
        "Rules:",
        "- Use only tool names listed above, spelled exactly.",
        "- `arguments` must be a JSON object, never a string.",
        "- You may include several entries in `tool_calls` to request parallel calls.",
        "- Tool results come back as messages beginning with 'Result of tool'. After you have "
        "the results you need, answer the user normally in plain text.",
    ]
    if forced:
        lines.append(f"- You must call the tool `{forced}` now, using the JSON format above.")
    elif tool_choice in ("required", "any"):
        lines.append("- You must call at least one tool now, using the JSON format above.")
    else:
        lines.append("- If no tool is needed, reply normally in plain text instead of JSON.")
    return "\n".join(lines)


def _render_assistant_tool_calls(content: str, tool_calls: Any) -> str:
    """Re-render a prior assistant tool call as the JSON envelope it 'sent'."""
    rendered = []
    for call in tool_calls if isinstance(tool_calls, list) else []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = function.get("name")
        if not isinstance(name, str):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"input": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        rendered.append({"name": name, "arguments": arguments})
    if not rendered:
        return content
    envelope = json.dumps({"tool_calls": rendered}, ensure_ascii=False)
    return f"{content}\n{envelope}".strip()


def render_messages(raw_messages: list[dict], specs: list[dict], tool_choice: Any = None) -> list[dict]:
    """Convert OpenAI-style messages into upstream {role, content} messages.

    - block-array content is flattened to text
    - assistant `tool_calls` become the JSON envelope the model is taught to emit
    - `role: "tool"` / `"function"` results become user turns the upstream accepts
    - when tools are active, a system instruction is prepended
    """
    rendered: list[dict] = []

    if specs:
        rendered.append({"role": "system", "content": tool_system_prompt(specs, tool_choice)})

    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or "user"
        content = flatten_content(message.get("content"))

        if role in TOOL_ROLES:
            name = message.get("name") or "tool"
            call_id = message.get("tool_call_id")
            header = f"Result of tool `{name}`"
            if call_id:
                header += f" (call {call_id})"
            rendered.append({"role": "user", "content": f"{header}:\n{content}"})
            continue

        if role == "assistant" and message.get("tool_calls"):
            content = _render_assistant_tool_calls(content, message["tool_calls"])

        if not content:
            # Upstream rejects empty content; drop the turn instead of erroring.
            continue
        rendered.append({"role": role, "content": content})

    return rendered


# ── Response translation ──────────────────────────────────────────────

def _first_json_object(text: str) -> tuple[Any, str]:
    """Find the first balanced {...} in text. Returns (parsed, preceding_text)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1]), text[:start]
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None, text


def extract_tool_calls(text: str, valid_names: set[str] | None = None) -> tuple[list[dict] | None, str]:
    """Parse a model reply into OpenAI tool_calls.

    Returns (tool_calls, leftover_text). tool_calls is None when the reply is
    ordinary prose, in which case leftover_text is the original text.
    Tolerates markdown fences, a bare {"name": ..., "arguments": ...} object,
    and arguments delivered as a JSON string.
    """
    if not text:
        return None, text

    candidates = [text]
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1))

    for candidate in candidates:
        parsed, leading = _first_json_object(candidate)
        if not isinstance(parsed, dict):
            continue

        raw_calls = parsed.get("tool_calls")
        if raw_calls is None and isinstance(parsed.get("name"), str):
            raw_calls = [parsed]  # bare single-call form
        if not isinstance(raw_calls, list):
            continue

        calls: list[dict] = []
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function") if isinstance(item.get("function"), dict) else item
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            if valid_names and name not in valid_names:
                continue  # hallucinated tool name: treat reply as prose
            arguments = function.get("arguments")
            if arguments is None:
                arguments = function.get("parameters")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            )

        if calls:
            return calls, leading.strip()

    return None, text


def apply_tool_calls(completion: dict, valid_names: set[str] | None = None) -> dict:
    """Rewrite a chat.completion in place to expose tool_calls when present."""
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        return completion
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return completion

    calls, leftover = extract_tool_calls(message.get("content") or "", valid_names)
    if not calls:
        return completion

    message["content"] = leftover or None
    message["tool_calls"] = calls
    choice["finish_reason"] = "tool_calls"
    return completion


def completion_to_chunks(completion: dict, include_usage: bool = False) -> list[dict]:
    """Convert a finished completion into OpenAI streaming chunks.

    Tool calls cannot be detected until the full reply is parsed, so streamed
    tool-calling requests are fulfilled non-streaming upstream and replayed as
    chunks here, preserving the SSE contract clients expect.
    """
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    base = {
        "id": completion.get("id"),
        "object": "chat.completion.chunk",
        "created": completion.get("created"),
        "model": completion.get("model"),
    }

    def chunk(delta: dict, finish_reason: Any = None) -> dict:
        return {
            **base,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
        }

    chunks = [chunk({"role": "assistant"})]
    if message.get("content"):
        chunks.append(chunk({"content": message["content"]}))
    for index, call in enumerate(message.get("tool_calls") or []):
        chunks.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": call["function"],
                        }
                    ]
                }
            )
        )
    final = chunk({}, choice.get("finish_reason") or "stop")
    if include_usage and completion.get("usage"):
        final["usage"] = completion["usage"]
    chunks.append(final)
    return chunks
