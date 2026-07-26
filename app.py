"""OmniRogue Agent API — authenticated OpenAI compatibility + MCP over SSE."""
from __future__ import annotations

import json
import secrets
import uuid
from typing import Any, Dict, Optional

import requests as rq
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, ConfigDict

from config import settings, users
from proxy import OmniRogueProxy
from tool_compat import (
    apply_tool_calls,
    completion_to_chunks,
    flatten_content,
    render_messages,
    tool_specs,
)

STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

try:
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.types import Resource, TextContent, Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

proxies = {user.api_key: OmniRogueProxy(user) for user in users}
default_proxy = next(iter(proxies.values()))

app = FastAPI(
    title="OmniRogue Agent API",
    description="OpenAI-compatible chat proxy + MCP server for OmniRogue.",
    version=settings.mcp_server_version,
)


# ═══════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════

def _extract_api_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-api-key")


def _resolve_proxy(key: str | None) -> OmniRogueProxy | None:
    if not settings.require_api_key:
        return default_proxy
    if not key:
        return None
    # Compare as bytes so non-ASCII keys cannot raise, and in constant time so
    # timing does not leak key prefixes.
    candidate = key.encode("utf-8")
    for configured_key, proxy in proxies.items():
        if secrets.compare_digest(candidate, configured_key.encode("utf-8")):
            return proxy
    return None


# Dependency-based auth (instead of HTTP middleware) so streaming responses and
# the MCP SSE transport keep direct access to the raw ASGI send channel.
def require_user(request: Request) -> OmniRogueProxy:
    proxy = _resolve_proxy(_extract_api_key(request))
    if proxy is None:
        raise HTTPException(
            status_code=401,
            detail={"message": "Invalid or missing API key", "type": "authentication_error"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return proxy


@app.exception_handler(StarletteHTTPException)
async def openai_style_http_errors(request: Request, exc: StarletteHTTPException):
    """Return errors as {"error": {...}} instead of FastAPI's {"detail": ...}.

    OpenAI client libraries read the top-level `error` object, so wrapping our
    payload in `detail` makes failures surface as unparsable responses.
    """
    if request.url.path.startswith("/mcp"):
        return await http_exception_handler(request, exc)
    detail = exc.detail
    error = detail if isinstance(detail, dict) else {"message": str(detail), "type": "invalid_request_error"}
    return JSONResponse({"error": error}, status_code=exc.status_code, headers=exc.headers)


# ═══════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """One OpenAI-style message.

    `content` accepts either a plain string or a block array (sent by clients
    when attaching files). Tool fields are accepted so agent frameworks can
    replay assistant tool calls and `role: "tool"` results.
    """

    model_config = ConfigDict(extra="allow")

    role: str = "user"
    content: Any = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = settings.default_model
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    # Tool calling. The upstream has no native support, so these are emulated
    # via a JSON protocol injected as a system instruction (see tool_compat).
    tools: Optional[list[Dict[str, Any]]] = None
    tool_choice: Any = None
    # Legacy OpenAI function-calling aliases.
    functions: Optional[list[Dict[str, Any]]] = None
    function_call: Any = None

    def resolved_tools(self) -> list[Dict[str, Any]]:
        return self.tools or self.functions or []

    def resolved_tool_choice(self) -> Any:
        return self.tool_choice if self.tool_choice is not None else self.function_call


class CookieUpdateRequest(BaseModel):
    jwt: Optional[str] = None
    client_cookie: Optional[str] = None
    client_uat: Optional[str] = None
    cf_bm: Optional[str] = None
    cfuvid: Optional[str] = None


class ConversationRequest(BaseModel):
    title: str = "Untitled"
    conversation_id: Optional[str] = None
    model: str = settings.default_model


# ═══════════════════════════════════════════════════════════════════════
#  OPENAI-COMPATIBLE ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, proxy: OmniRogueProxy = Depends(require_user)):
    if not req.messages:
        return JSONResponse({"error": {"message": "messages required", "type": "invalid_request_error"}}, status_code=400)

    tool_choice = req.resolved_tool_choice()
    specs = tool_specs(req.resolved_tools(), tool_choice)
    messages = render_messages([m.model_dump() for m in req.messages], specs, tool_choice)
    if not messages:
        return JSONResponse({"error": {"message": "no usable message content", "type": "invalid_request_error"}}, status_code=400)
    include_usage = (req.stream_options or {}).get("include_usage", False)

    if specs:
        # Emulated tool calling: a tool call is only detectable once the whole
        # reply is parsed, so always fetch non-streaming upstream, then replay
        # as SSE chunks if the client asked to stream.
        try:
            completion = apply_tool_calls(
                proxy.chat_sync(messages, req.model),
                {spec["name"] for spec in specs},
            )
        except rq.exceptions.HTTPError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "upstream_error", "detail": exc.response.text}},
                status_code=exc.response.status_code,
            )

        if not req.stream:
            return JSONResponse(completion)

        def replay():
            for chunk in completion_to_chunks(completion, include_usage=include_usage):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(replay(), media_type="text/event-stream", headers=STREAM_HEADERS)

    if req.stream:
        def generate():
            try:
                for chunk in proxy.chat_stream(messages, req.model, include_usage=include_usage):
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as exc:
                error = {"error": {"message": str(exc), "type": "upstream_error"}}
                yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return JSONResponse(proxy.chat_sync(messages, req.model))
    except rq.exceptions.HTTPError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "upstream_error", "detail": exc.response.text}},
            status_code=exc.response.status_code,
        )


@app.post("/v1/conversations")
def create_conversation(req: ConversationRequest, proxy: OmniRogueProxy = Depends(require_user)):
    conversation_id = req.conversation_id or str(uuid.uuid4())
    try:
        return JSONResponse(proxy.save_conversation(req.title, conversation_id, req.model))
    except rq.exceptions.HTTPError as exc:
        return JSONResponse(
            {"error": str(exc), "detail": exc.response.text},
            status_code=exc.response.status_code,
        )


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "configured_users": len(proxies)})


@app.post("/auth/refresh")
def force_refresh(proxy: OmniRogueProxy = Depends(require_user)):
    try:
        proxy.ensure_fresh(force=True)
        return JSONResponse({"status": "refreshed", "proxy": proxy.status()})
    except Exception as exc:
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)


@app.post("/auth/cookies")
def update_cookies(req: CookieUpdateRequest, proxy: OmniRogueProxy = Depends(require_user)):
    proxy.update_cookies(
        jwt=req.jwt,
        client_cookie=req.client_cookie,
        client_uat=req.client_uat,
        cf_bm=req.cf_bm,
        cfuvid=req.cfuvid,
    )
    return JSONResponse({"status": "updated", "proxy": proxy.status()})


# ═══════════════════════════════════════════════════════════════════════
#  MCP SERVER (SSE)
# ═══════════════════════════════════════════════════════════════════════

if MCP_AVAILABLE:
    sse_transport = SseServerTransport("/mcp/messages/")

    def build_mcp_server(proxy: OmniRogueProxy) -> Server:
        """Build a per-connection MCP server bound to one authenticated user."""
        server = Server(settings.mcp_server_name)

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="chat",
                    description="Send a chat message to OmniRogue and return the assistant's reply.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "List of {role, content} messages.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["role", "content"],
                                },
                            },
                            "model": {"type": "string", "default": settings.default_model},
                        },
                        "required": ["messages"],
                    },
                ),
                Tool(
                    name="save_conversation",
                    description="Persist a conversation to the OmniRogue backend.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "default": "Untitled"},
                            "conversation_id": {"type": "string"},
                            "model": {"type": "string", "default": settings.default_model},
                        },
                        "required": ["title"],
                    },
                ),
                Tool(
                    name="status",
                    description="Return this user's proxy/JWT status.",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "chat":
                # MCP clients can also send block-array content; normalize it.
                mcp_messages = [
                    {
                        "role": message.get("role", "user"),
                        "content": flatten_content(message.get("content")),
                    }
                    for message in arguments.get("messages", [])
                    if isinstance(message, dict)
                ]
                result = proxy.chat_sync(
                    mcp_messages,
                    arguments.get("model", settings.default_model),
                )
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return [TextContent(type="text", text=text)]
            if name == "save_conversation":
                result = proxy.save_conversation(
                    arguments.get("title", "Untitled"),
                    arguments.get("conversation_id") or str(uuid.uuid4()),
                    arguments.get("model", settings.default_model),
                )
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            if name == "status":
                return [TextContent(type="text", text=json.dumps(proxy.status(), indent=2))]
            raise ValueError(f"Unknown tool: {name}")

        @server.list_resources()
        async def list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="omnirogue://status",
                    name="Proxy Status",
                    mimeType="application/json",
                    description="Current JWT/session status for the authenticated user.",
                )
            ]

        @server.read_resource()
        async def read_resource(uri) -> str:
            if str(uri) == "omnirogue://status":
                return json.dumps(proxy.status(), indent=2)
            raise ValueError(f"Unknown resource: {uri}")

        return server

    @app.get("/mcp/sse")
    async def mcp_sse(request: Request, proxy: OmniRogueProxy = Depends(require_user)):
        server = build_mcp_server(proxy)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    # The SSE transport posts back to the mount path with ?session_id=...,
    # so this must be the bare path (not a /{session_id} path parameter).
    @app.post("/mcp/messages/")
    async def mcp_messages(request: Request, _: OmniRogueProxy = Depends(require_user)):
        await sse_transport.handle_post_message(request.scope, request.receive, request._send)

else:
    @app.get("/mcp/sse")
    async def mcp_sse_unavailable():
        return JSONResponse(
            {"error": "MCP SDK not installed. Add 'mcp' to requirements.txt and redeploy."},
            status_code=501,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
