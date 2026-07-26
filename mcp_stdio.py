"""OmniRogue MCP server in stdio mode for local agent clients."""
import asyncio
import json
import uuid
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from config import settings, users
from proxy import OmniRogueProxy

proxy = OmniRogueProxy(users[0])
server = Server(settings.mcp_server_name)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="chat", description="Send messages to OmniRogue.", inputSchema={
            "type": "object", "properties": {
                "messages": {"type": "array", "items": {"type": "object", "properties": {
                    "role": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["role", "content"]}},
                "model": {"type": "string", "default": settings.default_model}},
            "required": ["messages"]}),
        Tool(name="save_conversation", description="Persist a conversation.", inputSchema={
            "type": "object", "properties": {"title": {"type": "string"},
            "conversation_id": {"type": "string"}, "model": {"type": "string"}},
            "required": ["title"]}),
        Tool(name="status", description="Return proxy/JWT status.",
             inputSchema={"type": "object", "properties": {}}),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "chat":
        result = proxy.chat_sync(arguments.get("messages", []), arguments.get("model", settings.default_model))
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
async def list_resources() -> list[Any]:
    return [{"uri": "omnirogue://status", "name": "Proxy Status", "mimeType": "application/json"}]


@server.read_resource()
async def read_resource(uri: str) -> str:
    if str(uri) == "omnirogue://status":
        return json.dumps(proxy.status(), indent=2)
    raise ValueError(f"Unknown resource: {uri}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
