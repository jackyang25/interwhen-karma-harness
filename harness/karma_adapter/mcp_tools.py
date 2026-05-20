"""MedAI MCP client wrapper.

Exposes the three hierarchical meta-tools (discover, schema, compute) to the
model in the Anthropic Messages API tool-use format. The actual transport is
fastmcp; auth is injected by harness/_patches.py from EKA_API_TOKEN.

This module is *transport-only*: it does not know about clinical content, only
about translating Anthropic tool-use blocks into MCP calls and back.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastmcp import Client

MEDAI_MCP_URL = "https://medai-tools.eka.care/mcp"


def fetch_tool_schemas() -> list[dict[str, Any]]:
    """Pull MedAI's tool list and reshape into Anthropic tool-use schema.

    Returns the list[dict] expected by client.messages.create(tools=...).
    Each dict has name, description, input_schema.
    """

    async def _list() -> list[Any]:
        async with Client(MEDAI_MCP_URL) as c:
            return await c.list_tools()

    raw = asyncio.run(_list())
    out = []
    for t in raw:
        out.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
        )
    return out


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Invoke one MedAI tool and return its text output as a string.

    The result is whatever the tool returns — for the calculator path, this is
    the computed value. Stringified so it can go straight into a tool_result
    content block.
    """

    async def _call() -> Any:
        async with Client(MEDAI_MCP_URL) as c:
            return await c.call_tool(name, arguments)

    result = asyncio.run(_call())
    # FastMCP returns a CallToolResult with .content (list of content blocks).
    # Collapse to a single string for tool_result injection.
    if hasattr(result, "content"):
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(json.dumps(block, default=str))
        return "\n".join(parts) if parts else json.dumps(result, default=str)
    return json.dumps(result, default=str)
