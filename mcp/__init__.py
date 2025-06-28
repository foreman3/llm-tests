"""Utilities and built-in tools for MCP integration."""

from typing import Any, Dict, Callable

import logging


def uppercase(
    context: Dict[str, Any], args: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Return a copy of context with the value uppercased."""
    key = args.get("field", "title") if args else "title"
    new = dict(context)
    new[key] = str(context.get(key, "")).upper()
    return new


def char_count(
    context: Dict[str, Any], args: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Add a character count field to the context."""
    key = args.get("field", "title") if args else "title"
    new = dict(context)
    new[f"{key}_chars"] = len(str(context.get(key, "")))
    return new


MCP_TOOLS = {
    "uppercase": uppercase,
    "char_count": char_count,
}


def discover_remote_tools(server_url: str) -> Dict[str, Callable[[Dict[str, Any], Dict[str, Any] | None], Dict[str, Any]]]:
    """Return tool callables discovered from an MCP server.

    The server is expected to expose a ``/tools`` endpoint returning a JSON
    mapping of tool names to descriptions. Each discovered tool is invoked by
    sending the context to ``POST {server_url}/<tool_name>``.
    """

    try:
        import requests

        resp = requests.get(f"{server_url.rstrip('/')}/tools", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # pragma: no cover - network errors
        logging.debug("failed to discover MCP tools from %s: %s", server_url, exc)
        return {}

    tools: Dict[str, Callable[[Dict[str, Any], Dict[str, Any] | None], Dict[str, Any]]] = {}
    if isinstance(data, dict):
        for name, desc in data.items():
            url = f"{server_url.rstrip('/')}/{name}"

            def call(context: Dict[str, Any], args: Dict[str, Any] | None = None, *, _url=url) -> Dict[str, Any]:
                try:
                    payload = {"context": context, "args": args or {}}
                    resp = requests.post(_url, json=payload, timeout=10)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as exc:  # pragma: no cover - network errors
                    logging.debug("mcp tool %s failed: %s", _url, exc)
                    return {"error": str(exc)}

            call.__doc__ = desc
            tools[name] = call

    return tools
