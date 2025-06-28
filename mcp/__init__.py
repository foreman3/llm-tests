"""Simple MCP tools for the agentic pipeline."""

from typing import Any, Dict


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
