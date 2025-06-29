"""Agentic ingestion of knowledge into the knowledge server."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import requests
from llm_pipeline.llm_methods import AgenticGoalStep

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


class KnowledgeIngestor:
    """Ingest files or URLs into a knowledge server."""

    def __init__(
        self,
        server_url: str,
        *,
        confidence_threshold: float = 0.5,
        mcp_servers: Iterable[str] | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.confidence_threshold = confidence_threshold
        self.mcp_servers = list(mcp_servers or [])
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI:
            self.client = OpenAI(api_key=api_key)
        tools = {
            "extract": self._extract_tool,
            "store": self._store_tool,
        }
        self.agent = AgenticGoalStep(
            goal="ingest knowledge", tools=tools, mcp_servers=self.mcp_servers
        )

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _load_source(self, source: str) -> str:
        if source.startswith("http://") or source.startswith("https://"):
            try:
                resp = requests.get(source, timeout=10)
                resp.raise_for_status()
                text = resp.text
            except Exception:  # pragma: no cover - network errors
                text = ""
        else:
            with open(source, "r", encoding="utf-8") as fh:
                text = fh.read()
        return re.sub("<[^>]+>", " ", text)

    def _extract_facts(self, text: str) -> List[Tuple[str, str, str, float]]:
        if self.client is None:
            facts: List[Tuple[str, str, str, float]] = []
            for subj, obj in re.findall(r"(\w+) needs (\w+)", text.lower()):
                facts.append((subj, "needs", obj, 0.8))
            for subj, obj in re.findall(r"(\w+) is a (\w+)", text.lower()):
                facts.append((subj, "isa", obj, 0.8))
            return facts
        prompt = (
            "Extract factual relationships from the text below. "
            "Respond with JSON list of objects each containing 'subject', "
            "'predicate', 'object' and 'confidence' (0-1).\n\n" + text
        )
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            msg = completion.choices[0].message.content.strip()
            data = json.loads(msg)
        except Exception:  # pragma: no cover - network or parse errors
            return []
        facts = []
        if isinstance(data, list):
            for item in data:
                subj = item.get("subject")
                pred = item.get("predicate")
                obj = item.get("object")
                conf = float(item.get("confidence", 0))
                if subj and pred and obj:
                    facts.append((subj, pred, obj, conf))
        return facts

    # ------------------------------------------------------------------
    # Agent tools
    # ------------------------------------------------------------------
    def _extract_tool(self, context: Dict[str, Any], _: Dict[str, Any] | None = None) -> Dict[str, Any]:
        text = context.get("text", "")
        facts = self._extract_facts(text)
        return {"facts": facts}

    def _store_tool(self, context: Dict[str, Any], _: Dict[str, Any] | None = None) -> Dict[str, Any]:
        results = []
        for subj, pred, obj, conf in context.get("facts", []):
            if conf < self.confidence_threshold:
                continue
            try:
                resp = requests.post(
                    f"{self.server_url}/add",
                    json={"subject": subj, "predicate": pred, "object": obj},
                    timeout=5,
                )
                resp.raise_for_status()
                results.append({"subject": subj, "predicate": pred, "object": obj})
            except Exception as exc:  # pragma: no cover - network errors
                results.append({"error": str(exc)})
        return {"added": results}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest(self, source: str) -> Dict[str, Any]:
        """Load ``source`` and ingest discovered facts."""
        text = self._load_source(source)
        return self.agent._run_agent({"text": text})


__all__ = ["KnowledgeIngestor"]
