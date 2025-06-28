"""Minimal MCP server with a knowledge store."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import re
import threading
from typing import Dict, List, Tuple, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


class KnowledgeStore:
    """Simple in-memory knowledge graph with optional persistence."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.graph: Dict[str, List[Tuple[str, str]]] = {}
        if path:
            self.load()

    def load(self) -> None:
        """Load knowledge from ``self.path`` if it exists."""
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.graph = json.load(fh)
            except Exception:  # pragma: no cover - ignore corrupt file
                self.graph = {}

    def save(self) -> None:
        """Persist knowledge to ``self.path`` if set."""
        if self.path:
            try:
                with open(self.path, "w", encoding="utf-8") as fh:
                    json.dump(self.graph, fh)
            except Exception:  # pragma: no cover - disk errors
                pass

    def add_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Add a fact to the store."""
        self.graph.setdefault(subject, []).append((predicate, obj))
        self.save()

    def find(self, subject: str, predicate: str) -> List[str]:
        """Return all objects matching ``subject``/``predicate``."""
        return [o for p, o in self.graph.get(subject, []) if p == predicate]

    def parent(self, subject: str) -> Optional[str]:
        """Return the object for an ``isa`` relationship if present."""
        isa = self.find(subject, "isa")
        return isa[0] if isa else None


class _Handler(BaseHTTPRequestHandler):
    """Request handler created by ``make_handler``."""

    store: KnowledgeStore
    client: Optional[OpenAI]

    def _json_response(self, data: Dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802  - interface method
        if self.path == "/tools":
            self._json_response({"query": "ask a question", "add": "add a fact"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802  - interface method
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path == "/add":
            subj = body.get("subject")
            pred = body.get("predicate")
            obj = body.get("object")
            if subj and pred and obj:
                self.store.add_fact(subj, pred, obj)
                self._json_response({"status": "ok"})
            else:
                self._json_response({"error": "invalid"})
        elif self.path == "/query":
            text = body.get("text", "")
            resp = answer_question(text, self.store, self.client)
            self._json_response({"answer": resp})
        else:
            self.send_response(404)
            self.end_headers()


def make_handler(store: KnowledgeStore, client: Optional[OpenAI]) -> type:
    """Create a request handler bound to ``store`` and ``client``."""

    class Handler(_Handler):
        pass

    Handler.store = store
    Handler.client = client
    return Handler


def answer_question(text: str, store: KnowledgeStore, client: Optional[OpenAI]) -> str:
    """Return an answer for ``text`` using the store and optional LLM."""
    if client is None:
        match = re.search(r"what does (?:a |the )?(\w+) need", text.lower())
        if match:
            subj = match.group(1)
            needs = store.find(subj, "needs")
            if not needs:
                parent = store.parent(subj)
                if parent:
                    needs = store.find(parent, "needs")
            if needs:
                return needs[0]
        return text

    prompt = f"Answer the question using the provided facts: {store.graph}\nQ: {text}"
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content.strip()


def run_server(host: str = "localhost", port: int = 8000, *, path: Optional[str] = None) -> Tuple[HTTPServer, threading.Thread]:
    """Start the knowledge MCP server."""
    store = KnowledgeStore(path)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if (api_key and OpenAI) else None
    server = HTTPServer((host, port), make_handler(store, client))
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server, thread

__all__ = ["KnowledgeStore", "run_server", "answer_question"]
