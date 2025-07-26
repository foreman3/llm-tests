"""Tools for basic RAG workflows."""

from __future__ import annotations

import os
import re
from typing import List

import pandas as pd

from llm_pipeline.vector_store import VectorStore
from llm_pipeline.llm_methods import (
    generate_embeddings,
    openai_embedding_function,
)


class rag_tools:
    """Utility class to chunk text and persist embeddings to a ``VectorStore``."""

    def __init__(self, embedding_function=openai_embedding_function) -> None:
        self.embedding_function = embedding_function

    # ------------------------------------------------------------------
    # Text loading and chunking
    # ------------------------------------------------------------------
    def _load_text(self, source: str) -> str:
        """Return text from ``source`` which may be text, file path or directory."""
        if os.path.isdir(source):
            texts: List[str] = []
            for root, _, files in os.walk(source):
                for name in files:
                    if name.lower().endswith(".txt"):
                        path = os.path.join(root, name)
                        with open(path, "r", encoding="utf-8") as fh:
                            texts.append(fh.read())
            return "\n".join(texts)
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as fh:
                return fh.read()
        return source

    def chunk_text(
        self,
        source: str,
        *,
        chunk_size: int = 500,
        overlap: int = 0,
    ) -> pd.DataFrame:
        """Return a DataFrame of text chunks from ``source``."""
        text = self._load_text(source)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text else []

        chunks: List[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 > chunk_size and current:
                chunks.append(current.strip())
                if overlap > 0:
                    current = current[-overlap:].strip() + " " + sent
                else:
                    current = sent
            else:
                current = f"{current} {sent}".strip()
        if current:
            chunks.append(current.strip())

        return pd.DataFrame(
            {
                "chunk_id": [f"chunk_{i}" for i in range(len(chunks))],
                "text": chunks,
            }
        )

    # ------------------------------------------------------------------
    # Embedding storage
    # ------------------------------------------------------------------
    def store_chunks(
        self,
        source: str,
        store_path: str,
        *,
        chunk_size: int = 500,
        overlap: int = 0,
    ) -> VectorStore:
        """Create a ``VectorStore`` from ``source`` and persist it."""
        df = self.chunk_text(source, chunk_size=chunk_size, overlap=overlap)
        df = generate_embeddings(
            df, embedding_function=self.embedding_function, fields=["text"], output_key="embedding"
        )
        store = VectorStore(len(df["embedding"].iloc[0]), store_path=store_path)
        store.add(df["chunk_id"], df["embedding"])
        df[["chunk_id", "text"]].to_csv(f"{store_path}.csv", index=False)
        return store

    def query_store(self, query: str, store_path: str, k: int = 5) -> pd.DataFrame:
        """Query a stored ``VectorStore`` with ``query`` and return top ``k`` chunks."""
        store = VectorStore(1, store_path=store_path)
        embedding = self.embedding_function(query)
        results = store.query(embedding, k)
        if not results:
            return pd.DataFrame(columns=["chunk_id", "text", "distance"])
        mapping = pd.read_csv(f"{store_path}.csv")
        rows = []
        for chunk_id, dist in results:
            text = mapping.loc[mapping["chunk_id"] == chunk_id, "text"].values
            text_str = text[0] if len(text) > 0 else ""
            rows.append({"chunk_id": chunk_id, "text": text_str, "distance": dist})
        return pd.DataFrame(rows)


__all__ = ["rag_tools"]
