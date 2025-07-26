"""Tools for basic RAG workflows."""

from __future__ import annotations

import os
import re
from typing import List

import pandas as pd

from llm_pipeline.llm_methods import openai_embedding_function

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.embeddings import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


class rag_tools:
    """Utility class to chunk text and persist embeddings using ``LlamaIndex``."""

    def __init__(self, embedding_function=openai_embedding_function) -> None:
        self.embedding_function = embedding_function

    def _embed_model(self):
        """Return a LlamaIndex embedding model matching ``embedding_function``."""
        if self.embedding_function is openai_embedding_function:
            if os.getenv("OPENAI_API_KEY"):
                return OpenAIEmbedding()
            return MockEmbedding(embed_dim=32)

        class _FuncEmbed:
            def __init__(self, func):
                self.func = func
                self.embed_dim = len(func("test"))

            def __call__(self, texts):
                if isinstance(texts, str):
                    return [self.func(texts)]
                return [self.func(t) for t in texts]

        return _FuncEmbed(self.embedding_function)

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
    ) -> VectorStoreIndex:
        """Create a ``VectorStoreIndex`` from ``source`` and persist it."""
        df = self.chunk_text(source, chunk_size=chunk_size, overlap=overlap)
        Settings.embed_model = self._embed_model()
        Settings.llm = None
        documents = [Document(text=row["text"], id_=row["chunk_id"]) for _, row in df.iterrows()]
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=store_path)
        df[["chunk_id", "text"]].to_csv(f"{store_path}.csv", index=False)
        return index

    def query_store(self, query: str, store_path: str, k: int = 5) -> pd.DataFrame:
        """Query a stored ``VectorStoreIndex`` with ``query`` and return top ``k`` chunks."""
        Settings.embed_model = self._embed_model()
        Settings.llm = None
        storage_context = StorageContext.from_defaults(persist_dir=store_path)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query)
        if not results:
            return pd.DataFrame(columns=["chunk_id", "text", "distance"])
        mapping = pd.read_csv(f"{store_path}.csv")
        rows = []
        for item in results:
            chunk_id = item.node.ref_doc_id or item.node.node_id
            text = mapping.loc[mapping["chunk_id"] == chunk_id, "text"].values
            text_str = text[0] if len(text) > 0 else ""
            distance = 1 - float(item.score)
            rows.append({"chunk_id": chunk_id, "text": text_str, "distance": distance})
        return pd.DataFrame(rows)


__all__ = ["rag_tools"]
