"""Demonstration pipeline for ingesting text files into Chroma."""

from __future__ import annotations

import os
from typing import List
import openai

import pandas as pd
from llm_pipeline.llm_methods import openai_embedding_function
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.embeddings import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class ChromaIngestPipeline:
    """Pipeline that loads text files and stores embeddings in Chroma."""

    def __init__(self, embedding_function=openai_embedding_function) -> None:
        self.embedding_function = embedding_function
        self._vector_store = None  # Track for cleanup

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

    def _chunk_text(self, text: str, *, chunk_size: int, overlap: int) -> List[str]:
        """Split ``text`` into chunks using ``SimpleNodeParser``."""
        if not text:
            return []

        parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        nodes = parser.get_nodes_from_documents([Document(text=text)])
        return [node.text for node in nodes]

    def ingest_folder(
        self,
        folder: str,
        persist_path: str,
        *,
        chunk_size: int = 500,
        overlap: int = 0,
    ) -> None:
        """Read ``folder`` and store text chunks in a Chroma vector store."""
        documents: List[Document] = []
        rows = []
        for root, _, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(".txt"):
                    path = os.path.join(root, name)
                    with open(path, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    for i, chunk in enumerate(
                        self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                    ):
                        chunk_id = f"{name}_{i}"
                        documents.append(
                            Document(text=chunk, id_=chunk_id, metadata={"filename": name})
                        )
                        rows.append({"chunk_id": chunk_id, "filename": name, "text": chunk})

        Settings.embed_model = self._embed_model()
        Settings.llm = None
        vector_store = ChromaVectorStore.from_params(
            collection_name="chunks", persist_dir=persist_path
        )
        self._vector_store = vector_store  # Save reference for cleanup
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        index.storage_context.persist(persist_dir=persist_path)
        # Ensure CSV file is closed immediately
        with open(f"{persist_path}.csv", "w", encoding="utf-8", newline="") as f:
            pd.DataFrame(rows).to_csv(f, index=False)
        # Explicitly delete objects to release file handles
        del storage_context, index, vector_store

    def load_index(self, persist_path: str) -> VectorStoreIndex:
        """Load a persisted ``VectorStoreIndex`` from ``persist_path``."""
        vector_store = ChromaVectorStore.from_params(
            collection_name="chunks", persist_dir=persist_path
        )
        self._vector_store = vector_store  # Save reference for cleanup
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_path, vector_store=vector_store
        )
        index = load_index_from_storage(storage_context)
        # Explicitly delete storage_context after use
        del storage_context, vector_store
        return index

    def close(self):
        """Attempt to cleanup vector store resources."""
        if self._vector_store is not None:
            try:
                # ChromaVectorStore does not have a close method, but if it did:
                self._vector_store = None
            except Exception:
                pass

    def query_store(self, query: str, persist_path: str, k: int = 5) -> pd.DataFrame:
        """Query the stored index and return matching chunks."""
        Settings.embed_model = self._embed_model()
        Settings.llm = None
        index = self.load_index(persist_path)
        retriever = index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query)
        if not results:
            return pd.DataFrame(columns=["filename", "text", "distance"])
        mapping = pd.read_csv(f"{persist_path}.csv")
        rows = []
        for item in results:
            chunk_id = item.node.ref_doc_id or item.node.node_id
            rec = mapping.loc[mapping["chunk_id"] == chunk_id]
            filename = rec["filename"].values[0] if not rec.empty else ""
            text = rec["text"].values[0] if not rec.empty else ""
            distance = 1 - float(item.score)
            rows.append({"filename": filename, "text": text, "distance": distance})
        df = pd.DataFrame(rows)
        del index, retriever, results, mapping
        import gc

        gc.collect()
        return df

    def answer_with_llm(self, query: str, persist_path: str, k: int = 5) -> str:
        """
        Retrieve top-k relevant chunks and use OpenAI o4_mini model to answer the query.
        """
        import openai

        # Step 1: Retrieve top-k relevant chunks
        df = self.query_store(query, persist_path, k=k)
        if df.empty:
            return "No relevant context found to answer the question."

        # Step 2: Concatenate retrieved texts as context
        context = "\n\n".join(df["text"].tolist())

        # Step 3: Call OpenAI o4_mini model
        prompt = (
            f"Answer the following question using the provided context.  Deduplicate information to provide a clean response\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


__all__ = ["ChromaIngestPipeline"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest text files into Chroma vector store."
    )
    parser.add_argument(
        "--source_folder",
        required=True,
        help="Folder containing .txt files to ingest",
    )
    parser.add_argument(
        "--dest_folder",
        required=True,
        help="Destination folder for Chroma store",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Chunk size for splitting text",
    )
    parser.add_argument(
        "--overlap", type=int, default=0, help="Chunk overlap"
    )
    args = parser.parse_args()

    pipeline = ChromaIngestPipeline()
    print(
        f"Ingesting from '{args.source_folder}' to '{args.dest_folder}' ..."
    )
    pipeline.ingest_folder(
        args.source_folder,
        args.dest_folder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print("Ingestion complete.")
