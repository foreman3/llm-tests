"""Demonstration pipeline for ingesting text files into Chroma."""

from __future__ import annotations

import os
from typing import List

import pandas as pd
import chromadb
import openai

from llm_pipeline.llm_methods import openai_embedding_function


class ChromaIngestPipeline:
    """Pipeline that loads text files and stores embeddings in Chroma."""

    def __init__(self, embedding_function=openai_embedding_function) -> None:
        self.embedding_function = embedding_function
        self._client = None

    def _chunk_text(self, text: str, *, chunk_size: int, overlap: int) -> List[str]:
        """Split ``text`` into overlapping chunks of roughly ``chunk_size`` characters."""
        if not text:
            return []

        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            # +1 for the space when joining
            add_len = len(word) + (1 if current else 0)
            if current and current_len + add_len > chunk_size:
                chunks.append(" ".join(current))
                # start next chunk with overlap
                if overlap > 0:
                    overlap_words = current[-overlap:]
                    current = overlap_words + [word]
                    current_len = sum(len(w) for w in current) + max(len(current) - 1, 0)
                else:
                    current = [word]
                    current_len = len(word)
            else:
                current.append(word)
                current_len += add_len
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _embedding_wrapper(self):
        """Wrap the configured embedding function for Chroma."""

        class _Wrapper:
            def __init__(self, func):
                self.func = func
                self.default_space = "cosine"

            def embed_documents(self, input):
                texts = [input] if isinstance(input, str) else list(input)
                return [self.func(t) for t in texts]

            def embed_query(self, input):
                return [self.func(input if isinstance(input, str) else " ".join(input))]

            def __call__(self, input):
                return self.embed_documents(input)

            def name(self):
                return "default"

            def is_legacy(self):
                return False

            def name(self):
                return "default"

            def is_legacy(self):
                return False

        return _Wrapper(self.embedding_function)

    def _get_collection(self, persist_path: str):
        """Return a Chroma collection using the configured embedding function."""
        os.makedirs(persist_path, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_path)
        try:
            client.delete_collection("chunks")
        except Exception:
            pass
        return client.get_or_create_collection(
            name="chunks", embedding_function=self._embedding_wrapper()
        )

    def ingest_folder(
        self,
        folder: str,
        persist_path: str,
        *,
        chunk_size: int = 500,
        overlap: int = 0,
    ) -> None:
        """Read ``folder`` and store text chunks in a Chroma vector store."""
        rows = []
        texts = []
        ids = []
        metadatas = []
        collection = self._get_collection(persist_path)
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
                        ids.append(chunk_id)
                        texts.append(chunk)
                        metadatas.append({"filename": name})
                        rows.append({"chunk_id": chunk_id, "filename": name, "text": chunk})

        if ids:
            collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

        with open(f"{persist_path}.csv", "w", encoding="utf-8", newline="") as f:
            pd.DataFrame(rows).to_csv(f, index=False)

    def close(self):
        """Attempt to cleanup vector store resources."""
        self._client = None

    def query_store(self, query: str, persist_path: str, k: int = 5) -> pd.DataFrame:
        """Query the stored index and return matching chunks."""
        # Re-open the collection without deleting it
        client = chromadb.PersistentClient(path=persist_path)
        collection = client.get_or_create_collection(
            name="chunks", embedding_function=self._embedding_wrapper()
        )
        results = collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        if not docs:
            return pd.DataFrame(columns=["filename", "text", "distance"])

        rows = []
        for doc, meta, dist in zip(docs, metas, distances):
            rows.append(
                {
                    "filename": meta.get("filename", "") if isinstance(meta, dict) else "",
                    "text": doc,
                    "distance": float(dist) if dist is not None else None,
                }
            )
        return pd.DataFrame(rows)

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
