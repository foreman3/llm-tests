"""Simple example to ingest text files into a persistent Chroma store."""

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chroma_ingest import ChromaIngestPipeline


def main() -> None:
    """Ingest text under ``samples/sourceText`` and persist embeddings."""

    pipeline = ChromaIngestPipeline()
    pipeline.ingest_folder("./samples/sourceText", "./samples/store")
    print("Embeddings stored in ./samples/store")


if __name__ == "__main__":
    main()
