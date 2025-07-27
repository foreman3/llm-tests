"""Example program to query a previously persisted Chroma store."""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chroma_ingest import ChromaIngestPipeline


def main() -> None:
    """Load the store at ``./samples/store`` and query it."""
    if len(sys.argv) < 2:
        print("Usage: python query_chroma.py 'your question'")
        return
    query = sys.argv[1]
    pipeline = ChromaIngestPipeline()
    df = pipeline.query_store(query, "./samples/store", k=1)
    if df.empty:
        print("No results")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
