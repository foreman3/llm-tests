"""Simple example to ingest text files into a persistent Chroma store."""

from chroma_ingest import ChromaIngestPipeline


def main() -> None:
    """Ingest text under ``samples/sourceText`` and persist embeddings."""

    pipeline = ChromaIngestPipeline()
    pipeline.ingest_folder("./samples/sourceText", "./samples/store")
    print("Embeddings stored in ./samples/store")


if __name__ == "__main__":
    main()
