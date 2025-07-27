import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from chroma_ingest import ChromaIngestPipeline


def main():
    parser = argparse.ArgumentParser(description="Ask a question using LLM with Chroma context.")
    parser.add_argument("--persist_path", default="./samples/store", help="Path to Chroma persist directory")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top chunks to use as context")
    parser.add_argument("--question", type=str, help="Question to ask (if not provided, will prompt)")
    args = parser.parse_args()

    pipeline = ChromaIngestPipeline()

    question = args.question
    if not question:
        question = input("Enter your question: ")

    answer = pipeline.answer_with_llm(question, args.persist_path, k=args.top_k)
    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
