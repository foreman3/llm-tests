import asyncio
import logging
import os
import sys
from typing import List, Dict

from fastmcp import FastMCP

import pandas as pd
from llm_pipeline.llm_methods import openai_embedding_function
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import MockEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

mcp = FastMCP("MCP Server on Cloud Run")

CHROMA_DB_PATH = "./samples/store"
_index = None
_mapping = None

def _embed_model():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedding()
    return MockEmbedding(embed_dim=32)

def load_store_once(persist_path: str):
    global _index, _mapping
    if _index is None or _mapping is None:
        vector_store = ChromaVectorStore.from_params(
            collection_name="chunks", persist_dir=persist_path
        )
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_path, vector_store=vector_store
        )
        _index = load_index_from_storage(storage_context)
        _mapping = pd.read_csv(f"{persist_path}.csv")
        del storage_context, vector_store

def kNN_search(query: str, k: int = 5) -> List[Dict[str, float]]:
    """
    General purpose kNN search function to query a database of documents.

    Args:
        query: The query string.
        k: Number of top results to return.

    Returns:
        List of dicts with 'text' and 'distance' fields.
    """
    logger.info(f">>> Tool: 'kNN_search' called with query '{query}' and k={k}")
    load_store_once(CHROMA_DB_PATH)
    Settings.embed_model = _embed_model()
    Settings.llm = None
    retriever = _index.as_retriever(similarity_top_k=k)
    results = retriever.retrieve(query)
    if not results:
        return []
    rows = []
    for item in results:
        chunk_id = item.node.ref_doc_id or item.node.node_id
        rec = _mapping.loc[_mapping["chunk_id"] == chunk_id]
        text = rec["text"].values[0] if not rec.empty else ""
        distance = 1 - float(item.score)
        rows.append({"text": text, "distance": distance})
    del retriever, results
    return rows

@mcp.tool()
def dnd_query_tool(query: str, k: int = 5) -> List[Dict[str, float]]:
    """
    Use this to perform a query a database of documents about Dungeons and Dragons.
    Use this for all questions about Dungeons and Dragons or other fantasy topics.
    This will return the top k results closely matching the query.

    Args:
        query: The query string.
        k: Number of top results to return.

    Returns:
        List of dicts with 'text' and 'distance' fields.
    """
    results = kNN_search(query, k)
    logger.info(f">>> dnd_query_tool results: {results}")
    return results

@mcp.tool()
def add(a: int, b: int) -> int:
    """Use this to add two numbers together.
    
    Args:
        a: The first number.
        b: The second number.
    
    Returns:
        The sum of the two numbers.
    """
    logger.info(f">>> Tool: 'add' called with numbers '{a}' and '{b}'")
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Use this to subtract two numbers.
    
    Args:
        a: The first number.
        b: The second number.
    
    Returns:
        The difference of the two numbers.
    """
    logger.info(f">>> Tool: 'subtract' called with numbers '{a}' and '{b}'")
    return a - b

if __name__ == "__main__":
    logger.info(f" MCP server started on port {os.getenv('PORT', 8080)}")
    asyncio.run(
        mcp.run_async(
            transport="streamable-http",
            host="127.0.0.1",
            port=int(os.getenv("PORT", 8080)),
        )
    )