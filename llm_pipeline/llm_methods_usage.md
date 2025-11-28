# llm_methods.py Usage Guide

This document provides an overview and usage examples for the main functions and classes in `llm_methods.py`. It is intended for external developers who want to leverage the LLM pipeline and data processing utilities in their own projects.

---

## Table of Contents
- [OpenAI Embedding Function](#openai-embedding-function)
- [apply_llm_call](#apply_llm_call)
- [generate_embeddings](#generate_embeddings)
- [knn_filter](#knn_filter)
- [call_llm_with_dataframe](#call_llm_with_dataframe)
- [cluster_dbscan](#cluster_dbscan)
- [call_llm_in_batches](#call_llm_in_batches)
- [DataFrameProcessor](#dataframeprocessor)
- [Pipeline Classes](#pipeline-classes)

---

## OpenAI Embedding Function

**Function:** `openai_embedding_function(text: str) -> List[float]`

Returns an embedding vector for the input text using OpenAI's API, or a deterministic pseudo-embedding if no API key is set.

**Example:**
```python
embedding = openai_embedding_function("example text")
```

---

## apply_llm_call

**Function:**
```python
def apply_llm_call(df, prompt_template, output_key="llm_output", fields=None) -> pd.DataFrame:
```
Calls an LLM for each row in a DataFrame, using a prompt template and storing the result in a new column.

**Example:**
```python
import pandas as pd
from llm_pipeline.llm_methods import apply_llm_call

df = pd.DataFrame({"text": ["foo", "bar"]})
prompt = "Evaluate: {record_details}"
result = apply_llm_call(df, prompt, output_key="result")
```

---

## generate_embeddings

**Function:**
```python
def generate_embeddings(df, embedding_function=openai_embedding_function, output_key="embedding", fields=None) -> pd.DataFrame:
```
Generates embeddings for each row in a DataFrame, storing them in a new column.

**Example:**
```python
import pandas as pd
from llm_pipeline.llm_methods import generate_embeddings

df = pd.DataFrame({"text": ["foo", "bar"]})
result = generate_embeddings(df, fields=["text"])
```

---

## knn_filter

**Function:**
```python
def knn_filter(df, query, k, embedding_function=openai_embedding_function, embedding_column="embedding") -> pd.DataFrame:
```
Filters the DataFrame to the top-k rows most similar to the query (by cosine similarity of embeddings).

**Example:**
```python
import pandas as pd
from llm_pipeline.llm_methods import knn_filter, generate_embeddings

df = pd.DataFrame({"text": ["foo", "bar", "baz"]})
df = generate_embeddings(df, fields=["text"])
filtered = knn_filter(df, query="foo", k=2)
```

---

## call_llm_with_dataframe

**Function:**
```python
def call_llm_with_dataframe(df, prompt_template, fields=None) -> str:
```
Combines all records into a single prompt and calls the LLM, returning the response as a string.

**Example:**
```python
import pandas as pd
from llm_pipeline.llm_methods import call_llm_with_dataframe

df = pd.DataFrame({"text": ["foo", "bar"]})
prompt = "Summarize: {record_details}"
summary = call_llm_with_dataframe(df, prompt)
```

---

## cluster_dbscan

**Function:**
```python
def cluster_dbscan(df, embedding_column, output_key="cluster_id", eps=0.5, min_samples=5, **dbscan_kwargs) -> pd.DataFrame:
```
Clusters embeddings in a DataFrame using DBSCAN, adding a cluster label column.

**Example:**
```python
import pandas as pd
from llm_pipeline.llm_methods import generate_embeddings, cluster_dbscan

df = pd.DataFrame({"text": ["foo", "bar", "baz"]})
df = generate_embeddings(df, fields=["text"])
df = cluster_dbscan(df, embedding_column="embedding")
```

---

## call_llm_in_batches

**Function:**
```python
def call_llm_in_batches(df, prompt_template, fields=None, batch_size=100, consolidation_prefix=...) -> str:
```
Processes the DataFrame in batches, calls the LLM on each batch, and consolidates the responses.

**Example:**
```python
import pandas as pd
from llm_pipeline.llm_methods import call_llm_in_batches

df = pd.DataFrame({"text": ["foo", "bar", "baz"]})
prompt = "Summarize: {record_details}"
final_summary = call_llm_in_batches(df, prompt, batch_size=2)
```

---

## DataFrameProcessor

A chainable wrapper for DataFrame processing.

**Example:**
```python
from llm_pipeline.llm_methods import DataFrameProcessor
import pandas as pd

df = pd.DataFrame({"text": ["foo", "bar"]})
proc = DataFrameProcessor(df)
proc = proc.generate_embeddings(fields=["text"])
proc = proc.llm_call("Evaluate: {record_details}")
result_df = proc.get_df()
```

---

## Pipeline Classes

- `PipelineStep`: Abstract base for pipeline steps.
- `LLMCallStep`: LLM call per record, with optional caching.
- `FixedProcessingStep`: Applies a fixed function to the DataFrame.
- `FilterStep`: Filters rows based on a function.
- `GenerateEmbeddingsStep`: Embedding generation per record.
- `kNNFilterStep`: Top-k similarity filter.
- `LLMCallWithDataFrame`: LLM call with all records combined.
- `AgenticGoalStep`: Agentic step using tools and/or MCP servers.
- `SummarizationStep`: Summarizes the DataFrame.
- `DataPipeline` / `AsyncDataPipeline`: Run a sequence of steps synchronously or asynchronously.

**Example Pipeline:**
```python
from llm_pipeline.llm_methods import DataPipeline, LLMCallStep, GenerateEmbeddingsStep
import pandas as pd

df = pd.DataFrame({"text": ["foo", "bar"]})
steps = [
    GenerateEmbeddingsStep(fields=["text"]),
    LLMCallStep(prompt_template="Evaluate: {record_details}")
]
pipeline = DataPipeline(steps)
result_df = pipeline.run(df)
```

---

For more details, see the docstrings in `llm_methods.py` or contact the maintainers.
