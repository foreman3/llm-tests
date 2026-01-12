# llm-orchestration

## Project Overview
The `llm-orchestration` project provides a flexible framework for building end‑to‑end pipelines around large language models (LLMs). It bundles data preparation utilities, a modular processing pipeline, and a growing collection of tools that can run either locally or via Model Context Protocol (MCP) servers. The included pipeline supports caching, asynchronous execution, embeddings, and agent-driven workflows.

## Project Structure
```
llm-orchestration
├── llm_pipeline
│   ├── __init__.py
│   ├── llm_methods.py
│   ├── plugin_loader.py
│   └── vector_store.py
├── monster_pipeline
│   ├── __init__.py
│   ├── create_embeddings.py
│   ├── query_process.py
│   └── data/
├── mcp
│   └── __init__.py
├── tools
│   ├── __init__.py
│   └── utils.py
├── plugins
│   ├── __init__.py
│   └── example_plugin.py
├── tests
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_data_pipeline.csv
│   └── test_data_pipeline_embeddings.pkl
├── examples
│   ├── monster_pipeline
│   └── joust_game
├── AGENTS.md
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

## Components


### LLM Processing Pipeline
- **Module**: `llm_pipeline.llm_methods`
- **Classes**:
  - `DataPipeline`: runs a list of `PipelineStep` objects over a DataFrame.
  - `DataFrameProcessor`: a chainable helper for common processing functions.

### Utilities
- **Module**: `tools`
- **Functions**:
  - `log_message`: Uses Python's ``logging`` library to record progress.
  - `validate_input`: Validates input data for the pipeline.

### New Features
- **Response caching**: `LLMCallStep` can persist responses to disk to avoid repeated API calls.
- **Asynchronous pipeline**: `AsyncDataPipeline` runs steps concurrently when supported.
- **RAG tools**: `rag_tools` can chunk text files, build a vector store using LlamaIndex, and query it.
- **Vector store**: `VectorStore` allows embeddings to be persisted and queried using FAISS.
- **Chroma ingest**: `ChromaIngestPipeline` now chunks text using LlamaIndex's
  `SimpleNodeParser` and stores the embeddings in a Chroma vector store.
- **Chroma ingestion sample**: `samples/ingest_chroma.py` ingests the text under
  `samples/sourceText` and persists the embeddings. `samples/query_chroma.py`
  demonstrates querying the stored vector database.
- **Data validation & logging**: `DataPipeline` automatically validates inputs and logs each step.
- **Summarization step**: `SummarizationStep` generates a report for the entire DataFrame.
- **Plugin architecture**: additional `PipelineStep` classes can be discovered from a plugin directory.
- **Agentic goal step**: `AgenticGoalStep` now supports multi-step tool use and can load built-in MCP tools for advanced agent behavior.
- **MCP servers**: remote MCP servers can be listed for use by `AgenticGoalStep` to augment local tools.
- **MCP tool discovery**: `AgenticGoalStep` automatically queries remote MCP servers for available tools when possible.

## Testing
The project includes a set of basic test cases located in the `tests` directory to ensure the functionality of the `DataPipeline` class.

### Running the tests
Install the main requirements and additional test dependencies, then execute the test suite:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd llm-orchestration
pip install -r requirements.txt
pip install faiss-cpu  # or faiss-gpu
pip install .  # install package
```

FAISS provides the index used by `VectorStore`. If your machine has a compatible GPU you can
replace `faiss-cpu` with `faiss-gpu`.

## Usage
After setting up the project, you can use the `DataPipeline` or `DataFrameProcessor` classes from `llm_pipeline.llm_methods` to process data through the LLM pipeline. Refer to the individual module documentation for more details on usage.

## Joust Game Example
An additional example in `examples/joust_game/joust.html` demonstrates a simple
horizontal movement simulation implemented in JavaScript. The player's speed
starts 30% faster than the enemy and decreases by 5% with each level. Both the
player and enemy wrap around the screen horizontally, while enemies maintain
their movement direction when wrapping.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.
