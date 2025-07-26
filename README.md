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
│   └── monster_pipeline
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
- **RAG tools**: `rag_tools` can chunk text files, build a vector store, and query it.
- **Vector store**: `VectorStore` allows embeddings to be persisted and queried using FAISS.
- **Data validation & logging**: `DataPipeline` automatically validates inputs and logs each step.
- **Summarization step**: `SummarizationStep` generates a report for the entire DataFrame.
- **Plugin architecture**: additional `PipelineStep` classes can be discovered from a plugin directory.
- **Agentic goal step**: `AgenticGoalStep` now supports multi-step tool use and can load built-in MCP tools for advanced agent behavior.
- **MCP servers**: remote MCP servers can be listed for use by `AgenticGoalStep` to augment local tools.
- **MCP tool discovery**: `AgenticGoalStep` automatically queries remote MCP servers for available tools when possible.
- **Knowledge server**: `knowledge_server` provides a minimal MCP server with a
  persistent knowledge store. It answers simple questions and falls back to
  deterministic responses when `OPENAI_API_KEY` is not set. A starter dataset of
  basic facts is available in `knowledge_server/basic_knowledge.json` and can be
  loaded when starting the server:

- **Knowledge ingest**: `KnowledgeIngestor` can read text files or URLs,
  extract facts using an agent and store them in the knowledge server.

```python
from knowledge_server import run_server
server, thread = run_server(path="knowledge_server/basic_knowledge.json")
```

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

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.
