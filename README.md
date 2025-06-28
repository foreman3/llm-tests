# llm-orchestration

## Project Overview
The `llm-orchestration` project provides a flexible framework for building end‑to‑end pipelines around large language models (LLMs). It bundles data preparation utilities, a modular processing pipeline, and a growing collection of tools that can run either locally or via Model Context Protocol (MCP) servers. The included pipeline supports caching, asynchronous execution, embeddings, and agent-driven workflows.

## Project Structure
```
llm-orchestration
├── data_preparation
│   ├── __init__.py
│   └── prepare_data.py
├── llm_pipeline
│   ├── __init__.py
│   └── process_pipeline.py
├── mcp
│   └── __init__.py
├── tools
│   ├── __init__.py
│   └── utils.py
├── plugins
│   └── <custom steps>
├── tests
│   ├── __init__.py
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Components

### Data Preparation
- **Module**: `data_preparation`
- **Class**: `DataPreparer`
  - **Methods**:
    - `retrieve_data`: Fetches data from services.
    - `sanitize_data`: Normalizes the data for processing.

### LLM Processing Pipeline
- **Module**: `llm_pipeline`
- **Class**: `LLMPipeline`
  - **Methods**:
    - `define_steps`: Sets up the workflow for processing.
    - `add_prompt`: Adds prompts to the pipeline.
    - `process_batches`: Processes prepared data in batches.

### Utilities
- **Module**: `tools`
- **Functions**:
  - `log_message`: Logs messages for tracking.
  - `validate_input`: Validates input data for the pipeline.

### New Features
- **Response caching**: `LLMCallStep` can persist responses to disk to avoid repeated API calls.
- **Asynchronous pipeline**: `AsyncDataPipeline` runs steps concurrently when supported.
- **Vector store**: `VectorStore` allows embeddings to be persisted and queried using FAISS.
- **Data validation & logging**: `DataPipeline` automatically validates inputs and logs each step.
- **Summarization step**: `SummarizationStep` generates a report for the entire DataFrame.
- **Plugin architecture**: additional `PipelineStep` classes can be discovered from a plugin directory.
- **Agentic goal step**: `AgenticGoalStep` now supports multi-step tool use and can load built-in MCP tools for advanced agent behavior.
- **MCP servers**: remote MCP servers can be listed for use by `AgenticGoalStep` to augment local tools.
- **MCP tool discovery**: `AgenticGoalStep` automatically queries remote MCP servers for available tools when possible.

## Testing
The project includes a set of basic test cases located in the `tests` directory to ensure the functionality of the `LLMPipeline` class.

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
pip install .  # install package
```

## Usage
After setting up the project, you can use the `DataPreparer` and `LLMPipeline` classes to prepare data and process it through the LLM pipeline. Refer to the individual module documentation for more details on usage.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.
