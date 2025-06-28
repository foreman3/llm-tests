# llm-orchestration

## Project Overview
The `llm-orchestration` project is designed to facilitate the orchestration of large language model (LLM) processing pipelines. It includes components for data preparation, an LLM processing pipeline, and reusable tools and processors.

## Project Structure
```
llm-orchestration
├── data_preparation
│   ├── __init__.py
│   └── prepare_data.py
├── llm_pipeline
│   ├── __init__.py
│   └── process_pipeline.py
├── tools
│   ├── __init__.py
│   └── utils.py
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

## Testing
The project includes a set of basic test cases located in the `tests` directory to ensure the functionality of the `LLMPipeline` class.

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd llm-orchestration
pip install -r requirements.txt
```

## Usage
After setting up the project, you can use the `DataPreparer` and `LLMPipeline` classes to prepare data and process it through the LLM pipeline. Refer to the individual module documentation for more details on usage.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.