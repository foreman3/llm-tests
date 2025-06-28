# Agent Development Guidelines

These instructions apply to the entire repository.

## Workflow

- Run the test suite before committing:
  ```bash
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  pytest
  ```
- Update the `README.md` when new capabilities are added.

## Coding Conventions

- Follow PEP8 style and include docstrings for new functions and classes.
- Keep the public API of `llm_pipeline` stable when possible.
- New pipeline steps must subclass `PipelineStep` and implement `process` (and `process_async` for async behavior).
- Provide offline fallbacks for network-dependent features so tests can run without an API key.
- Add unit tests for new behavior under `tests/`.

## Agentic Features

- When extending `AgenticGoalStep` or related components, document available tools and ensure multi-step execution remains deterministic when `OPENAI_API_KEY` is unset.
- Avoid hard-coded credentials or personal data in code or tests.

