---
description: Run the Continuous Integration pipeline (Linting, Type Checking, Testing)
---

This workflow runs the full CI pipeline to ensure code quality and correctness.

1. **Sync Dependencies**: Ensure all dependencies (including dev) are installed and up to date.

    ```bash
    uv sync --all-extras
    ```

2. **Linting (Ruff)**: Check for code style issues and potential errors. Auto-fix if possible.

    ```bash
    uv run ruff check --fix
    ```

3. **Formatting (Ruff)**: Enforce standard code formatting.

    ```bash
    uv run ruff format
    ```

4. **Type Checking (Ty)**: Run static type analysis.

    ```bash
    uv run ty check
    ```

5. **Unit Tests (Pytest)**: Run the test suite.

    ```bash
    uv run pytest
    ```

// turbo-all
