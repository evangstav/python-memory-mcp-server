# Memory MCP Server Development Guide

## Commands
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_semantic_search.py

# Run specific test function
uv run pytest tests/test_semantic_search.py::test_function_name -v

# Type checking
uv run mypy memory_mcp_server

# Linting
uv run ruff check .

# Formatting
uv run ruff format .
```

## Code Style
- Python 3.12+ with strict typing (disallow_untyped_defs = true)
- Line length: 88 characters
- Naming: snake_case for functions/variables, PascalCase for classes
- Docstrings: Google style with args/returns/raises sections
- Imports: standard library → third-party → local, grouped and sorted
- Error handling: Custom exceptions in exceptions.py with specific error types
- Async: Use async/await for all I/O operations, proper error handling
- Tests: pytest with pytest-asyncio, high coverage (>90%)