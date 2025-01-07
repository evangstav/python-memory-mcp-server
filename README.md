# Memory MCP Server

[![CI](https://github.com/estav/python-memory-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/estav/python-memory-mcp-server/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/estav/python-memory-mcp-server/branch/main/graph/badge.svg)](https://codecov.io/gh/estav/python-memory-mcp-server)

An implementation of the Model Context Protocol (MCP) server for managing Claude's memory and knowledge graph.

## Installation

You can install the package using `uv`:

```bash
uvx memory-mcp-server
```

Or install it from the repository:

```bash
uv pip install git+https://github.com/estav/python-memory-mcp-server.git
```

## Usage

Once installed, you can run the server using:

```bash
uvx memory-mcp-server
```

### Configuration

The server uses a JSONL file for storage:

```bash
# Use default memory.jsonl in package directory
memory-mcp-server

# Specify custom file location
memory-mcp-server --path /path/to/memory.jsonl

# Configure cache TTL (default: 60 seconds)
memory-mcp-server --path /path/to/memory.jsonl --cache-ttl 120
```

### Integration with Claude Desktop

To use this MCP server with Claude Desktop, add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": ["memory-mcp-server"]
    }
  }
}
```

## Development

1. Clone the repository:
```bash
git clone https://github.com/estav/python-memory-mcp-server.git
cd python-memory-mcp-server
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[test]"  # Include test dependencies
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Run tests:
```bash
pytest                    # Run all tests
pytest -v                # Run with verbose output
pytest -v --cov         # Run with coverage report
```

5. Run the server locally:
```bash
python -m memory_mcp_server  # Run with default memory.jsonl
```

## Testing

The project uses pytest for testing. The test suite includes:

### Unit Tests
- `test_knowledge_graph_manager.py`: Tests for knowledge graph operations
- `test_server.py`: Tests for MCP server implementation
- `test_backends/`: Tests for backend implementations
  - `test_jsonl.py`: JSONL backend tests

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=memory_mcp_server

# Run specific test file
pytest tests/test_server.py

# Run tests with verbose output
pytest -v
```

### Test Fixtures
The `conftest.py` file provides common test fixtures:
- `temp_jsonl_path`: Creates a temporary JSONL file
- `knowledge_graph_manager`: Provides a KnowledgeGraphManager instance

## Code Quality

The project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pre-commit hooks**: Automated code quality checks
- **Interrogate**: Docstring coverage checking

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
