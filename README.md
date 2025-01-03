# Memory MCP Server

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

### Usage

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
      "args": ["memory-mcp-server"],
      "env": {
        // Optional: Neo4j configuration
        "NEO4J_URI": "neo4j://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-password"
      }
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

3. Run tests:
```bash
pytest                    # Run all tests
pytest -v                # Run with verbose output
pytest -v --cov         # Run with coverage report
```

4. Run the server locally:
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.