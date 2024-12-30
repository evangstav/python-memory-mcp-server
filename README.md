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

For Neo4j support, include the neo4j extras:

```bash
uv pip install "memory-mcp-server[neo4j]"
```

## Usage

Once installed, you can run the server using:

```bash
uvx memory-mcp-server
```

### Backend Options

The server supports two backend storage options:

#### 1. JSONL Backend (Default)

Uses a simple JSONL file for storage. Suitable for smaller graphs and development:

```bash
# Use default memory.jsonl in package directory
memory-mcp-server

# Specify custom file location
memory-mcp-server --path /path/to/memory.jsonl

# Configure cache TTL (default: 60 seconds)
memory-mcp-server --path /path/to/memory.jsonl --cache-ttl 120
```

#### 2. Neo4j Backend

Uses Neo4j for storage. Recommended for larger graphs and production use:

```bash
# Using command line arguments
memory-mcp-server --backend neo4j \
  --neo4j-uri "neo4j://localhost:7687" \
  --neo4j-user "neo4j" \
  --neo4j-password "password"

# Or using environment variables
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
memory-mcp-server --backend neo4j
```

### Migration Tool

To migrate data from JSONL to Neo4j:

```bash
# Using command line arguments
python -m memory_mcp_server.tools.migrate_to_neo4j \
  --jsonl-path /path/to/memory.jsonl \
  --neo4j-uri "neo4j://localhost:7687" \
  --neo4j-user "neo4j" \
  --neo4j-password "password"

# Or using environment variables for Neo4j config
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
python -m memory_mcp_server.tools.migrate_to_neo4j --jsonl-path /path/to/memory.jsonl
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
uv pip install -e ".[test,neo4j]"  # Include test and Neo4j dependencies
```

3. Run tests:
```bash
pytest                    # Run all tests
pytest -v                # Run with verbose output
pytest -v --cov         # Run with coverage report
```

4. Run the server locally:
```bash
python -m memory_mcp_server  # Use default JSONL backend
# or
python -m memory_mcp_server --backend neo4j  # Use Neo4j backend
```

## Testing

The project uses pytest for testing. The test suite includes:

### Unit Tests
- `test_knowledge_graph_manager.py`: Tests for knowledge graph operations
- `test_server.py`: Tests for MCP server implementation
- `test_backends/`: Tests for backend implementations
  - `test_jsonl.py`: JSONL backend tests
  - `test_neo4j.py`: Neo4j backend tests

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
- `temp_neo4j`: Provides a temporary Neo4j instance
- `knowledge_graph_manager`: Provides a KnowledgeGraphManager instance

## Backend Comparison

Feature | JSONL | Neo4j
--------|-------|-------
Storage Type | File-based | Graph Database
Query Performance | Good for small graphs | Excellent for large graphs
Scalability | Limited by file size | Highly scalable
Concurrent Access | Basic (file locks) | Advanced (ACID transactions)
Memory Usage | Loads full graph | On-demand loading
Setup Complexity | Simple | Requires database setup
Best For | Development, small graphs | Production, large graphs

## License

This project is licensed under the MIT License - see the LICENSE file for details.