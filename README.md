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

### Configuration

The server expects certain environment variables to be set:
- `DATABASE_URL`: SQLite database URL for storing the knowledge graph
- Add any other configuration variables here...

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
uv pip install -e .
```

3. Run the server locally:
```bash
python -m memory_mcp_server
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.