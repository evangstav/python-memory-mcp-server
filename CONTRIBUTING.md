# Contributing to Memory MCP Server

Thank you for your interest in contributing to the Memory MCP Server! This document provides guidelines and information for contributors.

## Project Overview

The Memory MCP Server is an implementation of the Model Context Protocol (MCP) that provides Claude with a persistent knowledge graph capability. The server manages entities and relations in a graph structure, with features like caching, indexing, and atomic file operations.

### Key Components

1. **Core Data Structures**
   - `Entity`: Nodes in the graph containing name, type, and observations
   - `Relation`: Edges between entities with relation types
   - `KnowledgeGraph`: Container for entities and relations

2. **Knowledge Graph Manager**
   - Handles persistent storage in JSONL format
   - Implements caching with TTL
   - Provides indexing for fast lookups
   - Ensures atomic file operations
   - Manages CRUD operations for entities and relations

3. **MCP Server Implementation**
   - Exposes tools for graph manipulation
   - Handles serialization/deserialization
   - Provides error handling and logging

   Available MCP Tools:
   - `create_entities`: Create multiple new entities in the knowledge graph
   - `create_relations`: Create relations between entities (in active voice)
   - `add_observations`: Add new observations to existing entities
   - `delete_entities`: Delete entities and their relations
   - `delete_observations`: Delete specific observations from entities
   - `delete_relations`: Delete specific relations
   - `read_graph`: Read the entire knowledge graph
   - `search_nodes`: Search entities and relations by query
   - `open_nodes`: Retrieve specific nodes by name

   Each tool has a defined input schema that validates the arguments. See the tool schemas in `main.py` for detailed parameter specifications.

## Getting Started

1. **Prerequisites**
   - Python 3.12 or higher
   - uv package manager

2. **Setup Development Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/estav/python-memory-mcp-server.git
   cd python-memory-mcp-server

   # Create virtual environment with Python 3.12+
   uv venv
   source .venv/bin/activate

   # Install dependencies including test dependencies
   uv pip install -e ".[test]"  # This will install pytest, pytest-asyncio, pytest-cov, and pytest-mock
   ```

2. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=memory_mcp_server

   # Run specific test file
   pytest tests/test_server.py
   ```

3. **Run the Server Locally**
   ```bash
   # Using the installed command (uses default memory.jsonl in package directory)
   memory-mcp-server

   # Specify custom memory file location
   memory-mcp-server --path /path/to/memory.jsonl

   # Or using the module directly
   python -m memory_mcp_server --path /path/to/memory.jsonl
   ```

   The server accepts the following arguments:
   - `--path`: Path to the memory file (default: memory.jsonl in package directory)

   The server runs on stdio for MCP communication, making it compatible with any MCP client.

## Development Guidelines

### Code Style

1. **Python Standards**
   - Follow PEP 8 style guide
   - Use type hints for function parameters and return values
   - Document classes and functions using docstrings

2. **Project-Specific Conventions**
   - Use async/await for I/O operations
   - Implement proper error handling with custom exceptions
   - Maintain atomic file operations for data persistence
   - Add appropriate logging statements

### Testing

1. **Test Structure**
   - Tests use pytest with pytest-asyncio for async testing
   - Test files must follow pattern `test_*.py` in the `tests/` directory
   - Async tests are automatically detected (asyncio_mode = "auto")
   - Test fixtures use function-level event loop scope

2. **Test Coverage**
   - Write unit tests for new functionality
   - Ensure tests cover error cases
   - Maintain high test coverage (aim for >90%)
   - Use pytest-cov for coverage reporting

3. **Test Categories**
   - Unit tests for individual components
   - Integration tests for MCP server functionality
   - Performance tests for operations on large graphs
   - Async tests for I/O operations and concurrency

4. **Test Configuration**
   - Configured in pyproject.toml under [tool.pytest.ini_options]
   - Uses quiet mode by default (-q)
   - Shows extra test summary (-ra)
   - Test discovery in tests/ directory

### Adding New Features

1. **Knowledge Graph Operations**
   - Implement new operations in `KnowledgeGraphManager`
   - Add appropriate indices if needed
   - Ensure atomic operations
   - Add validation and error handling

2. **MCP Tools**
   - Define tool schema in `main.py`
   - Implement tool handler function
   - Add to `TOOLS` dictionary
   - Include appropriate error handling

3. **Performance Considerations**
   - Consider caching implications
   - Optimize for large graphs
   - Handle memory efficiently

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Add tests for new functionality
   - Update documentation
   - Follow code style guidelines

2. **PR Description**
   - Clearly describe the changes
   - Reference any related issues
   - Explain testing approach
   - Note any breaking changes

3. **Review Process**
   - Address reviewer comments
   - Keep changes focused and atomic
   - Ensure CI checks pass

## Common Development Tasks

### Adding a New Tool

1. Define the tool schema in `list_tools()`:
   ```python
   types.Tool(
       name="new_tool_name",
       description="Description of what the tool does",
       inputSchema={
           "type": "object",
           "properties": {
               "param1": {
                   "type": "string",
                   "description": "Description of param1"
               }
           },
           "required": ["param1"]
       }
   )
   ```

2. Implement the tool handler:
   ```python
   async def tool_new_tool_name(
       manager: KnowledgeGraphManager,
       arguments: Dict[str, Any]
   ) -> List[types.TextContent]:
       # Implementation
       return [types.TextContent(type="text", text="result")]
   ```

3. Add to `TOOLS` dictionary:
   ```python
   TOOLS = {
       ...,
       "new_tool_name": tool_new_tool_name
   }
   ```

### Adding New Graph Operations

1. Add method to `KnowledgeGraphManager`:
   ```python
   async def new_operation(self, params):
       async with self._write_lock:  # If modifying graph
           graph = await self._check_cache()
           # Implementation
           self._dirty = True  # If graph was modified
           await self._save_graph(graph)
   ```

2. Add appropriate tests in `tests/test_knowledge_graph_manager.py`

## Troubleshooting

### Common Issues

1. **Cache Inconsistency**
   - Check cache TTL settings
   - Verify dirty flag handling
   - Ensure proper lock usage

2. **File Operations**
   - Check file permissions
   - Verify atomic write operations
   - Monitor temp file cleanup

3. **Performance Issues**
   - Review indexing strategy
   - Check cache effectiveness
   - Profile large operations

## Additional Resources

- [Model Context Protocol Documentation](https://github.com/ModelContext/protocol)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.