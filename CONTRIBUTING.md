# Contributing to Memory MCP Server

Thank you for your interest in contributing to the Memory MCP Server! This document provides guidelines and information for contributors.

## Project Overview

The Memory MCP Server is an implementation of the Model Context Protocol (MCP) that provides Claude with a persistent knowledge graph capability. The server manages entities and relations in a graph structure, supporting multiple backend storage options with features like caching, indexing, and atomic operations.

### Key Components

1. **Core Data Structures**
   - `Entity`: Nodes in the graph containing name, type, and observations
   - `Relation`: Edges between entities with relation types
   - `KnowledgeGraph`: Container for entities and relations

2. **Backend System**
   - `Backend`: Abstract interface defining storage operations
   - `JsonlBackend`: File-based storage using JSONL format
   - Extensible design for adding new backends

3. **Knowledge Graph Manager**
   - Backend-agnostic manager layer
   - Implements caching with TTL
   - Provides indexing for fast lookups
   - Ensures atomic operations
   - Manages CRUD operations for entities and relations

4. **MCP Server Implementation**
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

   # Install all dependencies (including test)
   uv pip install -e ".[test]"

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=memory_mcp_server

   # Run specific backend tests
   pytest tests/test_backends/test_jsonl.py
   ```

4. **Run the Server Locally**
   ```bash
   # Using JSONL backend
   memory-mcp-server --path /path/to/memory.jsonl
   ```

## Development Guidelines

### Code Style

1. **Python Standards**
   - Follow PEP 8 style guide
   - Use type hints for function parameters and return values
   - Document classes and functions using docstrings
   - Maintain 95% or higher docstring coverage

2. **Project-Specific Conventions**
   - Use async/await for I/O operations
   - Implement proper error handling with custom exceptions
   - Maintain atomic operations for data persistence
   - Add appropriate logging statements
   - Follow backend interface for new implementations

### Code Quality Tools

1. **Pre-commit Hooks**
   - Ruff for linting and formatting
   - MyPy for static type checking
   - Interrogate for docstring coverage
   - Additional checks for common issues

2. **CI/CD Pipeline**
   - Automated testing
   - Code coverage reporting
   - Performance benchmarking
   - Security scanning

### Testing

1. **Test Structure**
   - Tests use pytest with pytest-asyncio for async testing
   - Test files must follow pattern `test_*.py` in the `tests/` directory
   - Backend-specific tests in `tests/test_backends/`
   - Async tests are automatically detected (asyncio_mode = "auto")
   - Test fixtures use function-level event loop scope

2. **Test Coverage**
   - Write unit tests for new functionality
   - Ensure tests cover error cases
   - Maintain high test coverage (aim for >90%)
   - Use pytest-cov for coverage reporting

3. **Test Categories**
   - Unit tests for individual components
   - Backend-specific tests for storage implementations
   - Integration tests for MCP server functionality
   - Performance tests for operations on large graphs
   - Async tests for I/O operations and concurrency

4. **Test Configuration**
   - Configured in pyproject.toml under [tool.pytest.ini_options]
   - Uses quiet mode by default (-q)
   - Shows extra test summary (-ra)
   - Test discovery in tests/ directory

### Adding New Features

1. **New Backend Implementation**
   - Create new class implementing `Backend` interface
   - Implement all required methods
   - Add backend-specific configuration options
   - Create comprehensive tests
   - Update documentation and CLI

2. **Knowledge Graph Operations**
   - Implement operations in backend classes
   - Update KnowledgeGraphManager if needed
   - Add appropriate indices
   - Ensure atomic operations
   - Add validation and error handling

   Key operations include:
   - Entity creation/deletion
   - Relation creation/deletion
   - Observation management (adding/removing observations to entities)
   - Graph querying and search
   - Atomic write operations with locking

3. **MCP Tools**
   - Define tool schema in `main.py`
   - Implement tool handler function
   - Add to `TOOLS` dictionary
   - Include appropriate error handling

4. **Performance Considerations**
   - Consider backend-specific optimizations
   - Implement efficient caching strategies
   - Optimize for large graphs
   - Handle memory efficiently

### Adding a New Backend

1. Create new backend class:
   ```python
   from .base import Backend

   class NewBackend(Backend):
       def __init__(self, config_params):
           self.config = config_params

       async def initialize(self) -> None:
           # Setup connection, create indices, etc.
           pass

       async def create_entities(self, entities: List[Entity]) -> List[Entity]:
           # Implementation
           pass

       # Implement other required methods...
   ```

2. Add backend tests:
   ```python
   # tests/test_backends/test_new_backend.py
   @pytest.mark.asyncio
   async def test_new_backend_operations():
       backend = NewBackend(test_config)
       await backend.initialize()
       # Test implementations
   ```

3. Update CLI and configuration

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Add tests for new functionality
   - Update documentation
   - Follow code style guidelines
   - Run pre-commit hooks

2. **PR Description**
   - Clearly describe the changes
   - Reference any related issues
   - Explain testing approach
   - Note any breaking changes

3. **Review Process**
   - Address reviewer comments
   - Keep changes focused and atomic
   - Ensure CI checks pass

## Troubleshooting

### Common Issues

1. **Backend-Specific Issues**
   - JSONL Backend:
     - Check file permissions
     - Verify atomic write operations
     - Monitor temp file cleanup

2. **Cache Inconsistency**
   - Check cache TTL settings
   - Verify dirty flag handling
   - Ensure proper lock usage

3. **Performance Issues**
   - Review backend-specific indexing
   - Check cache effectiveness
   - Profile large operations

## Additional Resources

- [Model Context Protocol Documentation](https://github.com/ModelContext/protocol)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
