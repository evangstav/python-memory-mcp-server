# Contributing to Memory MCP Server

Thank you for your interest in contributing to the Memory MCP Server! This document provides guidelines and information for contributors.

## Project Overview

The Memory MCP Server is an implementation of the Model Context Protocol (MCP) that provides Claude with a persistent knowledge graph capability. The server manages entities and relations in a graph structure, supporting multiple backend storage options with features like caching, indexing, semantic search, and atomic operations.

### Key Components

1. **Core Data Structures**
   - `Entity`: Nodes in the graph containing name, type, and observations
   - `Relation`: Edges between entities with relation types
   - `KnowledgeGraph`: Container for entities and relations
   - `SearchOptions`: Configuration for search behavior

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
   - Coordinates semantic search capabilities

4. **Services**
   - `EmbeddingService`: Generates vector embeddings for semantic search
   - `QueryAnalyzer`: Analyzes natural language queries to determine intent

5. **MCP Server Implementation**
   - Exposes tools for graph manipulation
   - Handles serialization/deserialization
   - Provides error handling and logging

   Available MCP Tools:
   - `create_entities`: Create multiple new entities in the knowledge graph
   - `create_relations`: Create relations between entities
   - `add_observation`: Add new observations to existing entities
   - `delete_entities`: Delete entities and their relations
   - `delete_relation`: Delete specific relations
   - `search_memory`: Search using semantic understanding and natural language
   - `search_nodes`: Search using exact or fuzzy matching
   - `get_entity`: Retrieve entity by name
   - `get_graph`: Get entire knowledge graph
   - `flush_memory`: Persist changes to storage
   - `regenerate_embeddings`: Rebuild embeddings for all entities

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

   # Install semantic search dependencies
   pip install -r requirements-semantic-search.txt

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=memory_mcp_server

   # Run semantic search tests
   pytest tests/test_semantic_search.py
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
   - Semantic search tests in `tests/test_semantic_search.py`
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
   - Semantic search tests for embedding and query analysis

### Adding New Features

1. **New Backend Implementation**
   - Create new class implementing `Backend` interface
   - Implement all required methods including embedding storage
   - Add backend-specific configuration options
   - Create comprehensive tests
   - Update documentation and CLI

2. **Knowledge Graph Operations**
   - Implement operations in backend classes
   - Update KnowledgeGraphManager if needed
   - Add appropriate indices
   - Ensure atomic operations
   - Add validation and error handling

3. **Semantic Search Enhancements**
   - Modify `EmbeddingService` for embedding generation
   - Update `QueryAnalyzer` for query understanding
   - Enhance the `enhanced_search` method in `KnowledgeGraphManager`
   - Add tests in `test_semantic_search.py`

4. **MCP Tools**
   - Define tool schema in `main.py`
   - Implement tool handler function
   - Add to server instructions
   - Include appropriate error handling

5. **Performance Considerations**
   - Consider backend-specific optimizations
   - Implement efficient caching strategies
   - Optimize for large graphs
   - Handle memory efficiently
   - Consider embedding computation costs

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

4. **Semantic Search Issues**
   - Verify sentence-transformers installation
   - Check embedding file permissions
   - Monitor memory usage with large graphs

## Additional Resources

- [Model Context Protocol Documentation](https://github.com/ModelContext/protocol)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Sentence Transformers Documentation](https://www.sbert.net/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
