# Claude Memory Python

A Python package for managing Claude's knowledge graph with optimized performance for reading and writing operations.

## Overview

Claude Memory Python provides an interface for managing a knowledge graph that stores information about entities and their relationships. It features an optimized implementation with in-memory caching, indexing, and batch operations for improved performance.

## Features

- In-memory caching with TTL
- Indexing for fast lookups
- Batch write operations
- Async I/O support
- Concurrent access handling
- Atomic file operations
- Comprehensive validation
- Error handling

## Installation

The package can be installed globally using pipx:

```bash
pipx install -e /path/to/claude-memory-python
```

For development installation:

```bash
git clone https://github.com/yourusername/claude-memory-python.git
cd claude-memory-python
pip install -e .
```

## Requirements

- Python 3.8+
- aiofiles>=23.2.1
- mcp>=1.1.2

## Usage

```python
from pathlib import Path
from claude_memory_python.optimized_knowledge_graph_manager import OptimizedKnowledgeGraphManager
from claude_memory_python.interfaces import Entity, Relation

# Initialize the manager
manager = OptimizedKnowledgeGraphManager(Path("memory.jsonl"))

# Create entities
entities = [
    Entity(name="John", entityType="person", observations=["Software Engineer"]),
    Entity(name="Google", entityType="company", observations=["Tech company"])
]
await manager.create_entities(entities)

# Create relations
relations = [
    Relation(from_="John", to="Google", relationType="works at")
]
await manager.create_relations(relations)

# Read the graph
graph = await manager.read_graph()

# Search nodes
results = await manager.search_nodes("John")
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.