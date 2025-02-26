# Memory MCP Server

A Model Context Protocol (MCP) server that provides knowledge graph functionality for managing entities, relations, and observations in memory, with strict validation rules to maintain data consistency and semantic search capabilities.

## Features

- **Knowledge Graph Storage**: Store entities and relationships in a persistent graph structure
- **Semantic Search**: Find conceptually similar entities using vector embeddings
- **Natural Language Understanding**: Analyze queries to understand intent and context
- **Temporal Awareness**: Handle time-based queries (recent, past, etc.)
- **Flexible Storage Backend**: Support for both JSONL and SQLite backends
- **Performance Optimizations**: Intelligent caching, chunked loading, and memory usage controls

## Installation

First, clone the repository:

```bash
git clone https://github.com/yourusername/python-memory-mcp-server.git
cd python-memory-mcp-server
```

Install the server in Claude Desktop:

```bash
mcp install main.py -v MEMORY_FILE_PATH=/path/to/memory.jsonl
```

For semantic search capabilities, install additional dependencies:

```bash
pip install -r requirements-semantic-search.txt
```

## Data Validation Rules

### Entity Names
- Must start with a lowercase letter
- Can contain lowercase letters, numbers, and hyphens
- Maximum length of 100 characters
- Must be unique within the graph
- Example valid names: `python-project`, `meeting-notes-2024`, `user-john`

### Entity Types
The following entity types are supported:
- `person`: Human entities
- `concept`: Abstract ideas or principles
- `project`: Work initiatives or tasks
- `document`: Any form of documentation
- `tool`: Software tools or utilities
- `organization`: Companies or groups
- `location`: Physical or virtual places
- `event`: Time-bound occurrences

### Observations
- Non-empty strings
- Maximum length of 500 characters
- Must be unique per entity
- Should be factual and objective statements
- Include timestamp when relevant

### Relations
The following relation types are supported:
- `knows`: Person to person connection
- `contains`: Parent/child relationship
- `uses`: Entity utilizing another entity
- `created`: Authorship/creation relationship
- `belongs-to`: Membership/ownership
- `depends-on`: Dependency relationship
- `related-to`: Generic relationship

Additional relation rules:
- Both source and target entities must exist
- Self-referential relations not allowed
- No circular dependencies allowed
- Must use predefined relation types

## Usage

The server provides tools for managing a knowledge graph:

### Get Entity
```python
result = await session.call_tool("get_entity", {
    "entity_name": "example"
})
if not result.success:
    if result.error_type == "NOT_FOUND":
        print(f"Entity not found: {result.error}")
    elif result.error_type == "VALIDATION_ERROR":
        print(f"Invalid input: {result.error}")
    else:
        print(f"Error: {result.error}")
else:
    entity = result.data
    print(f"Found entity: {entity}")
```

### Get Graph
```python
result = await session.call_tool("get_graph", {})
if result.success:
    graph = result.data
    print(f"Graph data: {graph}")
else:
    print(f"Error retrieving graph: {result.error}")
```

### Create Entities
```python
# Valid entity creation
entities = [
    Entity(
        name="python-project",  # Lowercase with hyphens
        entityType="project",   # Must be a valid type
        observations=["Started development on 2024-01-29"]
    ),
    Entity(
        name="john-doe",
        entityType="person",
        observations=["Software engineer", "Joined team in 2024"]
    )
]
result = await session.call_tool("create_entities", {
    "entities": entities
})
if not result.success:
    if result.error_type == "VALIDATION_ERROR":
        print(f"Invalid entity data: {result.error}")
    else:
        print(f"Error creating entities: {result.error}")
```

### Add Observation
```python
# Valid observation
result = await session.call_tool("add_observation", {
    "entity": "python-project",
    "observation": "Completed initial prototype"  # Must be unique for entity
})
if not result.success:
    if result.error_type == "NOT_FOUND":
        print(f"Entity not found: {result.error}")
    elif result.error_type == "VALIDATION_ERROR":
        print(f"Invalid observation: {result.error}")
    else:
        print(f"Error adding observation: {result.error}")
```

### Create Relation
```python
# Valid relation
result = await session.call_tool("create_relation", {
    "from_entity": "john-doe",
    "to_entity": "python-project",
    "relation_type": "created"  # Must be a valid type
})
if not result.success:
    if result.error_type == "NOT_FOUND":
        print(f"Entity not found: {result.error}")
    elif result.error_type == "VALIDATION_ERROR":
        print(f"Invalid relation data: {result.error}")
    else:
        print(f"Error creating relation: {result.error}")
```

### Search Memory (Semantic Search)
```python
result = await session.call_tool("search_memory", {
    "query": "recent projects related to machine learning",  # Natural language query
    "semantic": True,  # Enable semantic search
    "max_results": 5   # Limit results
})
if result.success:
    if result.error_type == "NO_RESULTS":
        print(f"No results found: {result.error}")
    else:
        results = result.data
        print(f"Search results: {results}")
else:
    print(f"Error searching memory: {result.error}")
```

The semantic search functionality supports:
- Natural language understanding
- Temporal queries (e.g., "most recent", "last", "latest")
- Entity type filtering (e.g., "people who know about Python")
- Relation-focused queries (e.g., "connections between people")
- Conceptual similarity without exact keyword matches

### Search Nodes (Traditional Search)
```python
result = await session.call_tool("search_nodes", {
    "query": "python",
    "fuzzy": True,      # Enable fuzzy matching
    "threshold": 80.0   # Similarity threshold
})
```

### Regenerate Embeddings
```python
result = await session.call_tool("regenerate_embeddings", {})
if result.success:
    print(f"Success: {result.message}")
else:
    print(f"Error regenerating embeddings: {result.error}")
```

### Delete Entities
```python
result = await session.call_tool("delete_entities", {
    "names": ["python-project", "john-doe"]
})
if not result.success:
    if result.error_type == "NOT_FOUND":
        print(f"Entity not found: {result.error}")
    else:
        print(f"Error deleting entities: {result.error}")
```

### Delete Relation
```python
result = await session.call_tool("delete_relation", {
    "from_entity": "john-doe",
    "to_entity": "python-project"
})
if not result.success:
    if result.error_type == "NOT_FOUND":
        print(f"Entity not found: {result.error}")
    else:
        print(f"Error deleting relation: {result.error}")
```

### Flush Memory
```python
result = await session.call_tool("flush_memory", {})
if not result.success:
    print(f"Error flushing memory: {result.error}")
```

## Error Types

The server uses the following error types:

- `NOT_FOUND`: Entity or resource not found
- `VALIDATION_ERROR`: Invalid input data
- `INTERNAL_ERROR`: Server-side error
- `ALREADY_EXISTS`: Resource already exists
- `INVALID_RELATION`: Invalid relation between entities
- `NO_RESULTS`: Search returned no matches

## Response Models

All tools return typed responses using these models:

### EntityResponse
```python
class EntityResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
```

### GraphResponse
```python
class GraphResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
```

### OperationResponse
```python
class OperationResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    message: Optional[str] = None  # For success messages
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run semantic search tests
pytest tests/test_semantic_search.py

# Run backend tests
pytest tests/test_backends/

# Run caching tests
pytest tests/test_caching/

# Run benchmarking tests
pytest tests/test_benchmarking/
```

## Performance Features

The server includes several performance optimization features:

### Intelligent Caching
The caching system minimizes expensive operations by:
- Caching full graph data for faster reads
- Caching entity data for quicker lookups
- Caching search results for repeated queries
- Automatic cache invalidation on entity/relation changes

### Memory-Aware Loading
The system monitors memory usage and adapts its behavior:
- Uses standard loading when memory usage is low
- Switches to chunked loading when memory usage is high
- Avoids memory spikes during large graph operations

### Multiple Backend Support
- **JSONL Backend**: Simple file-based storage for ease of use
- **SQLite Backend**: Better performance for large graphs with indexed queries

### Performance Benchmarking
The server provides tools for measuring performance:
- `run_performance_benchmarks`: Test search, creation, and read performance
- `get_performance_metrics`: Get detailed performance statistics
- `enable_performance_profiling`: Start tracking operation performance
- `disable_performance_profiling`: Stop tracking and get results
- `generate_synthetic_graph`: Create test data for benchmarking

### Usage
```python
# Run benchmarks
result = await session.call_tool("run_performance_benchmarks", {
    "include_search": True,
    "include_create": True,
    "include_read": True
})

# Enable profiling
await session.call_tool("enable_performance_profiling", {})

# Run some operations
await session.call_tool("get_graph", {})
await session.call_tool("search_nodes", {"query": "test"})

# Get performance metrics
metrics = await session.call_tool("disable_performance_profiling", {})
print(f"Performance metrics: {metrics}")
```

### Adding New Features

1. Update validation rules in `validation.py`
2. Add tests in `tests/test_validation.py`
3. Implement changes in `knowledge_graph_manager.py`
4. For semantic search enhancements, modify `services/embedding_service.py` and `services/query_analyzer.py`
