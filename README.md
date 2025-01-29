# Memory MCP Server

A Model Context Protocol (MCP) server that provides knowledge graph functionality for managing entities, relations, and observations in memory.

## Installation

Install the server in Claude Desktop:

```bash
mcp install main.py -v MEMORY_FILE_PATH=/path/to/memory.jsonl
```

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
entities = [
    Entity(name="example1", type="person"),
    Entity(name="example2", type="location")
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
result = await session.call_tool("add_observation", {
    "entity": "example",
    "observation": "This is a new observation"
})
if not result.success:
    if result.error_type == "NOT_FOUND":
        print(f"Entity not found: {result.error}")
    else:
        print(f"Error adding observation: {result.error}")
```

### Create Relation
```python
result = await session.call_tool("create_relation", {
    "from_entity": "person1",
    "to_entity": "location1",
    "relation_type": "visited"
})
if not result.success:
    if result.error_type == "NOT_FOUND":
        print(f"Entity not found: {result.error}")
    elif result.error_type == "VALIDATION_ERROR":
        print(f"Invalid relation data: {result.error}")
    else:
        print(f"Error creating relation: {result.error}")
```

### Search Memory
```python
result = await session.call_tool("search_memory", {
    "query": "most recent workout"  # Supports natural language queries
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

The search functionality supports:
- Temporal queries (e.g., "most recent", "last", "latest")
- Activity queries (e.g., "workout", "exercise")
- General entity searches

When using temporal queries with activity types (e.g., "most recent workout"),
the search will automatically return the latest matching activity.

### Delete Entities
```python
result = await session.call_tool("delete_entities", {
    "names": ["example1", "example2"]
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
    "from_entity": "person1",
    "to_entity": "location1"
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
