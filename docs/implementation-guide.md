# Memory MCP Server Implementation Guide

This document outlines the required changes to improve the Memory MCP server implementation based on MCP protocol best practices.

## 1. Server Configuration

### Current Issue
The FastMCP initialization includes an unsupported `environment` parameter:

```python
mcp = FastMCP(
    "Memory",
    dependencies=["pydantic", "jsonl"],
    environment=["MEMORY_FILE_PATH"],  # Incorrect
    version="0.1.0",
)
```

### Required Changes
1. Remove the `environment` parameter
2. Add server instructions to document available tools
3. Update the initialization:

```python
mcp = FastMCP(
    "Memory",
    dependencies=["pydantic", "jsonl"],
    version="0.1.0",
    instructions="""
    Memory MCP server providing knowledge graph functionality.
    Available tools:
    - get_entity: Retrieve entity by name
    - get_graph: Get entire knowledge graph
    - create_entities: Create multiple entities
    - add_observation: Add observation to entity
    - create_relation: Create relation between entities
    - search_memory: Search entities by query
    - delete_entities: Delete multiple entities
    - delete_relation: Delete relation between entities
    - flush_memory: Persist changes to storage
    """
)
```

## 2. Type Safety

### Current Issue
Tool responses use untyped dictionaries which could lead to inconsistent responses.

### Required Changes
1. Add Pydantic models for responses:

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any

class EntityResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

class GraphResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

class OperationResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
```

2. Update tool return types to use these models:

```python
@mcp.tool()
async def get_entity(entity_name: str) -> EntityResponse:
    """Get entity by name from memory."""
    try:
        result = await kg.search_nodes(entity_name)
        if result:
            return EntityResponse(success=True, data=serialize_to_dict(result))
        return EntityResponse(
            success=False,
            error=f"Entity '{entity_name}' not found",
            error_type="NOT_FOUND"
        )
    except ValueError as e:
        return EntityResponse(
            success=False,
            error=str(e),
            error_type="VALIDATION_ERROR"
        )
    except Exception as e:
        return EntityResponse(
            success=False,
            error=str(e),
            error_type="INTERNAL_ERROR"
        )
```

## 3. Error Handling

### Current Issue
Error handling is basic and doesn't provide specific error types.

### Required Changes
1. Define error types as constants:

```python
# Add at top of file
ERROR_TYPES = {
    "NOT_FOUND": "NOT_FOUND",
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "INTERNAL_ERROR": "INTERNAL_ERROR",
    "ALREADY_EXISTS": "ALREADY_EXISTS",
    "INVALID_RELATION": "INVALID_RELATION"
}
```

2. Update all tools to use specific error types:

```python
@mcp.tool()
async def create_relation(
    from_entity: str,
    to_entity: str,
    relation_type: str,
    ctx: Context = None
) -> OperationResponse:
    """Create a relation between entities."""
    try:
        if ctx:
            ctx.info(f"Creating relation: {from_entity} -{relation_type}-> {to_entity}")

        # Check if entities exist
        from_exists = await kg.search_nodes(from_entity)
        to_exists = await kg.search_nodes(to_entity)

        if not from_exists:
            return OperationResponse(
                success=False,
                error=f"Source entity '{from_entity}' not found",
                error_type=ERROR_TYPES["NOT_FOUND"]
            )

        if not to_exists:
            return OperationResponse(
                success=False,
                error=f"Target entity '{to_entity}' not found",
                error_type=ERROR_TYPES["NOT_FOUND"]
            )

        relations = await kg.create_relations(
            [Relation(from_=from_entity, to=to_entity, relationType=relation_type)]
        )
        return OperationResponse(success=True)
    except ValueError as e:
        return OperationResponse(
            success=False,
            error=str(e),
            error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return OperationResponse(
            success=False,
            error=str(e),
            error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )
```

## 4. Prompts

### Current Issue
Prompts are commented out and use old format.

### Required Changes
1. Uncomment and update prompts to use new Message format:

```python
from mcp.server.fastmcp.prompts.base import Message, UserMessage

@mcp.prompt()
def create_entity_prompt(name: str, entity_type: str) -> list[Message]:
    """Generate prompt for entity creation."""
    return [
        UserMessage(
            f"I want to create a new entity in memory:\n"
            f"Name: {name}\n"
            f"Type: {entity_type}\n\n"
            f"What observations should I record about this entity?"
        )
    ]

@mcp.prompt()
def search_prompt(query: str) -> list[Message]:
    """Generate prompt for memory search."""
    return [
        UserMessage(
            f"I want to search my memory for information about: {query}\n\n"
            f"What specific aspects of these results would you like me to explain?"
        )
    ]

@mcp.prompt()
def relation_prompt(from_entity: str, to_entity: str) -> list[Message]:
    """Generate prompt for creating a relation."""
    return [
        UserMessage(
            f"I want to establish a relationship between:\n"
            f"Source: {from_entity}\n"
            f"Target: {to_entity}\n\n"
            f"What type of relationship exists between these entities?"
        )
    ]
```

## 5. Client Documentation

### Required Changes
1. Create a README.md file with client usage instructions:

```markdown
# Memory MCP Server

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

[Add examples for all other tools...]
```

## Implementation Steps

1. Update FastMCP initialization
2. Add response models
3. Add error type constants
4. Update all tools to use typed responses and specific error types
5. Update prompts to use new Message format
6. Create client documentation

## Testing

After implementing changes:

1. Test each tool with:
   - Valid inputs
   - Invalid inputs
   - Missing entities
   - Error conditions
2. Verify error types are correct
3. Test prompts
4. Verify client can properly handle all response types
