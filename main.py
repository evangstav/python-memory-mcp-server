#!/usr/bin/env python3
"""Memory MCP server using FastMCP."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger as logging
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import Message, UserMessage
from pydantic import BaseModel

from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager

# Error type constants
ERROR_TYPES = {
    "NOT_FOUND": "NOT_FOUND",
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "INTERNAL_ERROR": "INTERNAL_ERROR",
    "ALREADY_EXISTS": "ALREADY_EXISTS",
    "INVALID_RELATION": "INVALID_RELATION",
    "NO_RESULTS": "NO_RESULTS",  # Used when search returns no matches
}


# Response models
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


# Create FastMCP server with dependencies and instructions
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
    """,
)

# Initialize knowledge graph manager using environment variable
# Default to ~/.claude/memory.jsonl if MEMORY_FILE_PATH not set
default_memory_path = Path.home() / ".claude" / "memory.jsonl"
memory_file = Path(os.getenv("MEMORY_FILE_PATH", str(default_memory_path)))

logging.info(f"Memory server using file: {memory_file}")

# Create KnowledgeGraphManager instance
kg = KnowledgeGraphManager(memory_file, 60)


def serialize_to_dict(obj: Any) -> Dict:
    """Helper to serialize objects to dictionaries."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


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
            error_type=ERROR_TYPES["NOT_FOUND"],
        )
    except ValueError as e:
        return EntityResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return EntityResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def get_graph() -> GraphResponse:
    """Get the entire knowledge graph."""
    try:
        graph = await kg.read_graph()
        return GraphResponse(success=True, data=serialize_to_dict(graph))
    except Exception as e:
        return GraphResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def create_entities(entities: List[Entity]) -> OperationResponse:
    """Create multiple new entities."""
    try:
        await kg.create_entities(entities)
        return OperationResponse(success=True)
    except ValueError as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def add_observation(
    entity: str, observation: str, ctx: Context = None
) -> OperationResponse:
    """Add an observation to an existing entity."""
    try:
        if ctx:
            ctx.info(f"Adding observation to {entity}")

        # Check if entity exists
        exists = await kg.search_nodes(entity)
        if not exists:
            return OperationResponse(
                success=False,
                error=f"Entity '{entity}' not found",
                error_type=ERROR_TYPES["NOT_FOUND"],
            )

        await kg.add_observations(entity, [observation])
        return OperationResponse(success=True)
    except ValueError as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def create_relation(
    from_entity: str, to_entity: str, relation_type: str, ctx: Context = None
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
                error_type=ERROR_TYPES["NOT_FOUND"],
            )

        if not to_exists:
            return OperationResponse(
                success=False,
                error=f"Target entity '{to_entity}' not found",
                error_type=ERROR_TYPES["NOT_FOUND"],
            )

        await kg.create_relations(
            [Relation(from_=from_entity, to=to_entity, relationType=relation_type)]
        )
        return OperationResponse(success=True)
    except ValueError as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def search_memory(query: str, ctx: Context = None) -> EntityResponse:
    """Search memory using natural language queries.

    Handles:
    - Temporal queries (e.g., "most recent", "last", "latest")
    - Activity queries (e.g., "workout", "exercise")
    - General entity searches
    """
    try:
        if ctx:
            ctx.info(f"Searching for: {query}")

        # Handle temporal queries
        temporal_keywords = ["recent", "last", "latest"]
        is_temporal = any(keyword in query.lower() for keyword in temporal_keywords)

        # Extract activity type from query
        activity_type = None
        if "workout" in query.lower():
            activity_type = "workout"
        elif "exercise" in query.lower():
            activity_type = "exercise"
        elif "physical activity" in query.lower():
            activity_type = "physical_activity"

        # Search for entities
        results = await kg.search_nodes(activity_type if activity_type else query)

        if not results:
            return EntityResponse(
                success=True,
                data={"entities": [], "relations": []},
                error="No matching activities found in memory",
                error_type="NO_RESULTS",
            )

        # For temporal queries, sort by timestamp if available
        if is_temporal and isinstance(results, list):
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            if results:
                results = results[0]  # Get most recent

        return EntityResponse(success=True, data=serialize_to_dict(results))
    except ValueError as e:
        return EntityResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return EntityResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def delete_entities(names: List[str], ctx: Context = None) -> OperationResponse:
    """Delete multiple entities and their relations."""
    try:
        if ctx:
            ctx.info(f"Deleting entities: {', '.join(names)}")

        await kg.delete_entities(names)
        return OperationResponse(success=True)
    except ValueError as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def delete_relation(
    from_entity: str, to_entity: str, ctx: Context = None
) -> OperationResponse:
    """Delete relations between two entities."""
    try:
        if ctx:
            ctx.info(f"Deleting relations between {from_entity} and {to_entity}")

        # Check if entities exist
        from_exists = await kg.search_nodes(from_entity)
        to_exists = await kg.search_nodes(to_entity)

        if not from_exists:
            return OperationResponse(
                success=False,
                error=f"Source entity '{from_entity}' not found",
                error_type=ERROR_TYPES["NOT_FOUND"],
            )

        if not to_exists:
            return OperationResponse(
                success=False,
                error=f"Target entity '{to_entity}' not found",
                error_type=ERROR_TYPES["NOT_FOUND"],
            )

        await kg.delete_relations(from_entity, to_entity)
        return OperationResponse(success=True)
    except ValueError as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def flush_memory(ctx: Context = None) -> OperationResponse:
    """Ensure all changes are persisted to storage."""
    try:
        if ctx:
            ctx.info("Flushing memory to storage")

        await kg.flush()
        return OperationResponse(success=True)
    except Exception as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


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
