#!/usr/bin/env pythn3
"""Memory MCP server using FastMCP."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP

from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager

# Create FastMCP server with dependencies and environment variables
mcp = FastMCP(
    "Memory",
    dependencies=["pydantic", "jsonl"],
    environment=["MEMORY_FILE_PATH"],
    version="0.1.0",
)

# Initialize knowledge graph manager using environment variable
memory_file = Path(os.getenv("MEMORY_FILE_PATH", "memory.jsonl"))
kg = KnowledgeGraphManager(memory_file, 60)


def serialize_to_dict(obj: Any) -> Dict:
    """Helper to serialize objects to dictionaries."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


@mcp.resource("memory://{entity_name}")
async def get_entity(entity_name: str) -> Dict[str, Any]:
    """Get entity by name from memory."""
    try:
        result = await kg.search_nodes(entity_name)
        if result:
            return {"success": True, "data": serialize_to_dict(result)}
        return {"success": False, "error": f"Entity '{entity_name}' not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.resource("memory://graph")
async def get_graph() -> Dict[str, Any]:
    """Get the entire knowledge graph."""
    try:
        graph = await kg.read_graph()
        return {"success": True, "data": serialize_to_dict(graph)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def create_entity(
    name: str,
    entity_type: str,
    observations: Optional[List[str]] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Create a new entity in the knowledge graph."""
    try:
        if ctx:
            ctx.info(f"Creating entity: {name} of type {entity_type}")

        obs_list = [] if observations is None else observations
        entities = await kg.create_entities(
            [Entity(name=name, entityType=entity_type, observations=obs_list)]
        )
        return {"success": True, "created": [serialize_to_dict(e) for e in entities]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def add_observation(
    entity: str, observation: str, ctx: Context = None
) -> Dict[str, Any]:
    """Add an observation to an existing entity."""
    try:
        if ctx:
            ctx.info(f"Adding observation to {entity}")

        await kg.add_observations(entity, [observation])
        return {"success": True, "entity": entity}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def create_relation(
    from_entity: str, to_entity: str, relation_type: str, ctx: Context = None
) -> Dict[str, Any]:
    """Create a relation between entities."""
    try:
        if ctx:
            ctx.info(f"Creating relation: {from_entity} -{relation_type}-> {to_entity}")

        relations = await kg.create_relations(
            [Relation(from_=from_entity, to=to_entity, relationType=relation_type)]
        )
        return {"success": True, "created": [serialize_to_dict(r) for r in relations]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def search_memory(query: str, ctx: Context = None) -> Dict[str, Any]:
    """Search memory using a query string."""
    try:
        if ctx:
            ctx.info(f"Searching for: {query}")

        results = await kg.search_nodes(query)
        return {"success": True, "results": serialize_to_dict(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def delete_entities(names: List[str], ctx: Context = None) -> Dict[str, Any]:
    """Delete multiple entities and their relations."""
    try:
        if ctx:
            ctx.info(f"Deleting entities: {', '.join(names)}")

        deleted = await kg.delete_entities(names)
        return {"success": True, "deleted": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def delete_relation(
    from_entity: str, to_entity: str, ctx: Context = None
) -> Dict[str, Any]:
    """Delete relations between two entities."""
    try:
        if ctx:
            ctx.info(f"Deleting relations between {from_entity} and {to_entity}")

        await kg.delete_relations(from_entity, to_entity)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def flush_memory(ctx: Context = None) -> Dict[str, Any]:
    """Ensure all changes are persisted to storage."""
    try:
        if ctx:
            ctx.info("Flushing memory to storage")

        await kg.flush()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.prompt()
def create_entity_prompt(name: str, entity_type: str) -> str:
    """Generate prompt for entity creation."""
    return f"""I want to create a new entity in memory:
Name: {name}
Type: {entity_type}

What observations should I record about this entity?"""


@mcp.prompt()
def search_prompt(query: str) -> str:
    """Generate prompt for memory search."""
    return f"""I want to search my memory for information about: {query}

What specific aspects of these results would you like me to explain?"""


@mcp.prompt()
def relation_prompt(from_entity: str, to_entity: str) -> str:
    """Generate prompt for creating a relation."""
    return f"""I want to establish a relationship between:
Source: {from_entity}
Target: {to_entity}

What type of relationship exists between these entities?"""
