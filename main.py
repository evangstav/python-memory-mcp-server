#!/usr/bin/env python3
"""Memory MCP server using FastMCP."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger as logging
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import Message, UserMessage
from pydantic import BaseModel

from memory_mcp_server.benchmarking.benchmarks import (
    benchmark_create_entities,
    benchmark_search,
    generate_test_graph,
)
from memory_mcp_server.benchmarking.profiler import profiler
from memory_mcp_server.interfaces import Entity, Relation, SearchOptions
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
    message: Optional[str] = None


# Create FastMCP server with dependencies and instructions
mcp = FastMCP(
    "Memory",
    dependencies=["pydantic", "numpy"],
    version="0.2.0",
    instructions="""
    Memory MCP server providing knowledge graph functionality with semantic search capabilities.

    Available tools:
    - get_entity: Retrieve entity by name
    - get_graph: Get entire knowledge graph
    - create_entities: Create multiple entities
    - add_observation: Add observation to entity
    - create_relation: Create relation between entities
    - search_memory: Search entities using semantic understanding and natural language
    - search_nodes: Search entities using exact or fuzzy matching
    - delete_entities: Delete multiple entities
    - delete_relation: Delete relation between entities
    - flush_memory: Persist changes to storage
    - regenerate_embeddings: Rebuild embeddings for all entities

    The semantic search capabilities allow for:
    - Finding conceptually similar entities even when exact terms don't match
    - Understanding temporal references (recent, past, etc.)
    - Identifying entity types and attributes in queries
    - Detecting relationship-focused queries
    """,
)


def get_backend(memory_path: Path):
    """Get appropriate backend based on file extension."""
    if memory_path.suffix.lower() == ".db":
        from memory_mcp_server.backends.sqlite import SQLiteBackend

        logging.info(f"Using SQLite backend with file: {memory_path}")
        return SQLiteBackend(memory_path)
    else:
        from memory_mcp_server.backends.jsonl import JsonlBackend

        logging.info(f"Using JSONL backend with file: {memory_path}")
        return JsonlBackend(memory_path, 60)


# Initialize knowledge graph manager using environment variable
# Default to ~/.claude/memory.jsonl if MEMORY_FILE_PATH not set
default_memory_path = Path.home() / ".claude" / "memory.json"
memory_file = Path(os.getenv("MEMORY_FILE_PATH", str(default_memory_path)))

logging.info(f"Memory server using file: {memory_file}")

# Create KnowledgeGraphManager instance with appropriate backend
backend = get_backend(memory_file)
kg = KnowledgeGraphManager(backend, 60)


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
async def create_entities(
    entities: List[Entity], ctx: Context = None
) -> EntityResponse:
    """Create multiple new entities."""
    try:
        if ctx:
            ctx.info(f"Creating {len(entities)} entities")

        created = await kg.create_entities(entities)
        return EntityResponse(
            success=True, data={"entities": [e.to_dict() for e in created]}
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
async def add_observation(
    entity: str, observation: str, ctx: Context = None
) -> OperationResponse:
    """Add an observation to an existing entity."""
    try:
        if ctx:
            ctx.info(f"Adding observation to {entity}")

        # Check if entity exists
        exists = await kg.search_nodes(entity)
        if not exists.entities:
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
async def search_memory(
    query: str, semantic: bool = True, max_results: int = 10, ctx: Context = None
) -> EntityResponse:
    """Search memory using natural language queries.

    Handles:
    - Semantic search for conceptually similar entities
    - Temporal queries (e.g., "most recent", "last", "latest")
    - Entity-specific queries (e.g., "people who know about Python")
    - General knowledge graph exploration

    Args:
        query: Natural language search query
        semantic: Whether to use semantic search (default: True)
        max_results: Maximum number of results to return (default: 10)
        ctx: Optional context for logging

    Returns:
        EntityResponse containing matches
    """
    try:
        if ctx:
            ctx.info(f"Enhanced search for: {query}")

        # Use enhanced search with semantic capabilities
        options = SearchOptions(
            semantic=semantic, max_results=max_results, include_relations=True
        )
        results = await kg.enhanced_search(query, options)

        if not results.entities:
            return EntityResponse(
                success=True,
                data={"entities": [], "relations": []},
                error="No matching entities found in memory",
                error_type="NO_RESULTS",
            )

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
async def search_nodes(
    query: str, fuzzy: bool = False, threshold: float = 80.0, ctx: Context = None
) -> EntityResponse:
    """Search for entities and relations matching query using exact or fuzzy matching.

    Args:
        query: Search query string
        fuzzy: Whether to use fuzzy matching (default: False)
        threshold: Threshold for fuzzy matching (default: 80.0)
        ctx: Optional context for logging

    Returns:
        EntityResponse containing matches
    """
    try:
        if ctx:
            ctx.info(f"Searching for: {query}")

        # Configure search options
        options = None
        if fuzzy:
            options = SearchOptions(fuzzy=True, threshold=threshold, semantic=False)

        # Use standard search
        results = await kg.search_nodes(query, options)

        if not results.entities:
            return EntityResponse(
                success=True,
                data={"entities": [], "relations": []},
                error="No matching entities found in memory",
                error_type="NO_RESULTS",
            )

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


@mcp.tool()
async def regenerate_embeddings(ctx: Context = None) -> OperationResponse:
    """Regenerate embeddings for all entities in the knowledge graph.

    This is useful after importing data or if embeddings become outdated.
    """
    try:
        if ctx:
            ctx.info("Regenerating embeddings for all entities")

        # Get all entities
        graph = await kg.read_graph()
        count = 0

        # Process each entity
        for entity in graph.entities:
            # Create a combined text representation of the entity
            entity_text = f"{entity.name} {entity.entityType} " + " ".join(
                entity.observations
            )
            # Generate embedding
            embedding = kg.embedding_service.encode_text(entity_text)
            # Store embedding
            await kg.backend.store_embedding(entity.name, embedding)
            count += 1

        return OperationResponse(
            success=True,
            message=f"Successfully regenerated embeddings for {count} entities",
        )
    except Exception as e:
        return OperationResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )


@mcp.tool()
async def run_performance_benchmarks(
    include_search: bool = True, include_create: bool = True, include_read: bool = True
) -> Dict[str, Any]:
    """Run performance benchmarks on the memory system."""
    results = {}

    if include_search:
        results["search"] = await benchmark_search(kg)

    if include_create:
        results["create_entities"] = await benchmark_create_entities(kg)

    if include_read:
        start_time = time.perf_counter()
        graph = await kg.read_graph()
        read_time = time.perf_counter() - start_time
        results["read_graph"] = {
            "time_ms": read_time * 1000,
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
        }

    return results


@mcp.tool()
async def get_performance_metrics() -> Dict[str, Any]:
    """Get collected performance metrics."""
    return profiler.get_metrics()


@mcp.tool()
async def enable_performance_profiling() -> Dict[str, Any]:
    """Enable performance profiling."""
    profiler.enable()
    return {"status": "enabled"}


@mcp.tool()
async def disable_performance_profiling() -> Dict[str, Any]:
    """Disable performance profiling."""
    profiler.disable()
    metrics = profiler.get_metrics()
    profiler.clear()
    return {"status": "disabled", "metrics": metrics}


@mcp.tool()
async def generate_synthetic_graph(
    node_count: int = 1000,
    observations_per_node: int = 5,
    relations_per_node: float = 2.0,
) -> Dict[str, Any]:
    """Generate a synthetic graph for testing purposes."""
    try:
        # Generate test graph
        test_graph = await generate_test_graph(
            node_count=node_count,
            observations_per_node=observations_per_node,
            relations_per_node=relations_per_node,
        )

        # Create entities and relations
        start_time = time.perf_counter()
        created_entities = await kg.create_entities(test_graph.entities)
        entity_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        created_relations = await kg.create_relations(test_graph.relations)
        relation_time = time.perf_counter() - start_time

        return {
            "success": True,
            "entities_created": len(created_entities),
            "relations_created": len(created_relations),
            "entity_creation_time_ms": entity_time * 1000,
            "relation_creation_time_ms": relation_time * 1000,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
            f"I'll use semantic search to find the most relevant information. "
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
