"""Benchmarking utilities for Memory MCP Server."""

import random
import string
import time
from typing import Dict, List, Optional

from memory_mcp_server.interfaces import Entity, KnowledgeGraph, Relation
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager


async def generate_test_graph(
    node_count: int = 1000,
    observations_per_node: int = 5,
    relations_per_node: float = 2.0,
    entity_types: Optional[List[str]] = None,
    relation_types: Optional[List[str]] = None,
) -> KnowledgeGraph:
    """Generate a test knowledge graph of specified size."""
    if not entity_types:
        entity_types = ["person", "project", "concept", "tool", "location"]
    if not relation_types:
        relation_types = ["knows", "contains", "uses", "created", "related-to"]

    # Generate entities
    entities = []
    for i in range(node_count):
        entity_type = random.choice(entity_types)
        name = f"{entity_type}-{i}"
        observations = [
            f"Observation {j} for {name}" for j in range(observations_per_node)
        ]
        entities.append(
            Entity(name=name, entityType=entity_type, observations=observations)
        )

    # Generate relations
    relations = []
    for entity in entities:
        # Each entity will have a random number of relations
        relation_count = int(random.expovariate(1.0 / relations_per_node))
        for _ in range(relation_count):
            target = random.choice(entities)
            # Avoid self-relations
            if target.name != entity.name:
                rel_type = random.choice(relation_types)
                relations.append(
                    Relation(from_=entity.name, to=target.name, relationType=rel_type)
                )

    return KnowledgeGraph(entities=entities, relations=relations)


async def benchmark_search(
    kg_manager: KnowledgeGraphManager, query_count: int = 100, query_length: int = 3
) -> Dict[str, float]:
    """Benchmark search performance."""
    # Generate random search queries
    queries = []
    for _ in range(query_count):
        query = " ".join(
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            for _ in range(query_length)
        )
        queries.append(query)

    # Measure search time
    start_time = time.perf_counter()
    for query in queries:
        await kg_manager.search_nodes(query)
    total_time = time.perf_counter() - start_time

    return {
        "total_time_ms": total_time * 1000,
        "avg_time_ms": (total_time * 1000) / query_count,
        "query_count": query_count,
    }


async def benchmark_create_entities(
    kg_manager: KnowledgeGraphManager, entity_count: int = 100, batch_size: int = 10
) -> Dict[str, float]:
    """Benchmark entity creation performance."""
    # Generate entities
    all_entities = []
    for i in range(entity_count):
        name = f"benchmark-entity-{i}"
        entity_type = random.choice(["person", "project", "concept", "tool"])
        observations = [f"Benchmark observation {j}" for j in range(3)]
        all_entities.append(
            Entity(name=name, entityType=entity_type, observations=observations)
        )

    # Create entities in batches
    start_time = time.perf_counter()
    for i in range(0, entity_count, batch_size):
        batch = all_entities[i : i + batch_size]
        await kg_manager.create_entities(batch)
    total_time = time.perf_counter() - start_time

    # Clean up benchmark entities
    await kg_manager.delete_entities([e.name for e in all_entities])

    return {
        "total_time_ms": total_time * 1000,
        "avg_time_per_entity_ms": (total_time * 1000) / entity_count,
        "entity_count": entity_count,
        "batch_size": batch_size,
    }
