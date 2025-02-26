"""Tests for benchmarking utilities."""

from unittest.mock import AsyncMock

import pytest

from memory_mcp_server.benchmarking.benchmarks import (
    benchmark_create_entities,
    benchmark_search,
    generate_test_graph,
)
from memory_mcp_server.interfaces import KnowledgeGraph


@pytest.mark.asyncio
async def test_generate_test_graph():
    # Generate small test graph
    graph = await generate_test_graph(
        node_count=10, observations_per_node=2, relations_per_node=1.0
    )

    # Check entities
    assert len(graph.entities) == 10

    # Check entity structure
    entity = graph.entities[0]
    assert entity.name is not None
    assert entity.entityType is not None
    assert len(entity.observations) == 2

    # Check relations
    # Since relations are generated randomly, we can't assert exact count
    # But we can check structures
    if graph.relations:
        relation = graph.relations[0]
        assert relation.from_ is not None
        assert relation.to is not None
        assert relation.relationType is not None

        # The target entity should exist
        assert any(e.name == relation.to for e in graph.entities)


@pytest.mark.asyncio
async def test_generate_test_graph_custom_types():
    # Generate graph with custom types
    entity_types = ["custom_type1", "custom_type2"]
    relation_types = ["custom_rel1", "custom_rel2"]

    graph = await generate_test_graph(
        node_count=10,
        observations_per_node=2,
        relations_per_node=1.0,
        entity_types=entity_types,
        relation_types=relation_types,
    )

    # Check that custom entity types were used
    entity_type_set = set(e.entityType for e in graph.entities)
    assert entity_type_set.issubset(set(entity_types))

    # Check that custom relation types were used (if relations exist)
    if graph.relations:
        relation_type_set = set(r.relationType for r in graph.relations)
        assert relation_type_set.issubset(set(relation_types))


@pytest.mark.asyncio
async def test_benchmark_search():
    # Create a mock KnowledgeGraphManager
    mock_kg_manager = AsyncMock()
    mock_kg_manager.search_nodes = AsyncMock(
        return_value=KnowledgeGraph(entities=[], relations=[])
    )

    # Run benchmark
    results = await benchmark_search(mock_kg_manager, query_count=5, query_length=2)

    # Check result structure
    assert "total_time_ms" in results
    assert "avg_time_ms" in results
    assert "query_count" in results
    assert results["query_count"] == 5

    # Check that search was called the correct number of times
    assert mock_kg_manager.search_nodes.call_count == 5


@pytest.mark.asyncio
async def test_benchmark_create_entities():
    # Create a mock KnowledgeGraphManager
    mock_kg_manager = AsyncMock()
    mock_kg_manager.create_entities = AsyncMock(return_value=[])
    mock_kg_manager.delete_entities = AsyncMock(return_value=[])

    # Run benchmark
    results = await benchmark_create_entities(
        mock_kg_manager, entity_count=10, batch_size=5
    )

    # Check result structure
    assert "total_time_ms" in results
    assert "avg_time_per_entity_ms" in results
    assert "entity_count" in results
    assert "batch_size" in results
    assert results["entity_count"] == 10
    assert results["batch_size"] == 5

    # Check that create_entities was called the correct number of times
    assert (
        mock_kg_manager.create_entities.call_count == 2
    )  # 10 entities ÷ 5 batch size = 2 batches

    # Check that delete_entities was called with all created entities
    mock_kg_manager.delete_entities.assert_called_once()
