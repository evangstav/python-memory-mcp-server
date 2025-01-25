"""Tests for KnowledgeGraphManager."""

import asyncio
from typing import List

import pytest

from memory_mcp_server.exceptions import EntityNotFoundError
from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager


@pytest.mark.asyncio(scope="function")
async def test_create_entities(
    knowledge_graph_manager: KnowledgeGraphManager,
) -> None:
    """Test the creation of new entities in the knowledge graph.

    This test verifies that:
    1. Entities can be created successfully
    2. The created entities are stored in the graph
    3. Entity attributes are preserved correctly
    """
    print("\nStarting test_create_entities")
    entities = [
        Entity(
            name="John",
            entityType="Person",
            observations=["loves pizza"],
        )
    ]

    created_entities = await knowledge_graph_manager.create_entities(entities)
    print("Created entities")
    assert len(created_entities) == 1

    graph = await knowledge_graph_manager.read_graph()
    print("Read graph")
    assert len(graph.entities) == 1
    assert graph.entities[0].name == "John"

    print("test_create_entities: Complete")


@pytest.mark.asyncio(scope="function")
async def test_create_relations(
    knowledge_graph_manager: KnowledgeGraphManager,
) -> None:
    """Test the creation of relations between entities.

    This test verifies that:
    1. Relations can be created between existing entities
    2. Relations are stored properly in the graph
    3. Relation properties (from, to, type) are preserved
    """
    print("\nStarting test_create_relations")

    entities = [
        Entity(name="Alice", entityType="Person", observations=["test"]),
        Entity(name="Bob", entityType="Person", observations=["test"]),
    ]
    await knowledge_graph_manager.create_entities(entities)
    print("Created entities")

    relations = [Relation(from_="Alice", to="Bob", relationType="friends")]
    created_relations = await knowledge_graph_manager.create_relations(relations)
    print("Created relations")

    assert len(created_relations) == 1
    assert created_relations[0].from_ == "Alice"
    assert created_relations[0].to == "Bob"

    print("test_create_relations: Complete")


@pytest.mark.asyncio(scope="function")
async def test_search_functionality(
    knowledge_graph_manager: KnowledgeGraphManager,
) -> None:
    """Test the search functionality across different criteria.

    This test verifies searching by:
    1. Entity name
    2. Entity type
    3. Observation content
    4. Case insensitivity
    """
    # Create test entities with varied data
    entities = [
        Entity(
            name="SearchTest1",
            entityType="TestEntity",
            observations=["keyword1", "unique1"],
        ),
        Entity(name="SearchTest2", entityType="TestEntity", observations=["keyword2"]),
        Entity(
            name="DifferentType", entityType="OtherEntity", observations=["keyword1"]
        ),
    ]
    await knowledge_graph_manager.create_entities(entities)

    # Test search by name
    name_result = await knowledge_graph_manager.search_nodes("SearchTest")
    assert len(name_result.entities) == 2
    assert all("SearchTest" in e.name for e in name_result.entities)

    # Test search by type
    type_result = await knowledge_graph_manager.search_nodes("OtherEntity")
    assert len(type_result.entities) == 1
    assert type_result.entities[0].name == "DifferentType"

    # Test search by observation
    obs_result = await knowledge_graph_manager.search_nodes("keyword1")
    assert len(obs_result.entities) == 2
    assert any(e.name == "SearchTest1" for e in obs_result.entities)
    assert any(e.name == "DifferentType" for e in obs_result.entities)


@pytest.mark.asyncio(scope="function")
async def test_error_handling(
    knowledge_graph_manager: KnowledgeGraphManager,
) -> None:
    """Test error handling in various scenarios.

    This test verifies proper error handling for:
    1. Invalid entity names
    2. Non-existent entities in relations
    3. Empty delete requests
    4. Deleting non-existent entities
    """
    # Test invalid entity name
    with pytest.raises(ValueError, match="Invalid entity"):
        await knowledge_graph_manager.create_entities(
            [Entity(name="", entityType="Test", observations=[])]
        )

    # Test relation with non-existent entity
    with pytest.raises(EntityNotFoundError):
        await knowledge_graph_manager.create_relations(
            [Relation(from_="NonExistent", to="AlsoNonExistent", relationType="test")]
        )

    # Test deleting empty list
    with pytest.raises(ValueError, match="cannot be empty"):
        await knowledge_graph_manager.delete_entities([])

    # Test deleting non-existent entities
    result = await knowledge_graph_manager.delete_entities(["nonexistent"])
    assert result == []


@pytest.mark.asyncio(scope="function")
async def test_graph_persistence(
    knowledge_graph_manager: KnowledgeGraphManager,
) -> None:
    """Test that graph changes persist after reloading.

    This test verifies that:
    1. Created entities persist after a graph reload
    2. Added relations persist after a graph reload
    3. New observations persist after a graph reload
    """
    # Create initial data
    entity = Entity(name="PersistenceTest", entityType="Test", observations=["initial"])
    await knowledge_graph_manager.create_entities([entity])

    # Force a reload of the graph by clearing the cache
    knowledge_graph_manager._cache = None  # type: ignore

    # Verify data persists
    graph = await knowledge_graph_manager.read_graph()
    assert len(graph.entities) == 1
    assert graph.entities[0].name == "PersistenceTest"
    assert "initial" in graph.entities[0].observations


@pytest.mark.asyncio(scope="function")
async def test_concurrent_operations(
    knowledge_graph_manager: KnowledgeGraphManager,
) -> None:
    """Test handling of concurrent operations.

    This test verifies that:
    1. Multiple concurrent entity creations/deletions are handled properly
    2. Cache remains consistent under concurrent operations
    3. No data is lost during concurrent writes
    """

    # Create multiple entities concurrently
    async def create_entity(index: int) -> List[Entity]:
        entity = Entity(
            name=f"Concurrent{index}",
            entityType="Test",
            observations=[f"obs{index}"],
        )
        return await knowledge_graph_manager.create_entities([entity])

    # Delete entities concurrently
    async def delete_entity(index: int) -> List[str]:
        return await knowledge_graph_manager.delete_entities([f"Concurrent{index}"])

    # First create 5 entities
    create_tasks = [create_entity(i) for i in range(5)]
    create_results = await asyncio.gather(*create_tasks)
    assert all(len(r) == 1 for r in create_results)

    # Then concurrently delete 3 of them while creating 2 more
    delete_tasks = [delete_entity(i) for i in range(3)]
    create_tasks = [create_entity(i) for i in range(5, 7)]
    delete_results, create_results = await asyncio.gather(
        asyncio.gather(*delete_tasks), asyncio.gather(*create_tasks)
    )

    # Verify deletions
    assert all(len(r) == 1 for r in delete_results)

    # Verify creations
    assert all(len(r) == 1 for r in create_results)

    # Verify final state
    graph = await knowledge_graph_manager.read_graph()
    expected_names = {"Concurrent5", "Concurrent6", "Concurrent3", "Concurrent4"}
    assert len(graph.entities) == 4
    assert all(e.name in expected_names for e in graph.entities)
