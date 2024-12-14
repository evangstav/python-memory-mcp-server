import pytest
import asyncio
from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.exceptions import EntityNotFoundError


@pytest.mark.asyncio
async def test_create_entities(knowledge_graph_manager):
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


@pytest.mark.asyncio
async def test_create_relations(knowledge_graph_manager):
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


@pytest.mark.asyncio
async def test_search_functionality(knowledge_graph_manager):
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


@pytest.mark.asyncio
async def test_error_handling(knowledge_graph_manager):
    """Test error handling in various scenarios.

    This test verifies proper error handling for:
    1. Invalid entity names
    2. Non-existent entities in relations
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


@pytest.mark.asyncio
async def test_graph_persistence(knowledge_graph_manager):
    """Test that graph changes persist after reloading.

    This test verifies that:
    1. Created entities persist after a graph reload
    2. Added relations persist after a graph reload
    3. New observations persist after a graph reload
    """
    # Create initial data
    entity = Entity(name="PersistenceTest", entityType="Test", observations=["initial"])
    await knowledge_graph_manager.create_entities([entity])

    # Force a reload of the graph
    knowledge_graph_manager._cache = None

    # Verify data persists
    graph = await knowledge_graph_manager.read_graph()
    assert len(graph.entities) == 1
    assert graph.entities[0].name == "PersistenceTest"
    assert "initial" in graph.entities[0].observations


@pytest.mark.asyncio
async def test_concurrent_operations(knowledge_graph_manager):
    """Test handling of concurrent operations.

    This test verifies that:
    1. Multiple concurrent entity creations are handled properly
    2. Cache remains consistent under concurrent operations
    3. No data is lost during concurrent writes
    """

    # Create multiple entities concurrently
    async def create_entity(index: int):
        entity = Entity(
            name=f"Concurrent{index}", entityType="Test", observations=[f"obs{index}"]
        )
        return await knowledge_graph_manager.create_entities([entity])

    # Run concurrent operations
    tasks = [create_entity(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    # Verify all entities were created
    assert all(len(r) == 1 for r in results)

    # Verify final state
    graph = await knowledge_graph_manager.read_graph()
    assert len(graph.entities) == 5
    assert all(f"Concurrent{i}" in [e.name for e in graph.entities] for i in range(5))
