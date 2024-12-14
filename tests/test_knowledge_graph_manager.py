import pytest
import asyncio
from pathlib import Path
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager
from memory_mcp_server.interfaces import Entity, Relation, KnowledgeGraph
from memory_mcp_server.exceptions import EntityNotFoundError

@pytest.mark.asyncio
async def test_create_entities(knowledge_graph_manager):
    """Test creating new entities."""
    entities = [
        Entity(
            name="John",
            entityType="Person",
            observations=["loves pizza", "works as developer"]
        ),
        Entity(
            name="Alice",
            entityType="Person",
            observations=["likes coding"]
        )
    ]
    
    # Create entities
    created_entities = await knowledge_graph_manager.create_entities(entities)
    assert len(created_entities) == 2

    # Verify entities exist in cache
    graph = await knowledge_graph_manager.read_graph()
    assert len(graph.entities) == 2
    assert any(e.name == "John" for e in graph.entities)
    assert any(e.name == "Alice" for e in graph.entities)

    # Verify cache was built properly
    john = knowledge_graph_manager._get_entity_by_name("John")
    assert john is not None
    assert "loves pizza" in john.observations

@pytest.mark.asyncio
async def test_create_relations_with_indices(knowledge_graph_manager):
    """Test creating relations between entities with index verification."""
    # First create entities
    entities = [
        Entity(name="Alice", entityType="Person", observations=["likes coding"]),
        Entity(name="Bob", entityType="Person", observations=["likes gaming"])
    ]
    await knowledge_graph_manager.create_entities(entities)

    # Create relation
    relations = [
        Relation(
            from_="Alice",
            to="Bob",
            relationType="friends_with"
        )
    ]
    
    created_relations = await knowledge_graph_manager.create_relations(relations)
    assert len(created_relations) == 1

    # Verify relation in main graph
    graph = await knowledge_graph_manager.read_graph()
    assert len(graph.relations) == 1
    
    # Verify relation in indices
    alice_relations = knowledge_graph_manager._indices["relations_from"]["Alice"]
    assert len(alice_relations) == 1
    assert alice_relations[0].to == "Bob"
    assert alice_relations[0].relationType == "friends_with"

@pytest.mark.asyncio
async def test_add_observations_with_cache(knowledge_graph_manager):
    """Test adding observations with cache verification."""
    # Create initial entity
    entities = [Entity(name="Eve", entityType="Person", observations=["initial observation"])]
    await knowledge_graph_manager.create_entities(entities)

    # Add new observations
    observations = [{
        "entityName": "Eve",
        "contents": ["likes tea", "works remotely"]
    }]
    
    results = await knowledge_graph_manager.add_observations(observations)
    assert len(results) == 1
    assert len(results[0]["addedObservations"]) == 2

    # Verify in cache
    eve = knowledge_graph_manager._get_entity_by_name("Eve")
    assert eve is not None
    assert "initial observation" in eve.observations
    assert "likes tea" in eve.observations
    assert "works remotely" in eve.observations

@pytest.mark.asyncio
async def test_search_with_indices(knowledge_graph_manager):
    """Test search functionality using indices."""
    # Create test entities
    entities = [
        Entity(name="SearchTest1", entityType="TestEntity", observations=["keyword1"]),
        Entity(name="SearchTest2", entityType="TestEntity", observations=["keyword2"]),
        Entity(name="DifferentType", entityType="OtherEntity", observations=["keyword1"])
    ]
    await knowledge_graph_manager.create_entities(entities)

    # Search by name
    result = await knowledge_graph_manager.search_nodes("SearchTest")
    assert len(result.entities) == 2

    # Search by type
    result = await knowledge_graph_manager.search_nodes("OtherEntity")
    assert len(result.entities) == 1
    assert result.entities[0].name == "DifferentType"

    # Search by observation
    result = await knowledge_graph_manager.search_nodes("keyword1")
    assert len(result.entities) == 2

@pytest.mark.asyncio
async def test_cache_invalidation(knowledge_graph_manager):
    """Test that cache is properly invalidated when needed."""
    # Create initial entity
    entity = Entity(name="CacheTest", entityType="Test", observations=["initial"])
    await knowledge_graph_manager.create_entities([entity])

    # Get initial cache state
    initial_cache = await knowledge_graph_manager._check_cache()

    # Wait for cache to expire (cache_ttl is 1 second in test fixture)
    await asyncio.sleep(1.1)

    # Add new entity
    new_entity = Entity(name="NewEntity", entityType="Test", observations=["test"])
    await knowledge_graph_manager.create_entities([new_entity])

    # Get new cache state
    new_cache = await knowledge_graph_manager._check_cache()

    assert len(new_cache.entities) > len(initial_cache.entities)

@pytest.mark.asyncio
async def test_batch_operations(knowledge_graph_manager):
    """Test write queue and batch operations."""
    large_entities = [
        Entity(name=f"Entity{i}", entityType="Test", observations=[f"obs{i}"])
        for i in range(knowledge_graph_manager._max_queue_size + 5)
    ]

    # This should trigger at least one batch write due to queue size
    created = await knowledge_graph_manager.create_entities(large_entities)
    assert len(created) == len(large_entities)

    # Force flush any remaining writes
    await knowledge_graph_manager.flush()

    # Verify all entities were written
    graph = await knowledge_graph_manager.read_graph()
    assert len(graph.entities) == len(large_entities)

@pytest.mark.asyncio
async def test_error_handling(knowledge_graph_manager):
    """Test error handling in various scenarios."""
    # Test invalid entity name
    with pytest.raises(ValueError):
        await knowledge_graph_manager.create_entities([
            Entity(name="", entityType="Test", observations=[])
        ])

    # Test non-existent entity in relations
    with pytest.raises(EntityNotFoundError):
        await knowledge_graph_manager.create_relations([
            Relation(from_="NonExistent", to="AlsoNonExistent", relationType="test")
        ])