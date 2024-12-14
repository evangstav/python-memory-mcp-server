import pytest
from memory_mcp_server.optimized_knowledge_graph_manager import OptimizedKnowledgeGraphManager

@pytest.mark.asyncio
async def test_batch_operations(optimized_knowledge_graph_manager):
    """Test batch operations for creating entities and relations."""
    # Test batch entity creation
    entities = [
        {
            "name": "Alice",
            "entity_type": "Person",
            "observations": ["likes coding", "drinks coffee"]
        },
        {
            "name": "Bob",
            "entity_type": "Person",
            "observations": ["likes gaming", "drinks tea"]
        }
    ]
    
    success = await optimized_knowledge_graph_manager.create_entities(entities)
    assert success is True

    # Verify entities were created
    graph = await optimized_knowledge_graph_manager.read_graph()
    created_entities = [e for e in graph["entities"] 
                       if e["name"] in ["Alice", "Bob"]]
    assert len(created_entities) == 2

    # Test batch relation creation
    relations = [
        {
            "from_entity": "Alice",
            "to_entity": "Bob",
            "relation_type": "friends_with"
        },
        {
            "from_entity": "Bob",
            "to_entity": "Alice",
            "relation_type": "friends_with"
        }
    ]
    
    success = await optimized_knowledge_graph_manager.create_relations(relations)
    assert success is True

    # Verify relations were created
    graph = await optimized_knowledge_graph_manager.read_graph()
    created_relations = [r for r in graph["relations"] 
                        if r["from"] in ["Alice", "Bob"]]
    assert len(created_relations) == 2

@pytest.mark.asyncio
async def test_optimized_search(optimized_knowledge_graph_manager):
    """Test optimized search functionality."""
    # Create test entities
    await optimized_knowledge_graph_manager.create_entity(
        name="SearchTest1",
        entity_type="TestEntity",
        observations=["keyword1", "keyword2"]
    )
    await optimized_knowledge_graph_manager.create_entity(
        name="SearchTest2",
        entity_type="TestEntity",
        observations=["keyword2", "keyword3"]
    )

    # Test search with different queries
    results = await optimized_knowledge_graph_manager.search_nodes("keyword1")
    assert len(results) == 1
    assert results[0]["name"] == "SearchTest1"

    results = await optimized_knowledge_graph_manager.search_nodes("keyword2")
    assert len(results) == 2

    results = await optimized_knowledge_graph_manager.search_nodes("TestEntity")
    assert len(results) == 2

@pytest.mark.asyncio
async def test_bulk_observations(optimized_knowledge_graph_manager):
    """Test adding multiple observations to multiple entities."""
    # Create test entities
    entities = ["BulkTest1", "BulkTest2"]
    for entity in entities:
        await optimized_knowledge_graph_manager.create_entity(
            name=entity,
            entity_type="TestEntity",
            observations=["initial"]
        )

    # Add bulk observations
    observations = [
        {
            "entity_name": "BulkTest1",
            "observations": ["bulk1", "bulk2"]
        },
        {
            "entity_name": "BulkTest2",
            "observations": ["bulk3", "bulk4"]
        }
    ]

    success = await optimized_knowledge_graph_manager.add_bulk_observations(observations)
    assert success is True

    # Verify observations were added
    for entity in entities:
        result = await optimized_knowledge_graph_manager.search_nodes(entity)
        assert len(result) == 1
        assert "initial" in result[0]["observations"]
        if entity == "BulkTest1":
            assert all(obs in result[0]["observations"] for obs in ["bulk1", "bulk2"])
        else:
            assert all(obs in result[0]["observations"] for obs in ["bulk3", "bulk4"])