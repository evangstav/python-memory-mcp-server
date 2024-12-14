import pytest
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager

@pytest.mark.asyncio
async def test_create_entity(knowledge_graph_manager):
    """Test creating a new entity."""
    entity_name = "John"
    entity_type = "Person"
    observations = ["loves pizza", "works as developer"]
    
    # Create entity
    success = await knowledge_graph_manager.create_entity(
        name=entity_name,
        entity_type=entity_type,
        observations=observations
    )
    assert success is True

    # Verify entity exists
    entities = await knowledge_graph_manager.search_nodes("John")
    assert len(entities) == 1
    assert entities[0]["name"] == entity_name
    assert entities[0]["entity_type"] == entity_type

@pytest.mark.asyncio
async def test_create_relation(knowledge_graph_manager):
    """Test creating a relation between entities."""
    # Create two entities
    await knowledge_graph_manager.create_entity(
        name="Alice",
        entity_type="Person",
        observations=["likes coding"]
    )
    await knowledge_graph_manager.create_entity(
        name="Bob",
        entity_type="Person",
        observations=["likes gaming"]
    )

    # Create relation
    success = await knowledge_graph_manager.create_relation(
        from_entity="Alice",
        to_entity="Bob",
        relation_type="friends_with"
    )
    assert success is True

    # Verify relation
    graph = await knowledge_graph_manager.read_graph()
    relations = [r for r in graph["relations"] 
                if r["from"] == "Alice" and r["to"] == "Bob"]
    assert len(relations) == 1
    assert relations[0]["relationType"] == "friends_with"

@pytest.mark.asyncio
async def test_add_observations(knowledge_graph_manager):
    """Test adding observations to an existing entity."""
    # Create entity
    await knowledge_graph_manager.create_entity(
        name="Eve",
        entity_type="Person",
        observations=["initial observation"]
    )

    # Add new observations
    new_observations = ["likes tea", "works remotely"]
    success = await knowledge_graph_manager.add_observations(
        entity_name="Eve",
        observations=new_observations
    )
    assert success is True

    # Verify observations
    entities = await knowledge_graph_manager.search_nodes("Eve")
    assert len(entities) == 1
    all_observations = entities[0]["observations"]
    assert "initial observation" in all_observations
    assert all(obs in all_observations for obs in new_observations)

@pytest.mark.asyncio
async def test_delete_entity(knowledge_graph_manager):
    """Test deleting an entity."""
    # Create entity
    await knowledge_graph_manager.create_entity(
        name="ToDelete",
        entity_type="TestEntity",
        observations=["test observation"]
    )

    # Delete entity
    success = await knowledge_graph_manager.delete_entity("ToDelete")
    assert success is True

    # Verify entity is deleted
    entities = await knowledge_graph_manager.search_nodes("ToDelete")
    assert len(entities) == 0

@pytest.mark.asyncio
async def test_delete_relation(knowledge_graph_manager):
    """Test deleting a relation."""
    # Create entities and relation
    await knowledge_graph_manager.create_entity(
        name="Alice",
        entity_type="Person",
        observations=[]
    )
    await knowledge_graph_manager.create_entity(
        name="Bob",
        entity_type="Person",
        observations=[]
    )
    await knowledge_graph_manager.create_relation(
        from_entity="Alice",
        to_entity="Bob",
        relation_type="knows"
    )

    # Delete relation
    success = await knowledge_graph_manager.delete_relation(
        from_entity="Alice",
        to_entity="Bob",
        relation_type="knows"
    )
    assert success is True

    # Verify relation is deleted
    graph = await knowledge_graph_manager.read_graph()
    relations = [r for r in graph["relations"] 
                if r["from"] == "Alice" and r["to"] == "Bob"]
    assert len(relations) == 0