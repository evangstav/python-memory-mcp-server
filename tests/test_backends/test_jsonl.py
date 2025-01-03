"""Tests for the JSONL backend implementation."""

import pytest
from pathlib import Path
import json

from memory_mcp_server.backends.jsonl import JsonlBackend
from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.exceptions import EntityNotFoundError


@pytest.fixture
async def jsonl_backend(tmp_path):
    """Create a temporary JSONL backend for testing."""
    backend = JsonlBackend(tmp_path / "test_memory.jsonl")
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_create_entities(jsonl_backend):
    """Test creating new entities."""
    entities = [
        Entity("test1", "person", ["observation1", "observation2"]),
        Entity("test2", "location", ["observation3"]),
    ]
    
    result = await jsonl_backend.create_entities(entities)
    assert len(result) == 2
    
    # Verify entities were saved
    graph = await jsonl_backend.read_graph()
    assert len(graph.entities) == 2
    assert any(e.name == "test1" and e.entityType == "person" for e in graph.entities)
    assert any(e.name == "test2" and e.entityType == "location" for e in graph.entities)


@pytest.mark.asyncio
async def test_create_relations(jsonl_backend):
    """Test creating relations between entities."""
    # Create entities first
    entities = [
        Entity("person1", "person", ["observation1"]),
        Entity("location1", "location", ["observation2"]),
    ]
    await jsonl_backend.create_entities(entities)
    
    # Create relation
    relations = [
        Relation(from_="person1", to="location1", relationType="visited")
    ]
    result = await jsonl_backend.create_relations(relations)
    assert len(result) == 1
    
    # Verify relation was saved
    graph = await jsonl_backend.read_graph()
    assert len(graph.relations) == 1
    assert graph.relations[0].from_ == "person1"
    assert graph.relations[0].to == "location1"
    assert graph.relations[0].relationType == "visited"


@pytest.mark.asyncio
async def test_create_relation_missing_entity(jsonl_backend):
    """Test creating relation with non-existent entity."""
    relations = [
        Relation(from_="nonexistent1", to="nonexistent2", relationType="test")
    ]
    
    with pytest.raises(EntityNotFoundError):
        await jsonl_backend.create_relations(relations)


@pytest.mark.asyncio
async def test_search_nodes(jsonl_backend):
    """Test searching nodes in the graph."""
    # Create test data
    entities = [
        Entity("test1", "person", ["likes coffee", "works at office"]),
        Entity("test2", "person", ["likes tea"]),
        Entity("office", "location", ["big building"]),
    ]
    await jsonl_backend.create_entities(entities)
    
    relations = [
        Relation(from_="test1", to="office", relationType="works_at")
    ]
    await jsonl_backend.create_relations(relations)
    
    # Test search
    result = await jsonl_backend.search_nodes("coffee")
    assert len(result.entities) == 1
    assert result.entities[0].name == "test1"
    
    result = await jsonl_backend.search_nodes("office")
    assert len(result.entities) == 2  # Both "office" entity and entity with "office" in observations
    assert "office" in {e.name for e in result.entities}
    assert "test1" in {e.name for e in result.entities}
    assert len(result.relations) == 1


@pytest.mark.asyncio
async def test_persistence(tmp_path):
    """Test that data persists between backend instances."""
    file_path = tmp_path / "persistence_test.jsonl"
    
    # Create first instance and add data
    backend1 = JsonlBackend(file_path)
    await backend1.initialize()
    
    entities = [Entity("test1", "person", ["observation1"])]
    await backend1.create_entities(entities)
    await backend1.close()
    
    # Create second instance and verify data
    backend2 = JsonlBackend(file_path)
    await backend2.initialize()
    
    graph = await backend2.read_graph()
    assert len(graph.entities) == 1
    assert graph.entities[0].name == "test1"
    await backend2.close()


@pytest.mark.asyncio
async def test_file_format(tmp_path):
    """Test that the JSONL file format is correct."""
    file_path = tmp_path / "format_test.jsonl"
    backend = JsonlBackend(file_path)
    await backend.initialize()
    
    # Add test data
    entities = [Entity("test1", "person", ["obs1"])]
    relations = [Relation(from_="test1", to="test1", relationType="self_ref")]
    
    await backend.create_entities(entities)
    await backend.create_relations(relations)
    await backend.close()
    
    # Verify file content
    with open(file_path) as f:
        lines = f.readlines()
        
    assert len(lines) == 2  # One entity and one relation
    
    # Verify entity format
    entity_line = json.loads(lines[0])
    assert entity_line["type"] == "entity"
    assert entity_line["name"] == "test1"
    assert entity_line["entityType"] == "person"
    assert entity_line["observations"] == ["obs1"]
    
    # Verify relation format
    relation_line = json.loads(lines[1])
    assert relation_line["type"] == "relation"
    assert relation_line["from"] == "test1"
    assert relation_line["to"] == "test1"
    assert relation_line["relationType"] == "self_ref"


@pytest.mark.asyncio
async def test_caching(jsonl_backend):
    """Test that caching works correctly."""
    entities = [Entity("test1", "person", ["obs1"])]
    await jsonl_backend.create_entities(entities)
    
    # First read should cache
    graph1 = await jsonl_backend.read_graph()
    
    # Second read should use cache
    graph2 = await jsonl_backend.read_graph()
    
    # Should be the same object if cached
    assert graph1 is graph2


@pytest.mark.asyncio
async def test_atomic_writes(tmp_path):
    """Test that writes are atomic using temp files."""
    file_path = tmp_path / "atomic_test.jsonl"
    temp_path = file_path.with_suffix(".tmp")
    
    backend = JsonlBackend(file_path)
    await backend.initialize()
    
    # Add some data
    entities = [Entity("test1", "person", ["obs1"])]
    await backend.create_entities(entities)
    await backend.close()
    
    # Verify temp file was cleaned up
    assert not temp_path.exists()
    assert file_path.exists()


@pytest.mark.asyncio
async def test_duplicate_entities(jsonl_backend):
    """Test handling of duplicate entities."""
    entity = Entity("test1", "person", ["obs1"])
    
    # First creation should succeed
    result1 = await jsonl_backend.create_entities([entity])
    assert len(result1) == 1
    
    # Second creation should return empty list (no new entities)
    result2 = await jsonl_backend.create_entities([entity])
    assert len(result2) == 0
    
    # Verify only one entity exists
    graph = await jsonl_backend.read_graph()
    assert len(graph.entities) == 1


@pytest.mark.asyncio
async def test_duplicate_relations(jsonl_backend):
    """Test handling of duplicate relations."""
    # Create test entities
    entities = [
        Entity("test1", "person", ["obs1"]),
        Entity("test2", "person", ["obs2"]),
    ]
    await jsonl_backend.create_entities(entities)
    
    relation = Relation(from_="test1", to="test2", relationType="knows")
    
    # First creation should succeed
    result1 = await jsonl_backend.create_relations([relation])
    assert len(result1) == 1
    
    # Second creation should return empty list (no new relations)
    result2 = await jsonl_backend.create_relations([relation])
    assert len(result2) == 0
    
    # Verify only one relation exists
    graph = await jsonl_backend.read_graph()
    assert len(graph.relations) == 1