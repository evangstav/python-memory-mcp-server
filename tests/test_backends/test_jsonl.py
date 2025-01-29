"""Tests for the JSONL backend implementation."""

import json
from pathlib import Path
from typing import AsyncGenerator

import pytest

from memory_mcp_server.backends.jsonl import JsonlBackend
from memory_mcp_server.exceptions import EntityNotFoundError
from memory_mcp_server.interfaces import Entity, Relation, SearchOptions


@pytest.fixture(scope="function")
async def jsonl_backend(tmp_path: Path) -> AsyncGenerator[JsonlBackend, None]:
    """Create a temporary JSONL backend for testing."""
    backend = JsonlBackend(tmp_path / "test_memory.jsonl")
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio(scope="function")
async def test_create_entities(jsonl_backend: JsonlBackend) -> None:
    """Test creating new entities."""


@pytest.mark.asyncio
async def test_fuzzy_search(jsonl_backend: JsonlBackend) -> None:
    """Test fuzzy search functionality."""
    # Create test entities
    entities = [
        Entity(
            name="John Smith",
            entityType="person",
            observations=["Software engineer at Tech Corp"],
        ),
        Entity(
            name="Jane Smith",
            entityType="person",
            observations=["Product manager at Tech Corp"],
        ),
        Entity(
            name="Tech Corporation",
            entityType="company",
            observations=["A technology company"],
        ),
    ]
    await jsonl_backend.create_entities(entities)

    # Test exact match (backward compatibility)
    result = await jsonl_backend.search_nodes("John")
    assert len(result.entities) == 1
    assert result.entities[0].name == "John Smith"

    # Test fuzzy match with high threshold
    options = SearchOptions(
        fuzzy=True,
        threshold=90,
        weights={"name": 1.0, "type": 0.5, "observations": 0.3},
    )
    result = await jsonl_backend.search_nodes("Jon Smith", options)
    assert len(result.entities) == 1
    assert result.entities[0].name == "John Smith"

    # Test fuzzy match with lower threshold
    options = SearchOptions(fuzzy=True, threshold=70)
    result = await jsonl_backend.search_nodes("Jane Smyth", options)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Jane Smith"

    # Test fuzzy match with observations
    options = SearchOptions(
        fuzzy=True,
        threshold=60,
        weights={"name": 0.3, "type": 0.3, "observations": 1.0},
    )
    result = await jsonl_backend.search_nodes("software dev", options)
    assert len(result.entities) == 1
    assert result.entities[0].name == "John Smith"

    # Test no matches with high threshold
    options = SearchOptions(fuzzy=True, threshold=95)
    result = await jsonl_backend.search_nodes("Bob Jones", options)
    assert len(result.entities) == 0


@pytest.mark.asyncio
async def test_fuzzy_search_weights(jsonl_backend: JsonlBackend) -> None:
    """Test fuzzy search with different weight configurations."""
    # Clear any existing entities
    await jsonl_backend.delete_entities(
        [e.name for e in (await jsonl_backend.read_graph()).entities]
    )

    entities = [
        Entity(
            name="Programming Guide",
            entityType="document",
            observations=["A guide about software development"],
        ),
        Entity(
            name="Software Manual",
            entityType="document",
            observations=["Programming tutorial and guide"],
        ),
    ]
    await jsonl_backend.create_entities(entities)

    # Test name-weighted search
    name_options = SearchOptions(
        fuzzy=True,
        threshold=60,
        weights={"name": 1.0, "type": 0.1, "observations": 0.1},
    )
    result = await jsonl_backend.search_nodes("programming", name_options)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Programming Guide"

    # Test observation-weighted search
    obs_options = SearchOptions(
        fuzzy=True,
        threshold=60,
        weights={"name": 0.1, "type": 0.1, "observations": 1.0},
    )
    result = await jsonl_backend.search_nodes("programming", obs_options)
    assert len(result.entities) == 2
    assert any(e.name == "Software Manual" for e in result.entities)

    # Clear again for create_entities test
    await jsonl_backend.delete_entities(
        [e.name for e in (await jsonl_backend.read_graph()).entities]
    )

    test_entities = [
        Entity("test1", "person", ["observation1", "observation2"]),
        Entity("test2", "location", ["observation3"]),
    ]
    result = await jsonl_backend.create_entities(test_entities)
    assert len(result) == 2

    # Verify entities were saved
    graph = await jsonl_backend.read_graph()
    assert len(graph.entities) == 2
    assert any(e.name == "test1" and e.entityType == "person" for e in graph.entities)
    assert any(e.name == "test2" and e.entityType == "location" for e in graph.entities)


@pytest.mark.asyncio(scope="function")
async def test_create_relations(jsonl_backend: JsonlBackend) -> None:
    """Test creating relations between entities."""
    # Create entities first
    entities = [
        Entity("person1", "person", ["observation1"]),
        Entity("location1", "location", ["observation2"]),
    ]
    await jsonl_backend.create_entities(entities)

    # Create relation
    relations = [Relation(from_="person1", to="location1", relationType="visited")]
    result = await jsonl_backend.create_relations(relations)
    assert len(result) == 1

    # Verify relation was saved
    graph = await jsonl_backend.read_graph()
    assert len(graph.relations) == 1
    assert graph.relations[0].from_ == "person1"
    assert graph.relations[0].to == "location1"
    assert graph.relations[0].relationType == "visited"


@pytest.mark.asyncio(scope="function")
async def test_create_relation_missing_entity(jsonl_backend: JsonlBackend) -> None:
    """Test creating relation with non-existent entity."""
    relations = [Relation(from_="nonexistent1", to="nonexistent2", relationType="test")]

    with pytest.raises(EntityNotFoundError):
        await jsonl_backend.create_relations(relations)


@pytest.mark.asyncio(scope="function")
async def test_search_nodes(jsonl_backend: JsonlBackend) -> None:
    """Test searching nodes in the graph."""
    # Create test data
    entities = [
        Entity("test1", "person", ["likes coffee", "works at office"]),
        Entity("test2", "person", ["likes tea"]),
        Entity("office", "location", ["big building"]),
    ]
    await jsonl_backend.create_entities(entities)

    relations = [Relation(from_="test1", to="office", relationType="works_at")]
    await jsonl_backend.create_relations(relations)

    # Test search
    result = await jsonl_backend.search_nodes("coffee")
    assert len(result.entities) == 1
    assert result.entities[0].name == "test1"

    result = await jsonl_backend.search_nodes("office")
    assert len(result.entities) == 2  # Both office entity and entity with office in obs
    assert "office" in {e.name for e in result.entities}
    assert "test1" in {e.name for e in result.entities}
    assert len(result.relations) == 1


@pytest.mark.asyncio(scope="function")
async def test_persistence(tmp_path: Path) -> None:
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


@pytest.mark.asyncio(scope="function")
async def test_file_format(tmp_path: Path) -> None:
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


@pytest.mark.asyncio(scope="function")
async def test_caching(jsonl_backend: JsonlBackend) -> None:
    """Test that caching works correctly."""
    entities = [Entity("test1", "person", ["obs1"])]
    await jsonl_backend.create_entities(entities)

    # First read should cache
    graph1 = await jsonl_backend.read_graph()

    # Second read should use cache
    graph2 = await jsonl_backend.read_graph()

    # Should be the same object if cached
    assert graph1 is graph2


@pytest.mark.asyncio(scope="function")
async def test_atomic_writes(tmp_path: Path) -> None:
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


@pytest.mark.asyncio(scope="function")
async def test_duplicate_entities(jsonl_backend: JsonlBackend) -> None:
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


@pytest.mark.asyncio(scope="function")
async def test_duplicate_relations(jsonl_backend: JsonlBackend) -> None:
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


@pytest.mark.asyncio(scope="function")
async def test_delete_entities(jsonl_backend: JsonlBackend) -> None:
    """Test deleting entities and related relations."""
    # Create test data
    entities = [
        Entity("test1", "person", ["obs1"]),
        Entity("test2", "location", ["obs2"]),
    ]
    await jsonl_backend.create_entities(entities)

    relations = [
        Relation(from_="test1", to="test2", relationType="visits"),
        Relation(from_="test2", to="test1", relationType="hosts"),
    ]
    await jsonl_backend.create_relations(relations)

    # Delete one entity
    deleted = await jsonl_backend.delete_entities(["test1"])
    assert deleted == ["test1"]

    # Verify entity removal and relation cleanup
    graph = await jsonl_backend.read_graph()
    assert len(graph.entities) == 1
    assert len(graph.relations) == 0
    assert "test2" in [e.name for e in graph.entities]

    # Verify cache consistency
    graph2 = await jsonl_backend.read_graph()
    assert graph is graph2  # Should return same cached object

    # Test deleting non-existent entity
    deleted = await jsonl_backend.delete_entities(["nonexistent"])
    assert deleted == []


@pytest.mark.asyncio(scope="function")
async def test_delete_relations(jsonl_backend: JsonlBackend) -> None:
    """Test deleting relations between entities."""
    # Create test data
    entities = [
        Entity("person1", "person", []),
        Entity("location1", "location", []),
    ]
    relations = [
        Relation(from_="person1", to="location1", relationType="visited"),
        Relation(from_="person1", to="location1", relationType="likes"),
    ]

    await jsonl_backend.create_entities(entities)
    await jsonl_backend.create_relations(relations)

    # Delete relations
    await jsonl_backend.delete_relations("person1", "location1")

    # Verify deletion
    graph = await jsonl_backend.read_graph()
    assert len(graph.relations) == 0


@pytest.mark.asyncio(scope="function")
async def test_delete_nonexistent_relations(jsonl_backend: JsonlBackend) -> None:
    """Test deleting relations that don't exist."""
    entities = [
        Entity("person1", "person", []),
        Entity("location1", "location", []),
    ]
    await jsonl_backend.create_entities(entities)

    # Try to delete non-existent relation
    await jsonl_backend.delete_relations("person1", "location1")

    # Verify no changes
    graph = await jsonl_backend.read_graph()
    assert len(graph.relations) == 0


@pytest.mark.asyncio(scope="function")
async def test_delete_bidirectional_relations(jsonl_backend: JsonlBackend) -> None:
    """Test deleting bidirectional relations."""
    entities = [
        Entity("A", "node", []),
        Entity("B", "node", []),
    ]
    relations = [
        Relation(from_="A", to="B", relationType="connects"),
        Relation(from_="B", to="A", relationType="connects"),
    ]

    await jsonl_backend.create_entities(entities)
    await jsonl_backend.create_relations(relations)

    # Delete relations in both directions
    await jsonl_backend.delete_relations("A", "B")
    await jsonl_backend.delete_relations("B", "A")

    graph = await jsonl_backend.read_graph()
    assert len(graph.relations) == 0


@pytest.mark.asyncio(scope="function")
async def test_delete_relations_missing_entity(jsonl_backend: JsonlBackend) -> None:
    """Test deleting relations with missing entities."""
    with pytest.raises(EntityNotFoundError):
        await jsonl_backend.delete_relations("ghost", "nonexistent")
