import json
from pathlib import Path

import pytest

from memory_mcp_server.backends.jsonl import JsonlBackend
from memory_mcp_server.exceptions import EntityNotFoundError, FileAccessError
from memory_mcp_server.interfaces import (
    BatchOperation,
    BatchOperationType,
    BatchResult,
    Entity,
    Relation,
    SearchOptions,
)

# --- Fixtures ---


@pytest.fixture
async def backend(tmp_path: Path) -> JsonlBackend:
    b = JsonlBackend(tmp_path / "test.jsonl")
    await b.initialize()
    yield b
    await b.close()


# --- Entity Creation / Duplication ---


@pytest.mark.asyncio
async def test_create_entities(backend: JsonlBackend):
    entities = [
        Entity(name="Alice", entityType="person", observations=["likes apples"]),
        Entity(name="Bob", entityType="person", observations=["enjoys biking"]),
    ]
    created = await backend.create_entities(entities)
    assert len(created) == 2, "Should create two new entities"

    graph = await backend.read_graph()
    assert len(graph.entities) == 2, "Graph should contain two entities"


@pytest.mark.asyncio
async def test_duplicate_entities(backend: JsonlBackend):
    entity = Entity(name="Alice", entityType="person", observations=["likes apples"])
    created1 = await backend.create_entities([entity])
    created2 = await backend.create_entities([entity])
    assert len(created1) == 1
    assert len(created2) == 0, "Duplicate entity creation should return empty list"


# --- Relation Creation / Deletion ---


@pytest.mark.asyncio
async def test_create_relations(backend: JsonlBackend):
    entities = [
        Entity(name="Alice", entityType="person", observations=[""]),
        Entity(name="Wonderland", entityType="place", observations=["fantasy land"]),
    ]
    await backend.create_entities(entities)
    relation = Relation(from_="Alice", to="Wonderland", relationType="visits")
    created_relations = await backend.create_relations([relation])
    assert len(created_relations) == 1

    graph = await backend.read_graph()
    assert len(graph.relations) == 1


@pytest.mark.asyncio
async def test_create_relation_missing_entity(backend: JsonlBackend):
    # No entities have been created.
    relation = Relation(from_="Alice", to="Nowhere", relationType="visits")
    with pytest.raises(EntityNotFoundError):
        await backend.create_relations([relation])


@pytest.mark.asyncio
async def test_delete_relations(backend: JsonlBackend):
    entities = [
        Entity(name="Alice", entityType="person", observations=[]),
        Entity(name="Bob", entityType="person", observations=[]),
    ]
    await backend.create_entities(entities)
    # Create two distinct relations.
    relation1 = Relation(from_="Alice", to="Bob", relationType="likes")
    relation2 = Relation(from_="Alice", to="Bob", relationType="follows")
    await backend.create_relations([relation1, relation2])
    await backend.delete_relations("Alice", "Bob")
    graph = await backend.read_graph()
    assert (
        len(graph.relations) == 0
    ), "All relations between Alice and Bob should be removed"


@pytest.mark.asyncio
async def test_delete_entities(backend: JsonlBackend):
    entities = [
        Entity(name="Alice", entityType="person", observations=["obs1"]),
        Entity(name="Bob", entityType="person", observations=["obs2"]),
    ]
    await backend.create_entities(entities)
    # Create a relation so that deletion cascades.
    relation = Relation(from_="Alice", to="Bob", relationType="knows")
    await backend.create_relations([relation])
    deleted = await backend.delete_entities(["Alice"])
    assert "Alice" in deleted

    graph = await backend.read_graph()
    # Only Bob should remain and the relation should have been removed.
    assert len(graph.entities) == 1
    assert graph.entities[0].name == "Bob"
    assert len(graph.relations) == 0


# --- Searching ---


@pytest.mark.asyncio
async def test_search_nodes_exact(backend: JsonlBackend):
    entities = [
        Entity(
            name="Alice Wonderland", entityType="person", observations=["loves tea"]
        ),
        Entity(name="Wonderland", entityType="place", observations=["magical"]),
    ]
    await backend.create_entities(entities)
    result = await backend.search_nodes("Wonderland")
    # Both entities should match the substring.
    assert len(result.entities) == 2
    # No relations were created.
    assert len(result.relations) == 0


@pytest.mark.asyncio
async def test_search_nodes_fuzzy(backend: JsonlBackend):
    entities = [
        Entity(
            name="John Smith", entityType="person", observations=["software engineer"]
        ),
        Entity(
            name="Jane Smith", entityType="person", observations=["product manager"]
        ),
    ]
    await backend.create_entities(entities)
    options = SearchOptions(
        fuzzy=True,
        threshold=90,
        weights={"name": 0.7, "type": 0.5, "observations": 0.3},
    )
    result = await backend.search_nodes("Jon Smith", options)
    assert len(result.entities) == 1, "Fuzzy search should match John Smith"
    assert result.entities[0].name == "John Smith"


@pytest.mark.asyncio
async def test_search_nodes_fuzzy_weights(backend: JsonlBackend):
    # Clear any existing entities.
    current = await backend.read_graph()
    if current.entities:
        await backend.delete_entities([e.name for e in current.entities])
    entities = [
        Entity(
            name="Programming Guide",
            entityType="document",
            observations=["A guide about programming development"],
        ),
        Entity(
            name="Software Manual",
            entityType="document",
            observations=["Programming tutorial and guide"],
        ),
    ]
    await backend.create_entities(entities)
    # With name-weight high, only one should match.
    options_name = SearchOptions(
        fuzzy=True,
        threshold=60,
        weights={"name": 1.0, "type": 0.1, "observations": 0.1},
    )
    result = await backend.search_nodes("programming", options_name)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Programming Guide"

    # With observation weight high, both should match.
    options_obs = SearchOptions(
        fuzzy=True,
        threshold=60,
        weights={"name": 0.1, "type": 0.1, "observations": 1.0},
    )
    result = await backend.search_nodes("programming", options_obs)
    assert len(result.entities) == 2


# --- Observations ---


@pytest.mark.asyncio
async def test_add_observations(backend: JsonlBackend):
    entity = Entity(name="Alice", entityType="person", observations=["initial"])
    await backend.create_entities([entity])
    await backend.add_observations("Alice", ["update"])
    graph = await backend.read_graph()
    alice = next(e for e in graph.entities if e.name == "Alice")
    assert "update" in alice.observations


@pytest.mark.asyncio
async def test_add_batch_observations(backend: JsonlBackend):
    entities = [
        Entity(name="Alice", entityType="person", observations=["obs1"]),
        Entity(name="Bob", entityType="person", observations=["obs2"]),
    ]
    await backend.create_entities(entities)
    observations_map = {"Alice": ["new1", "new2"], "Bob": ["new3"]}
    await backend.add_batch_observations(observations_map)
    graph = await backend.read_graph()
    alice = next(e for e in graph.entities if e.name == "Alice")
    bob = next(e for e in graph.entities if e.name == "Bob")
    assert set(alice.observations) == {"obs1", "new1", "new2"}
    assert set(bob.observations) == {"obs2", "new3"}


@pytest.mark.asyncio
async def test_add_batch_observations_empty_map(backend: JsonlBackend):
    with pytest.raises(ValueError, match="Observations map cannot be empty"):
        await backend.add_batch_observations({})


@pytest.mark.asyncio
async def test_add_batch_observations_missing_entity(backend: JsonlBackend):
    entity = Entity(name="Alice", entityType="person", observations=["obs1"])
    await backend.create_entities([entity])
    observations_map = {"Alice": ["new"], "Bob": ["obs"]}
    with pytest.raises(EntityNotFoundError):
        await backend.add_batch_observations(observations_map)


# --- Transaction Management ---


@pytest.mark.asyncio
async def test_transaction_management(backend: JsonlBackend):
    entities = [
        Entity(name="Alice", entityType="person", observations=["obs1"]),
        Entity(name="Bob", entityType="person", observations=["obs2"]),
    ]
    await backend.create_entities(entities)
    # Begin a transaction.
    await backend.begin_transaction()
    await backend.create_entities(
        [Entity(name="Charlie", entityType="person", observations=["obs3"])]
    )
    await backend.delete_entities(["Alice"])
    # Within transaction, changes are visible.
    graph = await backend.read_graph()
    names = {e.name for e in graph.entities}
    assert "Charlie" in names
    assert "Alice" not in names
    # Roll back.
    await backend.rollback_transaction()
    graph = await backend.read_graph()
    names = {e.name for e in graph.entities}
    assert "Alice" in names
    assert "Charlie" not in names

    # Test commit.
    await backend.begin_transaction()
    await backend.create_entities(
        [Entity(name="Dave", entityType="person", observations=["obs4"])]
    )
    await backend.commit_transaction()
    graph = await backend.read_graph()
    names = {e.name for e in graph.entities}
    assert "Dave" in names


# --- Persistence and File Format ---


@pytest.mark.asyncio
async def test_persistence(tmp_path: Path):
    file_path = tmp_path / "persist.jsonl"
    backend1 = JsonlBackend(file_path)
    await backend1.initialize()
    entity = Entity(name="Alice", entityType="person", observations=["obs"])
    await backend1.create_entities([entity])
    await backend1.close()

    backend2 = JsonlBackend(file_path)
    await backend2.initialize()
    graph = await backend2.read_graph()
    assert any(e.name == "Alice" for e in graph.entities)
    await backend2.close()


@pytest.mark.asyncio
async def test_atomic_writes(tmp_path: Path):
    file_path = tmp_path / "atomic.jsonl"
    backend = JsonlBackend(file_path)
    await backend.initialize()
    entity = Entity(name="Alice", entityType="person", observations=["obs"])
    await backend.create_entities([entity])
    await backend.close()
    temp_file = file_path.with_suffix(".tmp")
    assert not temp_file.exists(), "Temporary file should be removed after writing"
    assert file_path.exists()


@pytest.mark.asyncio
async def test_file_format(tmp_path: Path):
    file_path = tmp_path / "format.jsonl"
    backend = JsonlBackend(file_path)
    await backend.initialize()
    entity = Entity(name="Alice", entityType="person", observations=["obs"])
    relation = Relation(from_="Alice", to="Alice", relationType="self")
    await backend.create_entities([entity])
    await backend.create_relations([relation])
    await backend.close()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 2, "File should contain exactly two JSON lines"
    data1 = json.loads(lines[0])
    data2 = json.loads(lines[1])
    types = {data1.get("type"), data2.get("type")}
    assert "entity" in types and "relation" in types


# --- Error / Corruption Handling ---


@pytest.mark.asyncio
async def test_corrupted_file_handling(tmp_path: Path):
    file_path = tmp_path / "corrupted.jsonl"
    # Write one valid and one corrupted JSON line.
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(
            '{"type": "entity", "name": "Alice", "entityType": "person", "observations": []}\n'
        )
        f.write(
            '{"type": "relation", "from": "Alice", "to": "Bob"'
        )  # missing closing brace
    backend = JsonlBackend(file_path)
    await backend.initialize()
    with pytest.raises(FileAccessError, match="Error loading graph"):
        await backend.read_graph()
    await backend.close()


@pytest.mark.asyncio
async def test_file_access_error_propagation(tmp_path: Path):
    file_path = tmp_path / "error.jsonl"
    # Create a directory with the same name as the file.
    file_path.mkdir()
    backend = JsonlBackend(file_path)
    with pytest.raises(FileAccessError, match="is a directory"):
        await backend.initialize()
    await backend.close()


# --- Caching ---


@pytest.mark.asyncio
async def test_caching(backend: JsonlBackend):
    entity = Entity(name="Alice", entityType="person", observations=["obs"])
    await backend.create_entities([entity])
    graph1 = await backend.read_graph()
    graph2 = await backend.read_graph()
    assert graph1 is graph2, "Repeated reads should return the cached graph"


# --- Batch Operations ---


@pytest.mark.asyncio
async def test_execute_batch(backend: JsonlBackend):
    # Create an initial entity.
    await backend.create_entities(
        [Entity(name="Alice", entityType="person", observations=["obs"])]
    )
    operations = [
        BatchOperation(
            operation_type=BatchOperationType.CREATE_ENTITIES,
            data={
                "entities": [
                    Entity(name="Bob", entityType="person", observations=["obs2"])
                ]
            },
        ),
        BatchOperation(
            operation_type=BatchOperationType.CREATE_RELATIONS,
            data={
                "relations": [Relation(from_="Alice", to="Bob", relationType="knows")]
            },
        ),
        BatchOperation(
            operation_type=BatchOperationType.ADD_OBSERVATIONS,
            data={"observations_map": {"Alice": ["new_obs"]}},
        ),
    ]
    result: BatchResult = await backend.execute_batch(operations)
    print(result)
    assert result.success, "Batch operations should succeed"
    graph = await backend.read_graph()
    assert any(e.name == "Bob" for e in graph.entities)
    assert len(graph.relations) == 1
    alice = next(e for e in graph.entities if e.name == "Alice")
    assert "new_obs" in alice.observations


@pytest.mark.asyncio
async def test_execute_batch_failure(backend: JsonlBackend):
    # Create an initial entity.
    await backend.create_entities(
        [Entity(name="Alice", entityType="person", observations=["obs"])]
    )
    operations = [
        BatchOperation(
            operation_type=BatchOperationType.CREATE_RELATIONS,
            data={
                "relations": [
                    Relation(from_="Alice", to="NonExistent", relationType="knows")
                ]
            },
        ),
    ]
    result: BatchResult = await backend.execute_batch(operations)
    assert (
        not result.success
    ), "Batch operation should fail if a relation refers to a non-existent entity"
    # Verify that rollback occurred (no partial changes).
    graph = await backend.read_graph()
    assert len(graph.entities) == 1
    assert len(graph.relations) == 0
