"""Tests for the JSONL to Neo4j migration tool."""

import pytest
from pathlib import Path

from memory_mcp_server.tools.migrate_to_neo4j import migrate_data
from memory_mcp_server.backends.jsonl import JsonlBackend
from memory_mcp_server.backends.neo4j import Neo4jBackend
from memory_mcp_server.interfaces import Entity, Relation


@pytest.fixture
async def populated_jsonl_file(tmp_path, sample_entities, sample_relations):
    """Create a JSONL file with sample data."""
    file_path = tmp_path / "source.jsonl"
    backend = JsonlBackend(file_path)
    await backend.initialize()
    await backend.create_entities(sample_entities)
    await backend.create_relations(sample_relations)
    await backend.close()
    return file_path


@pytest.mark.asyncio
async def test_migration(populated_jsonl_file, neo4j_container):
    """Test migrating data from JSONL to Neo4j."""
    # Run migration
    await migrate_data(
        jsonl_path=populated_jsonl_file,
        neo4j_uri=neo4j_container.uri,
        neo4j_user=neo4j_container.user,
        neo4j_password=neo4j_container.password,
        batch_size=2  # Small batch size to test batching
    )

    # Verify data in Neo4j
    neo4j_backend = Neo4jBackend(
        neo4j_container.uri,
        neo4j_container.user,
        neo4j_container.password
    )
    await neo4j_backend.initialize()

    try:
        # Read both graphs
        jsonl_backend = JsonlBackend(populated_jsonl_file)
        await jsonl_backend.initialize()
        
        source_graph = await jsonl_backend.read_graph()
        target_graph = await neo4j_backend.read_graph()

        # Compare entities
        assert len(source_graph.entities) == len(target_graph.entities)
        source_entities = {(e.name, e.entityType, tuple(e.observations)) 
                         for e in source_graph.entities}
        target_entities = {(e.name, e.entityType, tuple(e.observations)) 
                         for e in target_graph.entities}
        assert source_entities == target_entities

        # Compare relations
        assert len(source_graph.relations) == len(target_graph.relations)
        source_relations = {(r.from_, r.to, r.relationType) 
                          for r in source_graph.relations}
        target_relations = {(r.from_, r.to, r.relationType) 
                          for r in target_graph.relations}
        assert source_relations == target_relations

    finally:
        await jsonl_backend.close()
        await neo4j_backend.close()


@pytest.mark.asyncio
async def test_migration_empty_source(tmp_path, neo4j_container):
    """Test migrating from an empty JSONL file."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()

    await migrate_data(
        jsonl_path=empty_file,
        neo4j_uri=neo4j_container.uri,
        neo4j_user=neo4j_container.user,
        neo4j_password=neo4j_container.password
    )

    # Verify Neo4j is empty
    neo4j_backend = Neo4jBackend(
        neo4j_container.uri,
        neo4j_container.user,
        neo4j_container.password
    )
    await neo4j_backend.initialize()

    try:
        graph = await neo4j_backend.read_graph()
        assert len(graph.entities) == 0
        assert len(graph.relations) == 0
    finally:
        await neo4j_backend.close()


@pytest.mark.asyncio
async def test_migration_large_dataset(tmp_path, neo4j_container):
    """Test migrating a larger dataset to verify batching."""
    # Create a larger dataset
    file_path = tmp_path / "large.jsonl"
    backend = JsonlBackend(file_path)
    await backend.initialize()

    # Create 100 entities and relations
    entities = []
    relations = []
    for i in range(100):
        entities.append(Entity(
            name=f"entity{i}",
            entityType="test",
            observations=[f"observation{i}"]
        ))
        if i > 0:  # Create relations between consecutive entities
            relations.append(Relation(
                from_=f"entity{i-1}",
                to=f"entity{i}",
                relationType="next"
            ))

    await backend.create_entities(entities)
    await backend.create_relations(relations)
    await backend.close()

    # Run migration with small batch size
    await migrate_data(
        jsonl_path=file_path,
        neo4j_uri=neo4j_container.uri,
        neo4j_user=neo4j_container.user,
        neo4j_password=neo4j_container.password,
        batch_size=10  # Small batch size to test batching
    )

    # Verify all data was migrated
    neo4j_backend = Neo4jBackend(
        neo4j_container.uri,
        neo4j_container.user,
        neo4j_container.password
    )
    await neo4j_backend.initialize()

    try:
        graph = await neo4j_backend.read_graph()
        assert len(graph.entities) == 100
        assert len(graph.relations) == 99
    finally:
        await neo4j_backend.close()


@pytest.mark.asyncio
async def test_migration_with_search(populated_jsonl_file, neo4j_container):
    """Test that search functionality works correctly after migration."""
    # Run migration
    await migrate_data(
        jsonl_path=populated_jsonl_file,
        neo4j_uri=neo4j_container.uri,
        neo4j_user=neo4j_container.user,
        neo4j_password=neo4j_container.password
    )

    # Test search in both backends
    jsonl_backend = JsonlBackend(populated_jsonl_file)
    neo4j_backend = Neo4jBackend(
        neo4j_container.uri,
        neo4j_container.user,
        neo4j_container.password
    )

    await jsonl_backend.initialize()
    await neo4j_backend.initialize()

    try:
        # Search for a common term
        jsonl_result = await jsonl_backend.search_nodes("observation")
        neo4j_result = await neo4j_backend.search_nodes("observation")

        # Compare search results
        assert len(jsonl_result.entities) == len(neo4j_result.entities)
        assert len(jsonl_result.relations) == len(neo4j_result.relations)

    finally:
        await jsonl_backend.close()
        await neo4j_backend.close()