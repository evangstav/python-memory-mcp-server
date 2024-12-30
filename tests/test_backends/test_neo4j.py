"""Tests for the Neo4j backend implementation."""

import pytest
import asyncio
from typing import AsyncGenerator
import docker
import time
import aiohttp

from memory_mcp_server.backends.neo4j import Neo4jBackend
from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.exceptions import EntityNotFoundError


class Neo4jContainer:
    """Manages a Neo4j test container."""

    def __init__(self):
        self.client = docker.from_env()
        self.container = None
        self.uri = None
        self.user = "neo4j"
        self.password = "testpassword"

    async def start(self) -> None:
        """Start Neo4j container and wait for it to be ready."""
        self.container = self.client.containers.run(
            "neo4j:5.14",
            environment={
                "NEO4J_AUTH": f"{self.user}/{self.password}",
                "NEO4J_PLUGINS": '["graph-data-science", "apoc"]'
            },
            ports={
                "7687/tcp": None,  # Let Docker assign a random port
            },
            detach=True,
        )
        
        # Get the assigned port
        container_info = self.client.api.inspect_container(self.container.id)
        bolt_port = container_info["NetworkSettings"]["Ports"]["7687/tcp"][0]["HostPort"]
        self.uri = f"neo4j://localhost:{bolt_port}"

        # Wait for Neo4j to be ready
        timeout = time.time() + 60
        async with aiohttp.ClientSession() as session:
            while time.time() < timeout:
                try:
                    # Try to connect to Neo4j
                    backend = Neo4jBackend(self.uri, self.user, self.password)
                    await backend.initialize()
                    await backend.close()
                    return
                except Exception:
                    await asyncio.sleep(1)
            raise TimeoutError("Neo4j container failed to start")

    async def stop(self) -> None:
        """Stop and remove the Neo4j container."""
        if self.container:
            self.container.stop()
            self.container.remove(force=True)


@pytest.fixture
async def neo4j_container() -> AsyncGenerator[Neo4jContainer, None]:
    """Provide a Neo4j container for testing."""
    container = Neo4jContainer()
    await container.start()
    yield container
    await container.stop()


@pytest.fixture
async def neo4j_backend(neo4j_container) -> AsyncGenerator[Neo4jBackend, None]:
    """Provide a configured Neo4j backend for testing."""
    backend = Neo4jBackend(
        neo4j_container.uri,
        neo4j_container.user,
        neo4j_container.password
    )
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_create_entities(neo4j_backend):
    """Test creating new entities."""
    entities = [
        Entity("test1", "person", ["observation1", "observation2"]),
        Entity("test2", "location", ["observation3"]),
    ]
    
    result = await neo4j_backend.create_entities(entities)
    assert len(result) == 2
    
    # Verify entities were saved
    graph = await neo4j_backend.read_graph()
    assert len(graph.entities) == 2
    assert any(e.name == "test1" and e.entityType == "person" for e in graph.entities)
    assert any(e.name == "test2" and e.entityType == "location" for e in graph.entities)


@pytest.mark.asyncio
async def test_create_relations(neo4j_backend):
    """Test creating relations between entities."""
    # Create entities first
    entities = [
        Entity("person1", "person", ["observation1"]),
        Entity("location1", "location", ["observation2"]),
    ]
    await neo4j_backend.create_entities(entities)
    
    # Create relation
    relations = [
        Relation(from_="person1", to="location1", relationType="visited")
    ]
    result = await neo4j_backend.create_relations(relations)
    assert len(result) == 1
    
    # Verify relation was saved
    graph = await neo4j_backend.read_graph()
    assert len(graph.relations) == 1
    assert graph.relations[0].from_ == "person1"
    assert graph.relations[0].to == "location1"
    assert graph.relations[0].relationType == "visited"


@pytest.mark.asyncio
async def test_create_relation_missing_entity(neo4j_backend):
    """Test creating relation with non-existent entity."""
    relations = [
        Relation(from_="nonexistent1", to="nonexistent2", relationType="test")
    ]
    
    with pytest.raises(EntityNotFoundError):
        await neo4j_backend.create_relations(relations)


@pytest.mark.asyncio
async def test_search_nodes(neo4j_backend):
    """Test searching nodes in the graph."""
    # Create test data
    entities = [
        Entity("test1", "person", ["likes coffee", "works at office"]),
        Entity("test2", "person", ["likes tea"]),
        Entity("office", "location", ["big building"]),
    ]
    await neo4j_backend.create_entities(entities)
    
    relations = [
        Relation(from_="test1", to="office", relationType="works_at")
    ]
    await neo4j_backend.create_relations(relations)
    
    # Test search
    result = await neo4j_backend.search_nodes("coffee")
    assert len(result.entities) == 1
    assert result.entities[0].name == "test1"
    
    result = await neo4j_backend.search_nodes("office")
    assert len(result.entities) == 1
    assert len(result.relations) == 1


@pytest.mark.asyncio
async def test_persistence(neo4j_container):
    """Test that data persists between backend instances."""
    # Create first instance and add data
    backend1 = Neo4jBackend(
        neo4j_container.uri,
        neo4j_container.user,
        neo4j_container.password
    )
    await backend1.initialize()
    
    entities = [Entity("test1", "person", ["observation1"])]
    await backend1.create_entities(entities)
    await backend1.close()
    
    # Create second instance and verify data
    backend2 = Neo4jBackend(
        neo4j_container.uri,
        neo4j_container.user,
        neo4j_container.password
    )
    await backend2.initialize()
    
    graph = await backend2.read_graph()
    assert len(graph.entities) == 1
    assert graph.entities[0].name == "test1"
    await backend2.close()


@pytest.mark.asyncio
async def test_duplicate_entities(neo4j_backend):
    """Test handling of duplicate entities."""
    entity = Entity("test1", "person", ["obs1"])
    
    # First creation should succeed
    result1 = await neo4j_backend.create_entities([entity])
    assert len(result1) == 1
    
    # Second creation should return empty list (no new entities)
    result2 = await neo4j_backend.create_entities([entity])
    assert len(result2) == 0
    
    # Verify only one entity exists
    graph = await neo4j_backend.read_graph()
    assert len(graph.entities) == 1


@pytest.mark.asyncio
async def test_duplicate_relations(neo4j_backend):
    """Test handling of duplicate relations."""
    # Create test entities
    entities = [
        Entity("test1", "person", ["obs1"]),
        Entity("test2", "person", ["obs2"]),
    ]
    await neo4j_backend.create_entities(entities)
    
    relation = Relation(from_="test1", to="test2", relationType="knows")
    
    # First creation should succeed
    result1 = await neo4j_backend.create_relations([relation])
    assert len(result1) == 1
    
    # Second creation should return empty list (no new relations)
    result2 = await neo4j_backend.create_relations([relation])
    assert len(result2) == 0
    
    # Verify only one relation exists
    graph = await neo4j_backend.read_graph()
    assert len(graph.relations) == 1


@pytest.mark.asyncio
async def test_constraint_creation(neo4j_backend):
    """Test that unique constraints are properly created."""
    # Create an entity
    entity = Entity("test1", "person", ["obs1"])
    await neo4j_backend.create_entities([entity])
    
    # Attempt to create entity with same name but different type
    duplicate = Entity("test1", "location", ["obs2"])
    result = await neo4j_backend.create_entities([duplicate])
    assert len(result) == 0  # Should not create duplicate


@pytest.mark.asyncio
async def test_complex_search(neo4j_backend):
    """Test more complex search scenarios."""
    # Create a more complex graph
    entities = [
        Entity("John", "person", ["software engineer", "likes coffee"]),
        Entity("Jane", "person", ["product manager", "likes tea"]),
        Entity("TechCorp", "company", ["software company"]),
        Entity("CoffeeShop", "location", ["serves coffee"]),
    ]
    await neo4j_backend.create_entities(entities)
    
    relations = [
        Relation(from_="John", to="TechCorp", relationType="works_at"),
        Relation(from_="Jane", to="TechCorp", relationType="works_at"),
        Relation(from_="John", to="CoffeeShop", relationType="visits"),
    ]
    await neo4j_backend.create_relations(relations)
    
    # Search for coffee-related nodes
    result = await neo4j_backend.search_nodes("coffee")
    assert len(result.entities) == 2  # John and CoffeeShop
    assert len(result.relations) == 1  # John visits CoffeeShop
    
    # Search for work-related nodes
    result = await neo4j_backend.search_nodes("work")
    assert len(result.entities) >= 1  # TechCorp
    assert len(result.relations) == 2  # Both work relations