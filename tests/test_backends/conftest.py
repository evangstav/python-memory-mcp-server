"""Common test fixtures for backend tests."""

import pytest
from pathlib import Path
from typing import AsyncGenerator

from memory_mcp_server.interfaces import Entity, Relation


@pytest.fixture
def sample_entities() -> list[Entity]:
    """Provide a list of sample entities for testing."""
    return [
        Entity("test1", "person", ["observation1", "observation2"]),
        Entity("test2", "location", ["observation3"]),
        Entity("test3", "organization", ["observation4", "observation5"]),
    ]


@pytest.fixture
def sample_relations(sample_entities) -> list[Relation]:
    """Provide a list of sample relations for testing."""
    return [
        Relation(from_="test1", to="test2", relationType="visited"),
        Relation(from_="test1", to="test3", relationType="works_at"),
        Relation(from_="test2", to="test3", relationType="located_in"),
    ]


@pytest.fixture
async def populated_jsonl_backend(jsonl_backend, sample_entities, sample_relations):
    """Provide a JSONL backend pre-populated with sample data."""
    await jsonl_backend.create_entities(sample_entities)
    await jsonl_backend.create_relations(sample_relations)
    return jsonl_backend


@pytest.fixture
async def populated_neo4j_backend(neo4j_backend, sample_entities, sample_relations):
    """Provide a Neo4j backend pre-populated with sample data."""
    await neo4j_backend.create_entities(sample_entities)
    await neo4j_backend.create_relations(sample_relations)
    return neo4j_backend


@pytest.fixture
def temp_jsonl_path(tmp_path) -> Path:
    """Provide a temporary path for JSONL files."""
    return tmp_path / "test_memory.jsonl"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Path to the docker-compose file for Neo4j testing."""
    return str(Path(__file__).parent / "docker-compose.yml")


@pytest.fixture(scope="session")
def docker_compose_project_name():
    """Project name for docker-compose to avoid conflicts."""
    return "memory_mcp_test"