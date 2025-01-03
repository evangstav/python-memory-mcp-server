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


def is_neo4j_responsive(host, port):
    """Check if Neo4j is responsive and database is ready."""
    import neo4j
    import time
    try:
        driver = neo4j.GraphDatabase.driver(
            f"neo4j://{host}:{port}",
            auth=("neo4j", "testpassword")
        )
        # Try multiple times with exponential backoff
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                with driver.session() as session:
                    # Check if we can write to the database
                    result = session.run(
                        "CREATE (n:TestNode {name: 'test'}) "
                        "DELETE n "
                        "RETURN true as success"
                    )
                    if result.single()[0]:
                        driver.close()
                        return True
            except Exception:
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        driver.close()
        return False
    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_service(docker_ip, docker_services):
    """Ensure that Neo4j service is up and responsive."""
    port = docker_services.port_for("neo4j", 7687)
    docker_services.wait_until_responsive(
        timeout=60.0,
        pause=0.1,
        check=lambda: is_neo4j_responsive(docker_ip, port)
    )
    return port


@pytest.fixture
async def neo4j_backend(neo4j_service, docker_ip):
    """Provide a Neo4j backend connected to test container."""
    from memory_mcp_server.backends.neo4j import Neo4jBackend
    
    backend = Neo4jBackend(
        uri=f"neo4j://{docker_ip}:{neo4j_service}",
        username="neo4j",
        password="testpassword"
    )
    await backend.initialize()

    # Clear any existing data
    await backend._driver.execute_query("MATCH (n) DETACH DELETE n")

    yield backend

    # Cleanup
    await backend._driver.execute_query("MATCH (n) DETACH DELETE n")
    await backend.close()


@pytest.fixture
async def neo4j_backend(docker_services):
    """Provide a Neo4j backend connected to test container."""
    # Wait for Neo4j to be ready
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: docker_services.port_for("neo4j", 7687) is not None,
    )

    # Create and initialize backend
    from memory_mcp_server.backends.neo4j import Neo4jBackend
    backend = Neo4jBackend(
        uri=f"neo4j://localhost:{docker_services.port_for('neo4j', 7687)}",
        username="neo4j",
        password="testpassword"
    )
    await backend.initialize()

    # Clear any existing data
    await backend._driver.execute_query("MATCH (n) DETACH DELETE n")

    yield backend

    # Cleanup
    await backend.close()