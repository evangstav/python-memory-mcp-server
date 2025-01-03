import pytest
import logging
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager
from memory_mcp_server.interfaces import Entity, Relation

# Configure logging
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_memory_file(tmp_path):
    """Create a temporary memory file."""
    logger.debug(f"Creating temp file in {tmp_path}")
    return tmp_path / "memory.jsonl"


@pytest.fixture
def sample_entities():
    """Provide sample entities for testing."""
    return [
        Entity("person1", "person", ["likes reading", "works in tech"]),
        Entity("company1", "company", ["tech company", "founded 2020"]),
        Entity("location1", "place", ["office building", "in city center"])
    ]


@pytest.fixture
def sample_relations():
    """Provide sample relations for testing."""
    return [
        Relation(from_="person1", to="company1", relationType="works_at"),
        Relation(from_="company1", to="location1", relationType="located_at")
    ]


@pytest.fixture
async def knowledge_graph_manager(temp_memory_file):
    """Create a KnowledgeGraphManager instance with a temporary memory file."""
    logger.debug("Creating KnowledgeGraphManager")
    manager = KnowledgeGraphManager(backend=temp_memory_file, cache_ttl=1)
    logger.debug("KnowledgeGraphManager created")
    await manager.initialize()
    yield manager
    logger.debug("Cleaning up KnowledgeGraphManager")
    await manager.flush()
    await manager.close()
    logger.debug("Cleanup complete")


@pytest.fixture
def sample_entities():
    """Provide sample entities for testing."""
    return [
        Entity("person1", "person", ["likes reading", "works in tech"]),
        Entity("company1", "company", ["tech company", "founded 2020"]),
        Entity("location1", "place", ["office building", "in city center"])
    ]


@pytest.fixture
def sample_relations():
    """Provide sample relations for testing."""
    return [
        Relation(from_="person1", to="company1", relationType="works_at"),
        Relation(from_="company1", to="location1", relationType="located_at")
    ]

