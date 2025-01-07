"""Common test fixtures for all tests."""

import logging
from pathlib import Path
from typing import AsyncGenerator, List

import pytest

from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager

# Configure logging
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def temp_memory_file(tmp_path: Path) -> Path:
    """Create a temporary memory file."""
    logger.debug(f"Creating temp file in {tmp_path}")
    return tmp_path / "memory.jsonl"


@pytest.fixture(scope="function")
def sample_entities() -> List[Entity]:
    """Provide sample entities for testing."""
    return [
        Entity("person1", "person", ["likes reading", "works in tech"]),
        Entity("company1", "company", ["tech company", "founded 2020"]),
        Entity("location1", "place", ["office building", "in city center"]),
    ]


@pytest.fixture(scope="function")
def sample_relations() -> List[Relation]:
    """Provide sample relations for testing."""
    return [
        Relation(from_="person1", to="company1", relationType="works_at"),
        Relation(from_="company1", to="location1", relationType="located_at"),
    ]


@pytest.fixture(scope="function")
async def knowledge_graph_manager(
    temp_memory_file: Path,
) -> AsyncGenerator[KnowledgeGraphManager, None]:
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
