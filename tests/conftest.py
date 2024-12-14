import pytest
import logging
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager

# Configure logging
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_memory_file(tmp_path):
    """Create a temporary memory file."""
    logger.debug(f"Creating temp file in {tmp_path}")
    return tmp_path / "memory.jsonl"


@pytest.fixture
async def knowledge_graph_manager(temp_memory_file):
    """Create a KnowledgeGraphManager instance with a temporary memory file."""
    logger.debug("Creating KnowledgeGraphManager")
    manager = KnowledgeGraphManager(memory_path=temp_memory_file, cache_ttl=1)
    logger.debug("KnowledgeGraphManager created")
    yield manager
    logger.debug("Cleaning up KnowledgeGraphManager")
    await manager.flush()
    logger.debug("Cleanup complete")

