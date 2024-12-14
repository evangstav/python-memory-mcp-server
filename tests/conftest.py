import pytest
import os
import tempfile
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager
from memory_mcp_server.optimized_knowledge_graph_manager import OptimizedKnowledgeGraphManager

@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    os.unlink(path)

@pytest.fixture
def knowledge_graph_manager(temp_db_path):
    """Create a KnowledgeGraphManager instance with a temporary database."""
    manager = KnowledgeGraphManager(db_path=temp_db_path)
    return manager

@pytest.fixture
def optimized_knowledge_graph_manager(temp_db_path):
    """Create an OptimizedKnowledgeGraphManager instance with a temporary database."""
    manager = OptimizedKnowledgeGraphManager(db_path=temp_db_path)
    return manager