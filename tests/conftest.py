import pytest
from pathlib import Path
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager

@pytest.fixture
def temp_memory_file(tmp_path):
    """Create a temporary memory file."""
    return tmp_path / "memory.jsonl"

@pytest.fixture
def knowledge_graph_manager(temp_memory_file):
    """Create a KnowledgeGraphManager instance with a temporary memory file."""
    return KnowledgeGraphManager(memory_path=temp_memory_file, cache_ttl=1)