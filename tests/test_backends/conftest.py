"""Common test fixtures for backend tests."""

from pathlib import Path
from typing import AsyncGenerator, List

import pytest

from memory_mcp_server.backends.jsonl import JsonlBackend
from memory_mcp_server.interfaces import Entity, Relation


@pytest.fixture(scope="function")
def sample_entities() -> List[Entity]:
    """Provide a list of sample entities for testing."""
    return [
        Entity("test1", "person", ["observation1", "observation2"]),
        Entity("test2", "location", ["observation3"]),
        Entity("test3", "organization", ["observation4", "observation5"]),
    ]


@pytest.fixture(scope="function")
def sample_relations(sample_entities: List[Entity]) -> List[Relation]:
    """Provide a list of sample relations for testing."""
    return [
        Relation(from_="test1", to="test2", relationType="visited"),
        Relation(from_="test1", to="test3", relationType="works_at"),
        Relation(from_="test2", to="test3", relationType="located_in"),
    ]


@pytest.fixture(scope="function")
async def populated_jsonl_backend(
    jsonl_backend: JsonlBackend,
    sample_entities: List[Entity],
    sample_relations: List[Relation],
) -> AsyncGenerator[JsonlBackend, None]:
    """Provide a JSONL backend pre-populated with sample data."""
    await jsonl_backend.create_entities(sample_entities)
    await jsonl_backend.create_relations(sample_relations)
    yield jsonl_backend


@pytest.fixture(scope="function")
def temp_jsonl_path(tmp_path: Path) -> Path:
    """Provide a temporary path for JSONL files."""
    return tmp_path / "test_memory.jsonl"
