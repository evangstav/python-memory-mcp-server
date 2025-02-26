"""Tests for conversation context tracking."""

import asyncio
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager
from memory_mcp_server.validation import ValidationError


@pytest.fixture
async def temp_kg():
    """Create a temporary knowledge graph manager with test data."""
    with NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    # Create a knowledge graph manager with a temporary file
    kg = KnowledgeGraphManager(temp_path)
    await kg.initialize()

    # Create some test entities
    entities = [
        Entity(
            name="python-project",
            entityType="project",
            observations=["Started on 2024-01-15", "Version 1.0 released"],
        ),
        Entity(
            name="user-john",
            entityType="person",
            observations=["Software engineer", "Python expert"],
        ),
        Entity(
            name="memory-system",
            entityType="tool",
            observations=["Knowledge graph tool", "Used for long-term memory"],
        ),
    ]
    await kg.create_entities(entities)

    # Create some test relations
    relations = [
        Relation(from_="user-john", to="python-project", relationType="created"),
        Relation(from_="python-project", to="memory-system", relationType="contains"),
    ]
    await kg.create_relations(relations)

    yield kg

    # Clean up
    await kg.close()
    try:
        os.unlink(temp_path)
    except Exception:
        pass


async def test_update_conversation_context(temp_kg):
    """Test creating a conversation context entry."""
    # Arrange
    current_topic = "Python Development"
    entities_mentioned = ["python-project", "user-john"]
    summary = "Discussing the Python project development status"
    importance = 0.8

    # Act
    context_id = await temp_kg.update_conversation_context(
        current_topic=current_topic,
        entities_mentioned=entities_mentioned,
        summary=summary,
        importance=importance,
    )

    # Assert
    assert context_id is not None
    assert context_id.startswith("context-")

    # Check that the entity was created
    graph = await temp_kg.read_graph()
    context_entity = next((e for e in graph.entities if e.name == context_id), None)
    assert context_entity is not None
    assert context_entity.entityType == "conversation_context"

    # Check observations
    assert len(context_entity.observations) == 5
    assert any(
        obs.startswith("Topic: Python Development")
        for obs in context_entity.observations
    )
    assert any(
        obs.startswith("Summary: Discussing") for obs in context_entity.observations
    )
    assert any(
        obs.startswith("Entities: python-project, user-john")
        for obs in context_entity.observations
    )
    assert any(obs.startswith("Importance: 0.8") for obs in context_entity.observations)

    # Check relations
    relations = [r for r in graph.relations if r.from_ == context_id]
    assert len(relations) == 2
    assert any(
        r.to == "python-project" and r.relationType == "mentions" for r in relations
    )
    assert any(r.to == "user-john" and r.relationType == "mentions" for r in relations)


async def test_update_context_nonexistent_entity(temp_kg):
    """Test creating a context with reference to a nonexistent entity."""
    # Arrange
    current_topic = "Error Test"
    entities_mentioned = ["nonexistent-entity", "python-project"]
    summary = "This should fail"

    # Act & Assert
    with pytest.raises(ValidationError):
        await temp_kg.update_conversation_context(
            current_topic=current_topic,
            entities_mentioned=entities_mentioned,
            summary=summary,
        )


async def test_get_relevant_context(temp_kg):
    """Test retrieving relevant context based on time and entities."""
    # Arrange - Create a few context entries with different timestamps

    # Sleep to ensure unique timestamps
    await asyncio.sleep(0.05)

    # Context 1 - mentions python-project and user-john
    context1_id = await temp_kg.update_conversation_context(
        current_topic="Past Meeting",
        entities_mentioned=["python-project", "user-john"],
        summary="Initial project planning",
        importance=0.9,
    )

    # Sleep to ensure unique timestamps
    await asyncio.sleep(0.05)

    # Create a more recent context - mentions memory-system
    await temp_kg.update_conversation_context(
        current_topic="Recent Discussion",
        entities_mentioned=["memory-system"],
        summary="Discussing memory system implementation",
        importance=0.7,
    )

    # Sleep to ensure unique timestamps
    await asyncio.sleep(0.05)

    # Create another context - mentions user-john
    context3_id = await temp_kg.update_conversation_context(
        current_topic="Very Recent Talk",
        entities_mentioned=["user-john"],
        summary="Talking about user requirements",
        importance=0.5,
    )

    # Act - Get relevant context for current entities
    current_entities = ["user-john", "python-project"]
    results = await temp_kg.get_relevant_context(
        current_entities=current_entities, lookback_hours=12.0, max_results=3
    )

    # Assert
    assert len(results) >= 2  # Should find at least context1 and context3

    # Check that contexts are ordered by relevance (most relevant first)
    # Context3 should be most relevant (most recent + mentions user-john)
    # Context1 should be next (older but mentions both entities)
    # Context2 might be included (recent but no entity overlap)
    context_ids = [result["id"] for result in results]
    assert context3_id in context_ids
    assert context1_id in context_ids

    # The first result should have highest relevance
    assert results[0]["relevance"] > results[1]["relevance"]


async def test_get_relevant_context_with_time_decay(temp_kg):
    """Test that time decay works correctly in relevance calculation."""
    # Arrange - Create context entries with controlled timestamps

    # Sleep to ensure unique timestamps
    await asyncio.sleep(0.05)

    # Create an old context with high entity overlap
    old_context_id = await temp_kg.update_conversation_context(
        current_topic="Old Discussion",
        entities_mentioned=["python-project", "user-john", "memory-system"],
        summary="Complete overlap but old",
        importance=0.5,
    )

    # Sleep to ensure unique timestamps
    await asyncio.sleep(0.05)

    # Create a recent context with less overlap
    recent_context_id = await temp_kg.update_conversation_context(
        current_topic="Recent Talk",
        entities_mentioned=["python-project"],
        summary="Less overlap but recent",
        importance=0.5,
    )

    # Act - Get relevant context
    current_entities = ["python-project", "user-john"]
    results = await temp_kg.get_relevant_context(
        current_entities=current_entities, lookback_hours=24.0, max_results=2
    )

    # Assert
    assert len(results) == 2

    # The recent context should be ranked higher despite less overlap
    assert results[0]["id"] == recent_context_id
    assert results[1]["id"] == old_context_id
