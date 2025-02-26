"""Tests for ConversationContext model."""

import time
from datetime import datetime

import pytest

from memory_mcp_server.models.conversation_context import ConversationContext


def test_conversation_context_creation():
    """Test valid creation of ConversationContext."""
    # Arrange
    now = time.time()

    # Act
    context = ConversationContext(
        id="context-12345",
        timestamp=now,
        topic="Test Topic",
        entities_mentioned=["entity1", "entity2"],
        importance=0.8,
        summary="This is a test context",
    )

    # Assert
    assert context.id == "context-12345"
    assert context.timestamp == now
    assert context.topic == "Test Topic"
    assert context.entities_mentioned == ["entity1", "entity2"]
    assert context.importance == 0.8
    assert context.summary == "This is a test context"


def test_conversation_context_validation():
    """Test validation rules for ConversationContext."""
    now = time.time()

    # Test invalid ID
    with pytest.raises(ValueError, match="Context ID must be a non-empty string"):
        ConversationContext(
            id="",
            timestamp=now,
            topic="Test Topic",
            entities_mentioned=["entity1"],
            importance=0.5,
            summary="Test summary",
        )

    # Test invalid timestamp
    with pytest.raises(ValueError, match="Timestamp must be a positive float"):
        ConversationContext(
            id="context-12345",
            timestamp=-1.0,
            topic="Test Topic",
            entities_mentioned=["entity1"],
            importance=0.5,
            summary="Test summary",
        )

    # Test invalid topic
    with pytest.raises(ValueError, match="Topic must be a non-empty string"):
        ConversationContext(
            id="context-12345",
            timestamp=now,
            topic="",
            entities_mentioned=["entity1"],
            importance=0.5,
            summary="Test summary",
        )

    # Test invalid entities_mentioned
    with pytest.raises(ValueError, match="Entities mentioned must be a list"):
        ConversationContext(
            id="context-12345",
            timestamp=now,
            topic="Test Topic",
            entities_mentioned="not-a-list",  # type: ignore
            importance=0.5,
            summary="Test summary",
        )

    # Test invalid importance (too high)
    with pytest.raises(
        ValueError, match="Importance must be a float between 0.0 and 1.0"
    ):
        ConversationContext(
            id="context-12345",
            timestamp=now,
            topic="Test Topic",
            entities_mentioned=["entity1"],
            importance=1.5,
            summary="Test summary",
        )

    # Test invalid importance (too low)
    with pytest.raises(
        ValueError, match="Importance must be a float between 0.0 and 1.0"
    ):
        ConversationContext(
            id="context-12345",
            timestamp=now,
            topic="Test Topic",
            entities_mentioned=["entity1"],
            importance=-0.1,
            summary="Test summary",
        )

    # Test invalid summary
    with pytest.raises(ValueError, match="Summary must be a non-empty string"):
        ConversationContext(
            id="context-12345",
            timestamp=now,
            topic="Test Topic",
            entities_mentioned=["entity1"],
            importance=0.5,
            summary="",
        )


def test_to_entity_observations():
    """Test conversion to entity observations."""
    # Arrange
    now = time.time()
    context = ConversationContext(
        id="context-12345",
        timestamp=now,
        topic="Test Topic",
        entities_mentioned=["entity1", "entity2"],
        importance=0.75,
        summary="This is a test summary",
    )

    # Act
    observations = context.to_entity_observations()

    # Assert
    assert len(observations) == 5
    assert observations[0] == "Topic: Test Topic"
    assert observations[1].startswith("Time: ")
    # Verify that the timestamp is formatted correctly
    time_str = observations[1].replace("Time: ", "")
    dt = datetime.fromisoformat(time_str)
    assert (
        abs(dt.timestamp() - now) < 1
    )  # Allow for a small difference due to formatting
    assert observations[2] == "Summary: This is a test summary"
    assert observations[3] == "Entities: entity1, entity2"
    assert observations[4] == "Importance: 0.75"
