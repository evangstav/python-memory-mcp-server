"""Tests for knowledge graph validation."""

import pytest

from memory_mcp_server.interfaces import Entity, KnowledgeGraph, Relation
from memory_mcp_server.validation import (
    EntityValidationError,
    KnowledgeGraphValidator,
    RelationValidationError,
)


def test_validate_entity_name():
    """Test entity name validation rules."""
    # Valid names
    KnowledgeGraphValidator.validate_entity_name("test")
    KnowledgeGraphValidator.validate_entity_name("test-123")
    KnowledgeGraphValidator.validate_entity_name("a" + "b" * 98)  # 100 chars

    # Invalid names
    with pytest.raises(EntityValidationError, match="Invalid entity name"):
        KnowledgeGraphValidator.validate_entity_name("Test")  # Uppercase
    with pytest.raises(EntityValidationError, match="Invalid entity name"):
        KnowledgeGraphValidator.validate_entity_name("test_123")  # Underscore
    with pytest.raises(EntityValidationError, match="Invalid entity name"):
        KnowledgeGraphValidator.validate_entity_name("123test")  # Starts with number
    with pytest.raises(EntityValidationError, match="Invalid entity name"):
        KnowledgeGraphValidator.validate_entity_name("a" * 101)  # Too long


def test_validate_entity_type():
    """Test entity type validation."""
    # Valid types
    KnowledgeGraphValidator.validate_entity_type("person")
    KnowledgeGraphValidator.validate_entity_type("concept")
    KnowledgeGraphValidator.validate_entity_type("project")

    # Invalid types
    with pytest.raises(EntityValidationError, match="Invalid entity type"):
        KnowledgeGraphValidator.validate_entity_type("invalid-type")
    with pytest.raises(EntityValidationError, match="Invalid entity type"):
        KnowledgeGraphValidator.validate_entity_type("Person")  # Case sensitive


def test_validate_observations():
    """Test observation validation rules."""
    # Valid observations
    KnowledgeGraphValidator.validate_observations(["test observation"])
    KnowledgeGraphValidator.validate_observations(["obs 1", "obs 2"])  # Multiple unique
    KnowledgeGraphValidator.validate_observations([])  # Empty list is valid

    # Invalid observations
    with pytest.raises(EntityValidationError, match="Empty observation"):
        KnowledgeGraphValidator.validate_observations([""])
    with pytest.raises(EntityValidationError, match="maximum length"):
        KnowledgeGraphValidator.validate_observations(["a" * 501])  # Too long
    with pytest.raises(EntityValidationError, match="Duplicate observation"):
        KnowledgeGraphValidator.validate_observations(["same", "same"])


def test_validate_entity():
    """Test complete entity validation."""
    # Valid entity
    entity = Entity(
        name="test-entity",
        entityType="concept",
        observations=["valid observation"],
    )
    KnowledgeGraphValidator.validate_entity(entity)

    # Invalid entity - multiple issues
    invalid_entity = Entity(
        name="Invalid Name",
        entityType="invalid-type",
        observations=["valid"],
    )
    with pytest.raises(EntityValidationError, match="Invalid entity name"):
        KnowledgeGraphValidator.validate_entity(invalid_entity)


def test_validate_relation_type():
    """Test relation type validation."""
    # Valid types
    KnowledgeGraphValidator.validate_relation_type("knows")
    KnowledgeGraphValidator.validate_relation_type("contains")
    KnowledgeGraphValidator.validate_relation_type("uses")

    # Invalid types
    with pytest.raises(RelationValidationError, match="Invalid relation type"):
        KnowledgeGraphValidator.validate_relation_type("invalid-type")
    with pytest.raises(RelationValidationError, match="Invalid relation type"):
        KnowledgeGraphValidator.validate_relation_type("Knows")  # Case sensitive


def test_validate_relation():
    """Test complete relation validation."""
    # Valid relation
    relation = Relation(from_="entity1", to="entity2", relationType="knows")
    KnowledgeGraphValidator.validate_relation(relation)

    # Invalid - self reference
    self_ref_relation = Relation(from_="same", to="same", relationType="knows")
    with pytest.raises(RelationValidationError, match="Self-referential"):
        KnowledgeGraphValidator.validate_relation(self_ref_relation)

    # Invalid type
    invalid_type_relation = Relation(from_="e1", to="e2", relationType="invalid")
    with pytest.raises(RelationValidationError, match="Invalid relation type"):
        KnowledgeGraphValidator.validate_relation(invalid_type_relation)
        Relation(from_="e1", to="e2", relationType="invalid")


def test_validate_no_cycles():
    """Test cycle detection in relations."""
    # Valid - no cycles
    relations = [
        Relation(from_="a", to="b", relationType="knows"),
        Relation(from_="b", to="c", relationType="knows"),
    ]
    KnowledgeGraphValidator.validate_no_cycles(relations)

    # Invalid - direct cycle
    relations = [
        Relation(from_="a", to="b", relationType="knows"),
        Relation(from_="b", to="a", relationType="knows"),
    ]
    with pytest.raises(RelationValidationError, match="Circular dependency"):
        KnowledgeGraphValidator.validate_no_cycles(relations)

    # Invalid - indirect cycle
    relations = [
        Relation(from_="a", to="b", relationType="knows"),
        Relation(from_="b", to="c", relationType="knows"),
        Relation(from_="c", to="a", relationType="knows"),
    ]
    with pytest.raises(RelationValidationError, match="Circular dependency"):
        KnowledgeGraphValidator.validate_no_cycles(relations)


def test_validate_graph():
    """Test complete graph validation."""
    # Valid graph
    graph = KnowledgeGraph(
        entities=[
            Entity(name="person1", entityType="person", observations=["obs1"]),
            Entity(name="person2", entityType="person", observations=["obs2"]),
        ],
        relations=[
            Relation(from_="person1", to="person2", relationType="knows"),
        ],
    )
    KnowledgeGraphValidator.validate_graph(graph)

    # Invalid - duplicate entity names
    graph = KnowledgeGraph(
        entities=[
            Entity(name="same", entityType="person", observations=["obs1"]),
            Entity(name="same", entityType="person", observations=["obs2"]),
        ],
        relations=[],
    )
    with pytest.raises(EntityValidationError, match="Duplicate entity name"):
        KnowledgeGraphValidator.validate_graph(graph)

    # Invalid - missing referenced entity
    graph = KnowledgeGraph(
        entities=[
            Entity(name="person1", entityType="person", observations=[]),
        ],
        relations=[
            Relation(from_="person1", to="missing", relationType="knows"),
        ],
    )
    with pytest.raises(RelationValidationError, match="not found in graph"):
        KnowledgeGraphValidator.validate_graph(graph)
