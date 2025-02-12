"""Tests for validation functionality."""

import pytest

from memory_mcp_server.interfaces import Entity, Relation
from memory_mcp_server.validation import (
    EntityValidationError,
    KnowledgeGraphValidator,
    RelationValidationError,
)


def test_validate_batch_entities() -> None:
    """Test batch entity validation."""
    # Valid batch
    entities = [
        Entity("test1", "person", ["obs1"]),
        Entity("test2", "person", ["obs2"]),
    ]
    existing_names = {"existing1", "existing2"}
    KnowledgeGraphValidator.validate_batch_entities(entities, existing_names)

    # Empty batch
    with pytest.raises(EntityValidationError, match="Entity list cannot be empty"):
        KnowledgeGraphValidator.validate_batch_entities([], existing_names)

    # Duplicate names within batch
    entities = [
        Entity("test1", "person", ["obs1"]),
        Entity("test1", "person", ["obs2"]),
    ]
    with pytest.raises(EntityValidationError, match="Duplicate entity name in batch"):
        KnowledgeGraphValidator.validate_batch_entities(entities, existing_names)

    # Conflict with existing names
    entities = [
        Entity("test1", "person", ["obs1"]),
        Entity("existing1", "person", ["obs2"]),
    ]
    with pytest.raises(EntityValidationError, match="Entities already exist"):
        KnowledgeGraphValidator.validate_batch_entities(entities, existing_names)

    # Invalid entity type
    entities = [
        Entity("test1", "invalid-type", ["obs1"]),
    ]
    with pytest.raises(EntityValidationError, match="Invalid entity type"):
        KnowledgeGraphValidator.validate_batch_entities(entities, existing_names)


def test_validate_batch_relations() -> None:
    """Test batch relation validation."""
    # Valid batch
    relations = [
        Relation(from_="entity1", to="entity2", relationType="knows"),
        Relation(from_="entity2", to="entity3", relationType="knows"),
    ]
    existing_relations = []
    entity_names = {"entity1", "entity2", "entity3"}
    KnowledgeGraphValidator.validate_batch_relations(
        relations, existing_relations, entity_names
    )

    # Empty batch
    with pytest.raises(RelationValidationError, match="Relations list cannot be empty"):
        KnowledgeGraphValidator.validate_batch_relations(
            [], existing_relations, entity_names
        )

    # Duplicate relations
    relations = [
        Relation(from_="entity1", to="entity2", relationType="knows"),
        Relation(from_="entity1", to="entity2", relationType="knows"),  # Same relation
    ]
    with pytest.raises(RelationValidationError, match="Duplicate relation"):
        KnowledgeGraphValidator.validate_batch_relations(
            relations, existing_relations, entity_names
        )

    # Missing entities
    relations = [
        Relation("entity1", "nonexistent", "knows"),
    ]
    with pytest.raises(RelationValidationError, match="Entities not found"):
        KnowledgeGraphValidator.validate_batch_relations(
            relations, existing_relations, entity_names
        )

    # Invalid relation type
    relations = [
        Relation("entity1", "entity2", "invalid-type"),
    ]
    with pytest.raises(RelationValidationError, match="Invalid relation type"):
        KnowledgeGraphValidator.validate_batch_relations(
            relations, existing_relations, entity_names
        )

    # Self-referential relation
    relations = [
        Relation("entity1", "entity1", "knows"),
    ]
    with pytest.raises(
        RelationValidationError, match="Self-referential relations not allowed"
    ):
        KnowledgeGraphValidator.validate_batch_relations(
            relations, existing_relations, entity_names
        )

    # Cycle detection
    relations = [
        Relation("entity1", "entity2", "knows"),
        Relation("entity2", "entity3", "knows"),
        Relation("entity3", "entity1", "knows"),
    ]
    with pytest.raises(RelationValidationError, match="Circular dependency detected"):
        KnowledgeGraphValidator.validate_batch_relations(
            relations, existing_relations, entity_names
        )


def test_validate_batch_observations() -> None:
    """Test batch observation validation."""
    # Valid batch
    existing_entities = {
        "entity1": Entity("entity1", "person", ["existing1"]),
        "entity2": Entity("entity2", "person", ["existing2"]),
    }
    observations_map = {
        "entity1": ["new1", "new2"],
        "entity2": ["new3"],
    }
    KnowledgeGraphValidator.validate_batch_observations(
        observations_map, existing_entities
    )

    # Empty batch
    with pytest.raises(EntityValidationError, match="Observations map cannot be empty"):
        KnowledgeGraphValidator.validate_batch_observations({}, existing_entities)

    # Missing entities
    observations_map = {
        "entity1": ["new1"],
        "nonexistent": ["new2"],
    }
    with pytest.raises(EntityValidationError, match="Entities not found"):
        KnowledgeGraphValidator.validate_batch_observations(
            observations_map, existing_entities
        )

    # Empty observations list is allowed (skipped)
    observations_map = {
        "entity1": [],
    }
    KnowledgeGraphValidator.validate_batch_observations(
        observations_map, existing_entities
    )

    # Invalid observation format
    observations_map = {
        "entity1": ["", "new2"],  # Empty observation
    }
    with pytest.raises(EntityValidationError, match="Empty observation"):
        KnowledgeGraphValidator.validate_batch_observations(
            observations_map, existing_entities
        )

    # Duplicate observations
    observations_map = {
        "entity1": ["existing1", "new2"],  # Duplicate with existing observation
    }
    with pytest.raises(EntityValidationError, match="Duplicate observations"):
        KnowledgeGraphValidator.validate_batch_observations(
            observations_map, existing_entities
        )
