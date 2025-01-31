"""Validation module for knowledge graph consistency."""

import re
from typing import List, Optional, Set

from .interfaces import Entity, KnowledgeGraph, Relation


class ValidationError(Exception):
    """Base class for validation errors."""

    pass


class EntityValidationError(ValidationError):
    """Raised when entity validation fails."""

    pass


class RelationValidationError(ValidationError):
    """Raised when relation validation fails."""

    pass


class KnowledgeGraphValidator:
    """Validator for ensuring knowledge graph consistency."""

    # Constants for validation rules
    ENTITY_NAME_PATTERN = r"^[a-z][a-z0-9-]{0,99}$"
    MAX_OBSERVATION_LENGTH = 500
    VALID_ENTITY_TYPES = {
        "person",
        "concept",
        "project",
        "document",
        "tool",
        "organization",
        "location",
        "event",
    }
    VALID_RELATION_TYPES = {
        "knows",
        "contains",
        "uses",
        "created",
        "belongs-to",
        "depends-on",
        "related-to",
    }

    @classmethod
    def validate_entity_name(cls, name: str) -> None:
        """Validate entity name follows naming convention.

        Args:
            name: Entity name to validate

        Raises:
            EntityValidationError: If name is invalid
        """
        if not re.match(cls.ENTITY_NAME_PATTERN, name):
            raise EntityValidationError(
                f"Invalid entity name '{name}'. Must start with lowercase letter, "
                "contain only lowercase letters, numbers and hyphens, "
                "and be 1-100 characters long."
            )

    @classmethod
    def validate_entity_type(cls, entity_type: str) -> None:
        """Validate entity type is from allowed set.

        Args:
            entity_type: Entity type to validate

        Raises:
            EntityValidationError: If type is invalid
        """
        if entity_type not in cls.VALID_ENTITY_TYPES:
            raise EntityValidationError(
                f"Invalid entity type '{entity_type}'. Must be one of: "
                f"{', '.join(sorted(cls.VALID_ENTITY_TYPES))}"
            )

    @classmethod
    def validate_observations(cls, observations: List[str]) -> None:
        """Validate entity observations.

        Args:
            observations: List of observations to validate

        Raises:
            EntityValidationError: If any observation is invalid
        """
        seen = set()
        for obs in observations:
            if not obs:
                raise EntityValidationError("Empty observation")
            if len(obs) > cls.MAX_OBSERVATION_LENGTH:
                raise EntityValidationError(
                    f"Observation exceeds length of {cls.MAX_OBSERVATION_LENGTH} chars"
                )
            if obs in seen:
                raise EntityValidationError(f"Duplicate observation: {obs}")
            seen.add(obs)

    @classmethod
    def validate_entity(cls, entity: Entity) -> None:
        """Validate an entity.

        Args:
            entity: Entity to validate

        Raises:
            EntityValidationError: If entity is invalid
        """
        cls.validate_entity_name(entity.name)
        cls.validate_entity_type(entity.entityType)
        cls.validate_observations(list(entity.observations))

    @classmethod
    def validate_relation_type(cls, relation_type: str) -> None:
        """Validate relation type is from allowed set.

        Args:
            relation_type: Relation type to validate

        Raises:
            RelationValidationError: If type is invalid
        """
        if relation_type not in cls.VALID_RELATION_TYPES:
            valid_types = ", ".join(sorted(cls.VALID_RELATION_TYPES))
            raise RelationValidationError(
                f"Invalid relation type '{relation_type}'. Valid types: {valid_types}"
            )

    @classmethod
    def validate_relation(cls, relation: Relation) -> None:
        """Validate a relation.

        Args:
            relation: Relation to validate

        Raises:
            RelationValidationError: If relation is invalid
        """
        if relation.from_ == relation.to:
            raise RelationValidationError("Self-referential relations not allowed")
        cls.validate_relation_type(relation.relationType)

    @classmethod
    def validate_no_cycles(
        cls,
        relations: List[Relation],
        existing_relations: Optional[List[Relation]] = None,
    ) -> None:
        """Validate that relations don't create cycles.

        Args:
            relations: New relations to validate
            existing_relations: Optional list of existing relations to check against

        Raises:
            RelationValidationError: If cycles are detected
        """
        # Build adjacency list
        graph: dict[str, Set[str]] = {}
        all_relations = list(relations)
        if existing_relations:
            all_relations.extend(existing_relations)

        for rel in all_relations:
            if rel.from_ not in graph:
                graph[rel.from_] = set()
            graph[rel.from_].add(rel.to)

        # Check for cycles using DFS
        def has_cycle(node: str, visited: Set[str], path: Set[str]) -> bool:
            visited.add(node)
            path.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, path):
                        return True
                elif neighbor in path:
                    return True

            path.remove(node)
            return False

        visited: Set[str] = set()
        path: Set[str] = set()

        for node in graph:
            if node not in visited:
                if has_cycle(node, visited, path):
                    raise RelationValidationError(
                        "Circular dependency detected in relations"
                    )

    @classmethod
    def validate_graph(cls, graph: KnowledgeGraph) -> None:
        """Validate entire knowledge graph.

        Args:
            graph: Knowledge graph to validate

        Raises:
            ValidationError: If any validation fails
        """
        # Validate all entities
        entity_names = set()
        for entity in graph.entities:
            cls.validate_entity(entity)
            if entity.name in entity_names:
                raise EntityValidationError(f"Duplicate entity name: {entity.name}")
            entity_names.add(entity.name)

        # Validate all relations
        for relation in graph.relations:
            cls.validate_relation(relation)
            if relation.from_ not in entity_names:
                raise RelationValidationError(
                    f"Source entity '{relation.from_}' not found in graph"
                )
            if relation.to not in entity_names:
                raise RelationValidationError(
                    f"Target entity '{relation.to}' not found in graph"
                )

        # Check for cycles
        cls.validate_no_cycles(graph.relations)
