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

    @classmethod
    def validate_batch_entities(
        cls, entities: List[Entity], existing_names: Set[str]
    ) -> None:
        """Validate a batch of entities efficiently.

        Args:
            entities: List of entities to validate
            existing_names: Set of existing entity names

        Raises:
            EntityValidationError: If validation fails
        """
        if not entities:
            raise EntityValidationError("Entity list cannot be empty")

        # Check for duplicates within the batch
        new_names = set()
        for entity in entities:
            if entity.name in new_names:
                raise EntityValidationError(
                    f"Duplicate entity name in batch: {entity.name}"
                )
            new_names.add(entity.name)

        # Check for conflicts with existing entities
        conflicts = new_names.intersection(existing_names)
        if conflicts:
            raise EntityValidationError(
                f"Entities already exist: {', '.join(conflicts)}"
            )

        # Validate all entities in one pass
        for entity in entities:
            cls.validate_entity(entity)

    @classmethod
    def validate_batch_relations(
        cls,
        relations: List[Relation],
        existing_relations: List[Relation],
        entity_names: Set[str],
    ) -> None:
        """Validate a batch of relations efficiently.

        Args:
            relations: List of relations to validate
            existing_relations: List of existing relations
            entity_names: Set of valid entity names

        Raises:
            RelationValidationError: If validation fails
        """
        if not relations:
            raise RelationValidationError("Relations list cannot be empty")

        # Track relation keys to prevent duplicates
        seen_relations: Set[tuple[str, str, str]] = set()

        # Validate all relations in one pass
        missing_entities = set()
        for relation in relations:
            # Basic validation
            cls.validate_relation(relation)

            # Check for duplicate relations
            key = (relation.from_, relation.to, relation.relationType)
            if key in seen_relations:
                raise RelationValidationError(
                    f"Duplicate relation: {relation.from_} -> {relation.to}"
                )
            seen_relations.add(key)

            # Collect missing entities
            if relation.from_ not in entity_names:
                missing_entities.add(relation.from_)
            if relation.to not in entity_names:
                missing_entities.add(relation.to)

        # Report all missing entities at once
        if missing_entities:
            raise RelationValidationError(
                f"Entities not found: {', '.join(missing_entities)}"
            )

        # Check for cycles including existing relations
        cls.validate_no_cycles(relations, existing_relations)

    @classmethod
    def validate_batch_observations(
        cls,
        observations_map: dict[str, List[str]],
        existing_entities: dict[str, Entity],
    ) -> None:
        """Validate a batch of observations efficiently.

        Args:
            observations_map: Dictionary mapping entity names to lists of observations
            existing_entities: Dictionary of existing entities

        Raises:
            EntityValidationError: If validation fails
        """
        if not observations_map:
            raise EntityValidationError("Observations map cannot be empty")

        # Check for missing entities first
        missing_entities = [
            name for name in observations_map if name not in existing_entities
        ]
        if missing_entities:
            raise EntityValidationError(
                f"Entities not found: {', '.join(missing_entities)}"
            )

        # Validate all observations in one pass
        for entity_name, observations in observations_map.items():
            if not observations:
                continue

            # Validate observation format
            cls.validate_observations(observations)

            # Check for duplicates against existing observations
            entity = existing_entities[entity_name]
            existing_observations = set(entity.observations)
            duplicates = [obs for obs in observations if obs in existing_observations]
            if duplicates:
                raise EntityValidationError(
                    f"Duplicate observations for {entity_name}: {', '.join(duplicates)}"
                )
