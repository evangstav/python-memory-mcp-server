"""Interface definitions for Memory MCP Server."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


@dataclass(frozen=True)
class Entity:
    """Entity in the knowledge graph."""

    name: str
    entityType: str
    observations: List[str]

    def __hash__(self) -> int:
        """Make Entity hashable based on name."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare Entity based on name."""
        if not isinstance(other, Entity):
            return NotImplemented
        return self.name == other.name

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "entityType": self.entityType,
            "observations": list(
                self.observations
            ),  # Convert to list in case it's a tuple
        }


@dataclass(frozen=True)
class Relation:
    """Relation between entities in the knowledge graph."""

    from_: str
    to: str
    relationType: str

    def __hash__(self) -> int:
        """Make Relation hashable based on all fields."""
        return hash((self.from_, self.to, self.relationType))

    def __eq__(self, other: object) -> bool:
        """Compare Relation based on all fields."""
        if not isinstance(other, Relation):
            return NotImplemented
        return (
            self.from_ == other.from_
            and self.to == other.to
            and self.relationType == other.relationType
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "from": self.from_,
            "to": self.to,
            "relationType": self.relationType,
        }


@dataclass
class KnowledgeGraph:
    """Knowledge graph containing entities and relations."""

    entities: List[Entity]
    relations: List[Relation]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
        }


@dataclass
class SearchOptions:
    """Options for configuring search behavior."""

    fuzzy: bool = False
    threshold: float = 80.0
    weights: Optional[dict[str, float]] = None


class BatchOperationType(Enum):
    """Types of batch operations."""

    CREATE_ENTITIES = "create_entities"
    DELETE_ENTITIES = "delete_entities"
    CREATE_RELATIONS = "create_relations"
    DELETE_RELATIONS = "delete_relations"
    ADD_OBSERVATIONS = "add_observations"


@dataclass
class BatchOperation:
    """Represents a single operation in a batch."""

    operation_type: BatchOperationType
    data: dict  # Operation-specific data


@dataclass
class BatchResult:
    """Result of a batch operation execution."""

    success: bool
    operations_completed: int
    failed_operations: List[tuple[BatchOperation, str]]  # Operation and error message
    error_message: Optional[str] = None
