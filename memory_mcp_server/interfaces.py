"""Interface definitions for the memory MCP server."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)  # Make it hashable by adding frozen=True
class Entity:
    """Entity class representing a node in the knowledge graph."""

    name: str
    entityType: str
    observations: Tuple[str, ...]  # Change list to tuple to make it hashable

    def __init__(self, name: str, entityType: str, observations: List[str]) -> None:
        """Initialize an Entity.

        Args:
            name: The name of the entity
            entityType: The type of the entity
            observations: List of observations about the entity
        """
        # We need to use object.__setattr__ because the class is frozen
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entityType", entityType)
        # Convert list to tuple
        object.__setattr__(self, "observations", tuple(observations))

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity to a dictionary representation."""
        return {
            "name": self.name,
            "type": self.entityType,
            "observations": list(self.observations),
        }


@dataclass
class Relation:
    """Relation class representing an edge in the knowledge graph."""

    from_: str  # Using from_ in code but will serialize as 'from'
    to: str
    relationType: str

    def __init__(self, **kwargs: str) -> None:
        """Initialize a Relation.

        Args:
            **kwargs: Keyword arguments for 'from'/'from_', 'to', 'relationType'
        """
        # Handle both 'from' and 'from_' in input
        if "from" in kwargs:
            self.from_ = kwargs["from"]
        elif "from_" in kwargs:
            self.from_ = kwargs["from_"]
        self.to = kwargs["to"]
        self.relationType = kwargs["relationType"]

    def to_dict(self) -> Dict[str, str]:
        """Convert the relation to a dictionary representation."""
        return {
            "from": self.from_,
            "to": self.to,
            "relationType": self.relationType,
        }


@dataclass
class KnowledgeGraph:
    """KnowledgeGraph class representing the entire graph structure."""

    entities: List[Entity]
    relations: List[Relation]

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert the knowledge graph to a dictionary representation."""
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
        }
