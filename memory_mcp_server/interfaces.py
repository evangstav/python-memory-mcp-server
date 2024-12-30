from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)  # Make it hashable by adding frozen=True
class Entity:
    name: str
    entityType: str
    observations: tuple  # Change list to tuple to make it hashable

    def __init__(self, name: str, entityType: str, observations: List[str]):
        # We need to use object.__setattr__ because the class is frozen
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entityType", entityType)
        object.__setattr__(
            self, "observations", tuple(observations)
        )  # Convert list to tuple

    @classmethod
    def from_dict(cls, data: dict) -> 'Entity':
        return cls(
            name=data["name"],
            entityType=data["entityType"],
            observations=data["observations"]
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entityType": self.entityType,
            "observations": list(self.observations)  # Convert tuple back to list
        }


@dataclass
class Relation:
    from_: str
    to: str
    relationType: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """Create a Relation instance from a dictionary."""
        return cls(
            from_=data.get("from_", data.get("from")),  # Handle both formats
            to=data["to"],
            relationType=data["relationType"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary format."""
        return {
            "from": self.from_,  # Use 'from' for external API
            "to": self.to,
            "relationType": self.relationType
        }


@dataclass
class KnowledgeGraph:
    entities: List[Entity]
    relations: List[Relation]
