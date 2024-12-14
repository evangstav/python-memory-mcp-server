from dataclasses import dataclass
from typing import List


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


@dataclass
class Relation:
    from_: str  # Using from_ in code but will serialize as 'from'
    to: str
    relationType: str

    def __init__(self, **kwargs):
        # Handle both 'from' and 'from_' in input
        if "from" in kwargs:
            self.from_ = kwargs["from"]
        elif "from_" in kwargs:
            self.from_ = kwargs["from_"]
        self.to = kwargs["to"]
        self.relationType = kwargs["relationType"]

    def to_dict(self):
        return {"from": self.from_, "to": self.to, "relationType": self.relationType}


@dataclass
class KnowledgeGraph:
    entities: List[Entity]
    relations: List[Relation]
