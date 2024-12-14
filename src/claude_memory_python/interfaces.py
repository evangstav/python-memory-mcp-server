from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    name: str
    entityType: str
    observations: List[str]


@dataclass
class Relation:
    from_: str  # Using from_ in code but will serialize as 'from'
    to: str
    relationType: str

    def __init__(self, **kwargs):
        # Handle both 'from' and 'from_' in input
        if 'from' in kwargs:
            self.from_ = kwargs['from']
        elif 'from_' in kwargs:
            self.from_ = kwargs['from_']
        self.to = kwargs['to']
        self.relationType = kwargs['relationType']
    
    def __repr__(self):
        return f"Relation(from={self.from_}, to={self.to}, relationType={self.relationType})"
    
    def to_dict(self):
        return {
            "from": self.from_,
            "to": self.to,
            "relationType": self.relationType
        }


@dataclass
class KnowledgeGraph:
    entities: List[Entity]
    relations: List[Relation]