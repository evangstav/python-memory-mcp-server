from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    name: str
    entityType: str
    observations: List[str]


@dataclass
class Relation:
    from_: str  # Using from_ to avoid Python keyword
    to: str
    relationType: str


@dataclass
class KnowledgeGraph:
    entities: List[Entity]
    relations: List[Relation]
