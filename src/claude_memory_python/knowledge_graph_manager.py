from pathlib import Path
import json
from typing import List, Dict, Any

from .interfaces import KnowledgeGraph, Entity, Relation


class KnowledgeGraphManager:
    def __init__(self, memory_path: Path):
        self.memory_path = memory_path

    async def load_graph(self) -> KnowledgeGraph:
        try:
            with self.memory_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                graph = KnowledgeGraph(entities=[], relations=[])

                for line in lines:
                    item = json.loads(line)
                    if item["type"] == "entity":
                        graph.entities.append(
                            Entity(
                                name=item["name"],
                                entityType=item["entityType"],
                                observations=item["observations"],
                            )
                        )
                    elif item["type"] == "relation":
                        graph.relations.append(
                            Relation(
                                from_=item["from"],
                                to=item["to"],
                                relationType=item["relationType"],
                            )
                        )
                return graph
        except FileNotFoundError:
            return KnowledgeGraph(entities=[], relations=[])

    async def save_graph(self, graph: KnowledgeGraph):
        lines = []
        for entity in graph.entities:
            lines.append(
                json.dumps(
                    {
                        "type": "entity",
                        "name": entity.name,
                        "entityType": entity.entityType,
                        "observations": entity.observations,
                    }
                )
            )
        for relation in graph.relations:
            lines.append(
                json.dumps(
                    {
                        "type": "relation",
                        "from": relation.from_,
                        "to": relation.to,
                        "relationType": relation.relationType,
                    }
                )
            )

        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))