from interfaces import KnowledgeGraph, Entity, Relation
import json
from typing import List, Dict, Any


class KnowledgeGraphManager:
    def __init__(self, memory_path: str):
        self.memory_path = memory_path

    async def load_graph(self) -> KnowledgeGraph:
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
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

        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        graph = await self.load_graph()
        new_entities = [
            e
            for e in entities
            if not any(existing.name == e.name for existing in graph.entities)
        ]
        graph.entities.extend(new_entities)
        await self.save_graph(graph)
        return new_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        graph = await self.load_graph()
        new_relations = [
            r
            for r in relations
            if not any(
                existing.from_ == r.from_
                and existing.to == r.to
                and existing.relationType == r.relationType
                for existing in graph.relations
            )
        ]
        graph.relations.extend(new_relations)
        await self.save_graph(graph)
        return new_relations

    async def add_observations(
        self, observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        graph = await self.load_graph()
        results = []

        for obs in observations:
            entity = next(
                (e for e in graph.entities if e.name == obs["entityName"]), None
            )
            if not entity:
                raise ValueError(f"Entity {obs['entityName']} not found")

            new_obs = [
                content
                for content in obs["contents"]
                if content not in entity.observations
            ]
            entity.observations.extend(new_obs)
            results.append(
                {"entityName": obs["entityName"], "addedObservations": new_obs}
            )

        await self.save_graph(graph)
        return results

    async def delete_entities(self, entity_names: List[str]):
        graph = await self.load_graph()
        graph.entities = [e for e in graph.entities if e.name not in entity_names]
        graph.relations = [
            r
            for r in graph.relations
            if r.from_ not in entity_names and r.to not in entity_names
        ]
        await self.save_graph(graph)

    async def delete_observations(self, deletions: List[Dict[str, Any]]):
        graph = await self.load_graph()
        for deletion in deletions:
            entity = next(
                (e for e in graph.entities if e.name == deletion["entityName"]), None
            )
            if entity:
                entity.observations = [
                    obs
                    for obs in entity.observations
                    if obs not in deletion["observations"]
                ]
        await self.save_graph(graph)

    async def delete_relations(self, relations: List[Relation]):
        graph = await self.load_graph()
        graph.relations = [
            r
            for r in graph.relations
            if not any(
                dr.from_ == r.from_
                and dr.to == r.to
                and dr.relationType == r.relationType
                for dr in relations
            )
        ]
        await self.save_graph(graph)

    async def read_graph(self) -> KnowledgeGraph:
        return await self.load_graph()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        graph = await self.load_graph()
        query = query.lower()

        filtered_entities = [
            e
            for e in graph.entities
            if (
                query in e.name.lower()
                or query in e.entityType.lower()
                or any(query in obs.lower() for obs in e.observations)
            )
        ]

        filtered_entity_names = {e.name for e in filtered_entities}
        filtered_relations = [
            r
            for r in graph.relations
            if r.from_ in filtered_entity_names and r.to in filtered_entity_names
        ]

        return KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)

    async def open_nodes(self, names: List[str]) -> KnowledgeGraph:
        graph = await self.load_graph()
        filtered_entities = [e for e in graph.entities if e.name in names]

        filtered_entity_names = {e.name for e in filtered_entities}
        filtered_relations = [
            r
            for r in graph.relations
            if r.from_ in filtered_entity_names and r.to in filtered_entity_names
        ]

        return KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)
