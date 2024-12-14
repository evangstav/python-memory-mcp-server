from pathlib import Path
import json
import os
from typing import List, Dict, Any, Set

from .interfaces import KnowledgeGraph, Entity, Relation
from .exceptions import (
    EntityNotFoundError,
    EntityAlreadyExistsError,
    RelationValidationError,
    FileAccessError,
    JsonParsingError,
)


class KnowledgeGraphManager:
    def __init__(self, memory_path: Path):
        self.memory_path = memory_path

    def _check_file_permissions(self) -> None:
        """Check if we have proper file permissions."""
        try:
            # Check if directory exists and is writable
            if self.memory_path.exists():
                if not os.access(self.memory_path, os.R_OK | os.W_OK):
                    raise FileAccessError(
                        f"Insufficient permissions for file: {self.memory_path}"
                    )
            else:
                # Check if we can create the file
                if not os.access(self.memory_path.parent, os.W_OK):
                    raise FileAccessError(
                        f"Cannot create file in directory: {self.memory_path.parent}"
                    )
        except OSError as e:
            raise FileAccessError(f"File system error: {str(e)}")

    def _validate_entity(self, entity: Entity, existing_entities: Set[str]) -> None:
        """Validate an entity before creation."""
        if not entity.name:
            raise ValueError("Entity name cannot be empty")
        if not entity.entityType:
            raise ValueError("Entity type cannot be empty")
        if entity.name in existing_entities:
            raise EntityAlreadyExistsError(entity.name)

    def _validate_relation(
        self, relation: Relation, existing_entities: Set[str]
    ) -> None:
        """Validate a relation before creation."""
        if not relation.from_:
            raise RelationValidationError("Relation 'from' field cannot be empty")
        if not relation.to:
            raise RelationValidationError("Relation 'to' field cannot be empty")
        if not relation.relationType:
            raise RelationValidationError("Relation type cannot be empty")

        if relation.from_ not in existing_entities:
            raise EntityNotFoundError(relation.from_)
        if relation.to not in existing_entities:
            raise EntityNotFoundError(relation.to)

    async def load_graph(self) -> KnowledgeGraph:
        """Load the knowledge graph from file with improved error handling."""
        self._check_file_permissions()

        try:
            if not self.memory_path.exists():
                return KnowledgeGraph(entities=[], relations=[])

            with self.memory_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                graph = KnowledgeGraph(entities=[], relations=[])

                for i, line in enumerate(lines, 1):
                    try:
                        item = json.loads(line)

                        # Validate required fields
                        if "type" not in item:
                            raise ValueError("Missing 'type' field")

                        if item["type"] == "entity":
                            if "name" not in item:
                                raise ValueError("Missing 'name' field in entity")
                            if "entityType" not in item:
                                raise ValueError("Missing 'entityType' field in entity")
                            if "observations" not in item:
                                raise ValueError(
                                    "Missing 'observations' field in entity"
                                )

                            graph.entities.append(
                                Entity(
                                    name=item["name"],
                                    entityType=item["entityType"],
                                    observations=item["observations"],
                                )
                            )
                        elif item["type"] == "relation":
                            if "from" not in item:
                                raise ValueError("Missing 'from' field in relation")
                            if "to" not in item:
                                raise ValueError("Missing 'to' field in relation")
                            if "relationType" not in item:
                                raise ValueError(
                                    "Missing 'relationType' field in relation"
                                )

                            graph.relations.append(
                                Relation(
                                    from_=item["from"],
                                    to=item["to"],
                                    relationType=item["relationType"],
                                )
                            )
                        else:
                            raise ValueError(f"Unknown type: {item['type']}")

                    except json.JSONDecodeError as e:
                        raise JsonParsingError(i, line, e)
                    except Exception as e:
                        raise ValueError(f"Error processing line {i}: {str(e)}")

                return graph

        except FileNotFoundError:
            return KnowledgeGraph(entities=[], relations=[])
        except (OSError, IOError) as e:
            raise FileAccessError(f"Error reading file: {str(e)}")

    async def save_graph(self, graph: KnowledgeGraph) -> None:
        """Save the knowledge graph to file with improved error handling."""
        self._check_file_permissions()

        try:
            lines = []
            existing_entities = {entity.name for entity in graph.entities}

            # Validate all entities and relations before saving
            for entity in graph.entities:
                if not entity.name or not entity.entityType:
                    raise ValueError(f"Invalid entity: {entity}")

            for relation in graph.relations:
                if not relation.from_ or not relation.to or not relation.relationType:
                    raise ValueError(f"Invalid relation: {relation}")
                if relation.from_ not in existing_entities:
                    raise EntityNotFoundError(relation.from_)
                if relation.to not in existing_entities:
                    raise EntityNotFoundError(relation.to)

            # Create the JSON lines
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

            # Ensure the directory exists
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to a temporary file first
            temp_path = self.memory_path.with_suffix(".tmp")
            try:
                with temp_path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                # Rename the temporary file to the actual file
                # This provides atomic writes
                temp_path.replace(self.memory_path)
            finally:
                # Clean up the temporary file if it still exists
                if temp_path.exists():
                    temp_path.unlink()

        except (OSError, IOError) as e:
            raise FileAccessError(f"Error writing to file: {str(e)}")

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities with validation."""
        graph = await self.load_graph()
        existing_entities = {entity.name for entity in graph.entities}
        new_entities = []

        for entity in entities:
            try:
                self._validate_entity(entity, existing_entities)
                new_entities.append(entity)
                existing_entities.add(entity.name)
            except EntityAlreadyExistsError:
                # Skip duplicates silently as per original behavior
                continue

        graph.entities.extend(new_entities)
        await self.save_graph(graph)
        return new_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations with validation."""
        graph = await self.load_graph()
        existing_entities = {entity.name for entity in graph.entities}
        new_relations = []

        for relation in relations:
            try:
                self._validate_relation(relation, existing_entities)
                if not any(
                    existing.from_ == relation.from_
                    and existing.to == relation.to
                    and existing.relationType == relation.relationType
                    for existing in graph.relations
                ):
                    new_relations.append(relation)
            except (EntityNotFoundError, RelationValidationError) as e:
                raise e

        graph.relations.extend(new_relations)
        await self.save_graph(graph)
        return new_relations

    async def add_observations(
        self, observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add new observations to existing entities with validation."""
        graph = await self.load_graph()
        results = []

        for obs in observations:
            if "entityName" not in obs:
                raise ValueError("Missing 'entityName' in observation")
            if "contents" not in obs:
                raise ValueError("Missing 'contents' in observation")

            entity = next(
                (e for e in graph.entities if e.name == obs["entityName"]), None
            )
            if not entity:
                raise EntityNotFoundError(obs["entityName"])

            # Validate observation contents
            if not isinstance(obs["contents"], list):
                raise ValueError("Contents must be a list of strings")
            if not all(isinstance(content, str) for content in obs["contents"]):
                raise ValueError("All contents must be strings")

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

    async def delete_entities(self, entity_names: List[str]) -> None:
        """Delete multiple entities and their associated relations."""
        if not entity_names:
            raise ValueError("Entity names list cannot be empty")

        graph = await self.load_graph()
        # Check if any of the entities exist before deleting
        existing_names = {entity.name for entity in graph.entities}
        for name in entity_names:
            if name not in existing_names:
                raise EntityNotFoundError(name)

        graph.entities = [e for e in graph.entities if e.name not in entity_names]
        graph.relations = [
            r
            for r in graph.relations
            if r.from_ not in entity_names and r.to not in entity_names
        ]
        await self.save_graph(graph)

    async def delete_observations(self, deletions: List[Dict[str, Any]]) -> None:
        """Delete specific observations from entities."""
        graph = await self.load_graph()
        for deletion in deletions:
            if "entityName" not in deletion:
                raise ValueError("Missing 'entityName' in deletion")
            if "observations" not in deletion:
                raise ValueError("Missing 'observations' in deletion")

            entity = next(
                (e for e in graph.entities if e.name == deletion["entityName"]), None
            )
            if not entity:
                raise EntityNotFoundError(deletion["entityName"])

            if not isinstance(deletion["observations"], list):
                raise ValueError("Observations must be a list of strings")

            entity.observations = [
                obs
                for obs in entity.observations
                if obs not in deletion["observations"]
            ]

        await self.save_graph(graph)

    async def delete_relations(self, relations: List[Relation]) -> None:
        """Delete specific relations from the graph."""
        if not relations:
            raise ValueError("Relations list cannot be empty")

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
        """Read the entire knowledge graph."""
        return await self.load_graph()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        """Search for nodes in the knowledge graph."""
        if not query:
            raise ValueError("Search query cannot be empty")

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
        """Open specific nodes by their names."""
        if not names:
            raise ValueError("Names list cannot be empty")

        graph = await self.load_graph()
        filtered_entities = [e for e in graph.entities if e.name in names]

        # Check if all requested entities were found
        found_names = {e.name for e in filtered_entities}
        missing_names = set(names) - found_names
        if missing_names:
            raise EntityNotFoundError(next(iter(missing_names)))

        filtered_entity_names = {e.name for e in filtered_entities}
        filtered_relations = [
            r
            for r in graph.relations
            if r.from_ in filtered_entity_names and r.to in filtered_entity_names
        ]

        return KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)
