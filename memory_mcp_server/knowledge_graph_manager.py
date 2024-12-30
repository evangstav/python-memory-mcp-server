import asyncio
from pathlib import Path
import json
from typing import List, Dict, Optional, TypedDict
from collections import defaultdict
import aiofiles
import time
import logging

from .interfaces import KnowledgeGraph, Entity, Relation
from .exceptions import (
    EntityNotFoundError,
    FileAccessError,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Indices(TypedDict):
    entity_names: Dict[str, Entity]
    entity_types: Dict[str, List[Entity]]
    relations_from: Dict[str, List[Relation]]
    relations_to: Dict[str, List[Relation]]


class KnowledgeGraphManager:
    def __init__(
        self, memory_path: Path, cache_ttl: int = 60, full_indexing: bool = True
    ):
        """
        Initialize the KnowledgeGraphManager.

        Args:
            memory_path: Path to the knowledge graph file
            cache_ttl: Time to live for cache in seconds (default: 60)
            full_indexing: Whether to build all indices. If False, may skip some.
        """
        self.memory_path = memory_path
        self.cache_ttl = cache_ttl
        self._cache: Optional[KnowledgeGraph] = None
        self._cache_timestamp: float = 0.0
        self._dirty = False
        self.full_indexing = full_indexing
        # Initialize indices as empty. They will be populated after loading.
        self._indices: Indices = {
            "entity_names": {},
            "entity_types": defaultdict(list),
            "relations_from": defaultdict(list),
            "relations_to": defaultdict(list),
        }

        self._lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    def _build_indices(self, graph: KnowledgeGraph) -> None:
        """
        Build indices for faster lookups if full_indexing is True.
        Indices:
            entity_names: {entity_name: Entity}
            entity_types: {entity_type: [Entity]}
            relations_from: {entity_name: [Relation]}
            relations_to: {entity_name: [Relation]}
        """
        if not self.full_indexing:
            return

        entity_names: Dict[str, Entity] = {}
        entity_types: Dict[str, List[Entity]] = defaultdict(list)
        relations_from: Dict[str, List[Relation]] = defaultdict(list)
        relations_to: Dict[str, List[Relation]] = defaultdict(list)

        for entity in graph.entities:
            entity_names[entity.name] = entity
            entity_types[entity.entityType].append(entity)

        for relation in graph.relations:
            relations_from[relation.from_].append(relation)
            relations_to[relation.to].append(relation)

        self._indices["entity_names"] = entity_names
        self._indices["entity_types"] = entity_types
        self._indices["relations_from"] = relations_from
        self._indices["relations_to"] = relations_to

    def _validate_entity(self, entity: Entity) -> None:
        """
        Validate the fields of an entity.
        Raises ValueError if invalid.
        """
        if not entity.name or not isinstance(entity.name, str):
            raise ValueError("Entity name must be a non-empty string")
        if not entity.entityType or not isinstance(entity.entityType, str):
            raise ValueError("Entity type must be a non-empty string")
        if not isinstance(entity.observations, (list, tuple)):
            raise ValueError("Observations must be a list or tuple")

    def _validate_relation(self, relation: Relation) -> None:
        """
        Validate the fields of a relation.
        Raises ValueError if invalid.
        """
        if not relation.from_ or not relation.to or not relation.relationType:
            raise ValueError(f"Invalid relation: {relation}")

    async def _check_cache(self) -> KnowledgeGraph:
        """
        Ensure the cache is up-to-date. If the cache is stale or dirty, reload from file.
        Returns the current cached KnowledgeGraph.
        Uses a monotonic time source to avoid issues with system clock changes.
        """
        current_time = time.monotonic()
        needs_refresh = (
            self._cache is None
            or (current_time - self._cache_timestamp > self.cache_ttl)
            or self._dirty
        )

        if needs_refresh:
            async with self._lock:
                # Double-check inside the lock
                current_time = time.monotonic()
                if (
                    self._cache is None
                    or (current_time - self._cache_timestamp > self.cache_ttl)
                    or self._dirty
                ):
                    try:
                        graph = await self._load_graph_from_file()
                        self._cache = graph
                        self._cache_timestamp = current_time
                        self._build_indices(graph)
                        self._dirty = False
                    except Exception as e:
                        logger.error(f"Error loading graph: {e}")
                        return KnowledgeGraph(entities=[], relations=[])

        return self._cache

    async def _load_graph_from_file(self) -> KnowledgeGraph:
        """
        Load the knowledge graph from file line-by-line using asynchronous I/O.
        If the file doesn't exist, returns an empty graph.
        """
        if not self.memory_path.exists():
            return KnowledgeGraph(entities=[], relations=[])

        graph = KnowledgeGraph(entities=[], relations=[])
        try:
            async with aiofiles.open(self.memory_path, mode="r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
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
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error parsing line: {e}")
            return graph
        except Exception as e:
            raise FileAccessError(f"Error reading file: {str(e)}")

    async def _save_graph(self, graph: KnowledgeGraph) -> None:
        """
        Save the knowledge graph to file using an atomic write strategy and asynchronous I/O.
        Writes line-by-line, each line containing one JSON object.
        """
        temp_path = self.memory_path.with_suffix(".tmp")
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(temp_path, mode="w", encoding="utf-8") as f:
                for entity in graph.entities:
                    line = json.dumps(
                        {
                            "type": "entity",
                            "name": entity.name,
                            "entityType": entity.entityType,
                            "observations": entity.observations,
                        }
                    )
                    await f.write(line + "\n")

                for relation in graph.relations:
                    line = json.dumps(
                        {
                            "type": "relation",
                            "from": relation.from_,
                            "to": relation.to,
                            "relationType": relation.relationType,
                        }
                    )
                    await f.write(line + "\n")

            temp_path.replace(self.memory_path)
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            raise
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up temp file: {e}")

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Create multiple new entities. Updates in-memory indices and persists to file.
        Returns a list of newly created entities.
        """
        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = self._indices["entity_names"]
            new_entities = []

            for entity in entities:
                self._validate_entity(entity)
                if entity.name not in existing_entities:
                    new_entities.append(entity)
                    existing_entities[entity.name] = entity
                    self._indices["entity_types"][entity.entityType].append(entity)

            if new_entities:
                graph.entities.extend(new_entities)
                self._dirty = True
                await self._save_graph(graph)
                # After saving, reset dirty and update cache timestamp
                self._dirty = False
                self._cache_timestamp = time.monotonic()

            return new_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """
        Create multiple new relations. Updates in-memory indices and persists to file.
        Returns a list of newly created relations.
        """
        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = self._indices["entity_names"]
            new_relations = []

            for relation in relations:
                self._validate_relation(relation)

                if relation.from_ not in existing_entities:
                    raise EntityNotFoundError(f"Entity not found: {relation.from_}")
                if relation.to not in existing_entities:
                    raise EntityNotFoundError(f"Entity not found: {relation.to}")

                # Check for duplicates
                existing_from = self._indices["relations_from"].get(relation.from_, [])
                if not any(
                    r.from_ == relation.from_
                    and r.to == relation.to
                    and r.relationType == relation.relationType
                    for r in existing_from
                ):
                    new_relations.append(relation)
                    self._indices["relations_from"][relation.from_].append(relation)
                    self._indices["relations_to"][relation.to].append(relation)

            if new_relations:
                graph.relations.extend(new_relations)
                self._dirty = True
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

            return new_relations

    async def read_graph(self) -> KnowledgeGraph:
        """
        Read the entire knowledge graph using the cached version if available.
        """
        return await self._check_cache()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        """
        Search for entities and relations by a query string.
        Returns a filtered KnowledgeGraph.
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        graph = await self._check_cache()
        q = query.lower()
        filtered_entities = set()

        # Basic search: checks name, entityType, and observations
        for entity in graph.entities:
            if (
                q in entity.name.lower()
                or q in entity.entityType.lower()
                or any(q in obs.lower() for obs in entity.observations)
            ):
                filtered_entities.add(entity)

        filtered_entity_names = {e.name for e in filtered_entities}
        filtered_relations = [
            relation
            for relation in graph.relations
            if relation.from_ in filtered_entity_names
            and relation.to in filtered_entity_names
        ]

        return KnowledgeGraph(
            entities=list(filtered_entities), relations=filtered_relations
        )

    async def flush(self) -> None:
        """
        Force save the graph to disk if dirty.
        """
        async with self._write_lock:
            if self._dirty:
                graph = await self._check_cache()
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()
