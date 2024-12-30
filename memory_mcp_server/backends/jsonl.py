"""JSONL backend implementation for Memory MCP Server."""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

import aiofiles

from ..interfaces import Entity, Relation, KnowledgeGraph
from ..exceptions import EntityNotFoundError, FileAccessError
from .base import Backend


class JsonlBackend(Backend):
    """JSONL file-based implementation of the knowledge graph backend."""

    def __init__(self, memory_path: Path, cache_ttl: int = 60):
        """Initialize JSONL backend.

        Args:
            memory_path: Path to the JSONL file
            cache_ttl: Cache time-to-live in seconds
        """
        self.memory_path = memory_path
        self.cache_ttl = cache_ttl
        self._cache: Optional[KnowledgeGraph] = None
        self._cache_timestamp: float = 0.0
        self._dirty = False
        self._write_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._indices = {
            "entity_names": {},
            "entity_types": defaultdict(list),
            "relations_from": defaultdict(list),
            "relations_to": defaultdict(list),
        }

    async def initialize(self) -> None:
        """Initialize the backend. Creates parent directory if needed."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        """Close the backend. Ensures any pending changes are saved."""
        await self.flush()

    def _build_indices(self, graph: KnowledgeGraph) -> None:
        """Build indices for faster lookups.

        Args:
            graph: KnowledgeGraph to index
        """
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

    async def _check_cache(self) -> KnowledgeGraph:
        """Check if cache needs refresh and reload if necessary.

        Returns:
            Current KnowledgeGraph from cache or file
        """
        current_time = time.monotonic()
        needs_refresh = (
            self._cache is None
            or (current_time - self._cache_timestamp > self.cache_ttl)
            or self._dirty
        )

        if needs_refresh:
            async with self._lock:
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
                        return KnowledgeGraph(entities=[], relations=[])

        return self._cache

    async def _load_graph_from_file(self) -> KnowledgeGraph:
        """Load the knowledge graph from JSONL file.

        Returns:
            KnowledgeGraph loaded from file

        Raises:
            FileAccessError: If file cannot be read
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
                        continue
            return graph
        except Exception as e:
            raise FileAccessError(f"Error reading file: {str(e)}")

    async def _save_graph(self, graph: KnowledgeGraph) -> None:
        """Save the knowledge graph to JSONL file atomically.

        Args:
            graph: KnowledgeGraph to save

        Raises:
            FileAccessError: If file cannot be written
        """
        temp_path = self.memory_path.with_suffix(".tmp")
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
            raise FileAccessError(f"Error saving file: {str(e)}")
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities.

        Args:
            entities: List of entities to create

        Returns:
            List of successfully created entities

        Raises:
            ValueError: If entity validation fails
            FileAccessError: If file operations fail
        """
        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = self._indices["entity_names"]
            new_entities = []

            for entity in entities:
                if not entity.name or not entity.entityType:
                    raise ValueError(f"Invalid entity: {entity}")
                if entity.name not in existing_entities:
                    new_entities.append(entity)
                    existing_entities[entity.name] = entity
                    self._indices["entity_types"][entity.entityType].append(entity)

            if new_entities:
                graph.entities.extend(new_entities)
                self._dirty = True
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

            return new_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations.

        Args:
            relations: List of relations to create

        Returns:
            List of successfully created relations

        Raises:
            ValueError: If relation validation fails
            EntityNotFoundError: If referenced entities don't exist
            FileAccessError: If file operations fail
        """
        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = self._indices["entity_names"]
            new_relations = []

            for relation in relations:
                if not relation.from_ or not relation.to or not relation.relationType:
                    raise ValueError(f"Invalid relation: {relation}")

                if relation.from_ not in existing_entities:
                    raise EntityNotFoundError(f"Entity not found: {relation.from_}")
                if relation.to not in existing_entities:
                    raise EntityNotFoundError(f"Entity not found: {relation.to}")

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
        """Read the entire knowledge graph.

        Returns:
            Current state of the knowledge graph
        """
        return await self._check_cache()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        """Search for entities and relations matching query.

        Args:
            query: Search query string

        Returns:
            KnowledgeGraph containing matches

        Raises:
            ValueError: If query is empty
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        graph = await self._check_cache()
        q = query.lower()
        filtered_entities = set()

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
            entities=list(filtered_entities),
            relations=filtered_relations
        )

    async def flush(self) -> None:
        """Ensure any pending changes are saved to disk."""
        async with self._write_lock:
            if self._dirty:
                graph = await self._check_cache()
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()