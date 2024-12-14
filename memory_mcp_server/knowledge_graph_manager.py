import asyncio
from pathlib import Path
import json
import os
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict
import aiofiles
import time
from functools import lru_cache

from .interfaces import KnowledgeGraph, Entity, Relation
from .exceptions import (
    EntityNotFoundError,
    FileAccessError,
    JsonParsingError,
)


class KnowledgeGraphManager:
    def __init__(self, memory_path: Path, cache_ttl: int = 60):
        """
        Initialize the OptimizedKnowledgeGraphManager.

        Args:
            memory_path: Path to the knowledge graph file
            cache_ttl: Time to live for cache in seconds (default: 60)
        """
        self.memory_path = memory_path
        self.cache_ttl = cache_ttl
        self._cache: Optional[KnowledgeGraph] = None
        self._cache_timestamp: float = 0
        self._indices: Dict = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._dirty = False
        self._write_queue: List[Dict] = []
        self._max_queue_size = 100

    def _build_indices(self, graph: KnowledgeGraph) -> None:
        """Build indices for faster lookups."""
        # Create new dictionaries instead of clearing existing ones
        entity_names = {}
        entity_types = defaultdict(list)
        relations_from = defaultdict(list)
        relations_to = defaultdict(list)

        # Build indices
        for entity in graph.entities:
            entity_names[entity.name] = entity
            entity_types[entity.entityType].append(entity)

        for relation in graph.relations:
            relations_from[relation.from_].append(relation)
            relations_to[relation.to].append(relation)

        # Update indices atomically
        self._indices = {
            "entity_names": entity_names,
            "entity_types": entity_types,
            "relations_from": relations_from,
            "relations_to": relations_to,
        }

    async def _check_cache(self) -> KnowledgeGraph:
        """Check if cache is valid and return cached graph."""
        current_time = time.time()
        needs_refresh = (
            self._cache is None
            or current_time - self._cache_timestamp > self.cache_ttl
            or self._dirty
        )

        if needs_refresh:
            try:
                async with self._lock:
                    # Double check pattern
                    if (
                        self._cache is None
                        or current_time - self._cache_timestamp > self.cache_ttl
                        or self._dirty
                    ):
                        graph = await self._load_graph_from_file()
                        self._cache = graph
                        self._cache_timestamp = current_time
                        self._build_indices(graph)
                        self._dirty = False
            except Exception as e:
                # If loading fails, return empty graph
                print(f"Error loading graph: {e}")
                return KnowledgeGraph(entities=[], relations=[])

        return self._cache

    async def _load_graph_from_file(self) -> KnowledgeGraph:
        """Load the knowledge graph from file with improved error handling."""
        if not self.memory_path.exists():
            return KnowledgeGraph(entities=[], relations=[])

        try:
            async with aiofiles.open(self.memory_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
                lines = [line.strip() for line in content.splitlines() if line.strip()]

            graph = KnowledgeGraph(entities=[], relations=[])

            for i, line in enumerate(lines, 1):
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
                    print(f"Error parsing line {i}: {e}")
                    continue

            return graph

        except Exception as e:
            raise FileAccessError(f"Error reading file: {str(e)}")

    async def _save_graph(self, graph: KnowledgeGraph) -> None:
        """Save the knowledge graph to file with batched writes."""
        lines = []

        # Prepare all lines first
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
            async with aiofiles.open(temp_path, mode="w", encoding="utf-8") as f:
                await f.write("\n".join(lines))

            # Atomic rename
            if os.name == 'nt':  # Windows
                if self.memory_path.exists():
                    self.memory_path.unlink()
                temp_path.rename(self.memory_path)
            else:  # Unix-like
                temp_path.replace(self.memory_path)

        except Exception as e:
            print(f"Error saving graph: {e}")
            raise
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    print(f"Error cleaning up temp file: {e}")

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities with validation and batching."""
        try:
            async with self._write_lock:  # Use separate lock for writes
                graph = await self._check_cache()
                existing_entities = self._indices["entity_names"]
                new_entities = []

                for entity in entities:
                    if not entity.name or not entity.entityType:
                        raise ValueError(f"Invalid entity: {entity}")

                    if entity.name not in existing_entities:
                        new_entities.append(entity)
                        self._indices["entity_names"][entity.name] = entity
                        self._indices["entity_types"][entity.entityType].append(entity)

                if new_entities:
                    graph.entities.extend(new_entities)
                    self._dirty = True
                    await self._save_graph(graph)  # Save immediately in tests

                return new_entities
        except Exception as e:
            print(f"Error creating entities: {e}")
            raise

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations with validation and batching."""
        async with self._write_lock:  # Use separate lock for writes
            graph = await self._check_cache()
            existing_entities = self._indices["entity_names"]
            new_relations = []

            for relation in relations:
                if not relation.from_ or not relation.to or not relation.relationType:
                    raise ValueError(f"Invalid relation: {relation}")

                if relation.from_ not in existing_entities:
                    raise EntityNotFoundError(relation.from_)
                if relation.to not in existing_entities:
                    raise EntityNotFoundError(relation.to)

                if not any(
                    r.from_ == relation.from_
                    and r.to == relation.to
                    and r.relationType == relation.relationType
                    for r in self._indices["relations_from"].get(relation.from_, [])
                ):
                    new_relations.append(relation)
                    self._indices["relations_from"][relation.from_].append(relation)
                    self._indices["relations_to"][relation.to].append(relation)

            if new_relations:
                graph.relations.extend(new_relations)
                self._dirty = True
                await self._save_graph(graph)  # Save immediately in tests

            return new_relations

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph using cache."""
        return await self._check_cache()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        """Search for nodes in the knowledge graph using indices."""
        if not query:
            raise ValueError("Search query cannot be empty")

        graph = await self._check_cache()
        query = query.lower()
        filtered_entities = set()

        for entity in graph.entities:
            if query in entity.name.lower():
                filtered_entities.add(entity)
            elif query in entity.entityType.lower():
                filtered_entities.add(entity)
            elif any(query in obs.lower() for obs in entity.observations):
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
        """Force flush any pending writes to disk."""
        async with self._write_lock:
            if self._dirty:
                graph = await self._check_cache()
                await self._save_graph(graph)
                self._dirty = False
