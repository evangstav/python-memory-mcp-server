"""JSONL backend implementation for Memory MCP Server."""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import aiofiles

from ..exceptions import EntityNotFoundError, FileAccessError
from ..interfaces import Entity, KnowledgeGraph, Relation
from .base import Backend


class ChangeEvent:
    def __init__(
        self,
        event_type: str,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
    ):
        self.event_type = event_type  # 'entities_created', 'entities_deleted', etc.
        self.entities = entities or []
        self.relations = relations or []


class JsonlBackend(Backend):
    """JSONL file-based implementation of the knowledge graph backend."""

    def __init__(self, memory_path: Path, cache_ttl: int = 60):
        self.memory_path = memory_path
        self.cache_ttl = cache_ttl
        self._cache: Optional[KnowledgeGraph] = None
        self._cache_timestamp: float = 0.0
        self._cache_file_mtime: float = 0.0
        self._dirty = False
        self._write_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._listeners: List[Callable[[ChangeEvent], None]] = []

        self._indices: Dict[str, Any] = {
            "entity_names": {},
            "entity_types": defaultdict(list),
            "relations_from": defaultdict(list),
            "relations_to": defaultdict(list),
            "relation_keys": set(),
            "observation_index": defaultdict(set),
        }

    def register_listener(self, callback: Callable[[ChangeEvent], None]):
        """Add a change listener (non-breaking addition)"""
        self._listeners.append(callback)

    def _notify_listeners(self, event: ChangeEvent):
        """Notify all registered listeners"""
        for callback in self._listeners:
            try:
                callback(event)
            except Exception:
                pass

    async def initialize(self) -> None:
        """Initialize the backend. Creates parent directory and validates path."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if self.memory_path.exists() and self.memory_path.is_dir():
            raise FileAccessError(f"Path {self.memory_path} is a directory")

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
        relation_keys: Set[Tuple[str, str, str]] = set()

        for entity in graph.entities:
            entity_names[entity.name] = entity
            entity_types[entity.entityType].append(entity)

        for relation in graph.relations:
            relations_from[relation.from_].append(relation)
            relations_to[relation.to].append(relation)
            relation_keys.add((relation.from_, relation.to, relation.relationType))

        self._indices["entity_names"] = entity_names
        self._indices["entity_types"] = entity_types
        self._indices["relations_from"] = relations_from
        self._indices["relations_to"] = relations_to
        self._indices["relation_keys"] = relation_keys

        # Add observation index
        observation_index = cast(
            Dict[str, Set[str]], self._indices["observation_index"]
        )
        observation_index.clear()

        for entity in graph.entities:
            for obs in entity.observations:
                for word in obs.lower().split():
                    observation_index[word].add(entity.name)

    async def _check_cache(self) -> KnowledgeGraph:
        """Check if cache needs refresh and reload if necessary.

        Returns:
            Current KnowledgeGraph from cache or file
        """
        current_time = time.monotonic()
        file_mtime = (
            self.memory_path.stat().st_mtime if self.memory_path.exists() else 0
        )
        needs_refresh = (
            self._cache is None
            or (current_time - self._cache_timestamp > self.cache_ttl)
            or self._dirty
            or (file_mtime > self._cache_file_mtime)
        )

        if needs_refresh:
            async with self._lock:
                current_time = time.monotonic()
                file_mtime = (
                    self.memory_path.stat().st_mtime if self.memory_path.exists() else 0
                )
                needs_refresh = (
                    self._cache is None
                    or (current_time - self._cache_timestamp > self.cache_ttl)
                    or self._dirty
                    or (file_mtime > self._cache_file_mtime)
                )
                if needs_refresh:
                    try:
                        graph = await self._load_graph_from_file()
                        self._cache = graph
                        self._cache_timestamp = current_time
                        self._cache_file_mtime = file_mtime
                        self._build_indices(graph)
                        self._dirty = False
                    except Exception:
                        empty_graph = KnowledgeGraph(entities=[], relations=[])
                        return empty_graph

        return cast(KnowledgeGraph, self._cache)

    async def _process_line(
        self, graph: KnowledgeGraph, line: str, hasher: hashlib._Hash = None
    ) -> None:
        """Process a single line from the file"""
        line = line.strip()
        if not line:
            return

        try:
            item = json.loads(line)
            if hasher:
                hasher.update(line.encode())

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
        except (json.JSONDecodeError, KeyError):
            pass

    async def _process_legacy_file(
        self,
        graph: KnowledgeGraph,
        first_line: str,
        f: aiofiles.threadpool.AsyncTextIOWrapper,
        hasher: hashlib._Hash,
    ) -> None:
        """Handle files without checksum header"""
        await self._process_line(graph, first_line, hasher)
        async for line in f:
            await self._process_line(graph, line, hasher)

    async def _load_graph_from_file(self) -> KnowledgeGraph:
        """Load with checksum validation (backward compatible)"""
        if not self.memory_path.exists():
            return KnowledgeGraph(entities=[], relations=[])

        graph = KnowledgeGraph(entities=[], relations=[])
        hasher = hashlib.sha256()
        expected_checksum = None

        try:
            async with aiofiles.open(self.memory_path, mode="r", encoding="utf-8") as f:
                first_line = await f.readline()
                try:
                    header = json.loads(first_line.strip())
                    expected_checksum = header.get("checksum")
                except json.JSONDecodeError:
                    # Handle files without checksum header
                    await self._process_legacy_file(graph, first_line, f, hasher)
                    return graph

                async for line in f:
                    hasher.update(line.encode())
                    await self._process_line(graph, line)

            if expected_checksum and expected_checksum != "pending":
                actual_checksum = hasher.hexdigest()
                if actual_checksum != expected_checksum:
                    raise FileAccessError("Data corruption detected: checksum mismatch")

            return graph
        except Exception as err:
            raise FileAccessError(f"Error reading file: {str(err)}") from err

    async def _save_graph(self, graph: KnowledgeGraph) -> None:
        """Save with checksum validation (backward compatible)"""
        temp_path = self.memory_path.with_suffix(".tmp")
        buffer_size = 1000
        hasher = hashlib.sha256()

        try:
            async with aiofiles.open(temp_path, mode="w+", encoding="utf-8") as f:
                # Write placeholder header
                await f.write(json.dumps({"checksum": "pending"}) + "\n")

                # Write content and compute checksum
                buffer = []
                for entity in graph.entities:
                    line = json.dumps(
                        {
                            "type": "entity",
                            "name": entity.name,
                            "entityType": entity.entityType,
                            "observations": entity.observations,
                        }
                    )
                    buffer.append(line)
                    hasher.update(line.encode())
                    if len(buffer) >= buffer_size:
                        await f.write("\n".join(buffer) + "\n")
                        buffer = []

                if buffer:
                    await f.write("\n".join(buffer) + "\n")
                    buffer = []

                for relation in graph.relations:
                    line = json.dumps(
                        {
                            "type": "relation",
                            "from": relation.from_,
                            "to": relation.to,
                            "relationType": relation.relationType,
                        }
                    )
                    buffer.append(line)
                    hasher.update(line.encode())
                    if len(buffer) >= buffer_size:
                        await f.write("\n".join(buffer) + "\n")
                        buffer = []

                if buffer:
                    await f.write("\n".join(buffer) + "\n")

                # Write actual checksum
                checksum = hasher.hexdigest()
                await f.seek(0)
                await f.write(json.dumps({"checksum": checksum}) + "\n")

            temp_path.replace(self.memory_path)
        except Exception as err:
            raise FileAccessError(f"Error saving file: {str(err)}") from err
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities with enhanced duplicate check."""
        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = cast(Dict[str, Entity], self._indices["entity_names"])
            new_entities = []

            for entity in entities:
                if not entity.name or not entity.entityType:
                    raise ValueError(f"Invalid entity: {entity}")
                if entity.name not in existing_entities:
                    new_entities.append(entity)
                    existing_entities[entity.name] = entity
                    cast(Dict[str, List[Entity]], self._indices["entity_types"])[
                        entity.entityType
                    ].append(entity)

            if new_entities:
                graph.entities.extend(new_entities)
                self._dirty = True
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

            return new_entities

    async def delete_entities(self, entity_names: List[str]) -> List[str]:
        """Delete entities with optimized relation cleanup."""
        if not entity_names:
            return []

        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = cast(Dict[str, Entity], self._indices["entity_names"])
            deleted_names = []
            relation_keys = cast(
                Set[Tuple[str, str, str]], self._indices["relation_keys"]
            )

            for name in entity_names:
                if name in existing_entities:
                    entity = existing_entities.pop(name)
                    entity_type_list = cast(
                        Dict[str, List[Entity]], self._indices["entity_types"]
                    )[entity.entityType]
                    entity_type_list.remove(entity)

                    # Remove associated relations
                    relations_from = cast(
                        Dict[str, List[Relation]], self._indices["relations_from"]
                    ).get(name, [])
                    relations_to = cast(
                        Dict[str, List[Relation]], self._indices["relations_to"]
                    ).get(name, [])
                    relations_to_remove = relations_from + relations_to

                    for relation in relations_to_remove:
                        graph.relations.remove(relation)
                        # Remove from relation indices
                        relation_keys.discard(
                            (relation.from_, relation.to, relation.relationType)
                        )
                        cast(
                            Dict[str, List[Relation]], self._indices["relations_from"]
                        )[relation.from_].remove(relation)
                        cast(Dict[str, List[Relation]], self._indices["relations_to"])[
                            relation.to
                        ].remove(relation)

                    deleted_names.append(name)

            if deleted_names:
                graph.entities = [
                    e for e in graph.entities if e.name not in deleted_names
                ]
                self._dirty = True
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

            return deleted_names

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create relations with optimized duplicate checking."""
        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = cast(Dict[str, Entity], self._indices["entity_names"])
            relation_keys = cast(
                Set[Tuple[str, str, str]], self._indices["relation_keys"]
            )
            new_relations = []

            for relation in relations:
                if not relation.from_ or not relation.to or not relation.relationType:
                    raise ValueError(f"Invalid relation: {relation}")

                if relation.from_ not in existing_entities:
                    raise EntityNotFoundError(f"Entity not found: {relation.from_}")
                if relation.to not in existing_entities:
                    raise EntityNotFoundError(f"Entity not found: {relation.to}")

                key = (relation.from_, relation.to, relation.relationType)
                if key not in relation_keys:
                    new_relations.append(relation)
                    relation_keys.add(key)
                    cast(Dict[str, List[Relation]], self._indices["relations_from"])[
                        relation.from_
                    ].append(relation)
                    cast(Dict[str, List[Relation]], self._indices["relations_to"])[
                        relation.to
                    ].append(relation)

            if new_relations:
                graph.relations.extend(new_relations)
                self._dirty = True
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

            return new_relations

    async def delete_relations(self, from_: str, to: str) -> None:
        """Delete relations between two entities.

        Args:
            from_: Source entity name
            to: Target entity name

        Raises:
            EntityNotFoundError: If either entity doesn't exist
        """
        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = cast(Dict[str, Entity], self._indices["entity_names"])

            # Validate entities exist
            if from_ not in existing_entities:
                raise EntityNotFoundError(f"Entity not found: {from_}")
            if to not in existing_entities:
                raise EntityNotFoundError(f"Entity not found: {to}")

            # Get relations to remove from indices
            relations_from = cast(
                Dict[str, List[Relation]], self._indices["relations_from"]
            ).get(from_, [])
            relations_to_remove = [rel for rel in relations_from if rel.to == to]

            if relations_to_remove:
                # Remove from graph
                graph.relations = [
                    rel for rel in graph.relations if rel not in relations_to_remove
                ]

                # Update indices
                relation_keys = cast(
                    Set[Tuple[str, str, str]], self._indices["relation_keys"]
                )
                for rel in relations_to_remove:
                    relation_keys.discard((rel.from_, rel.to, rel.relationType))
                    cast(Dict[str, List[Relation]], self._indices["relations_from"])[
                        from_
                    ].remove(rel)
                    cast(Dict[str, List[Relation]], self._indices["relations_to"])[
                        to
                    ].remove(rel)

                self._dirty = True
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

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
            entities=list(filtered_entities), relations=filtered_relations
        )

    async def flush(self) -> None:
        """Ensure any pending changes are saved to disk."""
        async with self._write_lock:
            if self._dirty:
                graph = await self._check_cache()
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

    async def add_observations(self, entity_name: str, observations: List[str]) -> None:
        """Add observations to an existing entity.

        Args:
            entity_name: Name of the entity to add observations to
            observations: List of observations to add

        Raises:
            EntityNotFoundError: If entity does not exist
            ValueError: If observations list is empty
            FileAccessError: If file operations fail
        """
        if not observations:
            raise ValueError("Observations list cannot be empty")

        async with self._write_lock:
            graph = await self._check_cache()
            existing_entities = cast(Dict[str, Entity], self._indices["entity_names"])

            if entity_name not in existing_entities:
                raise EntityNotFoundError(f"Entity not found: {entity_name}")

            entity = existing_entities[entity_name]
            # Create new Entity with updated observations since Entity is immutable
            updated_entity = Entity(
                name=entity.name,
                entityType=entity.entityType,
                observations=list(entity.observations) + observations,
            )

            # Update entity in graph and indices
            graph.entities = [
                updated_entity if e.name == entity_name else e for e in graph.entities
            ]
            existing_entities[entity_name] = updated_entity

            # Update entity type index
            entity_types = cast(Dict[str, List[Entity]], self._indices["entity_types"])
            entity_types[updated_entity.entityType] = [
                updated_entity if e.name == entity_name else e
                for e in entity_types[updated_entity.entityType]
            ]

            self._dirty = True
            await self._save_graph(graph)
            self._dirty = False
            self._cache_timestamp = time.monotonic()
