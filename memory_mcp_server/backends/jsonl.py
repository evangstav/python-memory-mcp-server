import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import aiofiles
from thefuzz import fuzz

from ..exceptions import EntityNotFoundError, FileAccessError
from ..interfaces import (
    BatchOperation,
    BatchOperationType,
    BatchResult,
    Entity,
    KnowledgeGraph,
    Relation,
    SearchOptions,
)
from .base import Backend


@dataclass
class SearchResult:
    entity: Entity
    score: float


class ReentrantLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0

    async def acquire(self):
        current = asyncio.current_task()
        if self._owner == current:
            self._count += 1
            return
        await self._lock.acquire()
        self._owner = current
        self._count = 1

    def release(self):
        current = asyncio.current_task()
        if self._owner != current:
            raise RuntimeError("Lock not owned by current task")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, tb):
        self.release()


class JsonlBackend(Backend):
    def __init__(self, memory_path: Path, cache_ttl: int = 60):
        self.memory_path = memory_path
        self.cache_ttl = cache_ttl
        self._cache: Optional[KnowledgeGraph] = None
        self._cache_timestamp: float = 0.0
        self._cache_file_mtime: float = 0.0
        self._dirty = False
        self._write_lock = ReentrantLock()
        self._lock = asyncio.Lock()

        # Transaction support: when a transaction is active, we work on separate copies.
        self._transaction_cache: Optional[KnowledgeGraph] = None
        self._transaction_indices: Optional[Dict[str, Any]] = None
        self._in_transaction = False

        self._indices: Dict[str, Any] = {
            "entity_names": {},
            "entity_types": defaultdict(list),
            "relations_from": defaultdict(list),
            "relations_to": defaultdict(list),
            "relation_keys": set(),
            "observation_index": defaultdict(set),
        }

    async def initialize(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if self.memory_path.exists() and self.memory_path.is_dir():
            raise FileAccessError(f"Path {self.memory_path} is a directory")

    async def close(self) -> None:
        await self.flush()

    def _build_indices(self, graph: KnowledgeGraph) -> None:
        # Build indices for faster lookups.
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

        # Build the observation index.
        observation_index = cast(
            Dict[str, Set[str]], self._indices["observation_index"]
        )
        observation_index.clear()
        for entity in graph.entities:
            for obs in entity.observations:
                for word in obs.lower().split():
                    observation_index[word].add(entity.name)

    async def _check_cache(self) -> KnowledgeGraph:
        # During a transaction, always use the transaction snapshot.
        if self._in_transaction:
            return self._transaction_cache  # type: ignore

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
                    except FileAccessError:
                        raise
                    except Exception as e:
                        raise FileAccessError(f"Error loading graph: {str(e)}") from e

        return cast(KnowledgeGraph, self._cache)

    async def _load_graph_from_file(self) -> KnowledgeGraph:
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
                    except json.JSONDecodeError as e:
                        raise FileAccessError(f"Error loading graph: {str(e)}") from e
                    except KeyError as e:
                        raise FileAccessError(
                            f"Error loading graph: Missing required key {str(e)}"
                        ) from e
            return graph
        except Exception as err:
            raise FileAccessError(f"Error reading file: {str(err)}") from err

    async def _save_graph(self, graph: KnowledgeGraph) -> None:
        # This function writes to disk. Note that during a transaction, it is only called on commit.
        temp_path = self.memory_path.with_suffix(".tmp")
        buffer_size = 1000  # Buffer size (number of lines)
        try:
            async with aiofiles.open(temp_path, mode="w", encoding="utf-8") as f:
                buffer = []
                # Write entities.
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
                    if len(buffer) >= buffer_size:
                        await f.write("\n".join(buffer) + "\n")
                        buffer = []
                if buffer:
                    await f.write("\n".join(buffer) + "\n")
                    buffer = []

                # Write relations.
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
                    if len(buffer) >= buffer_size:
                        await f.write("\n".join(buffer) + "\n")
                        buffer = []
                if buffer:
                    await f.write("\n".join(buffer) + "\n")
            temp_path.replace(self.memory_path)
        except Exception as err:
            raise FileAccessError(f"Error saving file: {str(err)}") from err
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    async def _get_current_state(self) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        # Returns the active graph and indices. If a transaction is in progress,
        # return the transaction copies; otherwise, return the persistent ones.
        if self._in_transaction:
            return self._transaction_cache, self._transaction_indices  # type: ignore
        else:
            graph = await self._check_cache()
            return graph, self._indices

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        async with self._write_lock:
            graph, indices = await self._get_current_state()
            existing_entities = cast(Dict[str, Entity], indices["entity_names"])
            new_entities = []

            for entity in entities:
                if not entity.name or not entity.entityType:
                    raise ValueError(f"Invalid entity: {entity}")
                if entity.name not in existing_entities:
                    new_entities.append(entity)
                    existing_entities[entity.name] = entity
                    cast(Dict[str, List[Entity]], indices["entity_types"]).setdefault(
                        entity.entityType, []
                    ).append(entity)

            if new_entities:
                graph.entities.extend(new_entities)
                # If not in a transaction, immediately persist the change.
                if not self._in_transaction:
                    self._dirty = True
                    await self._save_graph(graph)
                    self._dirty = False
                    self._cache_timestamp = time.monotonic()

            return new_entities

    async def delete_entities(self, entity_names: List[str]) -> List[str]:
        if not entity_names:
            return []

        async with self._write_lock:
            graph, indices = await self._get_current_state()
            existing_entities = cast(Dict[str, Entity], indices["entity_names"])
            deleted_names = []
            relation_keys = cast(Set[Tuple[str, str, str]], indices["relation_keys"])

            for name in entity_names:
                if name in existing_entities:
                    entity = existing_entities.pop(name)
                    entity_type_list = cast(
                        Dict[str, List[Entity]], indices["entity_types"]
                    ).get(entity.entityType, [])
                    if entity in entity_type_list:
                        entity_type_list.remove(entity)

                    # Remove associated relations.
                    relations_from = cast(
                        Dict[str, List[Relation]], indices["relations_from"]
                    ).get(name, [])
                    relations_to = cast(
                        Dict[str, List[Relation]], indices["relations_to"]
                    ).get(name, [])
                    relations_to_remove = relations_from + relations_to

                    for relation in relations_to_remove:
                        if relation in graph.relations:
                            graph.relations.remove(relation)
                        relation_keys.discard(
                            (relation.from_, relation.to, relation.relationType)
                        )
                        if relation in cast(
                            Dict[str, List[Relation]], indices["relations_from"]
                        ).get(relation.from_, []):
                            cast(Dict[str, List[Relation]], indices["relations_from"])[
                                relation.from_
                            ].remove(relation)
                        if relation in cast(
                            Dict[str, List[Relation]], indices["relations_to"]
                        ).get(relation.to, []):
                            cast(Dict[str, List[Relation]], indices["relations_to"])[
                                relation.to
                            ].remove(relation)

                    deleted_names.append(name)

            if deleted_names:
                graph.entities = [
                    e for e in graph.entities if e.name not in deleted_names
                ]
                if not self._in_transaction:
                    self._dirty = True
                    await self._save_graph(graph)
                    self._dirty = False
                    self._cache_timestamp = time.monotonic()

            return deleted_names

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        async with self._write_lock:
            graph, indices = await self._get_current_state()
            existing_entities = cast(Dict[str, Entity], indices["entity_names"])
            relation_keys = cast(Set[Tuple[str, str, str]], indices["relation_keys"])
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
                    cast(
                        Dict[str, List[Relation]], indices["relations_from"]
                    ).setdefault(relation.from_, []).append(relation)
                    cast(Dict[str, List[Relation]], indices["relations_to"]).setdefault(
                        relation.to, []
                    ).append(relation)

            if new_relations:
                graph.relations.extend(new_relations)
                if not self._in_transaction:
                    self._dirty = True
                    await self._save_graph(graph)
                    self._dirty = False
                    self._cache_timestamp = time.monotonic()

            return new_relations

    async def delete_relations(self, from_: str, to: str) -> None:
        async with self._write_lock:
            graph, indices = await self._get_current_state()
            existing_entities = cast(Dict[str, Entity], indices["entity_names"])

            if from_ not in existing_entities:
                raise EntityNotFoundError(f"Entity not found: {from_}")
            if to not in existing_entities:
                raise EntityNotFoundError(f"Entity not found: {to}")

            relations_from = cast(
                Dict[str, List[Relation]], indices["relations_from"]
            ).get(from_, [])
            relations_to_remove = [rel for rel in relations_from if rel.to == to]

            if relations_to_remove:
                graph.relations = [
                    rel for rel in graph.relations if rel not in relations_to_remove
                ]
                relation_keys = cast(
                    Set[Tuple[str, str, str]], indices["relation_keys"]
                )
                for rel in relations_to_remove:
                    relation_keys.discard((rel.from_, rel.to, rel.relationType))
                    if rel in cast(
                        Dict[str, List[Relation]], indices["relations_from"]
                    ).get(from_, []):
                        cast(Dict[str, List[Relation]], indices["relations_from"])[
                            from_
                        ].remove(rel)
                    if rel in cast(
                        Dict[str, List[Relation]], indices["relations_to"]
                    ).get(to, []):
                        cast(Dict[str, List[Relation]], indices["relations_to"])[
                            to
                        ].remove(rel)
                if not self._in_transaction:
                    self._dirty = True
                    await self._save_graph(graph)
                    self._dirty = False
                    self._cache_timestamp = time.monotonic()

    async def read_graph(self) -> KnowledgeGraph:
        return await self._check_cache()

    async def flush(self) -> None:
        async with self._write_lock:
            # During a transaction, disk is not touched until commit.
            if self._dirty and not self._in_transaction:
                graph = await self._check_cache()
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

    async def search_nodes(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> KnowledgeGraph:
        """
        Search for entities and relations matching the query.
        If options is provided and options.fuzzy is True, fuzzy matching is used with weights and threshold.
        Otherwise, a simple case‐insensitive substring search is performed.
        Relations are returned only if both endpoints are in the set of matched entities.
        """
        graph = await self._check_cache()
        matched_entities = []
        if options is not None and options.fuzzy:
            # Use provided weights or default to 1.0 if not provided.
            weights = (
                options.weights
                if options.weights is not None
                else {"name": 1.0, "type": 1.0, "observations": 1.0}
            )
            q = query.strip()
            for entity in graph.entities:
                # Compute robust scores for each field.
                name_score = fuzz.WRatio(q, entity.name)
                type_score = fuzz.WRatio(q, entity.entityType)
                obs_score = 0
                if entity.observations:
                    # For each observation, take the best between WRatio and partial_ratio.
                    scores = [
                        max(fuzz.WRatio(q, obs), fuzz.partial_ratio(q, obs))
                        for obs in entity.observations
                    ]
                    obs_score = max(scores) if scores else 0

                total_score = (
                    name_score * weights.get("name", 1.0)
                    + type_score * weights.get("type", 1.0)
                    + obs_score * weights.get("observations", 1.0)
                )
                if total_score >= options.threshold:
                    matched_entities.append(entity)
        else:
            q = query.lower()
            for entity in graph.entities:
                if (
                    q in entity.name.lower()
                    or q in entity.entityType.lower()
                    or any(q in obs.lower() for obs in entity.observations)
                ):
                    matched_entities.append(entity)

        matched_names = {entity.name for entity in matched_entities}
        matched_relations = [
            rel
            for rel in graph.relations
            if rel.from_ in matched_names and rel.to in matched_names
        ]
        return KnowledgeGraph(entities=matched_entities, relations=matched_relations)

    async def add_observations(self, entity_name: str, observations: List[str]) -> None:
        if not observations:
            raise ValueError("Observations list cannot be empty")

        async with self._write_lock:
            graph, indices = await self._get_current_state()
            existing_entities = cast(Dict[str, Entity], indices["entity_names"])

            if entity_name not in existing_entities:
                raise EntityNotFoundError(f"Entity not found: {entity_name}")

            entity = existing_entities[entity_name]
            updated_entity = Entity(
                name=entity.name,
                entityType=entity.entityType,
                observations=list(entity.observations) + observations,
            )

            graph.entities = [
                updated_entity if e.name == entity_name else e for e in graph.entities
            ]
            existing_entities[entity_name] = updated_entity

            entity_types = cast(Dict[str, List[Entity]], indices["entity_types"])
            if entity_name in [
                e.name for e in entity_types.get(updated_entity.entityType, [])
            ]:
                entity_types[updated_entity.entityType] = [
                    updated_entity if e.name == entity_name else e
                    for e in entity_types[updated_entity.entityType]
                ]

            if not self._in_transaction:
                self._dirty = True
                await self._save_graph(graph)
                self._dirty = False
                self._cache_timestamp = time.monotonic()

    async def add_batch_observations(
        self, observations_map: Dict[str, List[str]]
    ) -> None:
        if not observations_map:
            raise ValueError("Observations map cannot be empty")

        async with self._write_lock:
            graph, indices = await self._get_current_state()
            existing_entities = cast(Dict[str, Entity], indices["entity_names"])
            entity_types = cast(Dict[str, List[Entity]], indices["entity_types"])

            missing_entities = [
                name for name in observations_map if name not in existing_entities
            ]
            if missing_entities:
                raise EntityNotFoundError(
                    f"Entities not found: {', '.join(missing_entities)}"
                )

            updated_entities = {}
            for entity_name, observations in observations_map.items():
                if not observations:
                    continue
                entity = existing_entities[entity_name]
                updated_entity = Entity(
                    name=entity.name,
                    entityType=entity.entityType,
                    observations=list(entity.observations) + observations,
                )
                updated_entities[entity_name] = updated_entity

            if updated_entities:
                graph.entities = [
                    updated_entities.get(e.name, e) for e in graph.entities
                ]
                for updated_entity in updated_entities.values():
                    existing_entities[updated_entity.name] = updated_entity
                    et_list = entity_types.get(updated_entity.entityType, [])
                    for i, e in enumerate(et_list):
                        if e.name == updated_entity.name:
                            et_list[i] = updated_entity
                            break
                if not self._in_transaction:
                    self._dirty = True
                    await self._save_graph(graph)
                    self._dirty = False
                    self._cache_timestamp = time.monotonic()

    #
    # Transaction Methods
    #
    async def begin_transaction(self) -> None:
        async with self._write_lock:
            if self._in_transaction:
                raise ValueError("Transaction already in progress")
            graph = await self._check_cache()
            # Make deep (shallow for immutable entities) copies of state.
            self._transaction_cache = KnowledgeGraph(
                entities=list(graph.entities), relations=list(graph.relations)
            )
            self._transaction_indices = {
                "entity_names": dict(self._indices["entity_names"]),
                "entity_types": defaultdict(
                    list, {k: list(v) for k, v in self._indices["entity_types"].items()}
                ),
                "relations_from": defaultdict(
                    list,
                    {k: list(v) for k, v in self._indices["relations_from"].items()},
                ),
                "relations_to": defaultdict(
                    list, {k: list(v) for k, v in self._indices["relations_to"].items()}
                ),
                "relation_keys": set(self._indices["relation_keys"]),
                "observation_index": defaultdict(
                    set,
                    {k: set(v) for k, v in self._indices["observation_index"].items()},
                ),
            }
            self._in_transaction = True

    async def rollback_transaction(self) -> None:
        async with self._write_lock:
            if not self._in_transaction:
                raise ValueError("No transaction in progress")
            # Discard the transaction state; since disk writes were deferred, the file remains unchanged.
            self._transaction_cache = None
            self._transaction_indices = None
            self._in_transaction = False

    async def commit_transaction(self) -> None:
        async with self._write_lock:
            if not self._in_transaction:
                raise ValueError("No transaction in progress")
            # Persist the transaction state to disk.
            await self._save_graph(cast(KnowledgeGraph, self._transaction_cache))
            # Update the persistent state with the transaction snapshot.
            self._cache = self._transaction_cache
            self._indices = self._transaction_indices  # type: ignore
            self._transaction_cache = None
            self._transaction_indices = None
            self._in_transaction = False
            self._dirty = False
            self._cache_timestamp = time.monotonic()

    async def execute_batch(self, operations: List[BatchOperation]) -> BatchResult:
        if not operations:
            return BatchResult(
                success=True,
                operations_completed=0,
                failed_operations=[],
            )

        async with self._write_lock:
            try:
                # Start a transaction so that no disk writes occur until commit.
                await self.begin_transaction()

                completed = 0
                failed_ops: List[Tuple[BatchOperation, str]] = []

                # Execute each operation.
                for operation in operations:
                    try:
                        if (
                            operation.operation_type
                            == BatchOperationType.CREATE_ENTITIES
                        ):
                            await self.create_entities(operation.data["entities"])
                        elif (
                            operation.operation_type
                            == BatchOperationType.DELETE_ENTITIES
                        ):
                            await self.delete_entities(operation.data["entity_names"])
                        elif (
                            operation.operation_type
                            == BatchOperationType.CREATE_RELATIONS
                        ):
                            await self.create_relations(operation.data["relations"])
                        elif (
                            operation.operation_type
                            == BatchOperationType.DELETE_RELATIONS
                        ):
                            await self.delete_relations(
                                operation.data["from_"], operation.data["to"]
                            )
                        elif (
                            operation.operation_type
                            == BatchOperationType.ADD_OBSERVATIONS
                        ):
                            await self.add_batch_observations(
                                operation.data["observations_map"]
                            )
                        else:
                            raise ValueError(
                                f"Unknown operation type: {operation.operation_type}"
                            )
                        completed += 1
                    except Exception as e:
                        failed_ops.append((operation, str(e)))
                        if not operation.data.get("allow_partial", False):
                            # On failure, rollback and return.
                            await self.rollback_transaction()
                            return BatchResult(
                                success=False,
                                operations_completed=completed,
                                failed_operations=failed_ops,
                                error_message=f"Operation failed: {str(e)}",
                            )

                # Commit the transaction (persisting all changes) or report partial success.
                await self.commit_transaction()
                if failed_ops:
                    return BatchResult(
                        success=True,
                        operations_completed=completed,
                        failed_operations=failed_ops,
                        error_message="Some operations failed",
                    )
                else:
                    return BatchResult(
                        success=True,
                        operations_completed=completed,
                        failed_operations=[],
                    )

            except Exception as e:
                if self._in_transaction:
                    await self.rollback_transaction()
                return BatchResult(
                    success=False,
                    operations_completed=0,
                    failed_operations=[],
                    error_message=f"Batch execution failed: {str(e)}",
                )
