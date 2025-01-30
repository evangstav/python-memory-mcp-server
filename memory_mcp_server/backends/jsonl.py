"""JSONL backend implementation for Memory MCP Server."""

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
from ..interfaces import Entity, KnowledgeGraph, Relation, SearchOptions
from .base import Backend


@dataclass
class SearchResult:
    """Internal class for tracking search results with scores."""

    entity: Entity
    score: float


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

        self._indices: Dict[str, Any] = {
            "entity_names": {},
            "entity_types": defaultdict(list),
            "relations_from": defaultdict(list),
            "relations_to": defaultdict(list),
            "relation_keys": set(),
            "observation_index": defaultdict(set),
        }

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
                    except FileAccessError:
                        # Propagate file access errors
                        raise
                    except Exception as e:
                        # Convert unexpected errors to FileAccessError
                        raise FileAccessError(f"Error loading graph: {str(e)}") from e

        return cast(KnowledgeGraph, self._cache)

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
        """Save the knowledge graph to JSONL file atomically with buffered writes.

        Args:
            graph: KnowledgeGraph to save

        Raises:
            FileAccessError: If file cannot be written
        """
        temp_path = self.memory_path.with_suffix(".tmp")
        buffer_size = 1000  # Number of lines to buffer before writing

        try:
            async with aiofiles.open(temp_path, mode="w", encoding="utf-8") as f:
                buffer = []
                # Write entities
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
                # Write remaining entities
                if buffer:
                    await f.write("\n".join(buffer) + "\n")
                    buffer = []

                # Write relations
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
                # Write remaining relations
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

    def _calculate_fuzzy_score(
        self, query: str, entity: Entity, weights: Dict[str, float]
    ) -> float:
        """Calculate fuzzy match score for an entity.

        Args:
            query: Search query string
            entity: Entity to score
            weights: Field weights for scoring

        Returns:
            Weighted average score (0-100)
        """
        name_weight = weights.get("name", 0)
        type_weight = weights.get("type", 0)
        obs_weight = weights.get("observations", 0)
        total_weight = name_weight + type_weight + obs_weight

        if total_weight == 0:
            return 0.0

        query = query.lower()

        # Calculate name score using multiple ratios for better matching
        name_score = 0
        if name_weight > 0:
            # Combine different fuzzy ratios for better name matching
            token_sort = fuzz.token_sort_ratio(query, entity.name.lower())
            token_set = fuzz.token_set_ratio(query, entity.name.lower())
            partial = fuzz.partial_ratio(query, entity.name.lower())
            name_score = max(token_sort, token_set, partial)

        # Calculate type score using standard ratio
        type_score = 0
        if type_weight > 0:
            type_score = fuzz.ratio(query, entity.entityType.lower())

        # Calculate observation score using multiple matching strategies
        obs_score = 0
        if obs_weight > 0 and entity.observations:
            obs_scores = []
            for obs in entity.observations:
                obs_lower = obs.lower()
                # Try different matching strategies
                token_sort = fuzz.token_sort_ratio(query, obs_lower)
                token_set = fuzz.token_set_ratio(query, obs_lower)
                partial = fuzz.partial_ratio(query, obs_lower)
                # Take best match for this observation
                obs_scores.append(max(token_sort, token_set, partial))
            obs_score = max(obs_scores) if obs_scores else 0

        # If any individual score exceeds threshold, boost the final score
        max_score = (
            max(
                name_score * (name_weight / total_weight if total_weight > 0 else 0),
                type_score * (type_weight / total_weight if total_weight > 0 else 0),
                obs_score * (obs_weight / total_weight if total_weight > 0 else 0),
            )
            * total_weight
        )

        return max_score

    async def search_nodes(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> KnowledgeGraph:
        """Search for entities and relations matching query.

        Args:
            query: Search query string
            options: Optional search configuration

        Returns:
            KnowledgeGraph containing matches

        Raises:
            ValueError: If query is empty
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        graph = await self._check_cache()

        # Use exact matching if no options provided or fuzzy is False
        if not options or not options.fuzzy:
            q = query.lower()
            filtered_entities = set()

            for entity in graph.entities:
                if (
                    q in entity.name.lower()
                    or q in entity.entityType.lower()
                    or any(q in obs.lower() for obs in entity.observations)
                ):
                    filtered_entities.add(entity)
        else:
            # Fuzzy search with scoring
            search_results: List[SearchResult] = []

            for entity in graph.entities:
                # Get weights with defaults for no weights provided
                weights = options.weights if options.weights else {"name": 1.0}
                name_weight = weights.get("name", 0)
                type_weight = weights.get("type", 0)
                obs_weight = weights.get("observations", 0)
                total_weight = name_weight + type_weight + obs_weight

                if total_weight == 0:
                    continue

                query_lower = query.lower()
                query_words = set(query_lower.split())

                # Calculate scores for each field
                name_score = 0
                if name_weight > 0:
                    name_lower = entity.name.lower()
                    name_words = set(name_lower.split())

                    # Calculate word-level similarity
                    word_matches = sum(
                        1
                        for w in query_words
                        if any(fuzz.ratio(w, nw) >= 85 for nw in name_words)
                    )
                    word_match_ratio = (
                        word_matches / len(query_words) if query_words else 0
                    )

                    # Calculate token-based similarity
                    token_score = fuzz.token_sort_ratio(query_lower, name_lower)

                    # Combine scores based on threshold
                    if options.threshold >= 90:
                        # For high threshold, require high word match ratio
                        name_score = token_score if word_match_ratio > 0.9 else 0
                    else:
                        # For lower threshold, use weighted combination
                        name_score = (token_score * 0.6) + (
                            word_match_ratio * 100 * 0.4
                        )

                type_score = 0
                if type_weight > 0:
                    type_score = fuzz.ratio(query_lower, entity.entityType.lower())

                # Special handling for observation-weighted search
                is_obs_primary = obs_weight > name_weight and obs_weight > type_weight

                # Calculate observation score
                obs_score = 0
                if obs_weight > 0 and entity.observations:
                    obs_scores = []
                    for obs in entity.observations:
                        obs_lower = obs.lower()
                        # Calculate token-based similarity
                        token_score = fuzz.token_set_ratio(query_lower, obs_lower)

                        # Check for word presence
                        word_presence = sum(1 for w in query_words if w in obs_lower)
                        word_ratio = (
                            word_presence / len(query_words) if query_words else 0
                        )

                        # Combine scores based on search type
                        if is_obs_primary:
                            # For observation-weighted search, prioritize word presence
                            if word_ratio > 0:
                                # Give high score if any words match
                                obs_scores.append(max(90, token_score))
                            else:
                                # Still consider token matching
                                obs_scores.append(token_score)
                        else:
                            # Balance token matching (70%) with word presence (30%)
                            token_weight = token_score * 0.7
                            word_weight = word_ratio * 100 * 0.3
                            combined_score = token_weight + word_weight
                            obs_scores.append(combined_score)

                    obs_score = max(obs_scores) if obs_scores else 0

                # For observation-weighted search, also consider name matches
                if is_obs_primary and name_score >= options.threshold:
                    # If name matches threshold, treat as observation match
                    obs_score = max(obs_score, name_score)

                # Calculate final scores
                weighted_name = (
                    (name_score * name_weight) / total_weight if total_weight > 0 else 0
                )
                weighted_type = (
                    (type_score * type_weight) / total_weight if total_weight > 0 else 0
                )
                weighted_obs = (
                    (obs_score * obs_weight) / total_weight if total_weight > 0 else 0
                )

                # Calculate max weight to determine primary field
                max_weight = max(name_weight, type_weight, obs_weight)

                # For name-weighted search (default or explicit)
                if name_weight == max_weight:
                    # Use stricter name matching
                    qualifies = name_score >= options.threshold

                # For observation-weighted search
                elif obs_weight == max_weight:
                    # More lenient matching for observations
                    qualifies = (
                        # Strong observation match
                        obs_score >= options.threshold * 0.8
                        or
                        # Exact word match in name
                        (
                            name_score >= options.threshold
                            and any(w in entity.name.lower() for w in query_words)
                        )
                    )

                # For type-weighted search
                elif type_weight == max_weight:
                    qualifies = type_score >= options.threshold

                # No primary field (should not happen with defaults)
                else:
                    qualifies = False

                if qualifies:
                    final_score = (weighted_name + weighted_type + weighted_obs) * 100
                    search_results.append(
                        SearchResult(entity=entity, score=final_score)
                    )

            # Sort by score descending
            search_results.sort(key=lambda x: x.score, reverse=True)
            filtered_entities = {result.entity for result in search_results}

        # Filter relations to only include those between matched entities
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
