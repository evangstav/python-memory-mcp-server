"""Knowledge graph manager that delegates to a configured backend."""

import asyncio
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .backends.base import Backend
from .backends.jsonl import JsonlBackend
from .interfaces import Entity, KnowledgeGraph, Relation, SearchOptions
from .models.conversation_context import ConversationContext
from .services.embedding_service import EmbeddingService
from .services.query_analyzer import QueryAnalyzer, QueryType
from .validation import KnowledgeGraphValidator, ValidationError


class KnowledgeGraphManager:
    """Manages knowledge graph operations through a configured backend."""

    backend: Backend
    _write_lock: asyncio.Lock

    def __init__(
        self,
        backend: Union[Backend, Path],
        cache_ttl: int = 60,
    ):
        """Initialize the KnowledgeGraphManager.

        Args:
            backend: Either a Backend instance or Path to use default JSONL backend
            cache_ttl: Cache TTL in seconds (only used for JSONL backend)
        """
        if isinstance(backend, Path):
            self.backend = JsonlBackend(backend, cache_ttl)
        else:
            self.backend = backend
        self._write_lock = asyncio.Lock()

        # Initialize services for enhanced search capabilities
        self.embedding_service = EmbeddingService()
        self.query_analyzer = QueryAnalyzer()

    async def initialize(self) -> None:
        """Initialize the backend connection."""
        await self.backend.initialize()

    async def close(self) -> None:
        """Close the backend connection."""
        await self.backend.close()

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities.

        Args:
            entities: List of entities to create

        Returns:
            List of successfully created entities

        Raises:
            ValidationError: If any entity fails validation
        """
        # Get existing entities for validation
        graph = await self.read_graph()
        existing_names = {entity.name for entity in graph.entities}

        # Validate all entities in one pass
        KnowledgeGraphValidator.validate_batch_entities(entities, existing_names)

        async with self._write_lock:
            created_entities = await self.backend.create_entities(entities)

            # Generate and store embeddings for new entities
            for entity in created_entities:
                # Create a combined text representation of the entity
                entity_text = f"{entity.name} {entity.entityType} " + " ".join(
                    entity.observations
                )
                # Generate embedding
                embedding = self.embedding_service.encode_text(entity_text)
                # Store embedding
                await self.backend.store_embedding(entity.name, embedding)

            return created_entities

    async def delete_entities(self, entity_names: List[str]) -> List[str]:
        """Delete multiple existing entities by name.

        Args:
            entity_names: List of entity names to delete

        Returns:
            List of successfully deleted entity names

        Raises:
            ValueError: If entity_names list is empty
            EntityNotFoundError: If any entity is not found in the graph
            FileAccessError: If there are file system issues (backend specific)
        """
        if not entity_names:
            raise ValueError("Entity names list cannot be empty")

        async with self._write_lock:
            return await self.backend.delete_entities(entity_names)

    async def delete_relations(self, from_: str, to: str) -> None:
        """Delete relations between two entities.

        Args:
            from_: Source entity name
            to: Target entity name

        Raises:
            EntityNotFoundError: If either entity is not found
        """
        async with self._write_lock:
            return await self.backend.delete_relations(from_, to)

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations.

        Args:
            relations: List of relations to create

        Returns:
            List of successfully created relations

        Raises:
            ValidationError: If any relation fails validation
            EntityNotFoundError: If referenced entities don't exist
        """
        # Get existing graph for validation
        graph = await self.read_graph()
        existing_names = {entity.name for entity in graph.entities}

        # Validate all relations in one pass
        KnowledgeGraphValidator.validate_batch_relations(
            relations, graph.relations, existing_names
        )

        async with self._write_lock:
            return await self.backend.create_relations(relations)

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph.

        Returns:
            Current state of the knowledge graph
        """
        return await self.backend.read_graph()

    async def search_nodes(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> KnowledgeGraph:
        """Search for entities and relations matching query.

        Args:
            query: Search query string
            options: Optional SearchOptions for configuring search behavior.
                    If None, uses exact substring matching.

        Returns:
            KnowledgeGraph containing matches

        Raises:
            ValueError: If query is empty or options are invalid
        """
        # If semantic search is requested, use enhanced search
        if options and options.semantic:
            return await self.enhanced_search(query, options)

        # Otherwise use the standard search
        return await self.backend.search_nodes(query, options)

    async def flush(self) -> None:
        """Ensure any pending changes are persisted."""
        await self.backend.flush()

    async def enhanced_search(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> KnowledgeGraph:
        """Enhanced search implementation using semantic similarity and query understanding.

        Args:
            query: Search query string
            options: Optional SearchOptions for configuring search behavior

        Returns:
            KnowledgeGraph containing semantically relevant matches
        """
        # Analyze the query
        query_analysis = self.query_analyzer.analyze_query(query)

        # Get the base graph
        graph = await self.read_graph()

        # Initialize results
        results = []
        max_results = options.max_results if options and options.max_results else 10

        # Handle different query types
        if query_analysis.query_type == QueryType.TEMPORAL:
            # Temporal queries (most recent, etc.)
            entity_types = query_analysis.additional_params.get("entity_types", [])

            filtered_entities = graph.entities
            if entity_types:
                filtered_entities = [
                    e
                    for e in graph.entities
                    if any(et in e.entityType.lower() for et in entity_types)
                ]

            # Sort by recency if available (assuming observations contain timestamps)
            # This is a simplification - would need actual timestamp extraction
            if query_analysis.temporal_reference == "recent":
                filtered_entities.sort(
                    key=lambda e: max(
                        (obs for obs in e.observations if "date" in obs.lower()),
                        default="",
                    ),
                    reverse=True,
                )
            elif query_analysis.temporal_reference == "past":
                filtered_entities.sort(
                    key=lambda e: max(
                        (obs for obs in e.observations if "date" in obs.lower()),
                        default="",
                    )
                )

            results = filtered_entities[:max_results]

        elif query_analysis.query_type == QueryType.ENTITY:
            # Entity type specific search
            entity_types = query_analysis.additional_params.get("entity_types", [])
            if entity_types:
                results = [
                    e
                    for e in graph.entities
                    if any(et in e.entityType.lower() for et in entity_types)
                ]

                # If we have too many results, use semantic search to narrow down
                if len(results) > max_results:
                    results = await self._semantic_search(query, results, max_results)
            else:
                # General semantic search
                results = await self._semantic_search(
                    query, graph.entities, max_results
                )

        elif query_analysis.query_type == QueryType.RELATION:
            # Relation-focused search
            # First find entities that match the query semantically
            candidate_entities = await self._semantic_search(
                query, graph.entities, max_results * 2
            )
            entity_names = {entity.name for entity in candidate_entities}

            # Find relations between these entities
            matched_relations = [
                rel
                for rel in graph.relations
                if rel.from_ in entity_names and rel.to in entity_names
            ]

            # Get all entities involved in these relations
            relation_entity_names = {rel.from_ for rel in matched_relations}.union(
                {rel.to for rel in matched_relations}
            )

            results = [e for e in candidate_entities if e.name in relation_entity_names]

            # If we don't have enough results, add more from the candidates
            if len(results) < max_results:
                additional = [
                    e for e in candidate_entities if e.name not in relation_entity_names
                ]
                results.extend(additional[: max_results - len(results)])

        else:
            # General semantic search
            results = await self._semantic_search(query, graph.entities, max_results)

        # Get relations between the matched entities if requested
        matched_relations = []
        if options and options.include_relations:
            entity_names = {entity.name for entity in results}
            matched_relations = [
                rel
                for rel in graph.relations
                if rel.from_ in entity_names and rel.to in entity_names
            ]

        return KnowledgeGraph(entities=results, relations=matched_relations)

    async def _semantic_search(
        self, query: str, entities: List[Entity], max_results: int = 10
    ) -> List[Entity]:
        """Perform semantic search using embeddings.

        Args:
            query: Search query string
            entities: List of entities to search through
            max_results: Maximum number of results to return

        Returns:
            List of entities sorted by semantic relevance
        """
        if not entities:
            return []

        # Encode the query
        query_vector = self.embedding_service.encode_text(query)

        # Get embeddings for entities
        entity_vectors = []
        entity_map = {}

        for i, entity in enumerate(entities):
            # Create a combined text representation of the entity
            entity_text = f"{entity.name} {entity.entityType} " + " ".join(
                entity.observations
            )
            entity_vectors.append(entity_text)
            entity_map[i] = entity

        # Get vector embeddings for all entities
        if entity_vectors:
            embeddings = self.embedding_service.encode_batch(entity_vectors)

            # Compute similarities
            similarities = self.embedding_service.compute_similarity(
                query_vector, embeddings
            )

            # Get top results (indices sorted by similarity)
            top_indices = np.argsort(similarities)[::-1][:max_results]  # Top N results

            # Return entities in order of relevance
            return [entity_map[idx] for idx in top_indices]

        return []

    async def add_observations(self, entity_name: str, observations: List[str]) -> None:
        """Add observations to an existing entity.

        Args:
            entity_name: Name of the entity to add observations to
            observations: List of observations to add

        Raises:
            EntityNotFoundError: If the entity is not found
            ValidationError: If observations are invalid
            ValueError: If observations list is empty
        """
        if not observations:
            raise ValueError("Observations list cannot be empty")

        # Validate new observations
        KnowledgeGraphValidator.validate_observations(observations)

        # Get existing entity to check for duplicate observations
        graph = await self.read_graph()
        entity = next((e for e in graph.entities if e.name == entity_name), None)
        if not entity:
            raise ValidationError(f"Entity not found: {entity_name}")

        # Check for duplicates against existing observations
        existing_observations = set(entity.observations)
        duplicates = [obs for obs in observations if obs in existing_observations]
        if duplicates:
            raise ValidationError(f"Duplicate observations: {', '.join(duplicates)}")

        async with self._write_lock:
            await self.backend.add_observations(entity_name, observations)

            # Update embedding after adding observations
            updated_entity = next(
                (
                    e
                    for e in (await self.read_graph()).entities
                    if e.name == entity_name
                ),
                None,
            )
            if updated_entity:
                # Create a combined text representation of the updated entity
                entity_text = (
                    f"{updated_entity.name} {updated_entity.entityType} "
                    + " ".join(updated_entity.observations)
                )
                # Generate new embedding
                embedding = self.embedding_service.encode_text(entity_text)
                # Store updated embedding
                await self.backend.store_embedding(entity_name, embedding)

    async def add_batch_observations(
        self, observations_map: Dict[str, List[str]]
    ) -> None:
        """Add observations to multiple entities in a single operation.

        Args:
            observations_map: Dictionary mapping entity names to lists of observations

        Raises:
            ValidationError: If any observations are invalid
            EntityNotFoundError: If any entity is not found
            ValueError: If observations_map is empty
        """
        # Get existing graph for validation
        graph = await self.read_graph()
        entities_map = {entity.name: entity for entity in graph.entities}

        # Validate all observations in one pass
        KnowledgeGraphValidator.validate_batch_observations(
            observations_map, entities_map
        )

        # All validation passed, perform the batch update
        async with self._write_lock:
            await self.backend.add_batch_observations(observations_map)

            # Update embeddings for all modified entities
            updated_graph = await self.read_graph()
            updated_entities = {
                name: next((e for e in updated_graph.entities if e.name == name), None)
                for name in observations_map.keys()
            }

            for name, entity in updated_entities.items():
                if entity:
                    # Create a combined text representation of the updated entity
                    entity_text = f"{entity.name} {entity.entityType} " + " ".join(
                        entity.observations
                    )
                    # Generate new embedding
                    embedding = self.embedding_service.encode_text(entity_text)
                    # Store updated embedding
                    await self.backend.store_embedding(name, embedding)

    async def update_conversation_context(
        self,
        current_topic: str,
        entities_mentioned: List[str],
        summary: str,
        importance: float = 1.0,
    ) -> str:
        """Update the conversation context with current information.

        Args:
            current_topic: The main topic of the current conversation segment
            entities_mentioned: List of entity names mentioned in this segment
            summary: Brief summary of this conversation segment
            importance: Importance score (0.0-1.0) with higher values indicating
                        more significant context

        Returns:
            ID of the created context entity

        Raises:
            ValidationError: If parameters are invalid
            EntityNotFoundError: If mentioned entities don't exist
        """
        # Generate unique ID for this context point with microsecond precision
        # and a random component to ensure uniqueness
        timestamp = int(time.time())
        random_component = random.randint(1000, 9999)
        context_id = f"context-{timestamp}-{random_component}"

        # Create context entity
        context = ConversationContext(
            id=context_id,
            timestamp=time.time(),
            topic=current_topic,
            entities_mentioned=entities_mentioned,
            importance=importance,
            summary=summary,
        )

        # Create entity from context
        entity = Entity(
            name=context_id,
            entityType="conversation_context",
            observations=context.to_entity_observations(),
        )

        # Verify that all mentioned entities exist
        graph = await self.read_graph()
        existing_names = {entity.name for entity in graph.entities}
        missing_entities = [
            name for name in entities_mentioned if name not in existing_names
        ]
        if missing_entities:
            raise ValidationError(
                f"Referenced entities not found: {', '.join(missing_entities)}"
            )

        # Store entity
        await self.create_entities([entity])

        # Create relations to mentioned entities
        relations = []
        for entity_name in entities_mentioned:
            relations.append(
                Relation(from_=context_id, to=entity_name, relationType="mentions")
            )

        # Create the relations if there are any
        if relations:
            await self.create_relations(relations)

        return context_id

    async def get_relevant_context(
        self,
        current_entities: List[str],
        lookback_hours: float = 24.0,
        max_results: int = 5,
    ) -> List[Dict]:
        """Get relevant conversation context with time-based decay.

        Returns contexts ordered by a relevance score that combines:
        - Recency (exponential decay based on time)
        - Entity overlap (matching entities with current context)
        - Importance (manually set importance value)

        Args:
            current_entities: List of entity names in the current context
            lookback_hours: How many hours to look back for context
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing context information and relevance scores

        Raises:
            ValueError: If parameters are invalid
        """
        if not current_entities:
            raise ValueError("Current entities list cannot be empty")

        if lookback_hours <= 0:
            raise ValueError("Lookback hours must be positive")

        if max_results <= 0:
            raise ValueError("Max results must be positive")

        now = time.time()
        max_age = lookback_hours * 3600

        # Get contexts from recent history
        options = SearchOptions(
            entity_type="conversation_context",
            fuzzy=True,
            semantic=False,
            max_results=20,  # Get more than needed to filter
        )
        results = await self.search_nodes("conversation_context", options)

        # Calculate relevance scores
        contexts = []
        for entity in results.entities:
            if entity.entityType != "conversation_context":
                continue

            # Extract data from observations
            timestamp = None
            importance = 1.0
            entities_mentioned = []
            topic = ""
            summary = ""

            for obs in entity.observations:
                if obs.startswith("Time:"):
                    time_str = obs.replace("Time:", "").strip()
                    try:
                        dt = datetime.fromisoformat(time_str)
                        timestamp = dt.timestamp()
                    except Exception:
                        pass
                elif obs.startswith("Importance:"):
                    try:
                        importance = float(obs.replace("Importance:", "").strip())
                    except Exception:
                        pass
                elif obs.startswith("Entities:"):
                    entities_str = obs.replace("Entities:", "").strip()
                    if entities_str:
                        entities_mentioned = [
                            e.strip() for e in entities_str.split(",")
                        ]
                elif obs.startswith("Topic:"):
                    topic = obs.replace("Topic:", "").strip()
                elif obs.startswith("Summary:"):
                    summary = obs.replace("Summary:", "").strip()

            if timestamp is None:
                continue

            # Skip if too old
            age = now - timestamp
            if age > max_age:
                continue

            # Calculate time decay (exponential)
            time_factor = math.exp(-age / (max_age / 3))

            # Calculate entity overlap
            overlap = len(set(entities_mentioned) & set(current_entities))
            overlap_factor = overlap / max(1, len(entities_mentioned))

            # Combined relevance score
            relevance = (
                (0.5 * time_factor) + (0.3 * overlap_factor) + (0.2 * importance)
            )

            # Add to results
            contexts.append(
                {
                    "id": entity.name,
                    "relevance": relevance,
                    "age_hours": age / 3600,
                    "timestamp": timestamp,
                    "topic": topic,
                    "summary": summary,
                    "entities_mentioned": entities_mentioned,
                    "importance": importance,
                }
            )

        # Sort by relevance and return top results
        contexts.sort(key=lambda x: x["relevance"], reverse=True)
        return contexts[:max_results]
