"""Knowledge graph manager that delegates to a configured backend."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union

from .backends.base import Backend
from .backends.jsonl import JsonlBackend
from .interfaces import Entity, KnowledgeGraph, Relation, SearchOptions
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
                entity_text = f"{entity.name} {entity.entityType} " + " ".join(entity.observations)
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
                    e for e in graph.entities
                    if any(et in e.entityType.lower() for et in entity_types)
                ]

            # Sort by recency if available (assuming observations contain timestamps)
            # This is a simplification - would need actual timestamp extraction
            if query_analysis.temporal_reference == "recent":
                filtered_entities.sort(
                    key=lambda e: max((obs for obs in e.observations if "date" in obs.lower()),
                                    default=""),
                    reverse=True
                )
            elif query_analysis.temporal_reference == "past":
                filtered_entities.sort(
                    key=lambda e: max((obs for obs in e.observations if "date" in obs.lower()),
                                    default="")
                )

            results = filtered_entities[:max_results]

        elif query_analysis.query_type == QueryType.ENTITY:
            # Entity type specific search
            entity_types = query_analysis.additional_params.get("entity_types", [])
            if entity_types:
                results = [
                    e for e in graph.entities 
                    if any(et in e.entityType.lower() for et in entity_types)
                ]

                # If we have too many results, use semantic search to narrow down
                if len(results) > max_results:
                    results = await self._semantic_search(query, results, max_results)
            else:
                # General semantic search
                results = await self._semantic_search(query, graph.entities, max_results)

        elif query_analysis.query_type == QueryType.RELATION:
            # Relation-focused search
            # First find entities that match the query semantically
            candidate_entities = await self._semantic_search(query, graph.entities, max_results * 2)
            entity_names = {entity.name for entity in candidate_entities}
            
            # Find relations between these entities
            matched_relations = [
                rel for rel in graph.relations
                if rel.from_ in entity_names and rel.to in entity_names
            ]
            
            # Get all entities involved in these relations
            relation_entity_names = {rel.from_ for rel in matched_relations}.union(
                {rel.to for rel in matched_relations}
            )
            
            results = [e for e in candidate_entities if e.name in relation_entity_names]
            
            # If we don't have enough results, add more from the candidates
            if len(results) < max_results:
                additional = [e for e in candidate_entities if e.name not in relation_entity_names]
                results.extend(additional[:max_results - len(results)])

        else:
            # General semantic search
            results = await self._semantic_search(query, graph.entities, max_results)

        # Get relations between the matched entities if requested
        matched_relations = []
        if options and options.include_relations:
            entity_names = {entity.name for entity in results}
            matched_relations = [
                rel for rel in graph.relations
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
            entity_text = f"{entity.name} {entity.entityType} " + " ".join(entity.observations)
            entity_vectors.append(entity_text)
            entity_map[i] = entity

        # Get vector embeddings for all entities
        if entity_vectors:
            embeddings = self.embedding_service.encode_batch(entity_vectors)

            # Compute similarities
            similarities = self.embedding_service.compute_similarity(query_vector, embeddings)

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
            updated_entity = next((e for e in (await self.read_graph()).entities if e.name == entity_name), None)
            if updated_entity:
                # Create a combined text representation of the updated entity
                entity_text = f"{updated_entity.name} {updated_entity.entityType} " + " ".join(updated_entity.observations)
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
                    entity_text = f"{entity.name} {entity.entityType} " + " ".join(entity.observations)
                    # Generate new embedding
                    embedding = self.embedding_service.encode_text(entity_text)
                    # Store updated embedding
                    await self.backend.store_embedding(name, embedding)

    async def enhanced_search(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> KnowledgeGraph:
        """Enhanced search implementation using semantic similarity and query understanding."""
        # Analyze the query
        query_analysis = self.query_analyzer.analyze_query(query)

        # Get the base graph
        graph = await self.read_graph()

        # Initialize results
        results = []

        # Handle different query types
        if query_analysis.query_type == QueryType.TEMPORAL:
            # Temporal queries (most recent, etc.)
            filtered_entities = [
                e
                for e in graph.entities
                if query_analysis.target_entity in e.entityType.lower()
            ]

            # Sort by recency if available (assuming observations contain timestamps)
            # This is a simplification - would need actual timestamp extraction
            filtered_entities.sort(
                key=lambda e: max(
                    (obs for obs in e.observations if "date" in obs.lower()), default=""
                ),
                reverse=True,
            )

            results = filtered_entities[:5]  # Return top 5 most recent

        elif query_analysis.query_type == QueryType.ENTITY:
            # Entity type specific search
            entity_types = query_analysis.additional_params.get("entity_types", [])
            if entity_types:
                results = [e for e in graph.entities if e.entityType in entity_types]

                # If we have too many results, use semantic search to narrow down
                if len(results) > 10:
                    results = await self._semantic_search(query, results)

        else:
            # General semantic search
            results = await self._semantic_search(query, graph.entities)

        # Get relations between the matched entities
        entity_names = {entity.name for entity in results}
        matched_relations = [
            rel
            for rel in graph.relations
            if rel.from_ in entity_names and rel.to in entity_names
        ]

        return KnowledgeGraph(entities=results, relations=matched_relations)

    async def _semantic_search(
        self, query: str, entities: List[Entity]
    ) -> List[Entity]:
        """Perform semantic search using embeddings."""
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
            top_indices = np.argsort(similarities)[::-1][:10]  # Top 10 results

            # Return entities in order of relevance
            return [entity_map[idx] for idx in top_indices]

        return []
