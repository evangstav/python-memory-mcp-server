"""Knowledge graph manager that delegates to a configured backend."""

import asyncio
from pathlib import Path
from typing import List, Optional, Union

from .interfaces import Entity, Relation, KnowledgeGraph
from .backends.base import Backend
from .backends.jsonl import JsonlBackend


class KnowledgeGraphManager:
    """Manages knowledge graph operations through a configured backend."""

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
        """
        async with self._write_lock:
            return await self.backend.create_entities(entities)

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations.

        Args:
            relations: List of relations to create

        Returns:
            List of successfully created relations
        """
        async with self._write_lock:
            return await self.backend.create_relations(relations)

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph.

        Returns:
            Current state of the knowledge graph
        """
        return await self.backend.read_graph()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        """Search for entities and relations matching query.

        Args:
            query: Search query string

        Returns:
            KnowledgeGraph containing matches
        """
        return await self.backend.search_nodes(query)

    async def flush(self) -> None:
        """Ensure any pending changes are persisted."""
        await self.backend.flush()
