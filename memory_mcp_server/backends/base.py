"""Backend interface for Memory MCP Server storage implementations."""

from abc import ABC, abstractmethod
from typing import List

from ..interfaces import Entity, KnowledgeGraph, Relation, SearchOptions


class Backend(ABC):
    """Abstract base class for knowledge graph storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend connection and resources."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the backend connection and cleanup resources."""
        pass

    @abstractmethod
    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities in the backend.

        Args:
            entities: List of entities to create

        Returns:
            List of successfully created entities
        """
        pass

    @abstractmethod
    async def delete_entities(self, entity_names: List[str]) -> List[str]:
        """Create multiple new entities in the backend.

        Args:
            entities: List of entities to create

        Returns:
            List of successfully created entities
        """
        pass

    @abstractmethod
    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations in the backend.

        Args:
            relations: List of relations to create

        Returns:
            List of successfully created relations
        """
        pass

    @abstractmethod
    async def delete_relations(self, from_: str, to: str) -> None:
        """Delete relations between two entities.

        Args:
            from_: Source entity name
            to: Target entity name

        Raises:
            EntityNotFoundError: If either entity is not found
        """
        pass

    @abstractmethod
    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph from the backend.

        Returns:
            KnowledgeGraph containing all entities and relations
        """
        pass

    @abstractmethod
    async def search_nodes(
        self, query: str, options: SearchOptions = None
    ) -> KnowledgeGraph:
        """Search for entities and relations matching the query.

        Args:
            query: Search query string
            options: Optional SearchOptions for configuring search behavior.
                    If None, uses exact substring matching.

        Returns:
            KnowledgeGraph containing matching entities and relations

        Raises:
            ValueError: If query is empty or options are invalid
        """
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Ensure all pending changes are persisted to the backend."""
        pass

    @abstractmethod
    async def add_observations(self, entity_name: str, observations: List[str]) -> None:
        """Add observations to an existing entity.

        Args:
            entity_name: Name of the entity to add observations to
            observations: List of observations to add
        """
        pass
