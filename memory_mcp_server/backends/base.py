"""Backend interface for Memory MCP Server storage implementations."""

from abc import ABC, abstractmethod
from typing import List

from ..interfaces import (
    BatchOperation,
    BatchResult,
    Entity,
    KnowledgeGraph,
    Relation,
    SearchOptions,
)


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

    @abstractmethod
    async def add_batch_observations(
        self, observations_map: dict[str, List[str]]
    ) -> None:
        """Add observations to multiple entities in a single operation.

        Args:
            observations_map: Dictionary mapping entity names to lists of observations

        Raises:
            ValidationError: If any observations are invalid
            EntityNotFoundError: If any entity is not found
        """
        pass

    @abstractmethod
    async def execute_batch(self, operations: List[BatchOperation]) -> BatchResult:
        """Execute multiple operations in a single atomic batch.

        Args:
            operations: List of operations to execute

        Returns:
            BatchResult containing success/failure information

        Raises:
            ValidationError: If validation fails for any operation
        """
        pass

    @abstractmethod
    async def begin_transaction(self) -> None:
        """Begin a transaction for batch operations.

        This creates a savepoint that can be rolled back to if needed.
        """
        pass

    @abstractmethod
    async def rollback_transaction(self) -> None:
        """Rollback to the last transaction savepoint."""
        pass

    @abstractmethod
    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        pass
