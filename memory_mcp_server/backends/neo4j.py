"""Neo4j backend implementation for Memory MCP Server."""

import asyncio
from typing import List, Dict, Any
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError

from ..interfaces import Entity, Relation, KnowledgeGraph
from ..exceptions import EntityNotFoundError, FileAccessError
from .base import Backend


class Neo4jBackend(Backend):
    """Neo4j implementation of the knowledge graph backend."""

    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j backend.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: AsyncDriver | None = None
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Neo4j connection and create constraints."""
        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        
        # Create constraints for unique entity names
        async with self.driver.session() as session:
            await session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities in Neo4j.

        Args:
            entities: List of entities to create

        Returns:
            List of successfully created entities

        Raises:
            FileAccessError: If Neo4j operation fails
        """
        if not self.driver:
            raise FileAccessError("Neo4j connection not initialized")

        async with self._write_lock:
            try:
                async with self.driver.session() as session:
                    query = """
                    UNWIND $entities as entity
                    MERGE (e:Entity {name: entity.name})
                    ON CREATE SET 
                        e.entityType = entity.entityType,
                        e.created = timestamp()
                    WITH e, entity
                    UNWIND entity.observations as obs
                    CREATE (o:Observation {text: obs, created: timestamp()})
                    CREATE (e)-[:HAS_OBSERVATION]->(o)
                    RETURN e
                    """
                    result = await session.run(
                        query,
                        entities=[{
                            'name': e.name,
                            'entityType': e.entityType,
                            'observations': list(e.observations)
                        } for e in entities]
                    )
                    records = await result.values()
                    return entities
            except Neo4jError as e:
                raise FileAccessError(f"Neo4j error: {str(e)}")

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations in Neo4j.

        Args:
            relations: List of relations to create

        Returns:
            List of successfully created relations

        Raises:
            EntityNotFoundError: If referenced entities don't exist
            FileAccessError: If Neo4j operation fails
        """
        if not self.driver:
            raise FileAccessError("Neo4j connection not initialized")

        async with self._write_lock:
            try:
                async with self.driver.session() as session:
                    # First verify all entities exist
                    for relation in relations:
                        query = """
                        MATCH (e:Entity)
                        WHERE e.name IN [$from, $to]
                        RETURN e.name
                        """
                        result = await session.run(
                            query,
                            {"from": relation.from_, "to": relation.to}
                        )
                        existing = {record["e.name"] async for record in result}
                        if relation.from_ not in existing:
                            raise EntityNotFoundError(f"Entity not found: {relation.from_}")
                        if relation.to not in existing:
                            raise EntityNotFoundError(f"Entity not found: {relation.to}")

                    # Create relations
                    query = """
                    UNWIND $relations as rel
                    MATCH (from:Entity {name: rel.from})
                    MATCH (to:Entity {name: rel.to})
                    CREATE (from)-[r:RELATES {
                        type: rel.relationType,
                        created: timestamp()
                    }]->(to)
                    RETURN r
                    """
                    result = await session.run(
                        query,
                        relations=[r.to_dict() for r in relations]
                    )
                    await result.consume()
                    return relations
            except Neo4jError as e:
                raise FileAccessError(f"Neo4j error: {str(e)}")

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph from Neo4j.

        Returns:
            KnowledgeGraph containing all entities and relations

        Raises:
            FileAccessError: If Neo4j operation fails
        """
        if not self.driver:
            raise FileAccessError("Neo4j connection not initialized")

        try:
            async with self.driver.session() as session:
                # Get all entities with their observations
                entity_query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
                RETURN e.name, e.entityType, collect(o.text) as observations
                """
                result = await session.run(entity_query)
                entities = []
                async for record in result:
                    entities.append(Entity(
                        name=record["e.name"],
                        entityType=record["e.entityType"],
                        observations=record["observations"] or []
                    ))

                # Get all relations
                relation_query = """
                MATCH (from:Entity)-[r:RELATES]->(to:Entity)
                RETURN from.name, to.name, r.type
                """
                result = await session.run(relation_query)
                relations = []
                async for record in result:
                    relations.append(Relation(
                        from_=record["from.name"],
                        to=record["to.name"],
                        relationType=record["r.type"]
                    ))

                return KnowledgeGraph(entities=entities, relations=relations)
        except Neo4jError as e:
            raise FileAccessError(f"Neo4j error: {str(e)}")

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        """Search for entities and relations matching the query.

        Args:
            query: Search query string

        Returns:
            KnowledgeGraph containing matching entities and relations

        Raises:
            ValueError: If query is empty
            FileAccessError: If Neo4j operation fails
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        if not self.driver:
            raise FileAccessError("Neo4j connection not initialized")

        try:
            async with self.driver.session() as session:
                # Search entities and their observations
                entity_query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
                WHERE toLower(e.name) CONTAINS toLower($query)
                   OR toLower(e.entityType) CONTAINS toLower($query)
                   OR any(obs IN collect(o.text) WHERE toLower(obs) CONTAINS toLower($query))
                RETURN e.name, e.entityType, collect(o.text) as observations
                """
                result = await session.run(entity_query, {"query": query})
                entities = []
                entity_names = set()
                async for record in result:
                    entity_names.add(record["e.name"])
                    entities.append(Entity(
                        name=record["e.name"],
                        entityType=record["e.entityType"],
                        observations=record["observations"] or []
                    ))

                # Get relations between matching entities
                relation_query = """
                MATCH (from:Entity)-[r:RELATES]->(to:Entity)
                WHERE from.name IN $names AND to.name IN $names
                RETURN from.name, to.name, r.type
                """
                result = await session.run(
                    relation_query,
                    {"names": list(entity_names)}
                )
                relations = []
                async for record in result:
                    relations.append(Relation(
                        from_=record["from.name"],
                        to=record["to.name"],
                        relationType=record["r.type"]
                    ))

                return KnowledgeGraph(entities=entities, relations=relations)
        except Neo4jError as e:
            raise FileAccessError(f"Neo4j error: {str(e)}")

    async def flush(self) -> None:
        """Ensure all pending changes are persisted.
        
        Neo4j transactions are atomic, so this is a no-op.
        """
        pass