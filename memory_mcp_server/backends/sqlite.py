"""SQLite-based backend implementation for Memory MCP Server."""

import asyncio
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite
import numpy as np
from loguru import logger

from ..benchmarking.profiler import profiler
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


class SQLiteBackend(Backend):
    """SQLite-based backend implementation."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._write_lock = asyncio.Lock()
        self._transaction_lock = asyncio.Lock()
        self._in_transaction = False

    @profiler.track("sqlite_backend.initialize")
    async def initialize(self) -> None:
        """Initialize the database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Enable foreign keys
                await db.execute("PRAGMA foreign_keys = ON")

                # Create tables
                await db.execute(
                    """
                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT PRIMARY KEY,
                    entityType TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                )

                await db.execute(
                    """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_name) REFERENCES entities(name) ON DELETE CASCADE,
                    UNIQUE (entity_name, content)
                )
                """
                )

                await db.execute(
                    """
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_entity TEXT NOT NULL,
                    to_entity TEXT NOT NULL,
                    relationType TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (from_entity) REFERENCES entities(name) ON DELETE CASCADE,
                    FOREIGN KEY (to_entity) REFERENCES entities(name) ON DELETE CASCADE,
                    UNIQUE (from_entity, to_entity, relationType)
                )
                """
                )

                # Create indices
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entityType)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_observation_entity ON observations(entity_name)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relation_from ON relations(from_entity)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relation_to ON relations(to_entity)"
                )

                # Add FTS for observations
                await db.execute(
                    """
                CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
                    content, entity_name,
                    content=observations, content_rowid=id
                )
                """
                )

                # Create triggers to keep FTS in sync
                await db.execute(
                    """
                CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
                    INSERT INTO observations_fts(rowid, content, entity_name)
                    VALUES (new.id, new.content, new.entity_name);
                END
                """
                )

                await db.execute(
                    """
                CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
                    INSERT INTO observations_fts(observations_fts, rowid, content, entity_name)
                    VALUES ('delete', old.id, old.content, old.entity_name);
                END
                """
                )

                await db.execute(
                    """
                CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
                    INSERT INTO observations_fts(observations_fts, rowid, content, entity_name)
                    VALUES ('delete', old.id, old.content, old.entity_name);
                    INSERT INTO observations_fts(rowid, content, entity_name)
                    VALUES (new.id, new.content, new.entity_name);
                END
                """
                )

                # Create embeddings table
                await db.execute(
                    """
                CREATE TABLE IF NOT EXISTS embeddings (
                    entity_name TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_name) REFERENCES entities(name) ON DELETE CASCADE
                )
                """
                )

                await db.commit()
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {str(e)}")
            raise FileAccessError(
                f"Error initializing SQLite database: {str(e)}"
            ) from e

    async def close(self) -> None:
        """Close the database connection."""
        # SQLite connections are created as needed
        pass

    @profiler.track("sqlite_backend.create_entities")
    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple entities in the database."""
        if not entities:
            return []

        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")

                    # Use a transaction for all operations
                    await db.execute("BEGIN TRANSACTION")

                    created_entities = []

                    for entity in entities:
                        try:
                            # Insert entity
                            await db.execute(
                                "INSERT INTO entities (name, entityType) VALUES (?, ?)",
                                (entity.name, entity.entityType),
                            )

                            # Insert observations
                            for obs in entity.observations:
                                await db.execute(
                                    "INSERT INTO observations (entity_name, content) VALUES (?, ?)",
                                    (entity.name, obs),
                                )

                            created_entities.append(entity)
                        except sqlite3.IntegrityError:
                            # Entity already exists
                            pass

                    await db.commit()
                    return created_entities
            except Exception as e:
                logger.error(f"Error creating entities: {str(e)}")
                raise FileAccessError(f"Error creating entities: {str(e)}") from e

    @profiler.track("sqlite_backend.delete_entities")
    async def delete_entities(self, entity_names: List[str]) -> List[str]:
        """Delete multiple entities from the database."""
        if not entity_names:
            return []

        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")
                    await db.execute("BEGIN TRANSACTION")

                    deleted_names = []

                    for name in entity_names:
                        # Check if entity exists
                        async with db.execute(
                            "SELECT name FROM entities WHERE name = ?", (name,)
                        ) as cursor:
                            if await cursor.fetchone():
                                # Delete entity (cascade will handle observations and relations)
                                await db.execute(
                                    "DELETE FROM entities WHERE name = ?", (name,)
                                )
                                deleted_names.append(name)

                    await db.commit()
                    return deleted_names
            except Exception as e:
                logger.error(f"Error deleting entities: {str(e)}")
                raise FileAccessError(f"Error deleting entities: {str(e)}") from e

    @profiler.track("sqlite_backend.create_relations")
    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple relations in the database."""
        if not relations:
            return []

        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")
                    await db.execute("BEGIN TRANSACTION")

                    created_relations = []

                    for relation in relations:
                        try:
                            # Insert relation
                            await db.execute(
                                """
                                INSERT INTO relations (from_entity, to_entity, relationType)
                                VALUES (?, ?, ?)
                                """,
                                (relation.from_, relation.to, relation.relationType),
                            )
                            created_relations.append(relation)
                        except sqlite3.IntegrityError as e:
                            # Check if it's a foreign key constraint or duplicate
                            if "FOREIGN KEY constraint failed" in str(e):
                                # Entity doesn't exist
                                pass
                            else:
                                # Relation already exists
                                pass

                    await db.commit()
                    return created_relations
            except Exception as e:
                logger.error(f"Error creating relations: {str(e)}")
                raise FileAccessError(f"Error creating relations: {str(e)}") from e

    @profiler.track("sqlite_backend.delete_relations")
    async def delete_relations(self, from_: str, to: str) -> None:
        """Delete relations between two entities."""
        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")

                    # Check if entities exist
                    async with db.execute(
                        "SELECT name FROM entities WHERE name = ?", (from_,)
                    ) as cursor:
                        if not await cursor.fetchone():
                            raise EntityNotFoundError(f"Entity not found: {from_}")

                    async with db.execute(
                        "SELECT name FROM entities WHERE name = ?", (to,)
                    ) as cursor:
                        if not await cursor.fetchone():
                            raise EntityNotFoundError(f"Entity not found: {to}")

                    # Delete relations
                    await db.execute(
                        "DELETE FROM relations WHERE from_entity = ? AND to_entity = ?",
                        (from_, to),
                    )
                    await db.commit()
            except EntityNotFoundError:
                raise
            except Exception as e:
                logger.error(f"Error deleting relations: {str(e)}")
                raise FileAccessError(f"Error deleting relations: {str(e)}") from e

    @profiler.track("sqlite_backend.read_graph")
    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = sqlite3.Row

                # Get all entities
                entities = []
                entity_map = {}

                async with db.execute(
                    "SELECT name, entityType FROM entities"
                ) as cursor:
                    async for row in cursor:
                        entity = Entity(
                            name=row["name"],
                            entityType=row["entityType"],
                            observations=[],
                        )
                        entities.append(entity)
                        entity_map[entity.name] = entity

                # Get all observations
                async with db.execute(
                    "SELECT entity_name, content FROM observations ORDER BY entity_name"
                ) as cursor:
                    async for row in cursor:
                        entity_name = row["entity_name"]
                        if entity_name in entity_map:
                            entity_map[entity_name].observations.append(row["content"])

                # Get all relations
                relations = []

                async with db.execute(
                    "SELECT from_entity, to_entity, relationType FROM relations"
                ) as cursor:
                    async for row in cursor:
                        relations.append(
                            Relation(
                                from_=row["from_entity"],
                                to=row["to_entity"],
                                relationType=row["relationType"],
                            )
                        )

                return KnowledgeGraph(entities=entities, relations=relations)
        except Exception as e:
            logger.error(f"Error reading graph: {str(e)}")
            raise FileAccessError(f"Error reading graph: {str(e)}") from e

    @profiler.track("sqlite_backend.search_nodes")
    async def search_nodes(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> KnowledgeGraph:
        """Search for entities and relations."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = sqlite3.Row

                # Use options to determine search strategy
                if options and options.fuzzy:
                    # Fuzzy search using SQLite FTS5
                    search_query = f"{query}*"  # Add wildcard for prefix matching

                    # Search in observations using FTS
                    async with db.execute(
                        """
                        SELECT DISTINCT entity_name
                        FROM observations_fts
                        WHERE observations_fts MATCH ?
                        ORDER BY rank
                        """,
                        (search_query,),
                    ) as cursor:
                        entity_names = [row["entity_name"] async for row in cursor]

                    # Also search in entity names and types
                    async with db.execute(
                        """
                        SELECT name
                        FROM entities
                        WHERE name LIKE ? OR entityType LIKE ?
                        """,
                        (f"%{query}%", f"%{query}%"),
                    ) as cursor:
                        entity_names_direct = [row["name"] async for row in cursor]

                    # Combine results
                    all_names = set(entity_names + entity_names_direct)

                    # If no matches, return empty result
                    if not all_names:
                        return KnowledgeGraph(entities=[], relations=[])

                    # Fetch matching entities
                    placeholders = ",".join(["?"] * len(all_names))

                    entities = []
                    entity_map = {}

                    async with db.execute(
                        f"SELECT name, entityType FROM entities WHERE name IN ({placeholders})",
                        tuple(all_names),
                    ) as cursor:
                        async for row in cursor:
                            entity = Entity(
                                name=row["name"],
                                entityType=row["entityType"],
                                observations=[],
                            )
                            entities.append(entity)
                            entity_map[entity.name] = entity

                    # Get observations for matching entities
                    async with db.execute(
                        f"SELECT entity_name, content FROM observations WHERE entity_name IN ({placeholders})",
                        tuple(all_names),
                    ) as cursor:
                        async for row in cursor:
                            entity_name = row["entity_name"]
                            if entity_name in entity_map:
                                entity_map[entity_name].observations.append(
                                    row["content"]
                                )

                    # Get relations between matching entities
                    relations = []

                    async with db.execute(
                        f"""
                        SELECT from_entity, to_entity, relationType
                        FROM relations
                        WHERE from_entity IN ({placeholders})
                        AND to_entity IN ({placeholders})
                        """,
                        tuple(all_names) + tuple(all_names),
                    ) as cursor:
                        async for row in cursor:
                            relations.append(
                                Relation(
                                    from_=row["from_entity"],
                                    to=row["to_entity"],
                                    relationType=row["relationType"],
                                )
                            )

                    return KnowledgeGraph(entities=entities, relations=relations)
                else:
                    # Simple substring search
                    like_pattern = f"%{query}%"

                    # Search in entity names, types, and observations
                    async with db.execute(
                        """
                        SELECT DISTINCT e.name
                        FROM entities e
                        LEFT JOIN observations o ON e.name = o.entity_name
                        WHERE e.name LIKE ?
                        OR e.entityType LIKE ?
                        OR o.content LIKE ?
                        """,
                        (like_pattern, like_pattern, like_pattern),
                    ) as cursor:
                        entity_names = [row["name"] async for row in cursor]

                    # If no matches, return empty result
                    if not entity_names:
                        return KnowledgeGraph(entities=[], relations=[])

                    # Fetch matching entities
                    placeholders = ",".join(["?"] * len(entity_names))

                    entities = []
                    entity_map = {}

                    async with db.execute(
                        f"SELECT name, entityType FROM entities WHERE name IN ({placeholders})",
                        tuple(entity_names),
                    ) as cursor:
                        async for row in cursor:
                            entity = Entity(
                                name=row["name"],
                                entityType=row["entityType"],
                                observations=[],
                            )
                            entities.append(entity)
                            entity_map[entity.name] = entity

                    # Get observations for matching entities
                    async with db.execute(
                        f"SELECT entity_name, content FROM observations WHERE entity_name IN ({placeholders})",
                        tuple(entity_names),
                    ) as cursor:
                        async for row in cursor:
                            entity_name = row["entity_name"]
                            if entity_name in entity_map:
                                entity_map[entity_name].observations.append(
                                    row["content"]
                                )

                    # Get relations between matching entities
                    relations = []

                    async with db.execute(
                        f"""
                        SELECT from_entity, to_entity, relationType
                        FROM relations
                        WHERE from_entity IN ({placeholders})
                        AND to_entity IN ({placeholders})
                        """,
                        tuple(entity_names) + tuple(entity_names),
                    ) as cursor:
                        async for row in cursor:
                            relations.append(
                                Relation(
                                    from_=row["from_entity"],
                                    to=row["to_entity"],
                                    relationType=row["relationType"],
                                )
                            )

                    return KnowledgeGraph(entities=entities, relations=relations)

        except Exception as e:
            logger.error(f"Error searching nodes: {str(e)}")
            raise FileAccessError(f"Error searching nodes: {str(e)}") from e

    @profiler.track("sqlite_backend.add_observations")
    async def add_observations(self, entity_name: str, observations: List[str]) -> None:
        """Add observations to an entity."""
        if not observations:
            return

        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")

                    # Check if entity exists
                    async with db.execute(
                        "SELECT name FROM entities WHERE name = ?", (entity_name,)
                    ) as cursor:
                        if not await cursor.fetchone():
                            raise EntityNotFoundError(
                                f"Entity not found: {entity_name}"
                            )

                    # Add observations
                    await db.execute("BEGIN TRANSACTION")

                    for obs in observations:
                        try:
                            await db.execute(
                                "INSERT INTO observations (entity_name, content) VALUES (?, ?)",
                                (entity_name, obs),
                            )
                        except sqlite3.IntegrityError:
                            # Observation already exists (unique constraint)
                            pass

                    await db.commit()
            except EntityNotFoundError:
                raise
            except Exception as e:
                logger.error(f"Error adding observations: {str(e)}")
                raise FileAccessError(f"Error adding observations: {str(e)}") from e

    @profiler.track("sqlite_backend.add_batch_observations")
    async def add_batch_observations(
        self, observations_map: Dict[str, List[str]]
    ) -> None:
        """Add observations to multiple entities in a batch."""
        if not observations_map:
            return

        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")

                    # Check if all entities exist
                    entity_names = list(observations_map.keys())
                    placeholders = ",".join(["?"] * len(entity_names))

                    existing_entities = set()
                    async with db.execute(
                        f"SELECT name FROM entities WHERE name IN ({placeholders})",
                        tuple(entity_names),
                    ) as cursor:
                        async for row in cursor:
                            existing_entities.add(row["name"])

                    missing_entities = set(entity_names) - existing_entities
                    if missing_entities:
                        raise EntityNotFoundError(
                            f"Entities not found: {', '.join(missing_entities)}"
                        )

                    # Add observations in a transaction
                    await db.execute("BEGIN TRANSACTION")

                    for entity_name, observations in observations_map.items():
                        for obs in observations:
                            try:
                                await db.execute(
                                    "INSERT INTO observations (entity_name, content) VALUES (?, ?)",
                                    (entity_name, obs),
                                )
                            except sqlite3.IntegrityError:
                                # Observation already exists
                                pass

                    await db.commit()
            except EntityNotFoundError:
                raise
            except Exception as e:
                logger.error(f"Error adding batch observations: {str(e)}")
                raise FileAccessError(
                    f"Error adding batch observations: {str(e)}"
                ) from e

    async def flush(self) -> None:
        """Ensure all changes are persisted."""
        # SQLite commits after each transaction
        pass

    @profiler.track("sqlite_backend.store_embedding")
    async def store_embedding(self, entity_name: str, vector: np.ndarray) -> None:
        """Store embedding vector for an entity."""
        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")

                    # Check if entity exists
                    async with db.execute(
                        "SELECT name FROM entities WHERE name = ?", (entity_name,)
                    ) as cursor:
                        if not await cursor.fetchone():
                            raise EntityNotFoundError(
                                f"Entity not found: {entity_name}"
                            )

                    # Serialize the vector to binary
                    vector_blob = pickle.dumps(vector)

                    # Insert or replace the embedding
                    await db.execute(
                        """
                    INSERT OR REPLACE INTO embeddings (entity_name, vector, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                        (entity_name, vector_blob),
                    )

                    await db.commit()
            except EntityNotFoundError:
                raise
            except Exception as e:
                logger.error(f"Error storing embedding: {str(e)}")
                raise FileAccessError(f"Error storing embedding: {str(e)}") from e

    @profiler.track("sqlite_backend.get_embedding")
    async def get_embedding(self, entity_name: str) -> Optional[np.ndarray]:
        """Get embedding vector for an entity."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")

                # Get the embedding
                async with db.execute(
                    "SELECT vector FROM embeddings WHERE entity_name = ?",
                    (entity_name,),
                ) as cursor:
                    row = await cursor.fetchone()

                    if row:
                        # Deserialize the vector from binary
                        return pickle.loads(row[0])
                    return None
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise FileAccessError(f"Error getting embedding: {str(e)}") from e

    # Transaction methods
    async def begin_transaction(self) -> None:
        """Begin a transaction for batch operations."""
        async with self._transaction_lock:
            if self._in_transaction:
                raise ValueError("Transaction already in progress")
            self._in_transaction = True

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        async with self._transaction_lock:
            if not self._in_transaction:
                raise ValueError("No transaction in progress")
            self._in_transaction = False

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        async with self._transaction_lock:
            if not self._in_transaction:
                raise ValueError("No transaction in progress")
            self._in_transaction = False

    @profiler.track("sqlite_backend.execute_batch")
    async def execute_batch(self, operations: List[BatchOperation]) -> BatchResult:
        """Execute multiple operations in a single atomic batch."""
        if not operations:
            return BatchResult(
                success=True,
                operations_completed=0,
                failed_operations=[],
            )

        async with self._write_lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")
                    await db.execute("BEGIN TRANSACTION")

                    completed = 0
                    failed_ops: List[Tuple[BatchOperation, str]] = []

                    for operation in operations:
                        try:
                            if (
                                operation.operation_type
                                == BatchOperationType.CREATE_ENTITIES
                            ):
                                entities = operation.data["entities"]
                                for entity in entities:
                                    # Insert entity
                                    await db.execute(
                                        "INSERT INTO entities (name, entityType) VALUES (?, ?)",
                                        (entity.name, entity.entityType),
                                    )

                                    # Insert observations
                                    for obs in entity.observations:
                                        await db.execute(
                                            "INSERT INTO observations (entity_name, content) VALUES (?, ?)",
                                            (entity.name, obs),
                                        )

                            elif (
                                operation.operation_type
                                == BatchOperationType.DELETE_ENTITIES
                            ):
                                entity_names = operation.data["entity_names"]
                                for name in entity_names:
                                    await db.execute(
                                        "DELETE FROM entities WHERE name = ?", (name,)
                                    )

                            elif (
                                operation.operation_type
                                == BatchOperationType.CREATE_RELATIONS
                            ):
                                relations = operation.data["relations"]
                                for relation in relations:
                                    await db.execute(
                                        """
                                        INSERT INTO relations (from_entity, to_entity, relationType)
                                        VALUES (?, ?, ?)
                                        """,
                                        (
                                            relation.from_,
                                            relation.to,
                                            relation.relationType,
                                        ),
                                    )

                            elif (
                                operation.operation_type
                                == BatchOperationType.DELETE_RELATIONS
                            ):
                                from_ = operation.data["from_"]
                                to = operation.data["to"]
                                await db.execute(
                                    "DELETE FROM relations WHERE from_entity = ? AND to_entity = ?",
                                    (from_, to),
                                )

                            elif (
                                operation.operation_type
                                == BatchOperationType.ADD_OBSERVATIONS
                            ):
                                observations_map = operation.data["observations_map"]
                                for (
                                    entity_name,
                                    observations,
                                ) in observations_map.items():
                                    for obs in observations:
                                        await db.execute(
                                            "INSERT INTO observations (entity_name, content) VALUES (?, ?)",
                                            (entity_name, obs),
                                        )
                            else:
                                raise ValueError(
                                    f"Unknown operation type: {operation.operation_type}"
                                )

                            completed += 1
                        except Exception as e:
                            failed_ops.append((operation, str(e)))
                            if not operation.data.get("allow_partial", False):
                                await db.execute("ROLLBACK")
                                return BatchResult(
                                    success=False,
                                    operations_completed=completed,
                                    failed_operations=failed_ops,
                                    error_message=f"Operation failed: {str(e)}",
                                )

                    # Commit if all operations succeeded
                    await db.commit()

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
                logger.error(f"Error executing batch: {str(e)}")
                return BatchResult(
                    success=False,
                    operations_completed=0,
                    failed_operations=[],
                    error_message=f"Batch execution failed: {str(e)}",
                )
