import sqlite3
import logging
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from urllib.parse import urlparse

from .interfaces import Entity, Relation
from .exceptions import EntityNotFoundError, EntityAlreadyExistsError

logger = logging.getLogger(__name__)

class OptimizedSQLiteManager:
    def __init__(self, database_url: str, echo: bool = False):
        """Initialize SQLite manager with database path extracted from URL."""
        parsed_url = urlparse(database_url)
        if not parsed_url.path:
            raise ValueError("Database path not specified in URL")
            
        # Handle the database path, supporting both absolute and relative paths
        path = parsed_url.path.lstrip('/')
        if '/' in path:  # If path contains directories
            self.db_path = str(Path(path).absolute())
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        else:  # Simple filename in current directory
            self.db_path = path
        self.echo = echo
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def initialize(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                observations TEXT NOT NULL
            )
        """)
        
        # Create relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                from_entity TEXT NOT NULL,
                to_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                PRIMARY KEY (from_entity, to_entity, relation_type),
                FOREIGN KEY (from_entity) REFERENCES entities(name),
                FOREIGN KEY (to_entity) REFERENCES entities(name)
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_from_entity ON relations(from_entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_to_entity ON relations(to_entity)")
        
        conn.commit()

    async def cleanup(self):
        """Clean up database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        created_entities = []

        for entity in entities:
            try:
                cursor.execute(
                    "INSERT INTO entities (name, entity_type, observations) VALUES (?, ?, ?)",
                    (entity.name, entity.entityType, ','.join(entity.observations))
                )
                created_entities.append(entity)
            except sqlite3.IntegrityError:
                conn.rollback()
                raise EntityAlreadyExistsError(entity.name)

        conn.commit()
        return created_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        created_relations = []

        for relation in relations:
            # Check if entities exist
            cursor.execute("SELECT 1 FROM entities WHERE name = ?", (relation.from_,))
            if not cursor.fetchone():
                raise EntityNotFoundError(relation.from_)
                
            cursor.execute("SELECT 1 FROM entities WHERE name = ?", (relation.to,))
            if not cursor.fetchone():
                raise EntityNotFoundError(relation.to)

            try:
                cursor.execute(
                    "INSERT INTO relations (from_entity, to_entity, relation_type) VALUES (?, ?, ?)",
                    (relation.from_, relation.to, relation.relationType)
                )
                created_relations.append(relation)
            except sqlite3.IntegrityError:
                continue  # Skip duplicate relations

        conn.commit()
        return created_relations

    async def read_graph(self) -> Dict[str, List]:
        """Read the entire graph from the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all entities
        cursor.execute("SELECT * FROM entities")
        entities = []
        for row in cursor.fetchall():
            entities.append(Entity(
                name=row['name'],
                entityType=row['entity_type'],
                observations=row['observations'].split(',') if row['observations'] else []
            ))

        # Get all relations
        cursor.execute("SELECT * FROM relations")
        relations = []
        for row in cursor.fetchall():
            relations.append(Relation(
                from_=row['from_entity'],
                to=row['to_entity'],
                relationType=row['relation_type']
            ))

        return {"entities": entities, "relations": relations}

    async def search_nodes(self, query: str) -> Dict[str, List]:
        """Search for entities and relations containing the query string."""
        if not query:
            raise ValueError("Search query cannot be empty")

        conn = self._get_connection()
        cursor = conn.cursor()
        search_pattern = f"%{query}%"

        # Search entities
        cursor.execute("""
            SELECT * FROM entities 
            WHERE name LIKE ? 
            OR entity_type LIKE ? 
            OR observations LIKE ?
        """, (search_pattern, search_pattern, search_pattern))
        
        entities = []
        entity_names = set()
        for row in cursor.fetchall():
            entity = Entity(
                name=row['name'],
                entityType=row['entity_type'],
                observations=row['observations'].split(',') if row['observations'] else []
            )
            entities.append(entity)
            entity_names.add(entity.name)

        # Get related relations
        cursor.execute("""
            SELECT * FROM relations 
            WHERE from_entity IN (SELECT name FROM entities WHERE name IN ({}))
            AND to_entity IN (SELECT name FROM entities WHERE name IN ({}))
        """.format(
            ','.join('?' * len(entity_names)),
            ','.join('?' * len(entity_names))
        ), list(entity_names) * 2)

        relations = [
            Relation(
                from_=row['from_entity'],
                to=row['to_entity'],
                relationType=row['relation_type']
            )
            for row in cursor.fetchall()
        ]

        return {"entities": entities, "relations": relations}
