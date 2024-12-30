from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, TypeVar, AsyncIterator
from functools import wraps

from sqlalchemy import (
    create_engine,
    select,
    Text,
    ForeignKey,
    String,
    Integer,
    and_,
    or_,
    delete,
    text
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship
)
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.exc import IntegrityError

from .exceptions import (
    EntityNotFoundError,
    EntityAlreadyExistsError
)

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')

class Base(DeclarativeBase):
    pass

class Entity(Base):
    __tablename__ = 'entities'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    entityType: Mapped[str] = mapped_column(String, nullable=False)
    observations: Mapped[List[str]] = mapped_column(Text, nullable=False)
    
    # Relationships with cascade delete
    from_relations = relationship(
        'Relation',
        back_populates='from_entity',
        cascade='all, delete-orphan',
        foreign_keys='Relation.from_id'
    )
    to_relations = relationship(
        'Relation',
        back_populates='to_entity',
        cascade='all, delete-orphan',
        foreign_keys='Relation.to_id'
    )

class Relation(Base):
    __tablename__ = 'relations'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_id: Mapped[int] = mapped_column(ForeignKey('entities.id'), nullable=False)
    to_id: Mapped[int] = mapped_column(ForeignKey('entities.id'), nullable=False)
    relationType: Mapped[str] = mapped_column(String, nullable=False)
    
    from_entity = relationship('Entity', back_populates='from_relations', foreign_keys=[from_id])
    to_entity = relationship('Entity', back_populates='to_relations', foreign_keys=[to_id])

def retry_on_error(retries: int = 3, delay: float = 0.1):
    """Decorator for retrying database operations on transient errors."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    continue
            raise last_error
        return wrapper
    return decorator

class OptimizedSQLiteManager:
    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False
    ):
        """Initialize the SQLite manager with connection pooling."""
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            echo=echo,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def initialize(self):
        """Initialize database tables and indices."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
            # Create indices for common queries
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entityType);
                CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relationType);
            """))

    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Get a session from the connection pool with automatic cleanup."""
        session = self.async_session()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

    @retry_on_error()
    async def create_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create multiple entities in a single transaction."""
        async with self.get_session() as session:
            db_entities = [
                Entity(
                    name=e['name'],
                    entityType=e['entityType'],
                    observations=e['observations']
                )
                for e in entities
            ]
            
            session.add_all(db_entities)
            try:
                await session.commit()
                return [
                    {
                        'name': e.name,
                        'entityType': e.entityType,
                        'observations': e.observations
                    }
                    for e in db_entities
                ]
            except IntegrityError as e:
                await session.rollback()
                raise EntityAlreadyExistsError(str(e))

    @retry_on_error()
    async def create_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create multiple relations efficiently."""
        async with self.get_session() as session:
            # Batch fetch entities
            entity_names = {r['from'] for r in relations} | {r['to'] for r in relations}
            stmt = select(Entity).where(Entity.name.in_(entity_names))
            result = await session.execute(stmt)
            entities = {e.name: e for e in result.scalars().all()}
            
            # Create relations
            db_relations = []
            for r in relations:
                from_entity = entities.get(r['from'])
                to_entity = entities.get(r['to'])
                
                if not from_entity or not to_entity:
                    raise EntityNotFoundError(
                        f"Entity not found: {r['from'] if not from_entity else r['to']}"
                    )
                
                relation = Relation(
                    from_id=from_entity.id,
                    to_id=to_entity.id,
                    relationType=r['relationType']
                )
                session.add(relation)
                db_relations.append((relation, from_entity, to_entity))
            
            await session.commit()
            return [
                {
                    'from': fe.name,
                    'to': te.name,
                    'relationType': r.relationType
                }
                for r, fe, te in db_relations
            ]

    @retry_on_error()
    async def add_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add observations to multiple entities efficiently."""
        async with self.get_session() as session:
            # Batch fetch all entities
            entity_names = {obs['entityName'] for obs in observations}
            stmt = select(Entity).where(Entity.name.in_(entity_names))
            result = await session.execute(stmt)
            entities = {e.name: e for e in result.scalars().all()}
            
            updated_entities = []
            for obs_data in observations:
                entity = entities.get(obs_data['entityName'])
                if not entity:
                    raise EntityNotFoundError(obs_data['entityName'])
                
                # Extend observations efficiently
                entity.observations.extend(obs_data['contents'])
                updated_entities.append(entity)
            
            await session.commit()
            return {
                'entities': [
                    {
                        'name': e.name,
                        'entityType': e.entityType,
                        'observations': e.observations
                    }
                    for e in updated_entities
                ]
            }

    @retry_on_error()
    async def delete_entities(self, entity_names: List[str]) -> None:
        """Delete multiple entities and their relations efficiently."""
        async with self.get_session() as session:
            stmt = delete(Entity).where(Entity.name.in_(entity_names))
            await session.execute(stmt)
            await session.commit()

    @retry_on_error()
    async def delete_observations(self, deletions: List[Dict[str, Any]]) -> None:
        """Delete specific observations from multiple entities efficiently."""
        async with self.get_session() as session:
            # Batch fetch affected entities
            entity_names = {d['entityName'] for d in deletions}
            stmt = select(Entity).where(Entity.name.in_(entity_names))
            result = await session.execute(stmt)
            entities = {e.name: e for e in result.scalars().all()}
            
            for deletion in deletions:
                entity = entities.get(deletion['entityName'])
                if not entity:
                    raise EntityNotFoundError(deletion['entityName'])
                
                entity.observations = [
                    obs for obs in entity.observations 
                    if obs not in deletion['observations']
                ]
            
            await session.commit()

    @retry_on_error()
    async def delete_relations(self, relations: List[Dict[str, Any]]) -> None:
        """Delete multiple relations efficiently."""
        async with self.get_session() as session:
            # Batch fetch all involved entities
            entity_names = {r['from'] for r in relations} | {r['to'] for r in relations}
            stmt = select(Entity).where(Entity.name.in_(entity_names))
            result = await session.execute(stmt)
            entities = {e.name: e for e in result.scalars().all()}
            
            conditions = []
            for r in relations:
                from_entity = entities.get(r['from'])
                to_entity = entities.get(r['to'])
                if from_entity and to_entity:
                    conditions.append(
                        and_(
                            Relation.from_id == from_entity.id,
                            Relation.to_id == to_entity.id,
                            Relation.relationType == r['relationType']
                        )
                    )
            
            if conditions:
                stmt = delete(Relation).where(or_(*conditions))
                await session.execute(stmt)
                await session.commit()

    async def read_graph(self) -> Dict[str, Any]:
        """Read the entire graph efficiently."""
        async with self.get_session() as session:
            # Parallel fetching of entities and relations
            entity_task = asyncio.create_task(
                session.execute(select(Entity))
            )
            relation_task = asyncio.create_task(
                session.execute(select(Relation))
            )
            
            # Wait for both queries to complete
            entity_result, relation_result = await asyncio.gather(
                entity_task, relation_task
            )
            
            entities = entity_result.scalars().all()
            relations = relation_result.scalars().all()
            
            return {
                'entities': [
                    {
                        'name': e.name,
                        'entityType': e.entityType,
                        'observations': e.observations
                    }
                    for e in entities
                ],
                'relations': [
                    {
                        'from': r.from_entity.name,
                        'to': r.to_entity.name,
                        'relationType': r.relationType
                    }
                    for r in relations
                ]
            }

    async def search_nodes(self, query: str) -> Dict[str, Any]:
        """Search for entities and relations containing the query string."""
        if not query:
            raise ValueError("Search query cannot be empty")
            
        async with self.get_session() as session:
            # Use SQL LIKE for more efficient text search
            stmt = select(Entity).where(
                or_(
                    Entity.name.ilike(f'%{query}%'),
                    Entity.entityType.ilike(f'%{query}%'),
                    Entity.observations.cast(String).ilike(f'%{query}%')
                )
            )
            
            result = await session.execute(stmt)
            entities = result.scalars().all()
            
            # Fetch related relations if entities were found
            if entities:
                entity_ids = [e.id for e in entities]
                rel_stmt = select(Relation).where(
                    or_(
                        Relation.from_id.in_(entity_ids),
                        Relation.to_id.in_(entity_ids)
                    )
                )
                rel_result = await session.execute(rel_stmt)
                relations = rel_result.scalars().all()
            else:
                relations = []
            
            return {
                'entities': [
                    {
                        'name': e.name,
                        'entityType': e.entityType,
                        'observations': e.observations
                    }
                    for e in entities
                ],
                'relations': [
                    {
                        'from': r.from_entity.name,
                        'to': r.to_entity.name,
                        'relationType': r.relationType
                    }
                    for r in relations
                ]
            }

    async def open_nodes(self, names: List[str]) -> Dict[str, Any]:
        """Retrieve specific nodes and their relations efficiently."""
        async with self.get_session() as session:
            # Parallel fetch of entities and their relations
            entity_task = asyncio.create_task(
                session.execute(
                    select(Entity).where(Entity.name.in_(names))
                )
            )
            
            # Pre-fetch relations in parallel
            relation_task = asyncio.create_task(
                session.execute(
                    select(Relation).where(
                        or_(
                            Relation.from_entity.has(Entity.name.in_(names)),
                            Relation.to_entity.has(Entity.name.in_(names))
                        )
                    )
                )
            )
            
            entity_result, relation_result = await asyncio.gather(
                entity_task, relation_task
            )
            
            entities = entity_result.scalars().all()
            relations = relation_result.scalars().all()
            
            return {
                'entities': [
                    {
                        'name': e.name,
                        'entityType': e.entityType,
                        'observations': e.observations
                    }
                    for e in entities
                ],
                'relations': [
                    {
                        'from': r.from_entity.name,
                        'to': r.to_entity.name,
                        'relationType': r.relationType
                    }
                    for r in relations
                ]
            }

    async def cleanup(self):
        """Clean up resources when shutting down."""
        await self.engine.dispose()