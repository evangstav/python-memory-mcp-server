import asyncio
from pathlib import Path
import json
import os
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict
import aiofiles
import time
from functools import lru_cache

from .interfaces import KnowledgeGraph, Entity, Relation
from .exceptions import (
    EntityNotFoundError,
    EntityAlreadyExistsError,
    RelationValidationError,
    FileAccessError,
    JsonParsingError,
)


class OptimizedKnowledgeGraphManager:
    def __init__(self, memory_path: Path, cache_ttl: int = 60):
        """
        Initialize the OptimizedKnowledgeGraphManager.
        
        Args:
            memory_path: Path to the knowledge graph file
            cache_ttl: Time to live for cache in seconds (default: 60)
        """
        self.memory_path = memory_path
        self.cache_ttl = cache_ttl
        self._cache: Optional[KnowledgeGraph] = None
        self._cache_timestamp: float = 0
        self._indices: Dict = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._dirty = False
        self._write_queue: List[Dict] = []
        self._max_queue_size = 100
        
    def _build_indices(self, graph: KnowledgeGraph) -> None:
        """Build indices for faster lookups."""
        self._indices.clear()
        
        # Entity name index
        self._indices["entity_names"] = {
            entity.name: entity for entity in graph.entities
        }
        
        # Entity type index
        self._indices["entity_types"] = defaultdict(list)
        for entity in graph.entities:
            self._indices["entity_types"][entity.entityType].append(entity)
            
        # Relations index
        self._indices["relations_from"] = defaultdict(list)
        self._indices["relations_to"] = defaultdict(list)
        for relation in graph.relations:
            self._indices["relations_from"][relation.from_].append(relation)
            self._indices["relations_to"][relation.to].append(relation)

    async def _check_cache(self) -> KnowledgeGraph:
        """Check if cache is valid and return cached graph."""
        current_time = time.time()
        
        if (
            self._cache is None
            or current_time - self._cache_timestamp > self.cache_ttl
            or self._dirty
        ):
            async with self._lock:
                graph = await self._load_graph_from_file()
                self._cache = graph
                self._cache_timestamp = current_time
                self._build_indices(graph)
                self._dirty = False
                
        return self._cache

    async def _load_graph_from_file(self) -> KnowledgeGraph:
        """Load the knowledge graph from file with improved error handling."""
        if not self.memory_path.exists():
            return KnowledgeGraph(entities=[], relations=[])

        try:
            async with aiofiles.open(self.memory_path, mode='r', encoding='utf-8') as f:
                lines = [line.strip() for line in await f.readlines() if line.strip()]
                
            graph = KnowledgeGraph(entities=[], relations=[])
            
            for i, line in enumerate(lines, 1):
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
                        
                except (json.JSONDecodeError, KeyError) as e:
                    raise JsonParsingError(i, line, e)
                    
            return graph
            
        except Exception as e:
            raise FileAccessError(f"Error reading file: {str(e)}")

    async def _save_graph(self, graph: KnowledgeGraph) -> None:
        """Save the knowledge graph to file with batched writes."""
        lines = []
        
        for entity in graph.entities:
            lines.append(
                json.dumps(
                    {
                        "type": "entity",
                        "name": entity.name,
                        "entityType": entity.entityType,
                        "observations": entity.observations,
                    }
                )
            )
            
        for relation in graph.relations:
            lines.append(
                json.dumps(
                    {
                        "type": "relation",
                        "from": relation.from_,
                        "to": relation.to,
                        "relationType": relation.relationType,
                    }
                )
            )

        # Ensure the directory exists
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file first
        temp_path = self.memory_path.with_suffix('.tmp')
        try:
            async with aiofiles.open(temp_path, mode='w', encoding='utf-8') as f:
                await f.write("\n".join(lines))
            
            # Atomic rename
            temp_path.replace(self.memory_path)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @lru_cache(maxsize=1000)
    def _get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name using cache."""
        return self._indices["entity_names"].get(name)

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities with validation and batching."""
        async with self._lock:
            graph = await self._check_cache()
            existing_entities = self._indices["entity_names"]
            new_entities = []

            for entity in entities:
                if not entity.name or not entity.entityType:
                    raise ValueError(f"Invalid entity: {entity}")
                    
                if entity.name not in existing_entities:
                    new_entities.append(entity)
                    self._indices["entity_names"][entity.name] = entity
                    self._indices["entity_types"][entity.entityType].append(entity)

            if new_entities:
                graph.entities.extend(new_entities)
                self._dirty = True
                
                # Add to write queue
                self._write_queue.extend([{
                    "type": "entity",
                    "name": entity.name,
                    "entityType": entity.entityType,
                    "observations": entity.observations,
                } for entity in new_entities])
                
                # Flush queue if it's full
                if len(self._write_queue) >= self._max_queue_size:
                    await self._save_graph(graph)
                    self._write_queue.clear()
                    
            return new_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations with validation and batching."""
        async with self._lock:
            graph = await self._check_cache()
            existing_entities = self._indices["entity_names"]
            new_relations = []

            for relation in relations:
                if not relation.from_ or not relation.to or not relation.relationType:
                    raise ValueError(f"Invalid relation: {relation}")
                    
                if relation.from_ not in existing_entities:
                    raise EntityNotFoundError(relation.from_)
                if relation.to not in existing_entities:
                    raise EntityNotFoundError(relation.to)

                # Check for duplicates using indices
                if not any(
                    r.from_ == relation.from_
                    and r.to == relation.to
                    and r.relationType == relation.relationType
                    for r in self._indices["relations_from"][relation.from_]
                ):
                    new_relations.append(relation)
                    self._indices["relations_from"][relation.from_].append(relation)
                    self._indices["relations_to"][relation.to].append(relation)

            if new_relations:
                graph.relations.extend(new_relations)
                self._dirty = True
                
                # Add to write queue
                self._write_queue.extend([{
                    "type": "relation",
                    "from": relation.from_,
                    "to": relation.to,
                    "relationType": relation.relationType,
                } for relation in new_relations])
                
                # Flush queue if it's full
                if len(self._write_queue) >= self._max_queue_size:
                    await self._save_graph(graph)
                    self._write_queue.clear()
                    
            return new_relations

    async def add_observations(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add new observations to existing entities with batching."""
        async with self._lock:
            graph = await self._check_cache()
            results = []

            for obs in observations:
                entity_name = obs.get("entityName")
                contents = obs.get("contents", [])
                
                if not entity_name or not isinstance(contents, list):
                    raise ValueError("Invalid observation format")

                entity = self._get_entity_by_name(entity_name)
                if not entity:
                    raise EntityNotFoundError(entity_name)

                new_obs = [
                    content
                    for content in contents
                    if content not in entity.observations
                ]
                
                if new_obs:
                    entity.observations.extend(new_obs)
                    self._dirty = True
                    results.append({
                        "entityName": entity_name,
                        "addedObservations": new_obs
                    })

            if results:
                await self._save_graph(graph)
                
            return results

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph using cache."""
        return await self._check_cache()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        """Search for nodes in the knowledge graph using indices."""
        if not query:
            raise ValueError("Search query cannot be empty")

        graph = await self._check_cache()
        query = query.lower()
        
        # Use indices for faster searching
        filtered_entities = set()
        
        # Search by entity name
        filtered_entities.update(
            entity for entity in graph.entities
            if query in entity.name.lower()
        )
        
        # Search by entity type
        for entity_type, entities in self._indices["entity_types"].items():
            if query in entity_type.lower():
                filtered_entities.update(entities)
                
        # Search in observations
        for entity in graph.entities:
            if any(query in obs.lower() for obs in entity.observations):
                filtered_entities.add(entity)
                
        filtered_entity_names = {e.name for e in filtered_entities}
        
        # Use relation indices for faster filtering
        filtered_relations = [
            relation for relation in graph.relations
            if relation.from_ in filtered_entity_names 
            and relation.to in filtered_entity_names
        ]

        return KnowledgeGraph(
            entities=list(filtered_entities),
            relations=filtered_relations
        )

    async def open_nodes(self, names: List[str]) -> KnowledgeGraph:
        """Open specific nodes by their names using indices."""
        if not names:
            raise ValueError("Names list cannot be empty")

        graph = await self._check_cache()
        filtered_entities = []
        
        for name in names:
            entity = self._get_entity_by_name(name)
            if entity is None:
                raise EntityNotFoundError(name)
            filtered_entities.append(entity)
            
        filtered_entity_names = {e.name for e in filtered_entities}
        
        # Use relation indices for faster filtering
        filtered_relations = []
        for name in filtered_entity_names:
            filtered_relations.extend(
                relation for relation in self._indices["relations_from"][name]
                if relation.to in filtered_entity_names
            )

        return KnowledgeGraph(
            entities=filtered_entities,
            relations=filtered_relations
        )

    async def flush(self) -> None:
        """Force flush any pending writes to disk."""
        if self._write_queue:
            async with self._lock:
                graph = await self._check_cache()
                await self._save_graph(graph)
                self._write_queue.clear()
                self._dirty = False
