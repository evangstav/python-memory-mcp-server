# Enhanced Search Capabilities Implementation Plan

## Overview & Justification

The current search functionality in the Memory MCP Server relies on basic text matching and fuzzy matching techniques. This approach has limitations in understanding user intent, context, and semantic similarity. Enhancing search capabilities will significantly improve Claude's ability to retrieve relevant information from memory, especially for complex or ambiguous queries.

## Technical Approach

We'll implement a multi-layered search system that combines several techniques:

1. **Vector Embeddings for Semantic Search**:
   - Represent entities, observations, and queries as dense vector embeddings
   - Calculate semantic similarity between queries and stored information
   - Enable finding conceptually similar items even when exact words don't match

2. **Query Understanding**:
   - Implement query classification to determine search intent
   - Parse temporal references (e.g., "last week", "yesterday")
   - Identify entity references and attribute requests

3. **Hybrid Ranking**:
   - Combine scores from lexical matching, semantic similarity, and recency
   - Apply context-aware boosting based on query classification

## Implementation Steps

### 1. Add Vector Embedding Support (2 weeks)

```python
# Add necessary dependencies
# pyproject.toml
dependencies = [
    # ... existing dependencies
    "sentence-transformers>=2.2.2",
    "numpy>=1.24.0",
    "faiss-cpu>=1.7.0",  # For vector similarity search
]
```

1. Create a new embedding service class:

```python
# memory_mcp_server/services/embedding_service.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a vector."""
        return self.model.encode(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings to vectors."""
        return self.model.encode(texts)

    def compute_similarity(self, query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document vectors."""
        return np.dot(doc_vectors, query_vector) / (
            np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
        )
```

2. Extend the current backend interface to support vector storage:

```python
# memory_mcp_server/backends/base.py
# Add to Backend abstract class:
@abstractmethod
async def store_embedding(self, entity_name: str, vector: np.ndarray) -> None:
    """Store embedding vector for an entity."""
    pass

@abstractmethod
async def get_embedding(self, entity_name: str) -> Optional[np.ndarray]:
    """Get embedding vector for an entity."""
    pass
```

3. Update the JSONL backend to implement these methods:

```python
# memory_mcp_server/backends/jsonl.py
# In the JsonlBackend class:
async def store_embedding(self, entity_name: str, vector: np.ndarray) -> None:
    # Store as a separate file alongside the main JSONL file
    embedding_path = self.memory_path.with_suffix('.embeddings')
    async with self._write_lock:
        try:
            # Load existing embeddings
            embeddings = {}
            if embedding_path.exists():
                async with aiofiles.open(embedding_path, mode='rb') as f:
                    content = await f.read()
                    if content:
                        embeddings = pickle.loads(content)

            # Update with new embedding
            embeddings[entity_name] = vector.tolist()

            # Save back to file
            async with aiofiles.open(embedding_path, mode='wb') as f:
                await f.write(pickle.dumps(embeddings))
        except Exception as e:
            raise FileAccessError(f"Error storing embedding: {str(e)}")

async def get_embedding(self, entity_name: str) -> Optional[np.ndarray]:
    embedding_path = self.memory_path.with_suffix('.embeddings')
    if not embedding_path.exists():
        return None

    try:
        async with aiofiles.open(embedding_path, mode='rb') as f:
            content = await f.read()
            if not content:
                return None
            embeddings = pickle.loads(content)

        if entity_name in embeddings:
            return np.array(embeddings[entity_name])
        return None
    except Exception as e:
        raise FileAccessError(f"Error getting embedding: {str(e)}")
```

### 2. Implement Query Understanding (1.5 weeks)

```python
# memory_mcp_server/services/query_analyzer.py
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

class QueryType(Enum):
    ENTITY = "entity"
    TEMPORAL = "temporal"
    ATTRIBUTE = "attribute"
    RELATION = "relation"
    GENERAL = "general"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    temporal_reference: Optional[str] = None
    target_entity: Optional[str] = None
    target_attribute: Optional[str] = None
    relation_type: Optional[str] = None
    additional_params: Dict[str, Any] = None

class QueryAnalyzer:
    def __init__(self):
        # Temporal patterns (recency, specific dates, etc.)
        self.temporal_patterns = [
            (r"recent|latest|last|past", "recent"),
            (r"yesterday|today|this week|this month", "specific"),
            # Add more patterns
        ]

        # Entity type patterns
        self.entity_type_patterns = {
            entity_type: re.compile(f"\\b{entity_type}\\b", re.IGNORECASE)
            for entity_type in ["person", "project", "event", "location", "tool"]
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze the query and determine its type and parameters."""
        query_lower = query.lower()

        # Check for temporal references
        temporal_ref = None
        for pattern, temp_type in self.temporal_patterns:
            if re.search(pattern, query_lower):
                temporal_ref = temp_type
                break

        # Detect entity type references
        entity_types = []
        for entity_type, pattern in self.entity_type_patterns.items():
            if pattern.search(query_lower):
                entity_types.append(entity_type)

        # Determine query type based on analysis
        if temporal_ref and any(term in query_lower for term in ["workout", "exercise"]):
            return QueryAnalysis(
                query_type=QueryType.TEMPORAL,
                temporal_reference=temporal_ref,
                target_entity="workout"
            )
        elif "relation" in query_lower or "connected" in query_lower:
            return QueryAnalysis(
                query_type=QueryType.RELATION,
                additional_params={"entity_types": entity_types}
            )
        elif entity_types:
            return QueryAnalysis(
                query_type=QueryType.ENTITY,
                additional_params={"entity_types": entity_types}
            )

        # Default to general search
        return QueryAnalysis(
            query_type=QueryType.GENERAL,
            temporal_reference=temporal_ref
        )
```

### 3. Implement Enhanced Search Logic (2 weeks)

Update the KnowledgeGraphManager to leverage these new capabilities:

```python
# memory_mcp_server/knowledge_graph_manager.py
# Extend the search_nodes method:

from .services.embedding_service import EmbeddingService
from .services.query_analyzer import QueryAnalyzer, QueryType

# Initialize in constructor
self.embedding_service = EmbeddingService()
self.query_analyzer = QueryAnalyzer()

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
            e for e in graph.entities
            if query_analysis.target_entity in e.entityType.lower()
        ]

        # Sort by recency if available (assuming observations contain timestamps)
        # This is a simplification - would need actual timestamp extraction
        filtered_entities.sort(
            key=lambda e: max((obs for obs in e.observations if "date" in obs.lower()),
                            default=""),
            reverse=True
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
        rel for rel in graph.relations
        if rel.from_ in entity_names and rel.to in entity_names
    ]

    return KnowledgeGraph(entities=results, relations=matched_relations)

async def _semantic_search(self, query: str, entities: List[Entity]) -> List[Entity]:
    """Perform semantic search using embeddings."""
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
        top_indices = np.argsort(similarities)[::-1][:10]  # Top 10 results

        # Return entities in order of relevance
        return [entity_map[idx] for idx in top_indices]

    return []
```

### 4. Update MCP Server Interface (1 week)

Update the main MCP server interfaces to expose the enhanced search capabilities:

```python
# main.py
# Update the search_memory tool:
@mcp.tool()
async def search_memory(query: str, ctx: Context = None) -> EntityResponse:
    """Search memory using natural language queries.

    Handles:
    - Semantic search for conceptually similar entities
    - Temporal queries (e.g., "most recent", "last", "latest")
    - Entity-specific queries (e.g., "people who know about Python")
    - General knowledge graph exploration
    """
    try:
        if ctx:
            ctx.info(f"Enhanced search for: {query}")

        # Use the enhanced search implementation
        results = await kg.enhanced_search(query)

        if not results.entities:
            return EntityResponse(
                success=True,
                data={"entities": [], "relations": []},
                error="No matching entities found in memory",
                error_type="NO_RESULTS",
            )

        return EntityResponse(success=True, data=serialize_to_dict(results))
    except ValueError as e:
        return EntityResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["VALIDATION_ERROR"]
        )
    except Exception as e:
        return EntityResponse(
            success=False, error=str(e), error_type=ERROR_TYPES["INTERNAL_ERROR"]
        )
```

### 5. Implement Embedding Generation Hooks (1 week)

Ensure embeddings are generated and stored whenever entities are created or updated:

```python
# memory_mcp_server/knowledge_graph_manager.py
# Update the create_entities method:
async def create_entities(self, entities: List[Entity]) -> List[Entity]:
    """Create multiple new entities."""
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

# Similarly update add_observations to regenerate embeddings when observations change
```

## Potential Challenges

1. **Performance Impact**:
   - Generating embeddings can be computationally intensive
   - Solution: Implement background processing for embedding generation
   - Consider caching strategies to avoid redundant calculations

2. **Storage Requirements**:
   - Vector embeddings will increase storage requirements
   - Solution: Implement efficient storage formats (e.g., quantized vectors)
   - Consider compression techniques for large knowledge graphs

3. **Query Latency**:
   - Vector similarity search may increase query latency
   - Solution: Use approximate nearest neighbor search (FAISS) for large collections
   - Implement tiered search (fast exact match first, semantic search as fallback)

4. **Embedding Quality**:
   - Pre-trained models may not capture domain-specific semantics
   - Solution: Allow configuration of different embedding models
   - Consider fine-tuning models on domain-specific data

## Testing Strategy

1. **Unit Tests**:
   - Test individual components (embedding service, query analyzer)
   - Verify embedding generation and storage
   - Test different query types

2. **Integration Tests**:
   - Test end-to-end search functionality
   - Verify correct handling of different query types
   - Test with various knowledge graph sizes

3. **Benchmarking**:
   - Measure performance impact of embedding generation
   - Compare search latency with and without semantic search
   - Evaluate result quality improvement

4. **Quality Evaluation**:
   - Create a test set of queries with known expected results
   - Measure precision and recall of the enhanced search
   - Compare with baseline (current implementation)

## Timeline Estimate

- Research and design: 1 week
- Implementation of core components: 4 weeks
- Testing and benchmarking: 2 weeks
- Documentation and integration: 1 week

**Total estimated time**: 8 weeks

## Success Metrics

1. **Qualitative Improvement**:
   - Successfully handle complex natural language queries
   - Return semantically relevant results even with vocabulary mismatch
   - Properly understand temporal and contextual references

2. **Quantitative Metrics**:
   - Improve search precision by at least 25%
   - Maintain query latency under 200ms for 95% of queries
   - Storage overhead less than 20% of base knowledge graph size

3. **User Experience**:
   - Reduce need for specific, exact-match queries
   - Enable more conversational interaction with memory
   - Improve Claude's ability to retrieve relevant context
