# Semantic Search Capabilities

The Memory MCP Server includes advanced semantic search capabilities that allow for more natural and intuitive querying of the knowledge graph.

## Overview

Traditional search methods rely on exact keyword matching, which can miss conceptually similar content. Semantic search uses vector embeddings to find content based on meaning rather than exact wording.

## Key Components

### EmbeddingService

Located in `memory_mcp_server/services/embedding_service.py`, this service:

- Generates vector embeddings for entities using sentence-transformers
- Computes similarity between query vectors and entity vectors
- Handles batch encoding for efficient processing

### QueryAnalyzer

Located in `memory_mcp_server/services/query_analyzer.py`, this component:

- Analyzes natural language queries to determine intent
- Detects temporal references (recent, past, specific dates)
- Identifies entity types mentioned in queries
- Recognizes relation-focused queries

## Query Types

The system recognizes several types of queries:

1. **Entity Queries**: Focused on specific entity types (e.g., "Find all person entities")
2. **Temporal Queries**: Time-based searches (e.g., "Show me recent events")
3. **Relation Queries**: Exploring connections (e.g., "Show connections between people")
4. **Attribute Queries**: Searching for specific attributes (e.g., "Where is the conference located?")
5. **General Queries**: Default semantic search for any other queries

## Usage Examples

### Basic Semantic Search

```python
# Search with default settings
result = await search_memory("machine learning projects")

# Configure search options
options = SearchOptions(
    semantic=True,
    max_results=5,
    include_relations=True
)
result = await search_memory("machine learning projects", options)
```

### Temporal Queries

```python
# Find recent events
result = await search_memory("recent events")

# Find oldest projects
result = await search_memory("oldest projects")
```

### Relation Queries

```python
# Find connections between entities
result = await search_memory("connections between people")

# Find projects related to a concept
result = await search_memory("projects related to machine learning")
```

## Implementation Details

### Vector Embeddings

Entities are represented as vector embeddings by combining:
- Entity name
- Entity type
- All observations

These embeddings are stored alongside the knowledge graph and updated whenever entities are modified.

### Similarity Calculation

Similarity between queries and entities is calculated using cosine similarity:

```
similarity = dot(query_vector, entity_vector) / (|query_vector| * |entity_vector|)
```

### Embedding Storage

Embeddings are stored in a separate file with the `.embeddings` extension next to the main knowledge graph file.

## Regenerating Embeddings

If you need to regenerate embeddings (e.g., after importing data), use:

```python
await regenerate_embeddings()
```

## Performance Considerations

- Embedding generation can be computationally intensive
- First-time queries may be slower if embeddings need to be generated
- Consider regenerating embeddings after bulk imports
- The system uses caching to improve performance

## Limitations

- Requires sentence-transformers and related dependencies
- Embedding quality depends on the underlying model
- May not perform well with very domain-specific terminology
- Requires more computational resources than traditional search

## Future Enhancements

Potential areas for improvement:

- Support for multiple embedding models
- Fine-tuning embeddings for specific domains
- More sophisticated temporal understanding
- Improved relation inference
- Integration with external knowledge bases
