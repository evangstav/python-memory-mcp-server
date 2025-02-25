# Enhanced Memory Capabilities

This document outlines the enhanced memory capabilities implemented in the Memory MCP Server, focusing on semantic search and natural language understanding.

## Overview

The enhanced memory capabilities provide:

1. **Semantic Search**: Find conceptually similar entities even when exact terms don't match
2. **Query Understanding**: Analyze natural language queries to determine intent
3. **Temporal Awareness**: Handle time-based queries (recent, past, specific dates)
4. **Relation-Focused Search**: Explore connections between entities

## Semantic Search

### How It Works

1. **Vector Embeddings**: Entities are converted to vector embeddings using sentence-transformers
2. **Query Vectorization**: Search queries are also converted to vectors
3. **Similarity Calculation**: Cosine similarity is used to find the most relevant entities
4. **Ranking**: Results are ranked by similarity score

### Benefits

- Find conceptually similar content without exact keyword matches
- Understand synonyms and related concepts
- More natural and intuitive search experience

## Query Understanding

The system analyzes queries to determine:

- **Query Type**: Entity, temporal, relation, attribute, or general
- **Temporal References**: Recent, past, specific dates
- **Entity Types**: Person, project, event, etc.
- **Relations**: Connections between entities

### Example Query Types

- **Entity Query**: "Find all person entities"
- **Temporal Query**: "Show me recent events"
- **Relation Query**: "Show connections between people"
- **Attribute Query**: "Where is the conference located?"

## Implementation

### Embedding Generation

Embeddings are generated:
- When entities are created
- When observations are added
- When explicitly regenerated via the API

### Storage

Embeddings are stored in a separate file with the `.embeddings` extension, using pickle for serialization.

### Search Process

1. Query is analyzed to determine intent
2. Appropriate search strategy is selected based on query type
3. For semantic search, vector similarity is calculated
4. Results are filtered and ranked
5. Relations between matched entities are included if requested

## API Usage

### Search Memory

```python
# Basic search
result = await search_memory("machine learning projects")

# Advanced search with options
result = await search_memory(
    query="recent projects related to AI",
    semantic=True,
    max_results=5
)
```

### Search Nodes (Traditional Search)

```python
# Exact matching
result = await search_nodes("python")

# Fuzzy matching
result = await search_nodes(
    query="python",
    fuzzy=True,
    threshold=80.0
)
```

### Regenerate Embeddings

```python
# Regenerate all embeddings
result = await regenerate_embeddings()
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
