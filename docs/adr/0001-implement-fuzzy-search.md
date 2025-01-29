# ADR 0001: Implement Fuzzy Search for Knowledge Graph Queries

## Status
Proposed

## Context
The current search implementation in the knowledge graph uses exact substring matching, which may miss relevant results when queries contain typos or slight variations in wording. We need a more flexible search mechanism that can handle approximate matches.

## Decision
We will implement fuzzy string matching for the knowledge graph search functionality with the following approach:

1. Add fuzzy string matching capability using the `thefuzz` library (with `python-Levenshtein` for performance)
2. Extend the search interface to support:
   - Fuzzy matching with configurable similarity threshold (0-100)
   - Weighted scoring across different entity fields (name, type, observations)
   - Optional exact matching fallback

### Technical Details

#### Implementation Issues Found
The initial fuzzy search implementation had several issues:

1. Score Normalization: The scoring algorithm wasn't properly normalized, leading to scores being too low because:
   - Individual word scores were not combined effectively
   - Weights were applied before normalization
   - Final division by total weight reduced scores too much

2. Partial Word Matching: The implementation didn't effectively handle partial word matches, causing:
   - Common variations (e.g., "Jon" vs "John") to not match at high thresholds
   - Inconsistent scoring between name, type, and observation fields

#### Improved Implementation Strategy
1. Score Calculation:
   - Normalize individual field scores before applying weights
   - Use token set ratio for better partial matching
   - Combine multiple word scores using maximum score
   - Apply weights after normalization

2. Field-Specific Handling:
   - Names: Use token set ratio for better name variation matching
   - Types: Use direct ratio comparison
   - Observations: Use token set ratio with word-level matching

#### Interface Changes
- Extend `search_nodes()` in the Backend interface to support fuzzy search parameters
- Add new dataclass `SearchOptions` with fields:
  ```python
  @dataclass
  class SearchOptions:
      fuzzy: bool = False
      threshold: int = 80  # Similarity threshold (0-100)
      weights: Dict[str, float] = field(default_factory=lambda: {
          "name": 1.0,
          "type": 0.8,
          "observations": 0.6
      })
  ```

#### Implementation Strategy
1. JSONL Backend:
   - Implement fuzzy matching using `thefuzz.fuzz.ratio()` for string comparison
   - Calculate weighted scores across entity fields
   - Filter results based on threshold
   - Sort results by score
   - Cache fuzzy search results for performance

2. Future SQLite Backend:
   - Consider implementing fuzzy search using SQLite's FTS5 extension with custom tokenizer
   - Evaluate performance tradeoffs between in-memory vs. database fuzzy matching

## Consequences

### Advantages
1. More flexible and user-friendly search experience
2. Better handling of typos and variations in queries
3. Configurable matching sensitivity
4. Weighted scoring for more relevant results

### Disadvantages
1. Additional dependency on fuzzy matching library
2. Slightly increased memory usage for score calculations
3. Potential performance impact on large datasets
4. More complex codebase

### Performance Considerations
- Cache fuzzy search results with TTL
- Use python-Levenshtein for optimized string distance calculations
- Consider implementing batch processing for large result sets
- Add index for frequently searched terms

## Implementation Plan
1. Add new dependencies to pyproject.toml
2. Create SearchOptions interface
3. Update Backend base class
4. Implement fuzzy search in JSONL backend
5. Add tests for fuzzy matching
6. Update documentation

## Alternatives Considered
1. Elasticsearch integration
   - Pros: Powerful search capabilities, scalable
   - Cons: Additional infrastructure, complexity
2. Trigram-based matching
   - Pros: Fast, built-in PostgreSQL support
   - Cons: Less accurate for short strings
3. Phonetic matching (Soundex/Metaphone)
   - Pros: Good for name matching
   - Cons: Language-specific, less suitable for general text
