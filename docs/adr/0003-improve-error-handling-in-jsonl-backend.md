# Improve Error Handling in JSONL Backend

## Status
Proposed

## Context
The JSONL backend's error handling in the `_check_cache` method currently swallows all exceptions during graph loading and returns an empty graph instead. This causes issues because:

1. File access errors, corruption, or permission issues are hidden from the caller
2. The application appears to work but with an empty knowledge graph
3. Users have no way to know if their data failed to load
4. The main application expects errors to be propagated through exceptions

## Decision
We will modify the error handling in the JSONL backend to:

1. Properly propagate FileAccessError exceptions when there are issues reading the file
2. Only return empty graph for expected cases (new file)
3. Maintain proper error context and stack traces

The change will be in `_check_cache`:
```python
try:
    graph = await self._load_graph_from_file()
    self._cache = graph
    self._cache_timestamp = current_time
    self._cache_file_mtime = file_mtime
    self._build_indices(graph)
    self._dirty = False
except FileAccessError:
    # Propagate file access errors
    raise
except Exception as e:
    # Convert unexpected errors to FileAccessError
    raise FileAccessError(f"Error loading graph: {str(e)}") from e
```

## Consequences

### Positive
- Users will be properly notified when their data cannot be loaded
- The application can handle errors appropriately
- Maintains proper error context for debugging
- Follows Python's exception handling best practices

### Negative
- May surface errors that were previously hidden
- Requires error handling updates in dependent code

## Implementation Notes
1. Update JsonlBackend._check_cache to propagate errors
2. Test error cases in test suite
3. Update documentation to reflect new error handling behavior
