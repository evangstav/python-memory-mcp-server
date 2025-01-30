# Knowledge Graph Consistency Guidelines

## Context

The knowledge graph is a core component of our memory system, storing entities and their relationships. To maintain data quality and consistency, we need clear guidelines for how entities and relations should be structured and validated.

## Decision

We will implement the following guidelines for knowledge graph consistency:

### 1. Entity Naming Conventions

- Entity names must be unique within the graph
- Names should be descriptive and follow these rules:
  - Use lowercase with words separated by hyphens
  - Avoid special characters except hyphens
  - Maximum length of 100 characters
  - Must start with a letter
  - Examples: `python-project`, `meeting-notes-2024`, `user-john`

### 2. Entity Types

- Entity types should be predefined categories
- Types should be singular form, lowercase
- Common types include:
  - `person`: Human entities
  - `concept`: Abstract ideas or principles
  - `project`: Work initiatives or tasks
  - `document`: Any form of documentation
  - `tool`: Software tools or utilities
  - `organization`: Companies or groups
  - `location`: Physical or virtual places
  - `event`: Time-bound occurrences

### 3. Observations

- Each observation should be a complete, standalone statement
- Observations should be factual and objective
- Include timestamp when relevant
- Maximum length of 500 characters per observation
- Avoid duplicate observations

### 4. Relations

- Relations must have valid source and target entities
- Relation types should be verbs in present tense
- Common relation types include:
  - `knows`: Person to person connection
  - `contains`: Parent/child relationship
  - `uses`: Entity utilizing another entity
  - `created`: Authorship/creation relationship
  - `belongs-to`: Membership/ownership
  - `depends-on`: Dependency relationship
  - `related-to`: Generic relationship when others don't apply

### 5. Data Validation

- Implement validation at both API and storage layers
- Validate entity names against naming convention regex
- Check for entity existence before creating relations
- Prevent circular dependencies in relations
- Validate observation length and content
- Ensure relation types are from predefined list

### 6. Search Behavior

- Default to fuzzy search with 80% similarity threshold
- Search weights prioritize:
  1. Entity names (weight: 1.0)
  2. Entity types (weight: 0.8)
  3. Observations (weight: 0.6)
- Return partial matches with relevance scores

### 7. Data Integrity

- Use write locks for concurrent modifications
- Maintain referential integrity in relations
- Cascade deletes for related entities
- Regular validation of entire graph structure
- Backup before major operations

## Implementation

1. Add validation layer in KnowledgeGraphManager:
```python
class KnowledgeGraphValidator:
    ENTITY_NAME_PATTERN = r'^[a-z][a-z0-9-]{0,99}$'
    MAX_OBSERVATION_LENGTH = 500

    @staticmethod
    def validate_entity(entity: Entity) -> None:
        if not re.match(ENTITY_NAME_PATTERN, entity.name):
            raise ValueError("Invalid entity name format")
        if len(entity.observations) > 0:
            if any(len(obs) > MAX_OBSERVATION_LENGTH for obs in entity.observations):
                raise ValueError("Observation exceeds maximum length")
```

2. Enhance Entity and Relation classes with validation:
```python
@dataclass(frozen=True)
class Entity:
    name: str
    entityType: str
    observations: Tuple[str, ...]

    def __post_init__(self):
        KnowledgeGraphValidator.validate_entity(self)

@dataclass
class Relation:
    from_: str
    to: str
    relationType: str

    def __post_init__(self):
        KnowledgeGraphValidator.validate_relation(self)
```

3. Add integrity checks in backend operations:
```python
async def create_relations(self, relations: List[Relation]) -> List[Relation]:
    async with self._write_lock:
        # Check for circular dependencies
        if self._has_circular_dependency(relations):
            raise ValueError("Circular dependency detected")
        # Validate entity existence
        for relation in relations:
            if not await self._entity_exists(relation.from_):
                raise EntityNotFoundError(f"Source entity {relation.from_} not found")
            if not await self._entity_exists(relation.to):
                raise EntityNotFoundError(f"Target entity {relation.to} not found")
        return await self.backend.create_relations(relations)
```

## Consequences

### Positive

- Consistent data structure across the knowledge graph
- Reduced risk of data corruption
- Easier querying and maintenance
- Better search results due to standardized naming
- Clear guidelines for users and developers

### Negative

- Additional validation overhead
- More complex implementation
- Stricter constraints on data input
- Migration needed for existing data

## Status

Proposed

## References

- [Entity Interface](../../memory_mcp_server/interfaces.py)
- [Knowledge Graph Manager](../../memory_mcp_server/knowledge_graph_manager.py)
- [Implementation Guide](../implementation-guide.md)
