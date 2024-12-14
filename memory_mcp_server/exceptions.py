class KnowledgeGraphError(Exception):
    """Base exception for all knowledge graph errors."""

    pass


class EntityNotFoundError(KnowledgeGraphError):
    """Raised when an entity is not found in the graph."""

    def __init__(self, entity_name: str):
        self.entity_name = entity_name
        super().__init__(f"Entity '{entity_name}' not found in the graph")


class EntityAlreadyExistsError(KnowledgeGraphError):
    """Raised when trying to create an entity that already exists."""

    def __init__(self, entity_name: str):
        self.entity_name = entity_name
        super().__init__(f"Entity '{entity_name}' already exists in the graph")


class RelationValidationError(KnowledgeGraphError):
    """Raised when a relation is invalid."""

    pass


class FileAccessError(KnowledgeGraphError):
    """Raised when there are file access issues."""

    pass


class JsonParsingError(KnowledgeGraphError):
    """Raised when there are JSON parsing issues."""

    def __init__(self, line_number: int, line_content: str, original_error: Exception):
        self.line_number = line_number
        self.line_content = line_content
        self.original_error = original_error
        super().__init__(
            f"Failed to parse JSON at line {line_number}: {str(original_error)}\n"
            f"Content: {line_content}"
        )
