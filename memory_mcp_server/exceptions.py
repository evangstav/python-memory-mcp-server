class KnowledgeGraphError(Exception):
    """Base exception for all knowledge graph errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details

class ErrorResponse:
    def __init__(self, error: Exception):
        self.error_type = error.__class__.__name__
        self.message = str(error)
        self.details = getattr(error, 'details', None)

    def to_dict(self) -> dict:
        return {
            'error': {
                'type': self.error_type,
                'message': self.message,
                'details': self.details
            }
        }


class EntityNotFoundError(KnowledgeGraphError):
    """Raised when an entity is not found in the graph."""
    def __init__(self, entity_name: str):
        self.entity_name = entity_name
        super().__init__(
            f"Entity '{entity_name}' not found in the graph",
            details={'entity_name': entity_name}
        )


class EntityAlreadyExistsError(KnowledgeGraphError):
    """Raised when trying to create an entity that already exists."""
    def __init__(self, entity_name: str):
        self.entity_name = entity_name
        super().__init__(
            f"Entity '{entity_name}' already exists in the graph",
            details={'entity_name': entity_name}
        )


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
