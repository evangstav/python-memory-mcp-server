"""
Conversation context model for tracking conversation flow and topics over time.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class ConversationContext:
    """Special entity representing conversation context.

    Attributes:
        id: Unique identifier for this context point
        timestamp: Unix timestamp when this context was created
        topic: Main topic of this conversation segment
        entities_mentioned: List of entity names mentioned in this segment
        importance: Importance score (0.0-1.0) with higher values indicating more significant context
        summary: Brief summary of this conversation segment
    """

    id: str
    timestamp: float
    topic: str
    entities_mentioned: List[str]
    importance: float
    summary: str

    def __post_init__(self) -> None:
        """Validate conversation context fields."""
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Context ID must be a non-empty string")

        if not isinstance(self.timestamp, float) or self.timestamp <= 0:
            raise ValueError("Timestamp must be a positive float")

        if not isinstance(self.topic, str) or not self.topic:
            raise ValueError("Topic must be a non-empty string")

        if not isinstance(self.entities_mentioned, list):
            raise ValueError("Entities mentioned must be a list")

        if not isinstance(self.importance, float) or not (
            0.0 <= self.importance <= 1.0
        ):
            raise ValueError("Importance must be a float between 0.0 and 1.0")

        if not isinstance(self.summary, str) or not self.summary:
            raise ValueError("Summary must be a non-empty string")

    def to_entity_observations(self) -> List[str]:
        """Convert context to entity observations.

        Returns:
            List of observation strings for this context.
        """
        return [
            f"Topic: {self.topic}",
            f"Time: {datetime.fromtimestamp(self.timestamp).isoformat()}",
            f"Summary: {self.summary}",
            f"Entities: {', '.join(self.entities_mentioned)}",
            f"Importance: {self.importance}",
        ]
