import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


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
        if temporal_ref and any(
            term in query_lower for term in ["workout", "exercise"]
        ):
            return QueryAnalysis(
                query_type=QueryType.TEMPORAL,
                temporal_reference=temporal_ref,
                target_entity="workout",
            )
        elif "relation" in query_lower or "connected" in query_lower:
            return QueryAnalysis(
                query_type=QueryType.RELATION,
                additional_params={"entity_types": entity_types},
            )
        elif entity_types:
            return QueryAnalysis(
                query_type=QueryType.ENTITY,
                additional_params={"entity_types": entity_types},
            )

        # Default to general search
        return QueryAnalysis(
            query_type=QueryType.GENERAL, temporal_reference=temporal_ref
        )
