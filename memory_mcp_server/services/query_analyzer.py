"""Query analyzer for understanding search intent."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set


class QueryType(Enum):
    """Types of queries that can be identified."""
    
    ENTITY = "entity"           # Queries about specific entities
    TEMPORAL = "temporal"       # Queries with time references
    ATTRIBUTE = "attribute"     # Queries about entity attributes
    RELATION = "relation"       # Queries about relationships
    GENERAL = "general"         # General search queries


@dataclass
class QueryAnalysis:
    """Analysis result for a search query."""
    
    query_type: QueryType
    temporal_reference: Optional[str] = None
    target_entity: Optional[str] = None
    target_attribute: Optional[str] = None
    relation_type: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer:
    """Analyzes search queries to determine intent and extract parameters."""
    
    def __init__(self):
        """Initialize the query analyzer with pattern definitions."""
        # Temporal patterns (recency, specific dates, etc.)
        self.temporal_patterns = [
            (r"\b(recent|latest|last|newest|current)\b", "recent"),
            (r"\b(yesterday|today|this week|this month|last week|last month)\b", "specific"),
            (r"\b(old|oldest|earlier|previous|past)\b", "past"),
            (r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", "date"),
        ]

        # Entity type patterns
        self.entity_types = [
            "person", "project", "event", "location", "tool", 
            "concept", "organization", "document", "task"
        ]
        
        self.entity_type_patterns = {
            entity_type: re.compile(f"\\b{entity_type}s?\\b", re.IGNORECASE)
            for entity_type in self.entity_types
        }
        
        # Relation patterns
        self.relation_patterns = [
            (r"\b(related|connected|linked|associated)\b", "general"),
            (r"\b(knows|worked with|collaborated with)\b", "person"),
            (r"\b(part of|belongs to|member of)\b", "membership"),
            (r"\b(created|authored|developed|built)\b", "creation"),
            (r"\b(located in|situated at|found in)\b", "location"),
        ]
        
        # Attribute patterns
        self.attribute_patterns = [
            (r"\b(name|called|titled)\b", "name"),
            (r"\b(type|kind|category)\b", "type"),
            (r"\b(when|date|time|period)\b", "time"),
            (r"\b(where|location|place)\b", "location"),
            (r"\b(who|person|people)\b", "person"),
        ]

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze the query and determine its type and parameters.
        
        Args:
            query: The search query to analyze
            
        Returns:
            QueryAnalysis with detected intent and parameters
        """
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
                
        # Check for relation patterns
        relation_type = None
        for pattern, rel_type in self.relation_patterns:
            if re.search(pattern, query_lower):
                relation_type = rel_type
                break
                
        # Check for attribute patterns
        attribute_type = None
        for pattern, attr_type in self.attribute_patterns:
            if re.search(pattern, query_lower):
                attribute_type = attr_type
                break
        
        # Determine query type based on analysis
        if relation_type and ("between" in query_lower or "connecting" in query_lower):
            return QueryAnalysis(
                query_type=QueryType.RELATION,
                relation_type=relation_type,
                temporal_reference=temporal_ref,
                additional_params={"entity_types": entity_types}
            )
        elif temporal_ref and entity_types:
            return QueryAnalysis(
                query_type=QueryType.TEMPORAL,
                temporal_reference=temporal_ref,
                additional_params={"entity_types": entity_types}
            )
        elif attribute_type and entity_types:
            return QueryAnalysis(
                query_type=QueryType.ATTRIBUTE,
                target_attribute=attribute_type,
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
