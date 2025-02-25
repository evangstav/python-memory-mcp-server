"""Tests for semantic search capabilities."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from memory_mcp_server.services.embedding_service import EmbeddingService
from memory_mcp_server.services.query_analyzer import QueryAnalyzer, QueryType
from memory_mcp_server.interfaces import Entity, KnowledgeGraph, Relation, SearchOptions
from memory_mcp_server.knowledge_graph_manager import KnowledgeGraphManager


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service for testing."""
    with patch("memory_mcp_server.services.embedding_service.SentenceTransformer", autospec=True) as mock_st:
        service = EmbeddingService()
        # Mock the encode methods to return predictable vectors
        service.encode_text = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
        service.encode_batch = MagicMock(return_value=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]))
        service.compute_similarity = MagicMock(return_value=np.array([0.9, 0.6, 0.3]))
        yield service


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            name="Python",
            entityType="programming_language",
            observations=["Object-oriented language", "Created by Guido van Rossum", "Version 3.9 released in 2020"]
        ),
        Entity(
            name="JavaScript",
            entityType="programming_language",
            observations=["Used for web development", "Created in 1995", "ECMAScript standard"]
        ),
        Entity(
            name="Machine Learning",
            entityType="concept",
            observations=["Branch of AI", "Uses statistical methods", "Popular in data science"]
        ),
    ]


@pytest.fixture
def sample_relations():
    """Create sample relations for testing."""
    return [
        Relation(from_="Python", to="Machine Learning", relationType="used_for"),
        Relation(from_="JavaScript", to="Python", relationType="compared_to"),
    ]


@pytest.mark.asyncio
async def test_query_analyzer():
    """Test the query analyzer functionality."""
    analyzer = QueryAnalyzer()
    
    # Test entity type detection
    result = analyzer.analyze_query("Find all person entities")
    assert result.query_type == QueryType.ENTITY
    assert "person" in result.additional_params.get("entity_types", [])
    
    # Test temporal reference detection
    result = analyzer.analyze_query("Show me recent events")
    assert result.query_type == QueryType.TEMPORAL
    assert result.temporal_reference == "recent"
    assert "event" in result.additional_params.get("entity_types", [])
    
    # Test relation detection
    result = analyzer.analyze_query("Show connections between people")
    assert result.query_type == QueryType.RELATION
    assert "person" in result.additional_params.get("entity_types", [])


@pytest.mark.asyncio
async def test_embedding_service(mock_embedding_service):
    """Test the embedding service functionality."""
    # Test encoding a single text
    vector = mock_embedding_service.encode_text("test text")
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (3,)  # Our mock returns a 3-element vector
    
    # Test encoding a batch of texts
    vectors = mock_embedding_service.encode_batch(["text1", "text2", "text3"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 3)  # 3 vectors of 3 elements each
    
    # Test computing similarity
    query_vector = np.array([0.1, 0.2, 0.3])
    doc_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    similarities = mock_embedding_service.compute_similarity(query_vector, doc_vectors)
    assert isinstance(similarities, np.ndarray)
    assert similarities.shape == (3,)


@pytest.mark.asyncio
async def test_semantic_search(mock_embedding_service, sample_entities, sample_relations):
    """Test semantic search functionality."""
    # Create a mock backend
    mock_backend = MagicMock()
    # Make read_graph a coroutine that returns a KnowledgeGraph
    async def mock_read_graph():
        return KnowledgeGraph(
            entities=sample_entities,
            relations=sample_relations
        )
    mock_backend.read_graph = mock_read_graph
    
    # Make store_embedding a coroutine
    async def mock_store_embedding(entity_name, vector):
        pass
    mock_backend.store_embedding = mock_store_embedding
    
    # Make get_embedding a coroutine
    async def mock_get_embedding(entity_name):
        return None
    mock_backend.get_embedding = mock_get_embedding
    
    # Create a knowledge graph manager with the mock backend
    with patch("memory_mcp_server.knowledge_graph_manager.EmbeddingService", return_value=mock_embedding_service):
        kg_manager = KnowledgeGraphManager(mock_backend)
        
        # Test enhanced search
        options = SearchOptions(semantic=True, max_results=2)
        result = await kg_manager.enhanced_search("Find programming languages", options)
        
        # Verify results
        assert len(result.entities) <= 2  # Should respect max_results
        assert isinstance(result, KnowledgeGraph)
        
        # The mock similarity returns [0.9, 0.6, 0.3], so the first two entities should be returned
        assert result.entities[0].name == sample_entities[0].name
        assert result.entities[1].name == sample_entities[1].name


@pytest.mark.asyncio
async def test_embedding_generation_on_entity_creation(mock_embedding_service, sample_entities):
    """Test that embeddings are generated when entities are created."""
    # Create a mock backend
    mock_backend = MagicMock()
    
    # Make read_graph a coroutine
    async def mock_read_graph():
        return KnowledgeGraph(entities=[], relations=[])
    mock_backend.read_graph = mock_read_graph
    
    # Make create_entities a coroutine
    async def mock_create_entities(entities):
        return sample_entities
    mock_backend.create_entities = mock_create_entities
    
    # Make store_embedding a coroutine
    async def mock_store_embedding(entity_name, vector):
        pass
    mock_backend.store_embedding = mock_store_embedding
    
    # Create a knowledge graph manager with the mock backend
    with patch("memory_mcp_server.knowledge_graph_manager.EmbeddingService", return_value=mock_embedding_service):
        kg_manager = KnowledgeGraphManager(mock_backend)
        
        # Create entities
        await kg_manager.create_entities(sample_entities)
        
        # Verify that store_embedding was called for each entity
        assert mock_backend.store_embedding.call_count == len(sample_entities)


@pytest.mark.asyncio
async def test_embedding_update_on_observation_addition(mock_embedding_service, sample_entities):
    """Test that embeddings are updated when observations are added."""
    # Create a mock backend
    mock_backend = MagicMock()
    
    # Set up read_graph to return different values on successive calls
    read_graph_calls = 0
    async def mock_read_graph():
        nonlocal read_graph_calls
        read_graph_calls += 1
        return KnowledgeGraph(entities=sample_entities, relations=[])
    mock_backend.read_graph = mock_read_graph
    
    # Make add_observations a coroutine
    async def mock_add_observations(entity_name, observations):
        pass
    mock_backend.add_observations = mock_add_observations
    
    # Make store_embedding a coroutine
    async def mock_store_embedding(entity_name, vector):
        pass
    mock_backend.store_embedding = mock_store_embedding
    
    # Create a knowledge graph manager with the mock backend
    with patch("memory_mcp_server.knowledge_graph_manager.EmbeddingService", return_value=mock_embedding_service):
        kg_manager = KnowledgeGraphManager(mock_backend)
        
        # Add observations
        await kg_manager.add_observations("Python", ["New observation"])
        
        # Verify that store_embedding was called to update the embedding
        mock_backend.store_embedding.assert_called_once()
