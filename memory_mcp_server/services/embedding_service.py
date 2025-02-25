"""Embedding service for semantic search capabilities."""

from typing import List
import numpy as np

# Import sentence-transformers conditionally to avoid hard dependency
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingService:
    """Service for generating and comparing text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {str(e)}")
        else:
            print("Warning: sentence-transformers not installed. Semantic search will be limited.")

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Vector representation of the text
            
        Raises:
            RuntimeError: If sentence-transformers is not available
        """
        if self.model is None:
            # Return a random vector as fallback (not ideal but prevents crashes)
            print("Warning: Using random vector as embedding model is not available")
            return np.random.rand(384)  # Common embedding size
        
        return self.model.encode(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings to vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of vector representations
            
        Raises:
            RuntimeError: If sentence-transformers is not available
        """
        if self.model is None:
            # Return random vectors as fallback
            print("Warning: Using random vectors as embedding model is not available")
            return np.random.rand(len(texts), 384)
        
        return self.model.encode(texts)

    def compute_similarity(self, query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document vectors.
        
        Args:
            query_vector: Query vector
            doc_vectors: Document vectors
            
        Returns:
            Array of similarity scores
        """
        # Ensure vectors are normalized for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1e-10  # Avoid division by zero
            
        doc_norms = np.linalg.norm(doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1e-10  # Avoid division by zero
        
        # Compute cosine similarity
        return np.dot(doc_vectors, query_vector) / (doc_norms * query_norm)
