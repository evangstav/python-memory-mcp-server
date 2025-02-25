from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a vector."""
        return self.model.encode(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings to vectors."""
        return self.model.encode(texts)

    def compute_similarity(
        self, query_vector: np.ndarray, doc_vectors: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and document vectors."""
        return np.dot(doc_vectors, query_vector) / (
            np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
        )
