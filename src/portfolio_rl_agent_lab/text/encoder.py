from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class NewsEncoder:
    """
    Encode a list of news strings into sentence embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Returns embeddings with shape (len(texts), dim)
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)
