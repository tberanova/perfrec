"""
content.py

Implements a content-based recommender that ranks perfumes based on vector similarity.
Uses cosine or Jaccard similarity over precomputed feature vectors (e.g., tags, notes, embeddings).
"""
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Binarizer
from scipy.spatial.distance import pdist, squareform
from .base import BaseRecommender


class ContentSimilarityRecommender(BaseRecommender):
    """
    Content-based recommender using precomputed perfume vectors.

    Supports cosine or Jaccard similarity between items.
    """

    def __init__(self, perfume_vectors, index_to_perfume_id, config: dict = None):
        """
        Initialize the model with vector representations and similarity type.

        Parameters:
        - perfume_vectors (np.ndarray): Matrix of item feature vectors
        - index_to_perfume_id (dict): Maps matrix row index to perfume ID
        - config (dict): Config with 'similarity': 'cosine' | 'jaccard'
        """
        super().__init__(config)
        self.perfume_vectors = perfume_vectors
        self.index_to_perfume_id = index_to_perfume_id
        self.similarity = (self.config or {}).get("similarity", "cosine")
        self.similarity_matrix = None

    def _compute_similarity_matrix(self):
        """
        Compute item-item similarity matrix based on selected similarity metric.
        """
        if self.similarity == "cosine":
            return cosine_similarity(self.perfume_vectors)

        elif self.similarity == "jaccard":
            binarized = Binarizer(threshold=0.0).fit_transform(
                self.perfume_vectors)
            jaccard_distances = pdist(binarized, metric="jaccard")
            return 1 - squareform(jaccard_distances)

        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

    def fit(self, *args):
        """
        Precompute item-item similarity matrix.
        """
        self.similarity_matrix = self._compute_similarity_matrix()

    def get_similar(self, perfume_id: int, top_n: int = 5):
        """
        Get top-N perfumes most similar to the given perfume ID.

        Parameters:
        - perfume_id (int): ID of target perfume
        - top_n (int): Number of similar items to return

        Returns:
        - List of (index, similarity score) tuples
        """
        if perfume_id not in self.index_to_perfume_id.values():
            return []

        index = next(i for i, pid in self.index_to_perfume_id.items()
                     if pid == perfume_id)
        sims = self.similarity_matrix[index]
        sims[index] = -np.inf  # exclude self

        top_indices = np.argpartition(-sims, range(top_n))[:top_n]
        top_scores = sims[top_indices]
        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])

    def recommend(self, user_vector: ArrayLike, top_n: int = 5, **kwargs) -> List[Tuple[int, float]]:
        """
        Recommend items based on user preferences and content similarity.

        Parameters:
        - user_vector (np.ndarray): 1D binary array indicating user interactions.
        - top_n (int): Number of items to recommend.

        Returns:
        - List[Tuple[int, float]]: Sorted (item_index, score) recommendations.
        """
        user_vector = np.asarray(user_vector).ravel()

        # Compute similarity scores to unseen items
        scores = user_vector @ self.similarity_matrix
        scores[user_vector > 0] = -np.inf  # exclude seen items

        top_indices = np.argpartition(-scores, range(top_n))[:top_n]
        top_scores = scores[top_indices]
        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])
