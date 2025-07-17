"""
item_knn.py

This module implements item-based collaborative filtering using cosine similarity between item
columns in the user-item interaction matrix.

It recommends items to a user based on how similar they are to the items the user has 
already interacted with.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """
    Item-based collaborative filtering model using cosine similarity.

    Computes item-item similarities and ranks items for each user
    by aggregating the scores of similar items they've interacted with.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the model with optional config.

        Currently supports only cosine similarity.

        Parameters:
        - config (dict): Optional hyperparameters (e.g., 'metric')
        """
        super().__init__(config)
        self.metric = (self.config or {}).get("metric", "cosine")
        self.similarity_matrix = None

    def fit(self, interaction_matrix: np.ndarray):
        """
        Compute the item-item similarity matrix from the transposed interaction matrix.

        Parameters:
        - interaction_matrix (np.ndarray): Sparse user-item matrix.
        """
        super().fit()

        if self.metric != "cosine":
            raise ValueError(f"Unsupported similarity metric: {self.metric}")

        # Compute cosine similarity between item columns
        self.similarity_matrix = cosine_similarity(interaction_matrix.T)

    def get_similar(self, perfume_index: int, top_n=5):
        """
        Return top-N items most similar to the given item index.

        Parameters:
        - perfume_index (int): Target item index
        - top_n (int): Number of similar items to return

        Returns:
        - List[Tuple[int, float]]: (item index, similarity) pairs
        """
        sims = self.similarity_matrix[perfume_index]
        sims[perfume_index] = -np.inf  # exclude self

        top_indices = np.argpartition(-sims, range(top_n))[:top_n]
        top_scores = sims[top_indices]
        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])

    def recommend(self, user_vector: int, top_n: int = 5, *args, **kwargs):
        """
        Recommend items for the given user using item similarity aggregation.

        Parameters:
        - user_vector #TODO documentm e
        - top_n (int): Number of items to recommend

        Returns:
        - List[Tuple[int, float]]: (item index, score) pairs
        """

        row = user_vector.flatten()

        if np.count_nonzero(row) == 0:
            return []

        # Score all items based on similarity to user's interacted items
        scores = (user_vector @ self.similarity_matrix).ravel()

        # Filter out already interacted items
        seen_indices = seen_indices = np.nonzero(user_vector)[0]
        scores[seen_indices] = -np.inf

        valid_indices = np.where(scores != -np.inf)[0]
        if len(valid_indices) == 0:
            return []

        top_n = min(top_n, len(valid_indices))
        top_indices = np.argpartition(-scores, range(top_n))[:top_n]
        top_scores = scores[top_indices]
        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])
