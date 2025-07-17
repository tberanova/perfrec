"""
User-based k-Nearest Neighbors (UserKNN) Recommender.

This module implements a simple memory-based collaborative filtering algorithm that
computes cosine similarity between users. Each user's recommendation scores are
derived from the interactions of their most similar users.

The recommender supports:
- Top-K user similarity truncation
- Masking of already interacted items
- Efficient top-N recommendation generation

Usage:
- Call `fit(interaction_matrix)` to precompute similarities.
- Use `recommend(user_vector, user_index, top_n)` to get recommendations.

The model does not support cold-start users unless they are present in the interaction matrix.
"""
import warnings
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class UserKNNRecommender(BaseRecommender):
    """
    A user-based k-nearest neighbors recommender using cosine similarity.

    This model computes similarity between users based on the interaction matrix,
    then scores items for a target user by aggregating ratings from their top-k similar users.
    """

    def __init__(self, config: dict = None):
        """
        Initializes the UserKNNRecommender using parameters defined in the config.
        """
        super().__init__(config)
        self.top_k_neighbors = self.config.get("top_k_neighbors", 50)
        self.interaction_matrix = None
        self.user_similarity = None

    def fit(self, interaction_matrix: np.ndarray):
        """
        Computes the top-K user-user cosine similarity matrix and stores the matrix.

        Args:
            interaction_matrix (np.ndarray): User-item binary interaction matrix (CSR or dense).
        """
        self.interaction_matrix = interaction_matrix
        self._compute_similarity()

    def update(self):
        """
        Warns.
        """
        warnings.warn(
            "UserKNN does not support live updates. Please re-train with .fit().",
            category=UserWarning
        )

    def _compute_similarity(self):
        """Compute and store top-K cosine similarity matrix between users."""
        full_similarity = cosine_similarity(self.interaction_matrix)
        top_k_sim = np.zeros_like(full_similarity)

        for i in range(full_similarity.shape[0]):
            row = full_similarity[i]
            top_k_idx = np.argpartition(-row, self.top_k_neighbors +
                                        1)[:self.top_k_neighbors + 1]
            top_k_idx = top_k_idx[top_k_idx != i]  # remove self
            top_k_sim[i, top_k_idx] = row[top_k_idx]

        self.user_similarity = top_k_sim

    def recommend(
        self,
        user_vector: ArrayLike,
        user_index: int,
        top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N item recommendations for a given user by aggregating scores
        from top-k similar users.

        Args:
            user_vector (np.ndarray): 1D binary array of the user's interactions.
            user_index (int): Index of the user (used to access similarity row).
            top_n (int): Number of items to recommend.

        Returns:
            List[Tuple[int, float]]: List of (item_index, score) pairs.
        """
        seen_items = np.where(user_vector > 0)[0]
        similarities = self.user_similarity[user_index]

        scores = similarities @ self.interaction_matrix.toarray()
        scores[seen_items] = -np.inf

        valid_indices = np.where(scores != -np.inf)[0]
        if len(valid_indices) == 0:
            return []

        top_n = min(top_n, len(valid_indices))
        top_indices = np.argpartition(-scores, top_n)[:top_n]
        top_scores = scores[top_indices]

        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])
