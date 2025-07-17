"""
EASE (Embarrassingly Shallow AutoEncoder) Recommender.

Implements a linear item-item collaborative filtering model as described in:
"Embarrassingly Shallow Autoencoders for Sparse Data" (Steck, 2019).
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .base import BaseRecommender


class EASERecommender(BaseRecommender):
    """
    EASE (Embarrassingly Shallow AutoEncoder) recommender.

    Learns a linear item-item similarity matrix from co-occurrence and regularization.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the EASE model with optional regularization parameter.

        Parameters (via config):
        - lambda_ (float): L2 regularization weight (default: 250).
        """
        super().__init__(config)
        self.lambda_ = self.config.get("lambda_", 250)

        # Item similarity matrix (learned in fit)
        self.B = None

    def fit(self, interaction_matrix: csr_matrix):
        """
        Train the EASE model using closed-form ridge regression.

        Parameters:
        - interaction_matrix (csr_matrix): Sparse binary user-item matrix.
        """
        super().fit()

        X = interaction_matrix.astype(np.float32)
        G = X.T @ X                       # Gram matrix
        G += self.lambda_ * np.identity(G.shape[0])  # Regularization

        P = np.linalg.inv(G)
        B = -P / np.diag(P)[:, np.newaxis]
        np.fill_diagonal(B, 0.0)

        self.B = B

    def recommend(self, user_vector: np.ndarray, top_n: int = 5, **kwargs) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations using item-item scores.

        Parameters:
        - user_vector (np.ndarray): 1D binary interaction vector for the target user.
        - top_n (int): Number of items to recommend.

        Returns:
        - List[Tuple[int, float]]: Top-N (item_index, score) pairs, sorted descending by score.
        """
        scores = np.asarray(user_vector @ self.B).flatten()
        scores[user_vector > 0] = -np.inf  # Mask known items

        top_indices = np.argpartition(-scores, range(top_n))[:top_n]
        top_scores = scores[top_indices]

        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])
