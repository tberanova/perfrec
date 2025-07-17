"""
Bayesian Personalized Ranking (BPR) Recommender.

This model learns latent representations for users and items using pairwise ranking loss,
optimized via stochastic gradient descent with bootstrap sampling.

References:
- Rendle et al. (2012). BPR: Bayesian Personalized Ranking from Implicit Feedback. 
  arXiv:1205.2618. https://arxiv.org/abs/1205.2618
- Milogradskii et al. (2024). Revisiting BPR: A Replicability Study of a Common 
  Recommender System Baseline. 
  In *Proceedings of the 18th ACM Conference on Recommender Systems 
  (RecSys '24)*, pp. 267–277. https://doi.org/10.1145/3640457.3688073
"""

import warnings
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from .base import BaseRecommender


class BPRRecommender(BaseRecommender):
    """
    Bayesian Personalized Ranking recommender using matrix factorization.

    Learns user and item latent factors optimized to rank positive items above negative ones.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the BPR model with hyperparameters from config.

        Parameters (via config):
        - embedding_dim (int): Size of latent factor vectors.
        - lr (float): Learning rate.
        - reg (float): L2 regularization strength.
        - epochs (int): Number of training epochs.
        - biases (bool): Whether to use item biases.
        """
        super().__init__(config)
        cfg = self.config

        self.num_factors = cfg.get("embedding_dim", 128)
        self.learning_rate = cfg.get("lr", 0.05)
        self.reg = cfg.get("reg", 0.005)
        self.epochs = cfg.get("epochs", 40)
        self.use_bias = cfg.get("biases", True)

        # To be initialized later
        self.num_users = None
        self.num_items = None
        self.user_factors = None
        self.item_factors = None
        self.item_bias = None

    def fit(self, interaction_matrix: np.ndarray):
        """
        Train the model using pairwise ranking loss.

        Parameters:
        - interaction_matrix (np.ndarray): User-item sparse matrix (CSR format).
        """
        print(f"[INFO] Fitting model: {self.__class__.__name__}")
        self.num_users, self.num_items = interaction_matrix.shape
        self._train_matrix = interaction_matrix

        # Extract observed positive samples (S)
        user_ids, pos_items = interaction_matrix.nonzero()
        pos_samples = list(zip(user_ids, pos_items))
        num_samples = len(pos_samples)

        # Initialize latent factors
        self.user_factors = np.random.normal(
            scale=0.1, size=(self.num_users, self.num_factors)
        )
        self.item_factors = np.random.normal(
            scale=0.1, size=(self.num_items, self.num_factors)
        )
        if self.use_bias:
            self.item_bias = np.zeros(self.num_items)

        for epoch in range(self.epochs):
            sampled_indices = np.random.randint(num_samples, size=num_samples)
            pbar = tqdm(sampled_indices,
                        desc=f"BPR Epoch {epoch + 1}", leave=False)

            for idx in pbar:
                user, pos_item = pos_samples[idx]

                # In theory, BPR defines a training set of all triples (u, i, j) such that:
                # (u, i) ∈ S (positive interactions), and (u, j) ∉ S (negative items).
                # This leads to O(|S| * |I|) training samples, which is impractical to store
                # or iterate over explicitly for large datasets (especially in Python).
                # Therefore, we adopt stochastic sampling of j during training.
                neg_item = self._sample_negative(user)
                self._update_factors(user, pos_item, neg_item)

        self._train_matrix = None  # cleanup

    def _sample_negative(self, user: int) -> int:
        """Sample a negative (non-interacted) item for the given user."""
        while True:
            neg_item = np.random.randint(self.num_items)
            if self._train_matrix[user, neg_item] == 0:
                return neg_item

    def _update_factors(self, user: int, pos_item: int, neg_item: int):
        """
        Perform one SGD update on the sampled triple (user, positive item, negative item).
        """
        u = self.user_factors[user]
        i = self.item_factors[pos_item]
        j = self.item_factors[neg_item]

        x_uij = np.dot(u, i - j)
        if self.use_bias:
            x_uij += self.item_bias[pos_item] - self.item_bias[neg_item]

        sigmoid = 1 / (1 + np.exp(-x_uij))

        grad_u = (sigmoid - 1) * (i - j) + self.reg * u
        grad_i = (sigmoid - 1) * u + self.reg * i
        grad_j = -(sigmoid - 1) * u + self.reg * j

        self.user_factors[user] -= self.learning_rate * grad_u
        self.item_factors[pos_item] -= self.learning_rate * grad_i
        self.item_factors[neg_item] -= self.learning_rate * grad_j

        if self.use_bias:
            grad_bi = (sigmoid - 1) + self.reg * self.item_bias[pos_item]
            grad_bj = -(sigmoid - 1) + self.reg * self.item_bias[neg_item]
            self.item_bias[pos_item] -= self.learning_rate * grad_bi
            self.item_bias[neg_item] -= self.learning_rate * grad_bj

    def recommend(
        self,
        user_vector: ArrayLike,
        user_index: int,
        top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a given user using learned latent embeddings.

        Parameters:
        - user_vector (np.ndarray): Binary interaction vector (used only to mask seen items).
        - user_index (int): Index of the user in latent matrix.
        - top_n (int): Number of items to recommend.

        Returns:
        - List[Tuple[int, float]]: Sorted list of (item_index, predicted_score).
        """
        scores = np.dot(self.item_factors, self.user_factors[user_index])
        if self.use_bias:
            scores += self.item_bias

        seen_items = np.flatnonzero(user_vector)
        scores[seen_items] = -np.inf

        top_indices = np.argpartition(-scores, top_n)[:top_n]
        top_scores = scores[top_indices]
        sorted_indices = top_indices[np.argsort(-top_scores)]

        return [(int(idx), float(scores[idx])) for idx in sorted_indices]

    def update(self) -> None:
        """
        Warns.
        """
        warnings.warn(
            "BPR does not support live updates. Please re-train with .fit().",
        )
