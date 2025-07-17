"""
elsae.py

This module implements the ELSAERecommender — a hybrid recommendation model that combines
dense representation learning (via ELSA) with sparse interpretable encoding (via TopKSAE).

Key components:
- ELSA: Learns compact dense embeddings from interaction matrices using autoencoder-style training.
- TopKSAE: Enforces interpretability through sparsity and top-K filtering of neuron activations.

This model supports both traditional recommendation (top-N items) and explainable recommendation
via neuron-tag mappings built externally.

Expected usage:
- Instantiate with a config dictionary
- Call `fit(interaction_matrix)` to train
- Use `recommend(user_index)` or `recommend_with_explanations(...)` to generate recommendations

ADAPTED FROM https://anonymous.4open.science/r/knots-to-knobs.
"""

from typing import List, Tuple

import torch
import numpy as np
from scipy.sparse import csr_matrix

from .base import BaseRecommender
from .elsa import ELSA
from .sae import TopKSAE


class ELSAERecommender(BaseRecommender):
    """
    Embedded ELSA + TopKSAE recommender.

    Combines two stages:
    1. ELSA: A neural encoder producing dense user representations from interaction data.
    2. TopKSAE: A sparse autoencoder that filters those representations to extract interpretable 
       latent factors.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the model with hyperparameters loaded from config.
        """
        super().__init__(config)
        cfg = self.config

        # Load hyperparameters from config
        self.embedding_dim_elsa = cfg.get("embedding_dim_elsa", 128)
        self.embedding_dim_sae = cfg.get("embedding_dim_sae", 512)
        self.sae_k = cfg.get("sae_k", 16)
        self.sae_l1_coef = cfg.get("sae_l1_coef", 0.02)
        self.lr = cfg.get("lr", 0.01)
        self.batch_size = cfg.get("batch_size", 64)
        self.device = torch.device(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        # default True, but can be disabled
        self.elsa_epochs = cfg.get("elsa_epochs", 200)
        self.sae_epochs = cfg.get("sae_epochs", 100)
        self.early_stopping = cfg.get("early_stopping", True)
        self.patience = cfg.get("patience", 10)

        # Placeholders for internal model components
        self.elsa = None
        self.sae = None

        # To be set during fit
        self.num_users = None
        self.num_items = None

    def fit(self, interaction_matrix: csr_matrix):
        """
        Train both ELSA and TopKSAE using the given user-item interaction matrix.
        """
        super().fit()

        # Prepare data as dense tensor for ELSA
        X = interaction_matrix.astype(np.float32).toarray()
        self.num_users, self.num_items = X.shape
        X_tensor = torch.Tensor(X).to(self.device)

        # Initialize and train ELSA
        self.elsa = ELSA(input_dim=self.num_items,
                         embedding_dim=self.embedding_dim_elsa).to(self.device)
        self.elsa.train_model(
            X_tensor,
            lr=self.lr,
            epochs=self.elsa_epochs,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            patience=self.patience,
            verbose=True
        )

        # Encode all users using trained ELSA
        with torch.no_grad():
            user_embeddings = self.elsa.encode(X_tensor).detach()

        # Initialize and train TopKSAE on the ELSA embeddings
        self.sae = TopKSAE(
            input_dim=self.embedding_dim_elsa,
            embedding_dim=self.embedding_dim_sae,
            reconstruction_loss="Cosine",
            l1_coef=self.sae_l1_coef,
            k=self.sae_k
        ).to(self.device)
        self.sae.train_model(
            user_embeddings,
            lr=self.lr,
            epochs=self.sae_epochs,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            patience=self.patience,
        )

    def recommend(self, user_vector: np.ndarray, top_n: int = 5, **kwargs) -> List[Tuple[int, float]]:
        """
        Recommend top-N perfumes for a given user vector.

        Parameters:
        - user_vector (np.ndarray): 1D binary vector representing user interactions.
        - top_n (int): Number of recommendations to return.

        Returns:
        - List of (item_index, score) tuples, sorted by score descending.
        """
        user_tensor = torch.Tensor(
            user_vector.astype(np.float32)).to(self.device)

        with torch.no_grad():
            # ELSA encoding → SAE encoding → ELSA decoding
            elsa_emb = self.elsa.encode(user_tensor)
            sae_emb = self.sae(elsa_emb)[0]
            scores = self.elsa.decode(sae_emb).cpu().numpy().flatten()

        scores[user_vector > 0] = -np.inf
        top_indices = np.argpartition(-scores, range(top_n))[:top_n]
        top_scores = scores[top_indices]
        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])

    def recommend_with_explanations(
        self,
        user_vector: np.ndarray,
        neuron_tag_dict: dict,
        top_n: int = 5,
        top_k_neurons: int = 16,
        **kwargs
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, List[str]]]]:
        """
        Recommend top-N perfumes and return top-K active SAE neurons with their tag explanations.

        Parameters:
        - user_vector (np.ndarray): 1D binary array of user interactions.
        - neuron_tag_dict (dict): Dictionary mapping neuron IDs to tag-score mappings.
        - top_n (int): Number of perfumes to recommend.
        - top_k_neurons (int): Number of active neurons to explain.

        Returns:
        - List of (item_index, score) recommendation tuples.
        - List of (neuron_id, top_tags) explanation tuples.
        """
        user_tensor = torch.tensor(
            user_vector.astype(np.float32)).to(self.device)

        with torch.no_grad():
            elsa_emb = self.elsa.encode(user_tensor)
            sae_emb = self.sae(elsa_emb)[0].squeeze(0)
            scores = self.elsa.decode(sae_emb).cpu().numpy().flatten()

        scores[user_vector > 0] = -np.inf
        top_indices = np.argpartition(-scores, range(top_n))[:top_n]
        top_scores = scores[top_indices]
        recommendations = sorted(
            zip(top_indices, top_scores), key=lambda x: -x[1])

        # Explanation: top activated neurons
        sae_array = sae_emb.cpu().numpy()
        top_neuron_ids = np.argsort(-sae_array)[:top_k_neurons]
        top_neuron_tags = []

        for neuron_id in top_neuron_ids:
            tag_scores = neuron_tag_dict.get(str(neuron_id), {})
            top_tags = sorted(tag_scores.items(), key=lambda x: -x[1])
            tag_list = [tag for tag, _ in top_tags]
            top_neuron_tags.append((neuron_id, tag_list))

        print(top_neuron_tags)
        return recommendations, top_neuron_tags

    def encode_sae(self, perfume_multihot_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a multihot perfume input tensor through ELSA and SAE.

        Returns:
        - Tensor of shape (batch_size, sae_embedding_dim)
        """
        with torch.no_grad():
            elsa_emb = self.elsa.encode(perfume_multihot_tensor)
            sae_emb = self.sae(elsa_emb)[0]
        return sae_emb
