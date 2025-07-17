"""
elsa.py

Implementation of the Efficient Linear Sparse Autoencoder (ELSA) and its training logic.

ELSA learns a sparse, low-dimensional item embedding by projecting user-item interactions
into a latent space with normalized linear weights. It is trained using a reconstruction
loss computed over normalized outputs.

This module includes:
- A PyTorch `ELSA` model with encoder/decoder operations.
- Training with early stopping and norm constraints.
- Utility functions for normalization and loss computation.

ADAPTED FROM: https://anonymous.4open.science/r/knots-to-knobs
"""
import logging
import numpy as np
from typing import List, Tuple
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseRecommender

logger = logging.getLogger(__name__)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2-normalizes a tensor along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to normalize.

    Returns:
        torch.Tensor: L2-normalized tensor.
    """
    norm = x.norm(dim=dim, keepdim=True)
    return x / (norm + eps)


def normalized_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Computes mean squared error between L2-normalized predictions and ground truth.

    Args:
        y_pred (torch.Tensor): Predicted output tensor.
        y_true (torch.Tensor): Ground-truth tensor.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    return (l2_normalize(y_pred) - l2_normalize(y_true)).pow(2).sum(-1).mean()


class ELSA(nn.Module):
    """
    Efficient Linear Sparse Autoencoder (ELSA) model for user-item representation learning.

    This model learns an encoder matrix with unit-length rows and uses it to
    encode user interactions and reconstruct the original input.

    Attributes:
        encoder (nn.Parameter): Linear projection weights of shape (num_items, embedding_dim).
    """

    def __init__(self, input_dim: int, embedding_dim: int):
        """
        Initializes the ELSA model.

        Args:
            input_dim (int): Number of items (columns of the interaction matrix).
            embedding_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        self.encoder = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty([input_dim, embedding_dim])))
        self.normalize_encoder()

    @torch.no_grad()
    def normalize_encoder(self) -> None:
        """
        Normalizes the encoder weights to unit L2 norm for each row.
        This constraint ensures cosine-like behavior during optimization.
        """
        self.encoder.data = l2_normalize(self.encoder.data)
        if self.encoder.grad is not None:
            # Remove gradient component in direction of the encoder vector (projection-free update)
            self.encoder.grad -= (self.encoder.grad *
                                  self.encoder.data).sum(-1, keepdim=True) * self.encoder.data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the input into the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_items).

        Returns:
            torch.Tensor: Latent representation of shape (batch_size, embedding_dim).
        """
        return x @ self.encoder

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input from latent representations.

        Args:
            e (torch.Tensor): Latent tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, num_items).
        """
        return e @ self.encoder.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encodes and decodes input, then applies ReLU to the residual.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Residual reconstruction output.
        """
        return nn.ReLU()(self.decode(self.encode(x)) - x)

    def compute_loss_dict(self, batch: torch.Tensor) -> dict:
        """
        Computes the reconstruction loss on a batch.

        Args:
            batch (torch.Tensor): Batch of input data.

        Returns:
            dict: Dictionary containing the loss under key "Loss".
        """
        return {"Loss": normalized_mse_loss(self(batch), batch)}

    def train_model(
        self,
        X_tensor: torch.Tensor,
        lr: float,
        epochs: int,
        batch_size: int,
        early_stopping: bool,
        patience: int,
        verbose: bool = True
    ) -> None:
        """
        Trains the ELSA model on user-item interaction data.

        Args:
            X_tensor (torch.Tensor): Full dataset tensor of shape (num_users, num_items).
            lr (float): Learning rate for Adam optimizer.
            epochs (int): Maximum number of epochs to train.
            batch_size (int): Number of users per training batch.
            early_stopping (bool): Whether to apply early stopping based on validation loss.
            patience (int): Number of epochs with no improvement before stopping.
            verbose (bool): If True, prints loss progress during training.
        """
        device = next(self.parameters()).device
        dataloader = DataLoader(TensorDataset(
            X_tensor), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for (batch,) in dataloader:
                batch = batch.to(device)
                loss = self.compute_loss_dict(batch)["Loss"]

                if torch.isnan(loss):
                    logger.warning(
                        f"[ELSA] NaN loss detected at epoch {epoch + 1}. Skipping batch.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                self.normalize_encoder()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            avg_loss = total_loss / len(X_tensor)
            if verbose:
                logger.info(
                    f"[ELSA Epoch {epoch + 1}] Avg Loss: {avg_loss:.6f}")

            if early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        if verbose:
                            logger.info(
                                f"[INFO] Early stopping at epoch {epoch + 1}")
                        break

        self.eval()


class ELSARecommender(BaseRecommender):
    """
    Recommender wrapper for the ELSA model.

    This class fits the ELSA autoencoder to the user-item interaction matrix
    and provides top-N item recommendations for any user.

    Attributes:
        model (ELSA): The underlying trained autoencoder model.
        config (dict): Training and model parameters.
    """

    def __init__(self, config: dict = None):
        """
        Initializes the recommender using config parameters.

        Args:
            config (dict, optional): Configuration dictionary. Falls back to ALGO_PARAMS["elsa"] if not provided.
        """
        super().__init__(config)
        self.embedding_dim = self.config.get("embedding_dim", 64)
        self.lr = self.config.get("lr", 0.01)
        self.epochs = self.config.get("epochs", 200)
        self.batch_size = self.config.get("batch_size", 64)
        self.patience = self.config.get("patience", 10)
        self.device = torch.device(self.config.get("device", "cpu"))
        self.model = None
        self.input_dim = None

    def fit(self, interaction_matrix: csr_matrix):
        """
        Fits the ELSA model to the user-item interaction matrix.

        Args:
            interaction_matrix (csr_matrix): Binary implicit feedback matrix (users x items).
        """
        self.input_dim = interaction_matrix.shape[1]

        X = interaction_matrix.astype(np.float32).toarray()
        X_tensor = torch.tensor(X).to(self.device)

        self.model = ELSA(self.input_dim, self.embedding_dim).to(self.device)
        self.model.train_model(
            X_tensor,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            early_stopping=True,
            patience=self.patience
        )

    def recommend(self, user_vector: np.ndarray, top_n: int = 5, **kwargs) -> List[Tuple[int, float]]:
        """
        Recommends top-N unseen items for a given user vector.

        Args:
            user_vector (np.ndarray): Binary interaction vector (1D array) for the user.
            top_n (int): Number of items to recommend.

        Returns:
            List[Tuple[int, float]]: Ranked list of (item_index, score).
        """
        user_tensor = torch.tensor(
            user_vector.astype(np.float32)).to(self.device)

        with torch.no_grad():
            scores = self.model(user_tensor).cpu().numpy().flatten()

        scores[user_vector > 0] = -np.inf
        top_indices = np.argpartition(-scores, range(top_n))[:top_n]
        top_scores = scores[top_indices]

        return sorted(zip(top_indices, top_scores), key=lambda x: -x[1])

    def save_model(self, filepath: str):
        """
        Saves the model checkpoint and configuration to disk.

        Args:
            filepath (str): Path to save the model.
        """
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "input_dim": self.input_dim
        }, filepath)

    def load_model(self, filepath: str):
        """
        Loads the model checkpoint from disk.

        Args:
            filepath (str): Path to the saved model file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = ELSA(
            input_dim=checkpoint["input_dim"],
            embedding_dim=checkpoint["config"]["embedding_dim"]
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
