"""
sae.py

Implements variants of a sparse autoencoder (SAE) for user/item representation learning.

Core components:
- SAE: Base autoencoder architecture with normalization and flexible loss modes
- BasicSAE: L1-regularized version
- TopKSAE: Sparsity-enforcing variant that zeroes all but top-K activations

Supported loss modes: L2, cosine

This module is used in hybrid neural recommenders like ELSAE.
ADAPTED FROM: https://anonymous.4open.science/r/knots-to-knobs
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2-normalize a tensor along a given dimension."""
    return x / x.norm(dim=dim, keepdim=True)


class SAE(nn.Module):
    """
    Standard autoencoder with input standardization and custom loss composition.

    The encoder and decoder weights are separately parameterized and trained jointly.
    Decoder weights are normalized during training to encourage disentanglement.
    """

    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss

        # Encoder: linear layer with bias
        self.encoder_w = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty([input_dim, embedding_dim])))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))

        # Decoder: linear layer with bias (normalized)
        self.decoder_w = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty([embedding_dim, input_dim])))
        self.decoder_b = nn.Parameter(torch.zeros(input_dim))
        self.normalize_decoder()

    def standardize_input(self, x: torch.Tensor):
        """Standardize input (zero mean, unit variance) along the last dimension."""
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-7
        return (x - x_mean) / x_std, x_mean, x_std

    def destandardize_output(self, out: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor):
        """Undo standardization using saved mean and std."""
        return out * x_std + x_mean

    def encode(self, x: torch.Tensor):
        """Apply encoder to standardized input."""
        x_std, x_mean, x_std_dev = self.standardize_input(x)
        e_pre = F.relu((x_std - self.decoder_b) @
                       self.encoder_w + self.encoder_b)
        e = self.post_process_embedding(e_pre)
        return e, e_pre, x_mean, x_std_dev

    def decode(self, e: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor):
        """Decode latent embedding and re-scale to match original input range."""
        out = e @ self.decoder_w + self.decoder_b
        return self.destandardize_output(out, x_mean, x_std)

    def forward(self, x: torch.Tensor):
        """Full forward pass (encode → decode)"""
        e, e_pre, x_mean, x_std = self.encode(x)
        out = self.decode(e, x_mean, x_std)
        return out, e, e_pre, x_mean, x_std

    def compute_loss_dict(self, batch: torch.Tensor):
        """
        Compute reconstruction and sparsity losses on a batch.

        Returns:
        - dict: Loss values keyed by name 
        """
        out, e, _, _, _ = self(batch)

        losses = {
            "L2": F.mse_loss(out, batch),
            "Cosine": (1 - F.cosine_similarity(batch, out, dim=1)).mean(),
            "L1": e.abs().sum(-1).mean(),
        }
        losses["Loss"] = self.total_loss(losses)
        return losses

    def train_step(self, optimizer: torch.optim.Optimizer, batch: torch.Tensor):
        """
        Perform one gradient update step using a batch.
        """
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_decoder()
        optimizer.step()
        return losses

    @torch.no_grad()
    def normalize_decoder(self):
        """
        Normalize decoder weights (row-wise L2 norm) to encourage orthogonality.
        """
        self.decoder_w.data = l2_normalize(self.decoder_w.data)

        # Project gradient to be orthogonal to weights (if it exists)
        if self.decoder_w.grad is not None:
            self.decoder_w.grad -= (
                self.decoder_w.grad * self.decoder_w.data
            ).sum(-1, keepdim=True) * self.decoder_w.data

    def train_model(
        self,
        X_tensor: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 1024,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: bool = True
    ):
        """
        Full training loop with optional early stopping.

        Parameters:
        - X_tensor (torch.Tensor): Full dataset tensor
        - lr (float): Learning rate
        - epochs (int): Max number of epochs
        - batch_size (int): Mini-batch size
        - early_stopping (bool): Enable early stopping
        - patience (int): Max no. of epochs without improvement
        - verbose (bool): Print epoch logs
        """
        device = next(self.parameters()).device
        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=batch_size,
            shuffle=True
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for (batch,) in dataset:
                batch = batch.to(device)
                loss = self.compute_loss_dict(batch)["Loss"]
                optimizer.zero_grad()
                loss.backward()
                self.normalize_decoder()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            avg_loss = total_loss / len(X_tensor)
            if verbose:
                logger.info(f"[SAE Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")

            if early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        self.eval()


class BasicSAE(SAE):
    """
    Standard L1-regularized autoencoder.
    Applies an L1 penalty to encourage sparse activations without explicit top-k enforcement.
    """

    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str, l1_coef: float = 0.01):
        super().__init__(input_dim, embedding_dim, reconstruction_loss)
        self.l1_coef = l1_coef

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        return e

    def total_loss(self, losses: dict) -> torch.Tensor:
        return losses[self.reconstruction_loss] + self.l1_coef * losses["L1"]


class TopKSAE(SAE):
    """
    Top-K sparse autoencoder that zeroes all but the top-K values in each embedding vector.
    """

    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str, l1_coef: float = 0.01, k: int = 32):
        super().__init__(input_dim, embedding_dim, reconstruction_loss)
        self.l1_coef = l1_coef
        self.k = k

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        e_topk = torch.topk(e, self.k, dim=-1)
        return torch.zeros_like(e).scatter(-1, e_topk.indices, e_topk.values)

    def total_loss(self, losses: dict) -> torch.Tensor:
        return losses[self.reconstruction_loss] + self.l1_coef * losses["L1"]
