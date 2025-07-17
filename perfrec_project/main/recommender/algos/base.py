"""
Base Recommender Module.

This module defines the abstract base class `BaseRecommender`, which provides a unified interface
for all recommender algorithms in the system. It standardizes methods for model training, updating,
and persistence (saving/loading), and enforces implementation of a `recommend()` method.

Each recommender implementation (e.g., EASE, BPR, KNN-based) should inherit from this class to
ensure compatibility with the recommendation manager and broader application pipeline.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple

from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """
    Abstract base class for all recommender system implementations.
    Provides standard methods for model persistence and basic training/updating behavior.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def fit(self) -> None:
        """
        Default training method for models that require only access to the full interaction matrix.

        Subclasses can override this method for model-specific fitting logic.
        """
        logger.info(f"Fitting model: {self.__class__.__name__}")

    @abstractmethod
    def recommend(
        self,
        user_vector: ArrayLike,
        user_index: int,
        top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Abstract method to generate top-N recommendations for a given user.

        Parameters:
        - user_vector (np.ndarray): 1D binary array representing the user's interactions.
                                    Used by most models for scoring or masking.
        - user_index (int): Index of the user in the interaction matrix. Required for models
                            that use learned user embeddings (e.g., BPR).
        - top_n (int): Number of recommendations to return.

        Returns:
        - List[Tuple[int, float]]: Sorted list of (item_index, relevance_score) tuples.
        """

    def update(self) -> None:
        """
        Update the internal interaction matrix reference.

        Can be overridden to trigger partial retraining or other update behavior.
        """

    def save_model(self, filepath: str) -> None:
        """
        Persist the model's internal state to disk using pickle.

        Parameters:
        - filepath (str): Destination file path.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, filepath: str) -> None:
        """
        Load model state from a pickle file.

        Parameters:
        - filepath (str): Path to the saved model file.

        Raises:
        - FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such model file: {filepath}")

        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
