"""
Utilities for fitting, caching, and loading perfume vectorizers and vector matrices.
"""
import os
import pickle
import logging
import numpy as np

from .vectorizer import PerfumeVectorizer

logger = logging.getLogger(__name__)


def load_or_fit_vectors(perfumes, vectorizer_path: str, vectors_path: str):
    """
    Load cached vectorizer and vectors if available; otherwise fit and save them.

    Args:
        perfumes (list[Perfume]): List of ORM perfumes.
        vectorizer_path (str): Path to the serialized vectorizer.
        vectors_path (str): Path to the serialized numpy vector matrix.

    Returns:
        tuple: (vectorizer, np.ndarray of vectors)
    """
    if os.path.exists(vectorizer_path) and os.path.exists(vectors_path):
        logger.info("Loading cached vectorizer and vectors...")
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        vectors = np.load(vectors_path)
    else:
        logger.info("Fitting vectorizer on perfumes...")
        vectorizer = PerfumeVectorizer()
        vectors = vectorizer.fit_transform(perfumes)

        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        logger.info("Saving vectorizer and vectors...")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        np.save(vectors_path, vectors)

    return vectorizer, vectors
