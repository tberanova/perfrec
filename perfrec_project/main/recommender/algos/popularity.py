"""
popularity.py

This module implements a non-personalized popularity-based recommender that ranks items
based on their global `rating_count`, optionally filtering by user history and target gender.
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    Global popularity-based recommender.

    Ranks items by their overall `rating_count` field, optionally:
    - Filtering already-seen items for a specific user
    - Filtering by gender label
    """

    def __init__(self, perfumes):
        """
        Initialize the recommender.

        Parameters:
        - perfumes (List[Perfume]): ORM perfume objects, each with `.id`, `.gender`, `.rating_count`
        """
        self.perfumes = perfumes
        self.rating_counts = None

    def fit(self, *args):
        """
        Prepare item popularity scores.

        """
        super().fit()
        self.rating_counts = {
            p.id: p.rating_count or 0 for p in self.perfumes
        }

    def recommend(
        self,
        user_vector: Optional[np.ndarray] = None,
        top_n: int = 5,
        gender: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        Recommend top-N popular items, optionally filtered by gender and seen items.
        If gender is None, ensures roughly equal representation from Feminine, Masculine,
        and Unisex perfumes.

        Parameters:
        - user_vector (np.ndarray | None): If provided, filters out items the user has seen.
        - top_n (int): Number of items to return.
        - gender (str | None): If specified, filters perfumes by gender.

        Returns:
        - List[Tuple[int, float]]: List of (matrix_item_index, popularity_score) pairs.
        """
        candidates = {
            "female": [],
            "male": [],
            "unisex": []
        }

        for i, perfume in enumerate(self.perfumes):
            perfume_gender = (perfume.gender or "").lower()
            score = self.rating_counts.get(perfume.id, 0)

            # If filtering by a specific gender
            if gender and gender.lower() != "none":
                if perfume_gender != gender.lower():
                    continue
                candidates.setdefault(gender.lower(), []).append((i, score))
            else:
                # Bucket into gender categories
                if perfume_gender in candidates:
                    candidates[perfume_gender].append((i, score))

        # Remove seen items
        if user_vector is not None:
            seen = set(np.flatnonzero(user_vector))
            for g in candidates:
                candidates[g] = [(i, s)
                                 for i, s in candidates[g] if i not in seen]

        if gender and gender.lower() != "none":
            # Return top-N from the filtered gender
            return sorted(candidates.get(gender.lower(), []), key=lambda x: -x[1])[:top_n]

        result = []
        remaining = top_n
        genders = ["female", "male", "unisex"]

        # Sort all buckets in advance
        sorted_buckets = {
            g: sorted(candidates.get(g, []), key=lambda x: -x[1])
            for g in genders
        }

        while remaining > 0:
            added_any = False
            for g in genders:
                if sorted_buckets[g]:
                    result.append(sorted_buckets[g].pop(0))
                    remaining -= 1
                    added_any = True
                    if remaining == 0:
                        break
            if not added_any:
                break  # No more items in any bucket

        return result
