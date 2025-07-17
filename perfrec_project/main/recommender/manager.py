"""
High-level orchestration layer between raw recommendation algorithms and Django ORM.
This manager loads interaction data, trains or loads models, and exposes
personalized/top/content-based recommendation endpoints for use in views.

Configuration settings (enabled algorithms, file paths, feature toggles, and alg. hyperparameters)
are defined in config.py.
"""

import os
import json
import logging
import warnings

from main.config import (
    ENABLED_ALGORITHMS,
    NEURON_TAGGING_ENABLED,
    NEURON_TAG_PATH,
    VECTORIZER_PATH,
    VECTORS_PATH,
    MODEL_PATH_TEMPLATE,
    ALGO_PARAMS,
    CATEGORY_ATTRS,
)
from main.models import Perfume

from .utils.data_loader import load_perfume_interaction_data
from .utils.vectorizer_io import load_or_fit_vectors
from .utils.boosting import soft_boost
from .utils.user_matrix import add_user_row_to_matrix, update_user_row_in_matrix
from .utils.neuron_labeling import build_neuron_tag_dict

from .algos.i_knn import ItemKNNRecommender
from .algos.u_knn import UserKNNRecommender
from .algos.bpr import BPRRecommender
from .algos.ease import EASERecommender
from .algos.popularity import PopularityRecommender
from .algos.similarity import ContentSimilarityRecommender
from .algos.elsa import ELSARecommender
from .algos.elsa_sae import ELSAERecommender

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{message}")


warnings.showwarning = custom_warning_handler


class RecommenderManager:
    """
    Entry point for managing recommendation logic.
    Loads models, manages state, and facilitates view endpoints.

    Key Methods:
    - load_perfume_vectors(): Loads or fits content-based perfume vectors
    - train_algorithms(): Initializes, trains or loads all enabled recommenders
    - update_algorithms(): Refreshes models with updated interaction data
    - get_most_popular(): Returns top-N globally popular perfumes (optionally filtered by gender)
    - get_most_similar_by_content(): Finds similar perfumes via content-based similarity
    - get_most_similar_by_collab(): Finds similar perfumes using collaborative filtering
    - recommend_for_user(): Personalized recommendations with optional contextual boosting.
    - recommend_with_explanations(): Personalized ELSAE recommendations with neuron-level tags
    - add_user_row(): Appends a new user to the interaction matrix
    - update_user_row(): Updates a user’s interactions in the matrix based on current preferences
    """

    def __init__(self):
        # Load interaction data and mappings
        data = load_perfume_interaction_data()
        self.interaction_matrix = data["interaction_matrix"]
        self.real_users = data["real_users"]
        self.django_user_id_to_user = data["django_user_id_to_user"]
        self.django_user_id_to_matrix_id = data["django_user_id_to_matrix_id"]
        self.ext_perfume_id_to_matrix_id = data["ext_perfume_id_to_matrix_id"]
        self.matrix_id_to_ext_perfume_id = data["matrix_id_to_ext_perfume_id"]
        self.matrix_id_to_orm_perfume = data["matrix_id_to_orm_perfume"]

        # ORM perfumes with valid external ID
        self.perfumes = list(
            Perfume.objects.exclude(external_id__isnull=True).exclude(
                external_id__exact="")
        )

        # Load content vectors and train models
        self.load_perfume_vectors()
        self.train_algorithms()

    def load_perfume_vectors(self):
        """
        Loads or fits + caches content vectors for perfumes.
        Also builds ID → index mapping for content models.
        """
        self.vectorizer, self.perfume_vectors = load_or_fit_vectors(
            self.perfumes,
            vectorizer_path=VECTORIZER_PATH,
            vectors_path=VECTORS_PATH
        )
        self.perfume_id_to_vector_index = {
            p.id: i for i, p in enumerate(self.perfumes)
        }

    def train_algorithms(self):
        """
        Loads or fits each enabled algorithm and stores it in self.algorithms.
        Models are loaded from disk if possible; otherwise trained and saved.
        """
        self.algorithms = {}

        algorithm_classes = {
            "elsae": ELSAERecommender,
            "ease": EASERecommender,
            "user_knn": UserKNNRecommender,
            "bpr": BPRRecommender,
            "elsa": ELSARecommender,
            "item_knn": ItemKNNRecommender
        }

        # Initialize algorithms with parameters
        for name, cls in algorithm_classes.items():
            if ENABLED_ALGORITHMS.get(name, False):
                config = ALGO_PARAMS.get(name, {})
                self.algorithms[name] = cls(config)

        # Special case initializations without params
        if ENABLED_ALGORITHMS.get("popularity"):
            self.algorithms["popularity"] = PopularityRecommender(
                self.perfumes)
        if ENABLED_ALGORITHMS.get("content"):
            self.algorithms["content"] = ContentSimilarityRecommender(
                self.perfume_vectors,
                index_to_perfume_id={i: p.id for i,
                                     p in enumerate(self.perfumes)},
                config=ALGO_PARAMS.get("content", {})
            )

        # Train/load each model
        for name, algo in self.algorithms.items():
            model_path = MODEL_PATH_TEMPLATE.format(name=name)
            try:
                algo.load_model(model_path)
                algo.update()
                logger.info(f"Model loaded from {model_path}")
            except FileNotFoundError:
                logger.info(f"No saved model for {name}. Training...")
                algo.fit(self.interaction_matrix)
                algo.save_model(model_path)
                logger.info(f"Model saved to {model_path}.")

                if name == "elsae" and NEURON_TAGGING_ENABLED:
                    build_neuron_tag_dict(algo, self.matrix_id_to_orm_perfume)

        # Optional loading of precomputed neuron tags
        if "elsae" in self.algorithms and os.path.exists(NEURON_TAG_PATH):
            with open(NEURON_TAG_PATH, encoding="utf-8") as f:
                self.neuron_tag_dict = json.load(f)

    def update_algorithms(self):
        """Updates internal state of each model with new interaction matrix."""
        for algo in self.algorithms.values():
            algo.update()

    def get_perfume_by_id(self, perfume_id):
        """Fetches a perfume object by primary key. Returns None if not found."""
        try:
            return Perfume.objects.get(id=perfume_id)
        except Perfume.DoesNotExist:
            return None

    def get_most_popular(self, top_n=5, gender=None):
        """
        Wrapper around the popularity recommender.
        Returns top-N perfumes, optionally filtered by gender.
        """
        recommender = self.algorithms.get("popularity")
        if recommender is None:
            raise ValueError("PopularityRecommender not available")

        results = recommender.recommend(
            user_index=None, top_n=top_n, gender=gender)
        return [self.perfumes[i] for i, _ in results]

    def get_most_similar_by_content(self, perfume_id, top_n=5):
        """
        Wrapper around the content-based similarity recommender.
        Returns top-N similar perfumes by embedding similarity.
        """
        recommender = self.algorithms.get("content")
        if recommender is None:
            raise ValueError("ContentSimilarityRecommender not available")

        results = recommender.get_similar(perfume_id, top_n)
        return [self.get_perfume_by_id(recommender.index_to_perfume_id[i]) for i, _ in results]

    def get_most_similar_by_collab(self, perfume_id, top_n=5):
        """
        Uses collaborative filtering (item-item KNN) to find similar perfumes.
        """
        recommender = self.algorithms.get("item_knn")
        if recommender is None:
            raise ValueError("ItemKNNRecommender not available")

        # Translate perfume ID to matrix index
        matrix_index = next((idx for idx, p in self.matrix_id_to_orm_perfume.items()
                             if p and p.id == perfume_id), None)

        if matrix_index is None:
            return []

        results = recommender.get_similar(matrix_index, top_n)
        return [self.matrix_id_to_orm_perfume[i] for i, _ in results]

    def recommend_for_user(self, django_user_id=None, top_n=5, algorithm="item_knn",
                           filters=None, with_explanations=False, top_k_neurons=10):
        """
        Produces top-N personalized recommendations for a given Django user.
        - Applies soft boost based on filters.
        - Optionally enriches top results with neuron explanations (ELSAE only).
        Returns list of (perfume, score, explanation_or_None)
        """
        user_index = self.django_user_id_to_matrix_id.get(
            django_user_id) if django_user_id else None
        user_vector = self.interaction_matrix[user_index].toarray().ravel()
        recommender = self.algorithms.get(algorithm)
        if recommender is None:
            raise ValueError(f"Unknown recommender algorithm: {algorithm}")

        # Fallback for anonymous users
        if user_index is None:
            pop_recs = self.algorithms["popularity"].recommend(top_n=top_n)
            return [(self.perfumes[i], score, None) for i, score in pop_recs]

        buffer_size = 3 * top_n
        raw_results = recommender.recommend(
            user_vector=user_vector, user_index=user_index, top_n=buffer_size)

        # Normalize filters
        active_filters = {
            k.lower(): [v.lower() for v in vs]
            for k, vs in (filters or {}).items() if vs
        }

        # --- Boosting and ORM lookup ---
        boosted = []
        used_ids = set()
        for item_idx, base_score in raw_results:
            perfume = self.matrix_id_to_orm_perfume.get(item_idx)
            if not perfume or perfume.id in used_ids:
                continue
            used_ids.add(perfume.id)
            boost = soft_boost(perfume, active_filters=active_filters,
                               category_attr_map=CATEGORY_ATTRS, factor=0.6)
            boosted.append((perfume, item_idx, base_score + boost))

            if len(boosted) >= buffer_size:
                break

        # Sort by boosted score and truncate
        boosted.sort(key=lambda x: x[2], reverse=True)
        top_boosted = boosted[:top_n]

        # --- Explanation (only for visible items) ---
        # This is merely a placeholder for future implementation,
        # which might eventually differ substentially.
        #
        # explanations = {}
        # if with_explanations:
        #     try:
        #         item_indices = [item_idx for _, item_idx, _ in top_boosted]
        #         neuron_tags = recommender.explain_items(
        #             item_indices=item_indices,
        #             user_index=user_index,
        #             neuron_tag_dict=self.neuron_tag_dict,
        #             top_k_neurons=top_k_neurons
        #         )
        #         explanations = {
        #             item_idx: f"Neuron {neuron_id}: {', '.join(tags[:3])}"
        #             for item_idx, (neuron_id, tags) in zip(item_indices, neuron_tags)
        #         }
        #         except Exception as e:
        #             print(f"[WARN] Failed to get explanations: {e}")

        return [
            (perfume, score)
            for perfume, item_idx, score in top_boosted
        ]

    def recommend_with_explanations(self, django_user_id: int,
                                    top_n: int = 5, top_k_neurons: int = 10):
        """
        Returns top-N personalized recommendations with neuron explanation tags.
        Only available for ELSAE model.
        """
        user_index = self.django_user_id_to_matrix_id.get(django_user_id)
        user_vector = self.interaction_matrix[user_index].toarray().ravel()

        recommender = self.algorithms.get("elsae")
        if recommender is None:
            raise ValueError("ELSAERecommender not initialized.")

        raw_recs, neuron_tags = recommender.recommend_with_explanations(
            user_vector=user_vector,
            user_index=user_index,
            neuron_tag_dict=self.neuron_tag_dict,
            top_n=top_n,
            top_k_neurons=top_k_neurons
        )

        recs_with_orm = []
        for item_idx, score in raw_recs:
            perfume = self.matrix_id_to_orm_perfume.get(item_idx)
            if perfume:
                recs_with_orm.append((perfume, score))

        return recs_with_orm, neuron_tags

    def add_user_row(self, django_user):
        """
        Adds a new user row with no interactions to the matrix.
        Automatically updates models afterward.
        """
        self.interaction_matrix = add_user_row_to_matrix(
            self.interaction_matrix,
            real_users=self.real_users,
            user_id=django_user.id,
            id_to_user=self.django_user_id_to_user,
            id_to_index=self.django_user_id_to_matrix_id
        )
        self.django_user_id_to_user[django_user.id] = django_user
        self.update_algorithms()

    def update_user_row(self, user):
        """
        Updates an existing user's interaction row based on liked perfumes.
        Automatically updates models afterward.
        """
        self.interaction_matrix = update_user_row_in_matrix(
            self.interaction_matrix,
            user_id=user.id,
            user=user,
            id_to_index=self.django_user_id_to_matrix_id,
            ext_id_to_matrix_id=self.ext_perfume_id_to_matrix_id
        )
        self.update_algorithms()
