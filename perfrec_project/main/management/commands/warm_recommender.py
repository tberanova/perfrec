"""
Management command to pre-train and serialize the recommender before server startup.

This command:
- Triggers training of all configured recommender models
- Serializes model outputs (e.g., embeddings, similarity matrices)
- Ensures the recommender state is ready on disk for immediate loading at runtime

Note:
    This does not keep the recommender in memory (i.e., no shared instance),
    but prepares all necessary artifacts for fast instantiation when the app starts.

Usage:
    python manage.py warmup_recommender
"""

from django.core.management.base import BaseCommand
from main.recommender.manager_singleton import RecommenderManagerSingleton


class Command(BaseCommand):
    """Trains and serializes all recommender models before server startup."""

    help = (
        "Train and serialize the recommender. "
        "Use before starting the server to avoid cold-start delays."
    )

    def handle(self, *args, **opts):
        """
        Triggers model training and serialization via the singleton interface.

        This prepares all required files (e.g. vectorizers, scores) to ensure
        instant availability when the recommender is later instantiated in a live request.
        """
        RecommenderManagerSingleton.get()
        self.stdout.write(self.style.SUCCESS(
            "Recommender trained and serialized to disk."))
