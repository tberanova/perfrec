"""
Singleton wrapper for RecommenderManager.

Ensures the recommender is only initialized once per server run,
avoiding redundant loading of models and data.
"""

from main.recommender.manager import RecommenderManager


class RecommenderManagerSingleton:
    """
    Lazily instantiates and caches a single RecommenderManager instance.
    """
    _instance = None

    @classmethod
    def get(cls):
        """
        Returns the shared RecommenderManager instance.
        """
        if cls._instance is None:
            print("[INFO] Initializing recommender...")
            cls._instance = RecommenderManager()
        return cls._instance
