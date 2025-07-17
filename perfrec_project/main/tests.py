"""
Unit tests for the 'main' app.

This file contains basic scaffolding for writing tests related to
views, models, and recommender logic. Extend with functional, integration,
and regression tests as needed.
"""

from django.test import TestCase
from django.urls import reverse
from main.models import Perfume


class SmokeTest(TestCase):
    """
    Basic smoke test to ensure the homepage loads for anonymous users.
    """

    def test_homepage_loads(self):
        response = self.client.get(reverse("main_page"))
        self.assertEqual(response.status_code, 200)


# TODO: Add tests for:
# - PerfumeListView filtering (brand, accords, full-text)
# - main_page AJAX recommendation endpoint
# - PerfumeDetailView content & recommendations
# - User registration and profile update
# - RecommenderManager integration points
