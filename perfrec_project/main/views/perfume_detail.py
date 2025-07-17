"""
View for displaying detailed information about a specific perfume.

This module defines a detail page that shows:
- Metadata about the selected perfume
- Content-based and collaborative filtering recommendations
- User-vote charts for season, occasion, and type (in JSON for JavaScript use)

Uses:
    - Django's built-in DetailView
    - RecommenderManagerSingleton for related perfume suggestions
"""

import json
from django.views.generic import DetailView
from main.models import Perfume
from main.recommender.manager_singleton import RecommenderManagerSingleton


class PerfumeDetailView(DetailView):
    """
    Displays details of a single perfume, including recommendations and voting charts.

    Template:
        - perfume_detail.html

    Context:
        - perfume: The current perfume object
        - similar_perfumes: Top-N perfumes by content similarity
        - also_liked_perfumes: Top-N perfumes by collaborative filtering (user co-likes)
        - season_json, occasion_json, type_json: JSON-encoded chart data for JavaScript visualizations
    """

    model = Perfume
    template_name = 'perfume_detail.html'
    context_object_name = 'perfume'

    def get_context_data(self, **kwargs):
        """
        Adds recommendations and chart data to the template context.

        Returns:
            dict: Context dictionary containing the perfume, recommendations, and chart data.
        """
        recommender_manager = RecommenderManagerSingleton.get()
        context = super().get_context_data(**kwargs)
        current_perfume = context['perfume']

        # Content-based similarity using embedding vectors (e.g. TF-IDF, LLMs)
        context['similar_perfumes'] = recommender_manager.get_most_similar_by_content(
            perfume_id=current_perfume.id,
            top_n=20
        )

        # Collaborative similarity based on co-likes or co-occurrence in user collections
        context['also_liked_perfumes'] = recommender_manager.get_most_similar_by_collab(
            perfume_id=current_perfume.id,
            top_n=20
        )

        # Charts from user vote distributions (stored in DB as dictionaries)
        context["season_json"] = json.dumps(current_perfume.season_chart or {})
        context["occasion_json"] = json.dumps(
            current_perfume.occasion_chart or {})
        context["type_json"] = json.dumps(current_perfume.type_chart or {})

        return context
