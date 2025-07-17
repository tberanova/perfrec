"""
Main page view for the perfume recommender system.

This module defines the unified entry point (`main_page`) for both rendering the initial homepage
and handling dynamic AJAX requests for personalized recommendations. It integrates with the
RecommenderManagerSingleton to fetch:

- Popular perfumes for unauthenticated users
- Personalized recommendations for logged-in users
- Filtered recommendation updates via POST (AJAX)
- Neuron tag explanations for contextual tooltips (via ELSAE)

Dependencies:
    - Django views and templates
    - Custom recommendation logic
    - User profile model with liked perfumes and gender preference

Constants:
    - SEASON_FILTERS, OCCASION_FILTERS, TYPE_FILTERS (from config)
    - TOP_N: default number of recommendations to display
"""

import json
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.template.loader import render_to_string

from main.recommender.manager_singleton import RecommenderManagerSingleton
from main.config import SEASON_FILTERS, OCCASION_FILTERS, TYPE_FILTERS, TOP_N, MAX_N, DEFAULT_REC_ALG


@csrf_exempt
def main_page(request):
    """
    Unified view that renders the homepage and handles AJAX filter requests.

    For GET requests:
        - Renders the initial page with personalized and popular perfumes.
    For POST requests:
        - Returns a JSON response with filtered personalized recommendations based on user-selected filters.
    """
    recommender_manager = RecommenderManagerSingleton.get()

    # If user is not logged in, show popular perfumes only (no personalization)
    if not request.user.is_authenticated:
        return render(request, "main_page.html", {
            "popular_perfumes": recommender_manager.get_most_popular(top_n=TOP_N),
            "personalized_recommendations": [],
            "season_filters": SEASON_FILTERS,
            "occasion_filters": OCCASION_FILTERS,
            "type_filters": TYPE_FILTERS,
            "len_liked_perfumes": 0,
        })

    # Handle AJAX POST request to dynamically update recommendations based on filters
    if request.method == "POST":
        data = json.loads(request.body)
        filters = data.get("filters", {})
        # supports "Show More" on scroll/button
        top_n = min(data.get("top_n", TOP_N), MAX_N)

        recs = recommender_manager.recommend_for_user(
            django_user_id=request.user.id,
            top_n=top_n,
            filters=filters,
            algorithm=DEFAULT_REC_ALG
        )

        return JsonResponse({
            "success": True,
            "recommendations": [
                {
                    "id": perfume.id,
                    "html": render_to_string("partials/perfume_card.html", {"perfume": perfume}, request)
                }
                # Skip if slug is empty
                for perfume, _ in recs if perfume.slug
            ]
        })

    # Default GET request: fetch personalized and popular perfumes for initial render
    personalized_recs = recommender_manager.recommend_for_user(
        django_user_id=request.user.id,
        top_n=TOP_N,
        algorithm=DEFAULT_REC_ALG
    )

    # Get most-active concept tags for tooltip from explanation model (ELSAE)
    explained_recs = recommender_manager.recommend_with_explanations(
        django_user_id=request.user.id,
        top_n=1,
        top_k_neurons=8
    )

    seen = set()
    ordered_tags = []

    for _, tags in explained_recs[-1]:
        if tags:
            t = tags[0]
            if t not in seen:
                seen.add(t)
                ordered_tags.append(t)

    # fill in remaining tags (again preserving order and avoiding duplicates)
    for _, tags in explained_recs[-1]:
        for t in tags:
            if t not in seen:
                seen.add(t)
                ordered_tags.append(t)

    tooltip_text = ", ".join(ordered_tags[:7])

    return render(request, "main_page.html", {
        "popular_perfumes": recommender_manager.get_most_popular(
            top_n=TOP_N, gender=request.user.profile.preferred_perfume_gender),
        "personalized_recommendations": [perfume for perfume, _ in personalized_recs],
        "len_liked_perfumes": len(request.user.profile.liked_perfumes.all()),
        "top_neuron_tooltip": tooltip_text,
        "season_filters": SEASON_FILTERS,
        "occasion_filters": OCCASION_FILTERS,
        "type_filters": TYPE_FILTERS
    })
