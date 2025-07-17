"""
Provides search functionality via JSON-based autocomplete.

This module supports live search suggestions for perfumes by matching:
- Perfume name
- Brand name
- Full name (brand + name)

Functions:
    - get_perfume_search_queryset(queryset=None, query=None): 
        Returns a queryset filtered by full-text search.
    - search_suggestions(request): 
        Returns JSON-formatted search results for use in frontend autocomplete components.

Notes:
    This implementation does not rely on Django Autocomplete Light (DAL).
"""


from django.http import JsonResponse
from django.db.models import Q, Value
from django.db.models.functions import Concat
from main.models import Perfume


def get_perfume_search_queryset(queryset=None, query=None):
    """
    Returns a queryset filtered by search query.

    Supports:
        - Name-based matching
        - Brand name matching
        - Full name matching (brand + name concatenated)

    Args:
        queryset (QuerySet, optional): Initial queryset to filter (defaults to all perfumes).
        query (str, optional): User input string to search for.

    Returns:
        QuerySet: Filtered queryset matching the query.
    """
    if queryset is None:
        queryset = Perfume.objects.all()

    # Prefetch brand to avoid extra queries in annotations or formatting
    queryset = queryset.select_related("brand")

    if not query:
        return queryset

    # Annotate "full_name" = "Brand Name" for flexible matching
    return queryset.annotate(
        full_name=Concat('brand__name', Value(' '), 'name')
    ).filter(
        Q(name__icontains=query) |
        Q(brand__name__icontains=query) |
        Q(full_name__icontains=query)
    )


def search_suggestions(request):
    """
    Returns a JSON response with top perfume suggestions matching the user's input.

    Intended for use in JavaScript-based search/autocomplete fields.
    Shows up to 10 results, formatted as "Brand - Name".

    Args:
        request (HttpRequest): The GET request containing query parameter `q`.

    Returns:
        JsonResponse: A list of matching perfumes, each with:
            - id: database ID
            - name: brand + name (used internally)
            - label: formatted display string for dropdowns
            - slug: used for linking to detail page
    """
    query = request.GET.get("q", "")
    perfumes = get_perfume_search_queryset(query=query)[:10]

    return JsonResponse([
        {
            "id": p.id,
            "name": f"{p.brand} {p.name}",
            "label": f"{p.brand} - {p.name}",
            "slug": p.slug
        }
        for p in perfumes
    ], safe=False)
