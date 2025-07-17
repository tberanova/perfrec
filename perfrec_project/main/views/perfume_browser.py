"""
Defines the paginated perfume browser view with full-text search,
brand and accord filtering, and sorting.

View:
    - PerfumeListView: Renders the list of perfumes using query parameters.

Features:
    - Search by name/description using `get_perfume_search_queryset()`
    - Filtering by brand and multiple accords (with AND/OR logic)
    - Sorting by rating count or other fields
    - Separation of top brands vs. other brands for filter UI
"""

from django.views.generic import ListView
from django.db.models import Count

from main.models import Perfume, Brand, Accord
from main.views.search import get_perfume_search_queryset


class PerfumeListView(ListView):
    """
    Displays a paginated list of perfumes with optional filtering and sorting.

    Supported query parameters:
        - q: Full-text search string
        - brand: ID of selected brand to filter by
        - accords: List of selected accord IDs (multiple allowed)
        - filter_logic: 'or' (default) or 'and' to combine accords
        - sort: Field to sort by (default '-rating_count')

    Template:
        - perfume_browser.html

    Context:
        - perfumes: paginated queryset
        - top_brands, other_brands: for filter sidebar
        - accords: all available accord tags
        - selected_accords, current_brand, current_sort, filter_logic: UI state
    """

    model = Perfume
    template_name = 'perfume_browser.html'
    context_object_name = 'perfumes'
    paginate_by = 40

    def get_queryset(self):
        """
        Returns a filtered and sorted queryset of perfumes based on GET parameters.

        Filters:
            - Text search (`q`)
            - Brand ID
            - Accord IDs (with 'and' or 'or' logic)

        Returns:
            QuerySet: Filtered list of perfumes.
        """
        qs = super().get_queryset()
        query = self.request.GET.get('q', '').strip()
        sort_by = self.request.GET.get('sort', '-rating_count')
        gender_filter = self.request.GET.get("gender", "")
        brand_filter = self.request.GET.get('brand', '')
        accords_filter = self.request.GET.getlist('accords')
        filter_logic = self.request.GET.get('filter_logic', 'or')

        # Full-text search by name/description
        qs = get_perfume_search_queryset(queryset=qs, query=query)

        # Filter by brand ID
        if brand_filter:
            qs = qs.filter(brand__id=brand_filter)

        if gender_filter:
            qs = qs.filter(gender__iexact=gender_filter)

        # Filter by multiple accord IDs
        if accords_filter:
            if filter_logic == "and":
                for accord_id in accords_filter:
                    qs = qs.filter(perfumeaccord__accord_id=accord_id)
            else:  # Default OR logic
                qs = qs.filter(
                    perfumeaccord__accord_id__in=accords_filter).distinct()

        return qs.order_by(sort_by)

    def get_context_data(self, **kwargs):
        """
        Adds brand and accord filter metadata to the template context.

        Context additions:
            - top_brands, other_brands: split by popularity (top 100)
            - accords: all available accord tags
            - current filter values: for pre-filling UI
        """
        context = super().get_context_data(**kwargs)

        # Identify top 100 brands by number of perfumes
        top_brands_queryset = Brand.objects.annotate(
            perfume_count=Count('perfume')).order_by('-perfume_count')[:100]
        top_brand_ids = list(top_brands_queryset.values_list('id', flat=True))

        # Preload top and other brands separately
        top_brands = Brand.objects.filter(
            id__in=top_brand_ids).order_by('name')
        other_brands = Brand.objects.exclude(
            id__in=top_brand_ids).order_by('name')

        context.update({
            'top_brands': top_brands,
            'other_brands': other_brands,
            'accords': Accord.objects.all(),
            'gender': self.request.GET.get("gender", ""),
            'selected_accords': self.request.GET.getlist('accords', []),
            'current_brand':   self.request.GET.get('brand', ''),
            'current_sort':    self.request.GET.get('sort', '-rating_count'),
            'filter_logic':    self.request.GET.get('filter_logic', 'or'),
        })
        return context
