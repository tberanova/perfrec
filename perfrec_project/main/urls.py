"""
URL configuration for the 'main' app.

This module defines routing for:
- Landing and main recommendation page
- User authentication (login, logout, register, profile)
- Perfume browser and detail views
- AJAX endpoints for autocomplete and user interactions (likes)

Includes media file routing in development mode.

Routes:
    - /                        → main_page
    - /browser/                → PerfumeListView (search, filter, browse)
    - /register/              → User registration form
    - /login/, /logout/       → Custom auth views
    - /profile/               → Profile edit form
    - /search-suggestions/    → JSON autocomplete
    - /add-liked-perfume/     → POST: Add perfume to liked list
    - /remove-liked-perfume/  → POST: Remove perfume from liked list
    - /<slug>/                → PerfumeDetailView
"""

from django.urls import path
from django.contrib.auth.views import LogoutView
from django.conf import settings
from django.conf.urls.static import static

from .views.authentication import CustomLoginView, register, account_profile
from .views.main_page import main_page
from .views.perfume_browser import PerfumeListView
from .views.perfume_detail import PerfumeDetailView
from .views.search import search_suggestions
from .views.interactions import add_liked_perfume, remove_liked_perfume


urlpatterns = [
    path('', main_page, name='main_page'),
    path('browser/', PerfumeListView.as_view(), name='perfume_browser'),

    # Authentication
    path('register/', register, name='register'),
    path('profile/', account_profile, name='profile'),
    path('login/', CustomLoginView.as_view(next_page='main_page'), name='login'),
    path('logout/', LogoutView.as_view(next_page='main_page'), name='logout'),

    # Search autocomplete
    path('search-suggestions/', search_suggestions, name='search_suggestions'),

    # User interactions
    path("remove-liked-perfume/<int:perfume_id>/",
         remove_liked_perfume, name="remove-liked-perfume"),
    path("add-liked-perfume/", add_liked_perfume, name="add-liked-perfume"),

    # Perfume detail view
    path('<slug:slug>/', PerfumeDetailView.as_view(), name='perfume_detail'),
]

# Beware that this is not optimal for larger scale deployment
urlpatterns += static(settings.MEDIA_URL,
                      document_root=settings.MEDIA_ROOT)
