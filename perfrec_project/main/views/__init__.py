"""
This package (`views/`) contains all the views for the Django application.

Modules:
    - `authentication.py`: Handles user authentication (login, registration, profiles).
    - `perfume.py`: Provides views for perfume listings, detail pages, and recommendations.
    - `search.py`: Contains search-related views, including autocomplete suggestions.
    - `user_interactions.py`: Handles user interactions like adding/removing liked perfumes.

By importing all modules in this `__init__.py`, Django can recognize `views/` as a package.
"""

from .authentication import *
from .main_page import *
from .perfume_browser import *
from .perfume_detail import *
from .search import *
from .interactions import *
