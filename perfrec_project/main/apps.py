"""
Application configuration for the 'main' app.

This configuration sets the default auto-incrementing primary key type
and defines the app name used throughout Django's registry.
"""

from django.apps import AppConfig


class MainConfig(AppConfig):
    """
    Default configuration for the 'main' Django app.

    Attributes:
        default_auto_field (str): Specifies the type of primary key field to use for models.
        name (str): Internal Django name for this app.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'
