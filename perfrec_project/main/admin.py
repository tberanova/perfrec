"""
Registers core models with the Django admin interface.

This module allows administrators to manage perfume-related data through the admin panel,
including brands, perfumes, notes, and tag relationships.

"""

from django.contrib import admin
from .models import (
    Brand,
    Perfume,
    Accord,
    Note,
    PerfumeAccord,
    PerfumeNote,
)

# Register models for basic admin CRUD access
admin.site.register(Brand)
admin.site.register(Perfume)
admin.site.register(Accord)
admin.site.register(Note)
admin.site.register(PerfumeAccord)
admin.site.register(PerfumeNote)
