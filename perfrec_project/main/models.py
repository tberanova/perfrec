"""
models.py

Defines the core database models for the perfume recommender system.

Includes:
- Brand: Perfume manufacturers.
- Perfume: Central entity representing a fragrance.
- Accord / Note: Scent characteristics linked to perfumes.
- Profile: Extended user preferences.
"""

from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse
from django.utils.text import slugify
from .config import GENDER_CHOICES


class Brand(models.Model):
    """
    Represents a fragrance brand or house.
    """
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class Perfume(models.Model):
    """
    Main model representing a perfume product.
    """
    name = models.CharField(max_length=200)
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE)
    external_id = models.CharField(
        max_length=50, unique=True, blank=True, null=True)
    year = models.PositiveIntegerField(blank=True, null=True)
    gender = models.CharField(max_length=50, blank=True, null=True)
    description = models.TextField(blank=True)
    rating_score = models.FloatField(default=0.0)
    # Replaces previous `votes` field
    rating_count = models.IntegerField(default=0)
    fragrance_type = models.CharField(max_length=120, blank=True, null=True)
    url = models.URLField(blank=True, null=True)
    image_url = models.URLField(blank=True, null=True)
    image = models.ImageField(upload_to='perfume_images/', blank=True)
    official_url = models.URLField(blank=True, null=True)

    # Optional user-contributed chart data
    season_chart = models.JSONField(blank=True, null=True)
    occasion_chart = models.JSONField(blank=True, null=True)
    type_chart = models.JSONField(blank=True, null=True)
    style_chart = models.JSONField(blank=True, null=True)

    # URL slug
    slug = models.SlugField(max_length=200, unique=True,
                            blank=True, default="slug")

    def __str__(self):
        return f"{self.brand.name} - {self.name}"

    def get_absolute_url(self):
        """
        Returns the canonical URL for the perfume detail page.
        """
        return reverse('perfume_detail', kwargs={'slug': self.slug})

    def save(self, *args, **kwargs):
        """
        Automatically generates a unique slug if missing.
        """
        if not self.slug or self.slug == "slug":
            base_slug = slugify(self.name)
            slug = base_slug
            counter = 1
            while Perfume.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = slug
        super().save(*args, **kwargs)


class Accord(models.Model):
    """
    Represents a broad scent category (e.g. Floral, Woody).
    """
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class PerfumeAccord(models.Model):
    """
    Many-to-many relationship between perfumes and accords.
    """
    perfume = models.ForeignKey(Perfume, on_delete=models.CASCADE)
    accord = models.ForeignKey(Accord, on_delete=models.CASCADE)


class Note(models.Model):
    """
    Represents a specific olfactory note (e.g. Vanilla, Jasmine).
    """
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class PerfumeNote(models.Model):
    """
    Many-to-many relationship between perfumes and notes.
    """
    perfume = models.ForeignKey(Perfume, on_delete=models.CASCADE)
    note = models.ForeignKey(Note, on_delete=models.CASCADE)


class Profile(models.Model):
    """
    Extends Django's built-in User model with preferences.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    liked_perfumes = models.ManyToManyField(
        Perfume, related_name='liked_by', blank=True)

    preferred_perfume_gender = models.CharField(
        max_length=10, choices=GENDER_CHOICES, default='None')

    def __str__(self):
        return f"{self.user.username}'s Profile"
