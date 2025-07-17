"""
Management command to import perfumes from a CSV file.

This importer handles:
- Creating or caching brands, accords, and notes
- Parsing perfume metadata including charts and subratings
- Linking perfumes with Accord and Note relationships
- Populating charts (season, occasion, type, style) as normalized percentage dicts

Usage:
    python manage.py import_perfumes path/to/file.csv
"""

import csv
import os
from collections import defaultdict

from django.core.management.base import BaseCommand
from django.db import transaction
from main.models import (
    Brand, Perfume, Accord, Note,
    PerfumeAccord, PerfumeNote
)


ALLOWED = {"season", "occasion", "type", "style"}


def chart_scores_to_percents(raw: str) -> dict[str, dict[str, float]]:
    """
    Parse raw chart string into normalized percentage charts.

    Args:
        raw (str): Raw string like "Season=Summer:42|Winter:8, Occasion=Work:30|Night:70"

    Returns:
        dict[str, dict[str, float]]: 
            e.g., {'season': {'Summer': 84.0, 'Winter': 16.0}, 'occasion': {...}}
    """
    charts = defaultdict(dict)

    for block in filter(None, map(str.strip, raw.split(","))):
        if "=" not in block:
            continue
        category, items = map(str.strip, block.split("=", 1))
        key = category.lower()
        if key not in ALLOWED:
            continue

        pairs = [
            (label.strip(), float(score))
            for item in items.split("|")
            if ":" in item
            for label, score in [item.split(":", 1)]
        ]

        if not pairs:
            continue

        total = sum(score for _, score in pairs) or 1  # Avoid division by zero

        charts[key] = {
            label: round(score / total * 100, 1)
            for label, score in pairs
        }

    return charts


class Command(BaseCommand):
    """Import perfumes and related fields from a CSV file."""

    help = "Fast import of perfumes from CSV (no reviews)"

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to the CSV file")

    def handle(self, *args, **options):
        csv_path = options["csv_file"]
        if not os.path.exists(csv_path):
            self.stderr.write(self.style.ERROR(f"File not found: {csv_path}"))
            return

        with open(csv_path, newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)

            # Preload caches to avoid redundant DB lookups
            brand_cache = {b.name: b for b in Brand.objects.all()}
            accord_cache = {a.name: a for a in Accord.objects.all()}
            note_cache = {n.name: n for n in Note.objects.all()}

            accords_to_create = []
            notes_to_create = []

            with transaction.atomic():
                for row in reader:
                    # Brand (create if missing)
                    brand_name = row["brand"]
                    brand = brand_cache.get(brand_name)
                    if not brand:
                        brand = Brand.objects.create(name=brand_name)
                        brand_cache[brand_name] = brand

                    # Create perfume object
                    image_url = row.get("imageUrl")
                    filename = image_url.split(
                        "/")[-1].split("?")[0] if image_url else None
                    image_path = f"perfume_images/{filename}" if filename else None

                    perfume = Perfume(
                        name=row["name"],
                        brand=brand,
                        external_id=row.get("perfume_id"),
                        fragrance_type=row.get("type") or None,
                        year=int(row["releaseYear"]) if row.get(
                            "releaseYear") else None,
                        gender=row.get("gender") or None,
                        description=row.get("description") or "",
                        rating_score=float(row["rating"]) if row.get(
                            "rating") else 0.0,
                        rating_count=int(row["ratingCount"]) if row.get(
                            "ratingCount") else 0,
                        image_url=image_url,
                        image=image_path,
                        official_url=row.get("officialPageUrl"),
                        url=row.get("officialPageUrl"),
                    )

                    # Parse and attach chart scores
                    charts = chart_scores_to_percents(
                        row.get("chartScores", ""))
                    perfume.season_chart = charts.get("season")
                    perfume.occasion_chart = charts.get("occasion")
                    perfume.type_chart = charts.get("type")
                    perfume.style_chart = charts.get("style")

                    # Parse subratings from string
                    subratings = {}
                    for pair in (row.get("subratings") or "").split(","):
                        if ":" not in pair:
                            continue
                        label, value = pair.split(":", 1)
                        try:
                            subratings[label.strip()] = float(value)
                        except ValueError:
                            continue
                    perfume.subratings = subratings or None

                    perfume.save()  # Save perfume before M2M/related inserts

                    # Accords (many-to-many through PerfumeAccord)
                    for accord_name in row["accords"].split(","):
                        accord_name = accord_name.strip()
                        if not accord_name:
                            continue
                        accord = accord_cache.get(accord_name)
                        if not accord:
                            accord = Accord.objects.create(name=accord_name)
                            accord_cache[accord_name] = accord
                        accords_to_create.append(PerfumeAccord(
                            perfume=perfume, accord=accord))

                    # Notes (many-to-many through PerfumeNote)
                    for note_name in row["notes"].split(","):
                        note_name = note_name.strip()
                        if not note_name:
                            continue
                        note = note_cache.get(note_name)
                        if not note:
                            note = Note.objects.create(name=note_name)
                            note_cache[note_name] = note
                        notes_to_create.append(
                            PerfumeNote(perfume=perfume, note=note))

                # Bulk insert associations to minimize DB hits
                PerfumeAccord.objects.bulk_create(
                    accords_to_create, batch_size=500)
                PerfumeNote.objects.bulk_create(
                    notes_to_create, batch_size=500)
