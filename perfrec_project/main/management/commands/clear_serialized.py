"""
Management command to clear cached serialized model/vectorizer files.

This command deletes all files in the `main/serialized/` directory. It is
typically used to invalidate cached models, forcing a fresh retraining
or vectorization pipeline on the next run.

Usage:
    python manage.py clear_serialized
"""

import os
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    """Deletes cached model/vectorizer files to force retraining."""

    help = "Clears cached model/vectorizer files from main/serialized/ to force retraining."

    def handle(self, *args, **options):
        """
        Deletes all regular files from the `serialized` folder, which stores
        cached models, embeddings, and vectorizer outputs.
        """
        folder = os.path.join(settings.BASE_DIR, "serialized")

        if not os.path.exists(folder):
            self.stdout.write(self.style.WARNING(
                f"Folder does not exist: {folder}"))
            return

        deleted_count = 0

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {deleted_count} file(s) from {folder}")
        )
