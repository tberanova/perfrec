"""
Handles user interaction with perfumes, including liking/unliking perfumes,
autocomplete search functionality, and automatic profile creation.

Functions:
    - add_liked_perfume(request): Adds a perfume to the user's liked list.
    - remove_liked_perfume(request, perfume_id): Removes a perfume from the user's liked list.
    - create_user_profile(sender, instance, created, **kwargs): Signal handler that creates a Profile when a new User is registered.
    - save_user_profile(sender, instance, **kwargs): Signal handler that saves the Profile when a User is saved.

Note:
    This module assumes that each `User` has a related `Profile` instance via OneToOneField.
    It also hooks into `RecommenderManagerSingleton` to keep the recommendation model in sync.
"""

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User

from main.models import Perfume, Profile
from main.recommender.manager_singleton import RecommenderManagerSingleton


@login_required
def remove_liked_perfume(request, perfume_id):
    """
    Removes a perfume from the authenticated user's liked perfumes list.

    Args:
        request (HttpRequest): The request object (must be POST).
        perfume_id (int): The ID of the perfume to remove.

    Returns:
        JsonResponse: A success status and the removed perfume's ID.
    """
    if request.method == "POST":
        perfume = get_object_or_404(Perfume, id=perfume_id)
        request.user.profile.liked_perfumes.remove(perfume)
        RecommenderManagerSingleton.get().update_user_row(request.user)
        return JsonResponse({"success": True, "perfume_id": perfume_id})
    return JsonResponse({"success": False}, status=400)


@login_required
def add_liked_perfume(request):
    """
    Adds a perfume to the authenticated user's liked perfumes list.

    Expects a POST request with 'perfume_id' in form data.

    Args:
        request (HttpRequest): The request object (must be POST).

    Returns:
        JsonResponse: A success status and the added perfume's ID and name.
    """
    if request.method == "POST":
        perfume = get_object_or_404(Perfume, id=request.POST.get("perfume_id"))
        request.user.profile.liked_perfumes.add(perfume)
        RecommenderManagerSingleton.get().update_user_row(request.user)
        return JsonResponse({
            "success": True,
            "perfume_id": perfume.id,
            "perfume_name": perfume.name
        })
    return JsonResponse({"success": False}, status=400)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    Signal handler that creates a Profile and adds a user row to the recommender
    when a new User is created.

    Args:
        sender: The model class (User).
        instance: The actual instance being saved.
        created (bool): True if a new record was created.
        **kwargs: Additional keyword arguments.
    """
    if created:
        Profile.objects.create(user=instance)
        print("profile created now adding row")
        RecommenderManagerSingleton.get().add_user_row(instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """
    Signal handler that saves the Profile instance whenever the User is saved.

    Args:
        sender: The model class (User).
        instance: The actual instance being saved.
        **kwargs: Additional keyword arguments.
    """
    instance.profile.save()
