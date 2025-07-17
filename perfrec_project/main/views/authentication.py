"""
User authentication and profile management views.

This module integrates with Django's built-in authentication system and custom forms
to support user login, registration, and profile updates.

Classes:
    - CustomLoginView: A subclass of Django's LoginView that redirects already authenticated users.

Functions:
    - register(request): Renders and processes the user registration form.
    - account_profile(request): Renders and updates the logged-in user's profile using a custom form.
"""

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.contrib import messages
from main.forms import UserRegisterForm, ProfileUpdateForm


class CustomLoginView(LoginView):
    """
    View for handling user login using a custom login template.

    Redirects users who are already authenticated to the homepage.
    """
    redirect_authenticated_user = True
    template_name = 'login.html'


def register(request):
    """
    Handles new user registration.

    If the user is already authenticated, they are redirected to the main perfume browser.
    Otherwise, this view renders a registration form and saves the user if the form is valid.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered registration page or a redirect on success.
    """
    if request.user.is_authenticated:
        return redirect('perfume_browser')

    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(
                request, 'Your account has been created! You can now log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()

    return render(request, 'register.html', {'form': form})


@login_required
def account_profile(request):
    """
    Displays and allows updates to the authenticated user's profile.

    The profile is linked via a OneToOne relation to the Django user model.
    This view handles both form rendering and processing.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered profile page or a redirect after successful update.
    """
    profile = request.user.profile

    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('profile')
    else:
        form = ProfileUpdateForm(instance=profile)

    return render(request, 'profile.html', {'form': form, 'profile': profile})
