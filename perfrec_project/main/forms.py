"""
Form definitions for user registration and profile management.

Includes:
    - UserRegisterForm: Extends Django's built-in UserCreationForm to require email.
    - ProfileUpdateForm: Allows users to update profile preferences (e.g., gender preference).

These forms are integrated with Django’s authentication system and support
customization of user and profile data through the UI.
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile


class UserRegisterForm(UserCreationForm):
    """
    A registration form that extends Django's `UserCreationForm` to include an email field.

    Fields:
        - username (str): The desired username for the new user.
        - email (str): The user's email address (required).
        - password1 (str): First password input.
        - password2 (str): Password confirmation.

    This form handles user creation and is used on the registration page.
    """
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class ProfileUpdateForm(forms.ModelForm):
    """
    A form that allows users to update their profile preferences.

    Currently limited to:
        - preferred_perfume_gender (str): The user's selected gender-based perfume preference.
    """

    class Meta:
        model = Profile
        fields = ['preferred_perfume_gender']

    def save(self, commit=True):
        """
        Saves the profile instance with optional delayed commit.

        Args:
            commit (bool): If True, saves the instance and M2M relations immediately.

        Returns:
            Profile: The updated Profile instance.
        """
        profile = super().save(commit=False)

        if commit:
            profile.save()
            self.save_m2m()

        return profile
