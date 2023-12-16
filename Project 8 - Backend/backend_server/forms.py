from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm

class SignUpForm(UserCreationForm):
    fields = ['Username', 'Email', 'Password1', 'Password2']




class LoginForm(AuthenticationForm):
    class Meta:
        fields = ['username', 'password']  # Fields for the login form