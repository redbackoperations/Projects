from django.contrib import admin
from .models import Users

# Registers the Users model with the Django admin interface
admin.site.register(Users)
