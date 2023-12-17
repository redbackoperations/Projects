from django.contrib import admin
from .models import Users
from .models import warehouse

# Registers the Users model with the Django admin interface
admin.site.register(Users)
admin.site.register(warehouse)
# @admin.register(warehouse)
# class warehouseadmin(admin.ModelAdmin):
#     pass 