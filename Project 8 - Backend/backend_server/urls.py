"""
URL configuration for backend_server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from backend_server import views
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('admin/', admin.site.urls),  # URL for the Django admin interface
    path('users/', views.user_list),  # URL for handling user list operations (GET and POST)
    path('users/<int:id>', views.user_detail),  # URL for handling individual user operations (GET, PUT, DELETE)
    path('signup/', views.test_take_input, name='signup'),
    path('login/', views.login, name='login'),
    path('home/', views.home, name='home'),
    path('redirected_home.html/', views.login, name='home'),
]


#Getting Json format through browzer
urlpatterns = format_suffix_patterns(urlpatterns)