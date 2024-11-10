from django.urls import path
from . import  views

urlpatterns = [
    path('', views.AI2_page, name='AI2_page'),
]
