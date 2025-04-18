from django.urls import path
from . import views
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('predict/', views.predict, name='predict'),
    path('models/', views.models, name='models'),
    path('tsne/', views.tsne_visualization, name='tsne_visualization'),
    path('history/', views.history, name='history'),
    path('record/<int:record_id>/', views.record_detail, name='record_detail'),
    path('task_status/<str:task_id>/', views.get_task_status, name='get_task_status'),
]
