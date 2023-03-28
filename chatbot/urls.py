from django.urls import path
from . import views
from .models import Student

urlpatterns = [
    path("", views.index, name="index"),
    path('registration', views.registration_page, name='registration'),
    path('register', views.register, name="register"),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('interview/<str:student_email>/', views.interview, name='interview'),
    path('result/<int:student_id>', views.result, name='result'),
    path('download/<int:student_id>', views.download, name='download')
]
