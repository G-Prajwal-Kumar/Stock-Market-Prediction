from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home),
    path('home/train', views.train),
    path('image/<str:stockName>/<str:model>', views.image)
]