from django.urls import path
from . import views
from .SLR.views import slr_view

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('try-it/', views.try_it, name='try_it'),
    path('slr/', slr_view, name='slr'),
] 