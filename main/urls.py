from django.urls import path
from . import views
from .SLR.views import slr_view
from .MLR.views import mlr_view
from .LR.views import lr_view

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('try-it/', views.try_it, name='try_it'),
    path('slr/', slr_view, name='slr'),
    path('mlr/', mlr_view, name='mlr'),
    path('lr/', lr_view, name='lr'),
] 