from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('analysis-result/', views.analysis_result_view, name='result'),
]