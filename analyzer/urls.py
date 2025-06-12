from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'analyzer'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('preprocess/<str:dataset_id>/', views.preprocess_dataset, name='preprocess_dataset'),
    path('visualize/<str:dataset_id>/', views.visualize_dataset, name='visualize_dataset'),
    path('api/columns/<str:dataset_id>/', views.get_columns, name='get_columns'),
    path('api/visualize/single/<str:dataset_id>/', views.visualize_single_column, name='visualize_single_column'),
    path('api/visualize/comparison/<str:dataset_id>/', views.visualize_comparison, name='visualize_comparison'),
    path('api/visualize/advanced/<str:dataset_id>/', views.visualize_advanced, name='visualize_advanced'),
    path('api/preview/<str:dataset_id>/', views.preview_dataset, name='preview_dataset'),
    path('api/unique-values/<str:dataset_id>/', views.get_unique_values, name='get_unique_values'),
    path('download/<str:dataset_id>/', views.download_dataset, name='download_dataset'),
    path('delete/<str:dataset_id>/', views.delete_dataset, name='delete_dataset'),
] 