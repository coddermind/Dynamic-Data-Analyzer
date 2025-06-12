from django.contrib import admin
from .models import Dataset, Preprocessing

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('title', 'file_type', 'original_shape', 'current_shape', 'created_at')
    list_filter = ('file_type', 'has_preprocessing', 'created_at')
    search_fields = ('title',)
    readonly_fields = ('id', 'original_shape', 'current_shape', 'created_at')

@admin.register(Preprocessing)
class PreprocessingAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'missing_values_strategy', 'encoding_strategy', 'scaling_strategy')
    list_filter = ('missing_values_strategy', 'encoding_strategy', 'scaling_strategy', 'handle_outliers')
    search_fields = ('dataset__title',)
