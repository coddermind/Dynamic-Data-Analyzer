import uuid
import os
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

def dataset_file_path(instance, filename):
    """Generate file path for new dataset file."""
    ext = filename.split('.')[-1]
    filename = f"{instance.id}.{ext}"
    return os.path.join('datasets', filename)

class Dataset(models.Model):
    """Model representing an uploaded dataset."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to=dataset_file_path)
    file_type = models.CharField(max_length=10, choices=[('csv', 'CSV'), ('excel', 'Excel')])
    columns = models.JSONField(default=list)
    dropped_columns = models.JSONField(default=list)
    original_shape = models.CharField(max_length=50, blank=True, null=True)
    current_shape = models.CharField(max_length=50, blank=True, null=True)
    has_preprocessing = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='datasets')
    is_private = models.BooleanField(default=True)
    
    def __str__(self):
        return self.title

class Preprocessing(models.Model):
    """Model representing preprocessing settings for a dataset."""
    dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, related_name='preprocessing')
    missing_values_strategy = models.CharField(
        max_length=20, 
        choices=[
            ('drop', 'Drop rows with missing values'),
            ('mean', 'Fill with mean'),
            ('median', 'Fill with median'),
            ('mode', 'Fill with mode'),
            ('constant', 'Fill with constant value'),
        ],
        blank=True, null=True
    )
    fill_value = models.CharField(max_length=50, blank=True, null=True)
    encoding_strategy = models.CharField(
        max_length=20,
        choices=[
            ('none', 'No encoding'),
            ('onehot', 'One-Hot Encoding'),
            ('label', 'Label Encoding'),
        ],
        default='none'
    )
    scaling_strategy = models.CharField(
        max_length=20,
        choices=[
            ('none', 'No scaling'),
            ('minmax', 'Min-Max Scaling'),
            ('standard', 'Standard Scaling (Z-score)'),
            ('robust', 'Robust Scaling'),
        ],
        default='none'
    )
    handle_outliers = models.BooleanField(default=False)
    outlier_strategy = models.CharField(
        max_length=20,
        choices=[
            ('global', 'Use global strategy'),
            ('cap', 'Cap outliers'),
            ('trim', 'Remove outliers'),
        ],
        default='global'
    )
    pca_components = models.PositiveIntegerField(blank=True, null=True)
    feature_selection_strategy = models.CharField(
        max_length=20,
        choices=[
            ('none', 'No feature selection'),
            ('variance', 'Variance Threshold'),
            ('kbest', 'Select K Best'),
        ],
        default='none'
    )
    k_best_features = models.PositiveIntegerField(blank=True, null=True)
    
    def __str__(self):
        return f"Preprocessing for {self.dataset.title}"

class ColumnPreprocessing(models.Model):
    """Model representing column-specific preprocessing settings."""
    preprocessing = models.ForeignKey(Preprocessing, on_delete=models.CASCADE, related_name='column_preprocessings')
    column_name = models.CharField(max_length=255)
    
    # Strategy fields
    missing_values_strategy = models.CharField(max_length=50, default='global')
    fill_value = models.CharField(max_length=255, blank=True)
    encoding_strategy = models.CharField(max_length=50, default='global')
    scaling_strategy = models.CharField(max_length=50, default='global')
    handle_outliers = models.BooleanField(default=False)
    outlier_strategy = models.CharField(max_length=50, default='global')
    
    # Conditions for filtering data
    numeric_conditions = models.TextField(blank=True, null=True)  # Store as JSON string
    categorical_conditions = models.TextField(blank=True, null=True)  # Store as JSON string
    datetime_conditions = models.TextField(blank=True, null=True)  # Store as JSON string for date filtering
    
    # Value replacements
    value_replacements = models.TextField(blank=True, null=True)  # Store as JSON string
    
    class Meta:
        unique_together = ['preprocessing', 'column_name']
    
    def __str__(self):
        return f"Column preprocessing for {self.column_name}"
