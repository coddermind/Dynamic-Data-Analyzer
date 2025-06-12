import json
import os
import pandas as pd
import uuid
import tempfile
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.urls import reverse
import plotly.graph_objects as go
import plotly.io as pio
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.core.paginator import Paginator
from django.urls import reverse

from .models import Dataset, Preprocessing, ColumnPreprocessing
from . import utils  # Make sure utils is imported properly
from .utils import (
    load_dataset, get_column_types, apply_preprocessing,
    visualize_single_column as util_visualize_single_column, 
    visualize_comparison as util_visualize_comparison, 
    visualize_advanced as util_visualize_advanced,
    export_plotly_html,
    get_column_statistics,
    save_dataset
)

def index(request):
    """Home page view."""
    if request.user.is_authenticated:
        # When logged in, only show the user's own datasets
        datasets = Dataset.objects.filter(user=request.user).order_by('-created_at')
        return render(request, 'analyzer/index.html', {'datasets': datasets})
    else:
        # For anonymous users, just show login message - no datasets
        datasets = []
        return render(request, 'analyzer/index.html', {'datasets': datasets})

@login_required
@csrf_exempt
def upload_dataset(request):
    """Handle dataset upload."""
    if request.method == 'POST':
        title = request.POST.get('title', 'Untitled Dataset')
        file = request.FILES.get('dataset_file')
        
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)
        
        # Determine file type
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.csv':
            file_type = 'csv'
        elif file_extension in ['.xls', '.xlsx']:
            file_type = 'excel'
        else:
            return JsonResponse({'error': 'Unsupported file type'}, status=400)
        
        # All datasets are private and associated with the logged-in user
        # No need for privacy toggle anymore
        
        # Create a new dataset
        dataset = Dataset.objects.create(
            title=title,
            file=file,
            file_type=file_type,
            user=request.user,
            is_private=True
        )
        
        # Load the dataset to get columns and shape
        try:
            df = load_dataset(dataset.file.path, dataset.file_type)
            column_types = get_column_types(df)
            
            # Update dataset metadata
            dataset.columns = list(df.columns)
            dataset.original_shape = f"{df.shape[0]} rows × {df.shape[1]} columns"
            dataset.current_shape = dataset.original_shape
            dataset.save()
            
            return JsonResponse({
                'success': True,
                'dataset_id': str(dataset.id),
                'redirect_url': reverse('analyzer:preprocess_dataset', args=[str(dataset.id)])
            })
        
        except Exception as e:
            dataset.delete()  # Delete the dataset if processing fails
            return JsonResponse({'error': str(e)}, status=500)
    
    return render(request, 'analyzer/upload.html')

@require_http_methods(["GET", "POST"])
@login_required
def preprocess_dataset(request, dataset_id):
    """Handle dataset preprocessing."""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Check ownership permissions
        if dataset.user != request.user:
            messages.error(request, "You don't have permission to access this dataset.")
            return redirect('analyzer:index')
        
        if request.method == "POST":
            # Load the dataset
            df = load_dataset(dataset.file.path, dataset.file_type)
            
            # Check if this is a column-specific preprocessing request
            column_to_process = request.POST.get('process_column')
            
            # Get global preprocessing settings
            preprocessing_settings = {
                'missing_values_strategy': request.POST.get('missing_values_strategy'),
                'fill_value': request.POST.get('fill_value'),
                'encoding_strategy': request.POST.get('encoding_strategy'),
                'scaling_strategy': request.POST.get('scaling_strategy'),
                'handle_outliers': request.POST.get('handle_outliers') == 'on',
                'outlier_strategy': request.POST.get('outlier_strategy'),
                'feature_selection_strategy': request.POST.get('feature_selection_strategy'),
                'k_best_features': int(request.POST.get('k_best_features', 5)) if request.POST.get('k_best_features') else None,
                'pca_components': int(request.POST.get('pca_components', 0)) if request.POST.get('pca_components') else None
            }
            
            # Get column-specific settings
            column_settings = {}
            
            # Check for columns to drop
            columns_to_drop = request.POST.getlist('drop_columns')
            if columns_to_drop:
                print(f"Columns to drop: {columns_to_drop}")
                
                # Initialize dropped_columns if it doesn't exist
                if dataset.dropped_columns is None:
                    dataset.dropped_columns = []
                
                # Add new columns to drop list, avoiding duplicates
                for column in columns_to_drop:
                    if column not in dataset.dropped_columns:
                        dataset.dropped_columns.append(column)
                
                print(f"Updated dropped columns: {dataset.dropped_columns}")
                dataset.save()
                
                # Drop the columns from the dataframe too
                df = df.drop(columns=columns_to_drop, errors='ignore')
                
                messages.success(request, f'Column(s) dropped: {", ".join(columns_to_drop)}')
                
            # Skip dropped columns in further processing
            available_columns = [col for col in df.columns if col not in (dataset.dropped_columns or [])]
            print(f"Available columns for processing: {available_columns}")
            
            # If this is a column-specific preprocessing, just process that column
            if column_to_process:
                print(f"Processing column-specific preprocessing for: {column_to_process}")
                print(f"POST data related to column: {[(k, v) for k, v in request.POST.items() if column_to_process in k]}")
                
                if column_to_process in available_columns:
                    # Get column-specific settings from form fields for this column
                    col_settings = {}
                    
                    # Debug all form values related to this column
                    for key, value in request.POST.items():
                        if key.startswith(f"{column_to_process}_") and value:
                            print(f"Form field found: {key} = {value}")
                    
                    # Missing values strategy - direct from hidden field first
                    missing_strategy = request.POST.get(f"{column_to_process}_missing_values_strategy")
                    if missing_strategy and missing_strategy != 'global' and missing_strategy != '':
                        print(f"Setting missing value strategy to: {missing_strategy}")
                        col_settings['missing_values_strategy'] = missing_strategy
                        
                        # Handle fill value for constant strategy
                        if missing_strategy == 'constant':
                            fill_value = request.POST.get(f"{column_to_process}_fill_value")
                            if fill_value:
                                col_settings['fill_value'] = fill_value
                                print(f"Setting fill value to: {fill_value}")
                    
                    # Encoding strategy
                    encoding_strategy = request.POST.get(f"{column_to_process}_encoding_strategy")
                    if encoding_strategy and encoding_strategy != 'global':
                        print(f"Setting encoding strategy to: {encoding_strategy}")
                        col_settings['encoding_strategy'] = encoding_strategy
                    
                    # Scaling strategy
                    scaling_strategy = request.POST.get(f"{column_to_process}_scaling_strategy")
                    if scaling_strategy and scaling_strategy != 'global':
                        print(f"Setting scaling strategy to: {scaling_strategy}")
                        col_settings['scaling_strategy'] = scaling_strategy
                    
                    # Outlier handling
                    if request.POST.get(f"{column_to_process}_handle_outliers") == 'on':
                        col_settings['handle_outliers'] = True
                        outlier_strategy = request.POST.get(f"{column_to_process}_outlier_strategy")
                        if outlier_strategy and outlier_strategy != 'global':
                            col_settings['outlier_strategy'] = outlier_strategy
                            print(f"Setting outlier strategy to: {outlier_strategy}")
                    
                    # Numeric conditions
                    numeric_conditions = request.POST.get(f"{column_to_process}_numeric_conditions")
                    if numeric_conditions:
                        try:
                            print(f"Received numeric conditions for '{column_to_process}': {numeric_conditions}")
                            parsed_conditions = json.loads(numeric_conditions)
                            print(f"Parsed numeric conditions: {parsed_conditions}")
                            col_settings['numeric_conditions'] = parsed_conditions
                        except json.JSONDecodeError as e:
                            print(f"Error parsing numeric conditions for '{column_to_process}': {e}")
                            print(f"Raw value: {numeric_conditions}")
                    
                    # Categorical conditions
                    categorical_conditions = request.POST.get(f"{column_to_process}_categorical_conditions")
                    if categorical_conditions:
                        try:
                            print(f"Received categorical conditions for '{column_to_process}': {categorical_conditions}")
                            parsed_conditions = json.loads(categorical_conditions)
                            print(f"Parsed categorical conditions: {parsed_conditions}")
                            col_settings['categorical_conditions'] = parsed_conditions
                        except json.JSONDecodeError as e:
                            print(f"Error parsing categorical conditions for '{column_to_process}': {e}")
                            print(f"Raw value: {categorical_conditions}")
                    
                    # Datetime conditions
                    datetime_conditions = request.POST.get(f"{column_to_process}_datetime_conditions")
                    if datetime_conditions:
                        try:
                            print(f"Received datetime conditions for '{column_to_process}': {datetime_conditions}")
                            parsed_conditions = json.loads(datetime_conditions)
                            print(f"Parsed datetime conditions: {parsed_conditions}")
                            col_settings['datetime_conditions'] = parsed_conditions
                        except json.JSONDecodeError as e:
                            print(f"Error parsing datetime conditions for '{column_to_process}': {e}")
                            print(f"Raw value: {datetime_conditions}")
                    
                    # Value replacements
                    value_replacements = request.POST.get(f"{column_to_process}_value_replacements")
                    if value_replacements:
                        try:
                            print(f"Received value replacements for '{column_to_process}': {value_replacements}")
                            parsed_replacements = json.loads(value_replacements)
                            print(f"Parsed value replacements: {parsed_replacements}")
                            col_settings['value_replacements'] = parsed_replacements
                        except json.JSONDecodeError as e:
                            print(f"Error parsing value replacements for '{column_to_process}': {e}")
                            print(f"Raw value: {value_replacements}")
                    
                    # If no settings were provided but we still need to process the column
                    if not col_settings:
                        print("No specific preprocessing settings found, forcing column processing")
                        col_settings['force_process'] = True
                    
                    print(f"Final column settings for {column_to_process}: {col_settings}")
                    
                    # Store settings if we have any
                    if col_settings:
                        column_settings[column_to_process] = col_settings
                    else:
                        messages.warning(request, f'No preprocessing options selected for column: {column_to_process}')
                        return redirect('analyzer:preprocess_dataset', dataset_id=dataset_id)
                else:
                    messages.error(request, f'Column {column_to_process} not found in dataset')
                    return redirect('analyzer:preprocess_dataset', dataset_id=dataset_id)
            else:
                # Process all columns
                for column in available_columns:
                    col_settings = {}
                    
                    # Missing values strategy
                    missing_strategy = request.POST.get(f"{column}_missing_values_strategy")
                    if missing_strategy:
                        col_settings['missing_values_strategy'] = missing_strategy
                        fill_value = request.POST.get(f"{column}_fill_value")
                        if fill_value:
                            col_settings['fill_value'] = fill_value
                    
                    # Encoding strategy
                    encoding_strategy = request.POST.get(f"{column}_encoding_strategy")
                    if encoding_strategy and encoding_strategy != 'global':
                        col_settings['encoding_strategy'] = encoding_strategy
                    
                    # Scaling strategy
                    scaling_strategy = request.POST.get(f"{column}_scaling_strategy")
                    if scaling_strategy and scaling_strategy != 'global':
                        col_settings['scaling_strategy'] = scaling_strategy
                    
                    # Outlier handling
                    if request.POST.get(f"{column}_handle_outliers") == 'on':
                        col_settings['handle_outliers'] = True
                        outlier_strategy = request.POST.get(f"{column}_outlier_strategy")
                        if outlier_strategy and outlier_strategy != 'global':
                            col_settings['outlier_strategy'] = outlier_strategy
                    
                    # Numeric conditions
                    numeric_conditions = request.POST.get(f"{column}_numeric_conditions")
                    if numeric_conditions:
                        try:
                            print(f"Received numeric conditions for '{column}': {numeric_conditions}")
                            parsed_conditions = json.loads(numeric_conditions)
                            print(f"Parsed numeric conditions: {parsed_conditions}")
                            col_settings['numeric_conditions'] = parsed_conditions
                        except json.JSONDecodeError as e:
                            print(f"Error parsing numeric conditions for '{column}': {e}")
                            print(f"Raw value: {numeric_conditions}")
                    
                    # Categorical conditions
                    categorical_conditions = request.POST.get(f"{column}_categorical_conditions")
                    if categorical_conditions:
                        try:
                            print(f"Received categorical conditions for '{column}': {categorical_conditions}")
                            parsed_conditions = json.loads(categorical_conditions)
                            print(f"Parsed categorical conditions: {parsed_conditions}")
                            col_settings['categorical_conditions'] = parsed_conditions
                        except json.JSONDecodeError as e:
                            print(f"Error parsing categorical conditions for '{column}': {e}")
                            print(f"Raw value: {categorical_conditions}")
                    
                    # Datetime conditions
                    datetime_conditions = request.POST.get(f"{column}_datetime_conditions")
                    if datetime_conditions:
                        try:
                            print(f"Received datetime conditions for '{column}': {datetime_conditions}")
                            parsed_conditions = json.loads(datetime_conditions)
                            print(f"Parsed datetime conditions: {parsed_conditions}")
                            col_settings['datetime_conditions'] = parsed_conditions
                        except json.JSONDecodeError as e:
                            print(f"Error parsing datetime conditions for '{column}': {e}")
                            print(f"Raw value: {datetime_conditions}")
                    
                    # Value replacements
                    value_replacements = request.POST.get(f"{column}_value_replacements")
                    if value_replacements:
                        try:
                            print(f"Received value replacements for '{column}': {value_replacements}")
                            parsed_replacements = json.loads(value_replacements)
                            print(f"Parsed value replacements: {parsed_replacements}")
                            col_settings['value_replacements'] = parsed_replacements
                        except json.JSONDecodeError as e:
                            print(f"Error parsing value replacements for '{column}': {e}")
                            print(f"Raw value: {value_replacements}")
                    
                    if col_settings:
                        column_settings[column] = col_settings
            
            try:
                # Get column types
                column_types = get_column_types(df)
                
                # Apply preprocessing
                print("Applying preprocessing with the following settings:")
                print(f"Global settings: {preprocessing_settings}")
                print(f"Column settings: {column_settings}")
                
                # Combine global settings and column settings into one config object
                preprocessing_config = {
                    **preprocessing_settings,
                    'column_settings': column_settings
                }
                
                try:
                    # If the dataset_id is passed directly without a dict wrapper, there won't be an error
                    processed_df = apply_preprocessing(df, preprocessing_config, dataset_id=str(dataset.id))
                    
                    # Update original shape if not already set
                    if not dataset.original_shape:
                        dataset.original_shape = f"{df.shape[0]} rows × {df.shape[1]} columns"
                        
                    # Update current shape
                    dataset.current_shape = f"{processed_df.shape[0]} rows × {processed_df.shape[1]} columns"
                    
                    # Save preprocessed dataset back to the original file
                    if dataset.file_type == 'csv':
                        processed_df.to_csv(dataset.file.path, index=False)
                    elif dataset.file_type == 'excel':
                        processed_df.to_excel(dataset.file.path, index=False)
                    
                    # Also save to a preprocessed file as backup
                    preprocessed_path = os.path.join(settings.MEDIA_ROOT, 'datasets', f'preprocessed_{dataset.id}.csv')
                    
                    # Ensure datasets directory exists
                    datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
                    os.makedirs(datasets_dir, exist_ok=True)
                    
                    processed_df.to_csv(preprocessed_path, index=False)
                    
                    # Update dataset record
                    dataset.has_preprocessing = True
                    dataset.save()
                    
                    # Create or update preprocessing model
                    preprocessing, created = Preprocessing.objects.get_or_create(dataset=dataset)
                    
                    # Provide default values for required fields
                    preprocessing.missing_values_strategy = preprocessing_settings.get('missing_values_strategy') or 'none'
                    preprocessing.fill_value = preprocessing_settings.get('fill_value') or ''
                    preprocessing.encoding_strategy = preprocessing_settings.get('encoding_strategy') or 'none'
                    preprocessing.scaling_strategy = preprocessing_settings.get('scaling_strategy') or 'none'
                    preprocessing.handle_outliers = preprocessing_settings.get('handle_outliers', False)
                    preprocessing.outlier_strategy = preprocessing_settings.get('outlier_strategy') or 'none'
                    preprocessing.feature_selection_strategy = preprocessing_settings.get('feature_selection_strategy') or 'none'
                    preprocessing.k_best_features = preprocessing_settings.get('k_best_features') or 0
                    preprocessing.pca_components = preprocessing_settings.get('pca_components') or 0
                    preprocessing.save()
                    
                    # Delete old column-specific preprocessing settings
                    preprocessing.column_preprocessings.all().delete()
                    
                    # Create new column-specific preprocessing models
                    for column_name, column_setting in column_settings.items():
                        column_preprocessing = ColumnPreprocessing(
                            preprocessing=preprocessing,
                            column_name=column_name
                        )
                        
                        # Provide default values for required fields 
                        column_preprocessing.missing_values_strategy = column_setting.get('missing_values_strategy') or 'global'
                        column_preprocessing.fill_value = column_setting.get('fill_value') or ''
                        column_preprocessing.encoding_strategy = column_setting.get('encoding_strategy') or 'global'
                        column_preprocessing.scaling_strategy = column_setting.get('scaling_strategy') or 'global'
                        column_preprocessing.handle_outliers = column_setting.get('handle_outliers', False)
                        column_preprocessing.outlier_strategy = column_setting.get('outlier_strategy') or 'global'
                        
                        # Save conditions as JSON strings
                        if 'numeric_conditions' in column_setting:
                            column_preprocessing.numeric_conditions = json.dumps(column_setting['numeric_conditions'])
                        else:
                            column_preprocessing.numeric_conditions = json.dumps([])
                            
                        if 'categorical_conditions' in column_setting:
                            column_preprocessing.categorical_conditions = json.dumps(column_setting['categorical_conditions'])
                        else:
                            column_preprocessing.categorical_conditions = json.dumps([])
                        
                        if 'datetime_conditions' in column_setting:
                            column_preprocessing.datetime_conditions = json.dumps(column_setting['datetime_conditions'])
                        else:
                            column_preprocessing.datetime_conditions = json.dumps([])
                        
                        # Save value replacements as JSON string
                        if 'value_replacements' in column_setting:
                            column_preprocessing.value_replacements = json.dumps(column_setting['value_replacements'])
                        else:
                            column_preprocessing.value_replacements = json.dumps([])
                        
                        column_preprocessing.save()
                    
                    # Success message depending on whether it was column-specific or global preprocessing
                    if column_to_process:
                        messages.success(request, f'Column "{column_to_process}" preprocessed successfully!')
                    else:
                        messages.success(request, f'Dataset preprocessed successfully! Shape changed from {dataset.original_shape} to {dataset.current_shape}')
                    
                except Exception as e:
                    error_msg = f"Error applying preprocessing: {str(e)}"
                    print(error_msg)
                    messages.error(request, error_msg)
                    
                return redirect('analyzer:preprocess_dataset', dataset_id=dataset_id)
                
            except Exception as e:
                messages.error(request, f'Error applying preprocessing: {str(e)}')
                return redirect('analyzer:preprocess_dataset', dataset_id=dataset_id)
        
        # For GET request, prepare context
        df = load_dataset(dataset.file.path, dataset.file_type)
        column_types = get_column_types(df)
        
        # Get column statistics and convert to JSON-friendly format
        column_stats = get_column_statistics(df)
        
        # Ensure all values are JSON serializable
        for col, stats in column_stats.items():
            for key, value in stats.items():
                # Convert numpy types to Python native types
                if hasattr(value, 'item'):  # Check if it's a numpy type
                    try:
                        stats[key] = value.item()
                    except (AttributeError, ValueError) as e:
                        print(f"Error converting {key} for column {col}: {e}")
                        if key in ['mean', 'median', 'std', 'min', 'max']:
                            stats[key] = float(value) if value is not None else 0.0
                        elif key in ['missing_values', 'unique_values', 'most_common_count']:
                            stats[key] = int(value) if value is not None else 0
                        else:
                            stats[key] = str(value) if value is not None else ""
        
        # Add JSON string representation of column stats for easier JavaScript parsing
        column_stats_json = {}
        for col, stats in column_stats.items():
            try:
                # Ensure each statistic has the right type before serializing
                clean_stats = {}
                for key, value in stats.items():
                    if key in ['mean', 'median', 'std', 'min', 'max']:
                        clean_stats[key] = float(value) if value is not None else 0.0
                    elif key in ['missing_values', 'unique_values', 'most_common_count']:
                        clean_stats[key] = int(value) if value is not None else 0
                    elif key in ['most_common_value', 'min_date', 'max_date']:
                        clean_stats[key] = str(value) if value is not None else ""
                    else:
                        clean_stats[key] = value
                
                # Print debugging info for a couple of columns
                if col in list(df.columns)[:2]:
                    print(f"Stats for column {col}: {clean_stats}")
                
                column_stats_json[col] = json.dumps(clean_stats)
            except Exception as e:
                print(f"Error serializing stats for column {col}: {e}")
                # Provide default stats for this column
                default_stats = {"missing_values": 0, "unique_values": 0}
                column_stats_json[col] = json.dumps(default_stats)
        
        context = {
            'dataset': dataset,
            'columns': df.columns.tolist(),
            'column_types': column_types,
            'column_stats': column_stats,
            'column_stats_json': column_stats_json,
            'column_preprocessings': {}
        }
        
        # Add any existing preprocessing settings to the context
        if dataset.has_preprocessing:
            try:
                preprocessing = dataset.preprocessing
                # Add global preprocessing settings
                context['preprocessing'] = preprocessing
                
                # Add column-specific preprocessing settings
                column_preprocessings = {}
                for col_preprocessing in preprocessing.column_preprocessings.all():
                    column_settings = {
                        'missing_values_strategy': col_preprocessing.missing_values_strategy,
                        'fill_value': col_preprocessing.fill_value,
                        'encoding_strategy': col_preprocessing.encoding_strategy,
                        'scaling_strategy': col_preprocessing.scaling_strategy,
                        'handle_outliers': col_preprocessing.handle_outliers,
                        'outlier_strategy': col_preprocessing.outlier_strategy,
                    }
                    
                    # Parse JSON strings for conditions
                    if hasattr(col_preprocessing, 'numeric_conditions') and col_preprocessing.numeric_conditions:
                        try:
                            print(f"Loading numeric conditions from DB for '{col_preprocessing.column_name}': {col_preprocessing.numeric_conditions}")
                            column_settings['numeric_conditions'] = json.loads(col_preprocessing.numeric_conditions)
                            print(f"Parsed DB numeric conditions: {column_settings['numeric_conditions']}")
                        except json.JSONDecodeError as e:
                            print(f"Error parsing DB numeric conditions: {e}")
                            column_settings['numeric_conditions'] = []
                    
                    if hasattr(col_preprocessing, 'categorical_conditions') and col_preprocessing.categorical_conditions:
                        try:
                            print(f"Loading categorical conditions from DB for '{col_preprocessing.column_name}': {col_preprocessing.categorical_conditions}")
                            column_settings['categorical_conditions'] = json.loads(col_preprocessing.categorical_conditions)
                            print(f"Parsed DB categorical conditions: {column_settings['categorical_conditions']}")
                        except json.JSONDecodeError as e:
                            print(f"Error parsing DB categorical conditions: {e}")
                            column_settings['categorical_conditions'] = []
                    
                    if hasattr(col_preprocessing, 'datetime_conditions') and col_preprocessing.datetime_conditions:
                        try:
                            print(f"Loading datetime conditions from DB for '{col_preprocessing.column_name}': {col_preprocessing.datetime_conditions}")
                            column_settings['datetime_conditions'] = json.loads(col_preprocessing.datetime_conditions)
                            print(f"Parsed DB datetime conditions: {column_settings['datetime_conditions']}")
                        except json.JSONDecodeError as e:
                            print(f"Error parsing DB datetime conditions: {e}")
                            column_settings['datetime_conditions'] = []
                    
                    column_preprocessings[col_preprocessing.column_name] = column_settings
                
                context['column_preprocessings'] = column_preprocessings
            except Exception as e:
                print(f"Error retrieving preprocessing settings: {str(e)}")
        
        return render(request, 'analyzer/preprocess.html', context)
        
    except Dataset.DoesNotExist:
        messages.error(request, 'Dataset not found.')
        return redirect('analyzer:index')
    except Exception as e:
        messages.error(request, f'Error preprocessing dataset: {str(e)}')
        return redirect('analyzer:index')

@login_required
def visualize_dataset(request, dataset_id):
    """Visualize the preprocessed dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        messages.error(request, "You don't have permission to access this dataset.")
        return redirect('analyzer:index')
    
    try:
        # Load the dataset directly from file (preprocessing is already applied)
        print(f"Loading dataset for visualization: {dataset.file.path}")
        df = load_dataset(dataset.file.path, dataset.file_type)
        
        # Check if the dataset is empty
        if df.empty:
            return render(request, 'analyzer/visualize.html', {
                'dataset': dataset,
                'error': 'The dataset is empty. Please upload a non-empty dataset.'
            })
            
        # Drop selected columns
        if dataset.dropped_columns:
            # Make sure dropped_columns is a list
            dropped_columns = dataset.dropped_columns if isinstance(dataset.dropped_columns, list) else []
            if dropped_columns:
                print(f"Dropping columns for visualization: {dropped_columns}")
                existing_columns = set(df.columns)
                columns_to_drop = [col for col in dropped_columns if col in existing_columns]
                if columns_to_drop:
                    print(f"Actually dropping: {columns_to_drop}")
                    df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Get column types
        print("Determining column types...")
        column_types = get_column_types(df)
        
        # Get counts for each type of column
        numeric_columns = [col for col, type_ in column_types.items() if type_ == 'numeric']
        categorical_columns = [col for col, type_ in column_types.items() if type_ in ['categorical', 'binary']]
        datetime_columns = [col for col, type_ in column_types.items() if type_ == 'datetime']
        text_columns = [col for col, type_ in column_types.items() if type_ == 'text']
        
        # Print summary info
        print(f"Dataset shape: {df.shape}")
        print(f"Column types: {len(numeric_columns)} numeric, {len(categorical_columns)} categorical, "
              f"{len(datetime_columns)} datetime, {len(text_columns)} text")
        
        # Get dataset summary statistics
        df_summary = {
            'shape': df.shape,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_columns),
            'categorical_columns': len(categorical_columns),
            'datetime_columns': len(datetime_columns),
            'text_columns': len(text_columns),
            'missing_values': df.isna().sum().sum(),
            'missing_percentage': round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2) if df.shape[0] * df.shape[1] > 0 else 0
        }
        
        print(f"Dataset summary: {df_summary}")
        
        # Check if we have enough columns for visualization
        if len(df.columns) == 0:
            return render(request, 'analyzer/visualize.html', {
                'dataset': dataset,
                'error': 'The dataset has no columns. Please check your preprocessing steps.'
            })
            
        # Basic check for at least one numeric column for advanced visualizations
        has_numeric = len(numeric_columns) > 0
        has_multiple_numeric = len(numeric_columns) >= 2
        
        # Update current shape in dataset model
        dataset.current_shape = f"{df.shape[0]} rows × {df.shape[1]} columns"
        dataset.save()
        
        # Store the entire column types dictionary for more detailed information
        return render(request, 'analyzer/visualize.html', {
            'dataset': dataset,
            'columns': df.columns.tolist(),
            'column_types': column_types,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'datetime_columns': datetime_columns,
            'text_columns': text_columns,
            'df_summary': df_summary,
            'has_numeric': has_numeric,
            'has_multiple_numeric': has_multiple_numeric
        })
    
    except Exception as e:
        print(f"Error in visualize_dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return render(request, 'analyzer/visualize.html', {
            'dataset': dataset,
            'error': f'Error loading dataset: {str(e)}'
        })

@csrf_exempt
@login_required
def get_columns(request, dataset_id):
    """API endpoint to get columns of a dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        return JsonResponse({'error': 'You do not have permission to access this dataset'}, status=403)

    try:
        # Load directly from file - preprocessing is already applied to the file
        df = load_dataset(dataset.file.path, dataset.file_type)
        
        # Drop selected columns
        if dataset.dropped_columns:
            df = df.drop(columns=dataset.dropped_columns, errors='ignore')
        
        # Get column types after drops
        column_types = get_column_types(df)
        
        return JsonResponse({
            'columns': df.columns.tolist(),
            'column_types': column_types,
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@login_required
def visualize_single_column(request, dataset_id):
    """API endpoint to visualize a single column."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        return JsonResponse({'error': 'You do not have permission to access this dataset'}, status=403)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            column = data.get('column')
            
            if not column:
                return JsonResponse({'error': 'Column name is required'}, status=400)
            
            # Load dataset directly from file - preprocessing is already applied
            df = load_dataset(dataset.file.path, dataset.file_type)
            
            # Drop selected columns
            if dataset.dropped_columns:
                df = df.drop(columns=dataset.dropped_columns, errors='ignore')
            
            # Check if column exists
            if column not in df.columns:
                return JsonResponse({'error': f'Column "{column}" does not exist'}, status=400)
            
            # Get column type
            column_types = get_column_types(df)
            column_type = column_types.get(column)
            
            # Prevent visualization of text columns
            if column_type == 'text':
                return JsonResponse({'error': 'Text columns cannot be visualized', 'message': 'Text columns are not supported for visualization'}, status=400)
            
            # Generate visualization
            visualization = util_visualize_single_column(df, column, column_type)
            
            # Add column statistics
            if column_type == 'numeric':
                visualization['statistics'] = {
                    'count': int(df[column].count()),
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    '25%': float(df[column].quantile(0.25)),
                    '50%': float(df[column].median()),
                    '75%': float(df[column].quantile(0.75)),
                    'max': float(df[column].max()),
                    'missing': int(df[column].isna().sum())
                }
            elif column_type in ['categorical', 'binary']:
                value_counts = df[column].value_counts().to_dict()
                visualization['statistics'] = {
                    'count': int(df[column].count()),
                    'unique': int(df[column].nunique()),
                    'top': str(df[column].mode()[0]) if not df[column].mode().empty else None,
                    'freq': int(df[column].value_counts().iloc[0]) if not df[column].value_counts().empty else 0,
                    'missing': int(df[column].isna().sum()),
                    'value_counts': {str(k): int(v) for k, v in value_counts.items()}
                }
            
            return JsonResponse(visualization)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST request required'}, status=405)

@csrf_exempt
@login_required
def visualize_comparison(request, dataset_id):
    """API endpoint to visualize comparison between columns."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        return JsonResponse({'error': 'You do not have permission to access this dataset'}, status=403)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            columns = data.get('columns', [])
            
            if len(columns) < 2:
                return JsonResponse({'error': 'At least two columns are required'}, status=400)
            
            # Load dataset directly from file - preprocessing is already applied
            df = load_dataset(dataset.file.path, dataset.file_type)
            
            # Drop selected columns
            if dataset.dropped_columns:
                df = df.drop(columns=dataset.dropped_columns, errors='ignore')
            
            # Check if columns exist
            for column in columns:
                if column not in df.columns:
                    return JsonResponse({'error': f'Column "{column}" does not exist'}, status=400)
            
            # Get column types
            column_types = get_column_types(df)
            
            # Generate visualization
            visualization = util_visualize_comparison(df, columns, column_types)
            
            return JsonResponse(visualization)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST request required'}, status=405)

@csrf_exempt
@login_required
def visualize_advanced(request, dataset_id):
    """API endpoint to generate advanced visualizations."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        return JsonResponse({'error': 'You do not have permission to access this dataset'}, status=403)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            plot_type = data.get('plot_type')
            
            print(f"Advanced visualization requested: {plot_type} for dataset {dataset_id}")
            
            if not plot_type:
                return JsonResponse({'error': 'Plot type is required'}, status=400)
            
            # Load dataset directly from file - preprocessing is already applied
            df = load_dataset(dataset.file.path, dataset.file_type)
            
            # Drop selected columns
            if dataset.dropped_columns:
                df = df.drop(columns=dataset.dropped_columns, errors='ignore')
            
            # Get column types
            column_types = get_column_types(df)
            
            # Check for additional parameters
            additional_params = {}
            if plot_type == 'anomaly' and 'column' in data:
                additional_params['column'] = data.get('column')
                print(f"Anomaly detection requested for specific column: {data.get('column')}")
            
            # Generate visualization with additional parameters
            visualization = utils.visualize_advanced(df, column_types, plot_type, **additional_params)
            
            # Debug response size
            print(f"Visualization response keys: {visualization.keys()}")
            for key, value in visualization.items():
                if isinstance(value, str):
                    print(f"Response {key} size: {len(value)} bytes")
                else:
                    print(f"Response {key} type: {type(value)}")
            
            return JsonResponse(visualization)
        
        except Exception as e:
            print(f"Error in visualize_advanced: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST request required'}, status=405)

@csrf_exempt
@login_required
def preview_dataset(request, dataset_id):
    """API endpoint to get a preview of the dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        return JsonResponse({'error': 'You do not have permission to access this dataset'}, status=403)

    # Set CORS headers for all responses
    response_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, X-Requested-With',
        'Cache-Control': 'no-cache, no-store, must-revalidate'
    }
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = HttpResponse()
        for key, value in response_headers.items():
            response[key] = value
        return response
    
    try:
        print(f"Preview dataset requested for ID: {dataset_id}")
        
        # Always load from the original file which now contains the preprocessed data
        print(f"Loading dataset from: {dataset.file.path}")
        df = load_dataset(dataset.file.path, dataset.file_type)
        
        # Drop selected columns if any
        if dataset.dropped_columns:
            # Make sure dropped_columns is a list
            dropped_columns = dataset.dropped_columns if isinstance(dataset.dropped_columns, list) else []
            if dropped_columns:
                print(f"Dropping columns: {dropped_columns}")
                existing_columns = set(df.columns)
                columns_to_drop = [col for col in dropped_columns if col in existing_columns]
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop, errors='ignore')
                    print(f"Dropped columns: {columns_to_drop}")
        
        print(f"Dataset loaded, shape: {df.shape}")
        
        # Get the requested number of rows from the query parameters
        try:
            rows = int(request.GET.get('rows', 500))  # Default to 500 if not specified
            rows = max(1, min(rows, len(df)))  # Ensure rows is between 1 and dataset length
        except (TypeError, ValueError):
            rows = min(500, len(df))  # Fallback to default if invalid value
        
        # Get a sample of the data for preview
        preview_data = df.head(rows)
        
        # Convert to dict for JSON serialization
        column_names = preview_data.columns.tolist()
        
        # Convert rows to a list of dictionaries for better JSON compatibility
        rows = []
        for _, row in preview_data.iterrows():
            row_dict = {}
            for col in column_names:
                value = row[col]
                # Handle various data types for JSON serialization
                if isinstance(value, (pd.Timestamp, np.datetime64)):
                    row_dict[col] = str(value)
                elif isinstance(value, (np.integer, np.floating)):
                    row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.bool_):
                    row_dict[col] = bool(value)
                elif pd.isna(value):
                    row_dict[col] = None
                else:
                    row_dict[col] = str(value)
            rows.append(row_dict)
        
        response = JsonResponse({
            'columns': column_names,
            'data': rows,
            'total_rows': len(df),
            'preview_rows': len(preview_data),
            'is_processed': dataset.has_preprocessing
        })
        
        # Add CORS headers to the response
        for key, value in response_headers.items():
            response[key] = value
            
        return response
    
    except Exception as e:
        print(f"Error in preview_dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_response = JsonResponse({'error': str(e)}, status=500)
        
        # Add CORS headers to the error response
        for key, value in response_headers.items():
            error_response[key] = value
            
        return error_response

@login_required
def download_dataset(request, dataset_id):
    """Endpoint to download the preprocessed dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        messages.error(request, "You don't have permission to access this dataset.")
        return redirect('analyzer:index')

    try:
        print(f"Downloading dataset from: {dataset.file.path}")
        
        # Load from the original file which now contains the preprocessed data
        df = load_dataset(dataset.file.path, dataset.file_type)
        
        # Drop selected columns if any
        if dataset.dropped_columns:
            # Make sure dropped_columns is a list
            dropped_columns = dataset.dropped_columns if isinstance(dataset.dropped_columns, list) else []
            if dropped_columns:
                print(f"Dropping columns for download: {dropped_columns}")
                existing_columns = set(df.columns)
                columns_to_drop = [col for col in dropped_columns if col in existing_columns]
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Determine suffix based on whether preprocessing was applied
        suffix = "_processed" if dataset.has_preprocessing else ""
        
        # Create a CSV file in memory
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{dataset.title}{suffix}.csv"'
        
        # Write to response
        df.to_csv(response, index=False)
        
        return response
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@login_required
def delete_dataset(request, dataset_id):
    """Delete a dataset."""
    if request.method == 'POST':
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Check permissions - only owners can delete their datasets
        if dataset.user != request.user:
            return JsonResponse({'error': 'Permission denied. You can only delete your own datasets.'}, status=403)
        
        try:
            # First delete the file from storage
            if dataset.file and os.path.exists(dataset.file.path):
                os.remove(dataset.file.path)
            
            # Then delete the dataset record
            dataset.delete()
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST request required'}, status=405)

@csrf_exempt
@login_required
def download_visualization(request, dataset_id):
    """Endpoint to download visualization as HTML file."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Required parameters
            plot_type = data.get('plot_type')  # e.g., 'histogram', 'scatter', 'boxplot'
            plot_data = data.get('plot_data')  # The JSON data for the plot
            plot_layout = data.get('plot_layout')  # The layout data
            column = data.get('column')  # Column name for the plot
            
            if not all([plot_type, plot_data, plot_layout, column]):
                return JsonResponse({'error': 'Missing required parameters'}, status=400)
            
            # Debug the data received
            print(f"Creating figure for {plot_type} of {column}")
            print(f"Plot data type: {type(plot_data)}")
            
            # Handle case where plot_data might not be a list
            if not isinstance(plot_data, list):
                if isinstance(plot_data, dict):
                    print("Converting plot_data from dict to list")
                    plot_data = [plot_data]
                else:
                    print(f"Invalid plot_data format: {type(plot_data)}")
                    # Create a simple default data
                    plot_data = [{"type": "scatter", "x": [1, 2, 3], "y": [1, 3, 2]}]
            
            # Create a Plotly figure from the data
            try:
                fig = go.Figure(data=plot_data, layout=plot_layout)
            except Exception as e:
                print(f"Error creating figure: {str(e)}")
                # Create a simple fallback figure
                fig = go.Figure(
                    data=[go.Scatter(x=[1, 2, 3], y=[1, 3, 2])],
                    layout=go.Layout(title=f"{column} visualization (fallback)")
                )
            
            # Set a suitable filename
            filename = f"{column}_{plot_type}_{uuid.uuid4().hex[:8]}.html"
            
            # Export the figure as HTML
            file_path = export_plotly_html(fig, filename)
            
            # Serve the file for download
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    response = HttpResponse(f.read(), content_type='text/html')
                    response['Content-Disposition'] = f'attachment; filename="{filename}"'
                    
                    # Clean up the file after sending
                    os.remove(file_path)
                    
                    return response
            else:
                return JsonResponse({'error': 'Failed to create visualization file'}, status=500)
            
        except Exception as e:
            print(f"Exception in download_visualization: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST request required'}, status=405)

@csrf_exempt
@login_required
def get_unique_values(request, dataset_id):
    """API endpoint to get all unique values for a specific column."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check ownership permissions
    if dataset.user != request.user:
        return JsonResponse({'error': 'You do not have permission to access this dataset'}, status=403)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            column_name = data.get('column')
            
            if not column_name:
                return JsonResponse({'error': 'Column name is required'}, status=400)
            
            # Load the dataset
            print(f"Loading dataset to get unique values for column: {column_name}")
            df = load_dataset(dataset.file.path, dataset.file_type)
            
            # Drop selected columns if any
            if dataset.dropped_columns:
                df = df.drop(columns=dataset.dropped_columns or [], errors='ignore')
            
            # Check if column exists
            if column_name not in df.columns:
                return JsonResponse({'error': f'Column "{column_name}" not found'}, status=400)
            
            # Get unique values
            unique_values = df[column_name].dropna().unique().tolist()
            
            # Convert non-serializable types to strings
            for i, val in enumerate(unique_values):
                if isinstance(val, (np.integer, np.floating)):
                    unique_values[i] = float(val) if isinstance(val, np.floating) else int(val)
                elif isinstance(val, (np.bool_, bool)):
                    unique_values[i] = bool(val)
                elif not isinstance(val, (str, int, float, bool, type(None))):
                    unique_values[i] = str(val)
            
            # Sort values for readability
            unique_values.sort(key=lambda x: str(x).lower() if x is not None else '')
            
            return JsonResponse({
                'column': column_name,
                'unique_values': unique_values,
                'count': len(unique_values),
                'total_rows': len(df)
            })
            
        except Exception as e:
            print(f"Error in get_unique_values: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST request required'}, status=405)

def register_view(request):
    """Handle user registration."""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('analyzer:index')
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form}) 