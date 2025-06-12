import pandas as pd
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
import plotly
from io import BytesIO
from django.conf import settings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from scipy.stats import gaussian_kde

# Add NumpyEncoder class
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        # Handle basic numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            # Check for empty array to prevent errors
            if obj.size == 0:
                return []
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return str(obj)
            
        # Handle special numeric values
        try:
            if pd.isna(obj) or np.isnan(obj):
                return None
        except (TypeError, ValueError):
            pass
            
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict() if isinstance(obj, pd.DataFrame) else obj.tolist()
        
        # Handle dates and times
        if isinstance(obj, (pd.Timestamp, np.datetime64, pd._libs.tslibs.timestamps.Timestamp)):
            return str(obj)
            
        # Handle other iterables
        if hasattr(obj, 'tolist'):
            return obj.tolist()
            
        try:
            return super(NumpyEncoder, self).default(obj)
        except TypeError:
            # Fallback for any other numpy types
            return str(obj)

def load_dataset(file_path, file_type):
    """Load dataset from file."""
    try:
        if file_type == 'csv':
            try:
                # Try with default settings first
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                # If that fails, try different encodings
                df = pd.read_csv(file_path, encoding='latin1')
            except Exception as e:
                print(f"Error reading CSV with standard options: {str(e)}")
                # Try with more forgiving settings
                df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
        elif file_type == 'excel':
            try:
                # Try standard approach first
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"Error reading Excel with standard options: {str(e)}")
                # Try with more options
                df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Make column names string type to avoid problems with numeric/special columns
        df.columns = df.columns.astype(str)
        
        # Check if file is empty
        if df.empty:
            raise ValueError("The dataset is empty")
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset: {str(e)}")

def get_column_types(df):
    """Get column data types."""
    column_types = {}
    
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's binary (0/1 values only)
                unique_vals = set(df[col].dropna().unique())
                if unique_vals == {0, 1} or unique_vals == {0.0, 1.0} or unique_vals == {'0', '1'}:
                    column_types[col] = 'binary'
                else:
                    column_types[col] = 'numeric'
            elif pd.api.types.is_datetime64_dtype(df[col]):
                column_types[col] = 'datetime'
            else:
                # Check if this might be a date column that pandas didn't automatically detect
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                    # Sample some non-null values to test for date format
                    sample = df[col].dropna().sample(min(30, len(df[col].dropna())))
                    try:
                        # Try to convert the sample to datetime
                        pd.to_datetime(sample, errors='raise')
                        # If we reach here, the conversion was successful, likely a date column
                        column_types[col] = 'datetime'
                        # Attempt to convert the entire column to datetime
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            print(f"Converted column '{col}' to datetime format")
                        except:
                            print(f"Column '{col}' appears to contain dates but couldn't be fully converted")
                        continue
                    except:
                        # Not a valid date format, continue with regular classification
                        pass
                
                # For object/string columns, check if they're categorical
                nunique = df[col].nunique()
                if nunique < 10 or (len(df) > 0 and nunique / len(df) < 0.05):  # Categorical if few unique values or small ratio
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'
        except Exception as e:
            print(f"Error determining type for column {col}: {str(e)}")
            # Default to text type for problematic columns
            column_types[col] = 'text'
    
    return column_types

def handle_missing_values(df, strategy, fill_value=None, column_strategies=None):
    """
    Handle missing values in the dataframe.
    
    Args:
        df: The dataframe to process
        strategy: The global strategy to apply to columns without specific strategies
        fill_value: The global fill value for constant strategy
        column_strategies: Dict mapping column names to their specific preprocessing settings
    
    Returns:
        Processed dataframe
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    total_missing = df.isna().sum().sum()
    print(f"MISSING VALUES - Before processing: {total_missing} total missing values")
    
    if total_missing == 0:
        print("No missing values to handle, skipping")
        return df
    
    # No strategy provided and no column-specific strategies
    if not strategy and not column_strategies:
        print("No missing value strategy provided, skipping")
        return df
    
    # Process any column-specific strategies first
    if column_strategies:
        processed_columns = []
        print("Processing column-specific missing value strategies:")
        
        for col, col_settings in column_strategies.items():
            # Skip columns that don't exist
            if col not in df.columns:
                print(f"Column '{col}' not found in DataFrame, skipping")
                continue
                
            # Get column-specific strategy
            col_strategy = col_settings.get('missing_values_strategy')
            
            # Skip if no strategy or if strategy is 'global'
            if not col_strategy or col_strategy == 'global':
                print(f"  Column '{col}': Using global strategy ({strategy or 'none'})")
                continue
            
            # Skip columns without missing values
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                print(f"  Column '{col}': No missing values, skipping")
                continue
                
            # Process this column with its specific strategy
            print(f"  Column '{col}': {missing_count} missing values with strategy '{col_strategy}'")
            processed_columns.append(col)
            
            # Get fill value for column if provided
            col_fill_value = col_settings.get('fill_value')
            if not col_fill_value and col_strategy == 'constant':
                # Fall back to global fill value for constant strategy
                col_fill_value = fill_value
                print(f"    Using global fill value: {fill_value}")
            
            # Apply the appropriate strategy for this column
            if col_strategy == 'drop':
                print(f"    Marking for row dropping due to missing values")
                # We'll handle this at the end
                continue
                
            elif col_strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                if df[col].dropna().empty:
                    print(f"    Cannot calculate mean for column '{col}' - all values are missing")
                    continue
                    
                mean_value = df[col].mean()
                print(f"    Filling {missing_count} missing values with mean: {mean_value}")
                before_na_count = df[col].isna().sum()
                df[col] = df[col].fillna(mean_value)
                after_na_count = df[col].isna().sum()
                print(f"    Before: {before_na_count} NAs, After: {after_na_count} NAs")
                
            elif col_strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                if df[col].dropna().empty:
                    print(f"    Cannot calculate median for column '{col}' - all values are missing")
                    continue
                    
                median_value = df[col].median()
                print(f"    Filling {missing_count} missing values with median: {median_value}")
                before_na_count = df[col].isna().sum()
                df[col] = df[col].fillna(median_value)
                after_na_count = df[col].isna().sum()
                print(f"    Before: {before_na_count} NAs, After: {after_na_count} NAs")
                
            elif col_strategy == 'mode':
                if df[col].dropna().empty:
                    print(f"    Cannot calculate mode for column '{col}' - all values are missing")
                    continue
                    
                if not df[col].mode().empty:
                    mode_value = df[col].mode()[0]
                    print(f"    Filling {missing_count} missing values with mode: {mode_value}")
                    before_na_count = df[col].isna().sum()
                    df[col] = df[col].fillna(mode_value)
                    after_na_count = df[col].isna().sum()
                    print(f"    Before: {before_na_count} NAs, After: {after_na_count} NAs")
                    
            elif col_strategy == 'constant':
                print(f"    Filling {missing_count} missing values with constant: {col_fill_value}")
                before_na_count = df[col].isna().sum()
                df[col] = df[col].fillna(col_fill_value)
                after_na_count = df[col].isna().sum()
                print(f"    Before: {before_na_count} NAs, After: {after_na_count} NAs")
                
            elif col_strategy == 'ffill':
                print(f"    Forward filling {missing_count} missing values")
                before_na_count = df[col].isna().sum()
                df[col] = df[col].ffill()
                after_na_count = df[col].isna().sum()
                print(f"    Before: {before_na_count} NAs, After: {after_na_count} NAs")
                
            elif col_strategy == 'bfill':
                print(f"    Backward filling {missing_count} missing values")
                before_na_count = df[col].isna().sum()
                df[col] = df[col].bfill()
                after_na_count = df[col].isna().sum()
                print(f"    Before: {before_na_count} NAs, After: {after_na_count} NAs")
        
        # Now apply the global strategy to any columns that weren't specifically processed
        if strategy:
            remaining_columns = [col for col in df.columns if col not in processed_columns]
            missing_in_remaining = df[remaining_columns].isna().sum().sum()
            
            if missing_in_remaining > 0:
                print(f"\nApplying global strategy '{strategy}' to remaining {len(remaining_columns)} columns with {missing_in_remaining} missing values")
                
                # Create a new dataframe with just the remaining columns
                remaining_df = df[remaining_columns].copy()
                
                if strategy == 'drop':
                    # Create a mask for rows to keep
                    mask = ~remaining_df.isna().any(axis=1)
                    rows_to_drop = (~mask).sum()
                    if rows_to_drop > 0:
                        print(f"  Dropping {rows_to_drop} rows with missing values in remaining columns")
                        df = df[mask].copy()
                    else:
                        print("  No rows to drop in remaining columns")
                        
                elif strategy == 'mean':
                    numeric_cols = remaining_df.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        missing_count = remaining_df[col].isna().sum()
                        if missing_count > 0:
                            mean_value = remaining_df[col].mean()
                            print(f"  Filling {missing_count} missing values in column '{col}' with mean: {mean_value}")
                            df[col] = df[col].fillna(mean_value)
                            
                elif strategy == 'median':
                    numeric_cols = remaining_df.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        missing_count = remaining_df[col].isna().sum()
                        if missing_count > 0:
                            median_value = remaining_df[col].median()
                            print(f"  Filling {missing_count} missing values in column '{col}' with median: {median_value}")
                            df[col] = df[col].fillna(median_value)
                            
                elif strategy == 'mode':
                    for col in remaining_columns:
                        missing_count = remaining_df[col].isna().sum()
                        if missing_count > 0 and not remaining_df[col].mode().empty:
                            mode_value = remaining_df[col].mode()[0]
                            print(f"  Filling {missing_count} missing values in column '{col}' with mode: {mode_value}")
                            df[col] = df[col].fillna(mode_value)
                            
                elif strategy == 'constant':
                    for col in remaining_columns:
                        missing_count = remaining_df[col].isna().sum()
                        if missing_count > 0:
                            print(f"  Filling {missing_count} missing values in column '{col}' with constant: {fill_value}")
                            df[col] = df[col].fillna(fill_value)
                            
                elif strategy == 'ffill':
                    for col in remaining_columns:
                        missing_count = remaining_df[col].isna().sum()
                        if missing_count > 0:
                            print(f"  Forward filling {missing_count} missing values in column '{col}'")
                            df[col] = df[col].ffill()
                            
                elif strategy == 'bfill':
                    for col in remaining_columns:
                        missing_count = remaining_df[col].isna().sum()
                        if missing_count > 0:
                            print(f"  Backward filling {missing_count} missing values in column '{col}'")
                            df[col] = df[col].bfill()
    
    # If no column-specific strategies, apply global strategy to all columns
    elif strategy:
        if strategy == 'drop':
            print(f"Applying global DROP strategy")
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            print(f"Dropped {rows_dropped} rows with missing values")
            
        elif strategy == 'mean':
            print(f"Applying global MEAN strategy")
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  Filling {missing_count} missing values in column '{col}' with mean: {df[col].mean()}")
                    df[col] = df[col].fillna(df[col].mean())
                    
        elif strategy == 'median':
            print(f"Applying global MEDIAN strategy")
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  Filling {missing_count} missing values in column '{col}' with median: {df[col].median()}")
                    df[col] = df[col].fillna(df[col].median())
                    
        elif strategy == 'mode':
            print(f"Applying global MODE strategy")
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0 and not df[col].mode().empty:
                    mode_value = df[col].mode()[0]
                    print(f"  Filling {missing_count} missing values in column '{col}' with mode: {mode_value}")
                    df[col] = df[col].fillna(mode_value)
                    
        elif strategy == 'constant':
            print(f"Applying global CONSTANT strategy with value: {fill_value}")
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  Filling {missing_count} missing values in column '{col}' with constant: {fill_value}")
            df = df.fillna(fill_value)
            
        elif strategy == 'ffill':
            print(f"Applying global FORWARD FILL strategy")
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  Forward filling {missing_count} missing values in column '{col}'")
            df = df.ffill()
            
        elif strategy == 'bfill':
            print(f"Applying global BACKWARD FILL strategy")
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  Backward filling {missing_count} missing values in column '{col}'")
            df = df.bfill()
    
    # Handle 'drop' strategy for columns that specified it
    if column_strategies:
        cols_to_drop = []
        for col, col_settings in column_strategies.items():
            if col_settings.get('missing_values_strategy') == 'drop':
                cols_to_drop.append(col)
        
        if cols_to_drop:
            rows_before = len(df)
            print(f"Dropping rows with missing values in columns: {cols_to_drop}")
            df = df.dropna(subset=cols_to_drop)
            rows_dropped = rows_before - len(df)
            print(f"  Dropped {rows_dropped} rows")
    
    final_missing = df.isna().sum().sum()
    print(f"MISSING VALUES - After processing: {final_missing} total missing values")
    print(f"  Missing values filled: {total_missing - final_missing}")
    
    if final_missing > 0:
        print(f"  Remaining missing values by column: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
    
    return df

def encode_categorical(df, strategy, column_types, column_strategies=None):
    """
    Encode categorical variables in the dataframe.
    
    Args:
        df: The dataframe to process
        strategy: The global encoding strategy ('none', 'onehot', 'label')
        column_types: Dict mapping column names to data types
        column_strategies: Dict mapping column names to their specific preprocessing settings
    
    Returns:
        Processed dataframe
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    print(f"Starting categorical encoding with strategy: {strategy}")
    
    # Skip if global strategy is 'none' and all column strategies are also 'none'
    if strategy == 'none' and (not column_strategies or all(
            col_settings.get('encoding_strategy', 'global') in ['global', 'none'] 
            for col, col_settings in column_strategies.items() if col in df.columns)):
        print("No encoding needed (all strategies are 'none')")
        return df
    
    # Identify categorical columns
    categorical_cols = [col for col, type_ in column_types.items() 
                       if type_ in ['categorical', 'binary'] and col in df.columns]
    
    if not categorical_cols:
        print("No categorical columns found for encoding")
        return df
    
    print(f"Found {len(categorical_cols)} categorical columns to potentially encode")
    
    # For each categorical column, apply the appropriate encoding
    for col in categorical_cols:
        # Determine which strategy to use for this column
        col_strategy = strategy
        
        if column_strategies and col in column_strategies:
            col_settings = column_strategies[col]
            col_encoding = col_settings.get('encoding_strategy', 'global')
            
            if col_encoding != 'global':
                col_strategy = col_encoding
            
        # Skip columns with 'none' strategy
        if col_strategy == 'none':
            print(f"  Skipping encoding for column '{col}' (strategy: none)")
            continue
            
        # Count of unique values
        unique_count = df[col].nunique()
        print(f"  Column '{col}' has {unique_count} unique values")
            
        # Handle NaN values before encoding
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"  Column '{col}' has {missing_count} missing values, filling with '_MISSING_' before encoding")
            df[col] = df[col].fillna('_MISSING_')
        
        if col_strategy == 'label':
            # Label encoding
            try:
                from sklearn.preprocessing import LabelEncoder
                print(f"  Applying label encoding to column '{col}'")
                # Get distribution of values before encoding
                value_counts = df[col].value_counts().head(5).to_dict()
                print(f"    Top values before encoding: {value_counts}")
                
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                
                # Show the mapping for reference
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                top_mappings = {k: mapping[str(k)] for k in value_counts.keys() if str(k) in mapping}
                print(f"    Mapping examples: {top_mappings}")
                
            except Exception as e:
                print(f"    Error during label encoding: {str(e)}")
            
        elif col_strategy == 'onehot':
            # One-hot encoding
            try:
                print(f"  Applying one-hot encoding to column '{col}'")
                # Get number of unique values
                print(f"    Creating {df[col].nunique()} new binary columns")
                
                # Store the column names before encoding
                old_cols = set(df.columns)
                
                # Apply one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                # Drop the original column and add the dummy columns
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                
                # Calculate new columns added
                new_cols = set(df.columns) - old_cols
                print(f"    Added {len(new_cols)} new columns: {', '.join(list(new_cols)[:5])}{'...' if len(new_cols) > 5 else ''}")
                
            except Exception as e:
                print(f"    Error during one-hot encoding: {str(e)}")
    
    return df

def scale_features(df, strategy, column_strategies=None):
    """
    Scale numeric features in the dataframe.
    
    Args:
        df: The dataframe to process
        strategy: The global scaling strategy ('none', 'minmax', 'standard', 'robust')
        column_strategies: Dict mapping column names to their specific preprocessing settings
    
    Returns:
        Processed dataframe
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    print(f"Starting feature scaling with strategy: {strategy}")
    
    # Skip if strategy is 'none' and all column strategies are also 'none'
    if strategy == 'none' and (not column_strategies or all(
            col_settings.get('scaling_strategy', 'global') in ['global', 'none'] 
            for col, col_settings in column_strategies.items() if col in df.columns)):
        print("No scaling needed (all strategies are 'none')")
        return df
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns found for scaling")
        return df
    
    print(f"Found {len(numeric_cols)} numeric columns to potentially scale")
    
    # For each numeric column, apply the appropriate scaling
    for col in numeric_cols:
        # Determine which strategy to use for this column
        col_strategy = strategy
        
        if column_strategies and col in column_strategies:
            col_settings = column_strategies[col]
            col_scaling = col_settings.get('scaling_strategy', 'global')
            
            if col_scaling != 'global':
                col_strategy = col_scaling
        
        # Skip columns with 'none' strategy
        if col_strategy == 'none':
            print(f"  Skipping scaling for column '{col}' (strategy: none)")
            continue
        
        # Handle NaN values before scaling (replace with mean)
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"  Column '{col}' has {missing_count} missing values, filling with mean before scaling")
            df[col] = df[col].fillna(df[col].mean())
        
        # Create the appropriate scaler
        if col_strategy == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            print(f"  Applying MinMax scaling to column '{col}'")
            scaler = MinMaxScaler()
        elif col_strategy == 'standard':
            from sklearn.preprocessing import StandardScaler
            print(f"  Applying Standard scaling to column '{col}'")
            scaler = StandardScaler()
        elif col_strategy == 'robust':
            from sklearn.preprocessing import RobustScaler
            print(f"  Applying Robust scaling to column '{col}'")
            scaler = RobustScaler()
        else:
            print(f"  Unknown scaling strategy '{col_strategy}' for column '{col}', skipping")
            continue
        
        # Apply scaling to the column
        try:
            # Print some stats before scaling
            before_min = df[col].min()
            before_max = df[col].max()
            before_mean = df[col].mean()
            before_std = df[col].std()
            
            # Apply scaling
            df[col] = scaler.fit_transform(df[[col]])
            
            # Print some stats after scaling
            after_min = df[col].min()
            after_max = df[col].max()
            after_mean = df[col].mean()
            after_std = df[col].std()
            
            print(f"    Before scaling: min={before_min:.4f}, max={before_max:.4f}, mean={before_mean:.4f}, std={before_std:.4f}")
            print(f"    After scaling:  min={after_min:.4f}, max={after_max:.4f}, mean={after_mean:.4f}, std={after_std:.4f}")
        except Exception as e:
            print(f"    Error scaling column '{col}': {str(e)}")
    
    return df

def handle_outliers(df, column, strategy='cap', threshold=1.5):
    """Handle outliers in a numeric column using IQR method."""
    if strategy not in ['cap', 'trim', 'clip', 'remove']:
        return df
    
    # For backward compatibility
    if strategy == 'clip':
        strategy = 'cap'
    elif strategy == 'remove':
        strategy = 'trim'
    
    # Calculate IQR and bounds using the IQR method
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    # Calculate lower and upper bounds
    # The threshold parameter is the multiplier for IQR (default is 1.5 for standard outlier detection)
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Print information about the bounds
    print(f"  IQR outlier detection for column '{column}':")
    print(f"    Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    print(f"    Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
    
    if strategy == 'cap':
        # Count outliers before capping
        outliers_below = (df[column] < lower_bound).sum()
        outliers_above = (df[column] > upper_bound).sum()
        total_outliers = outliers_below + outliers_above
        
        if total_outliers > 0:
            print(f"  Capping {total_outliers} outliers in column '{column}':")
            print(f"    - {outliers_below} below lower bound, {outliers_above} above upper bound")
            # Cap the outliers using the IQR-based bounds
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        else:
            print(f"  No outliers found in column '{column}' using IQR method")
    else:  # trim
        # Create a mask for rows within bounds
        inlier_mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        outliers = (~inlier_mask).sum()
        
        if outliers > 0:
            rows_before = len(df)
            print(f"  Removing {outliers} rows with outliers in column '{column}'")
            df = df[inlier_mask]
            print(f"  Removed {rows_before - len(df)} rows")
        else:
            print(f"  No outliers found in column '{column}' using IQR method")
    
    return df

def apply_numeric_conditions(df, column, conditions):
    """
    Apply numeric conditions to filter rows.
    Rows that match the conditions will be REMOVED from the dataset.
    
    Args:
        df: DataFrame to filter
        column: Column to apply conditions to
        conditions: List of condition dictionaries with 'type' and 'value' keys
        
    Returns:
        Filtered DataFrame with rows removed that match the conditions
    """
    print(f"\n===== APPLYING NUMERIC CONDITIONS FOR COLUMN '{column}' =====")
    print(f"Original dataframe shape: {df.shape}")
    print(f"First 5 rows of original dataframe:\n{df.head()}")
    
    # Make a copy to avoid modifying the original
    df_result = df.copy()
    
    # Handle string JSON representation (might be coming from DB)
    if isinstance(conditions, str):
        import json
        try:
            conditions = json.loads(conditions)
            print(f"Parsed JSON conditions: {conditions}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse conditions: {e}")
            return df_result
    
    # Verify conditions is a list or has expected format
    if not conditions or not isinstance(conditions, list):
        print(f"No valid numeric conditions for column '{column}'")
        return df_result
    
    # Verify column exists
    if column not in df_result.columns:
        print(f"Column '{column}' not found in DataFrame")
        return df_result
    
    # Verify column is numeric
    if not pd.api.types.is_numeric_dtype(df_result[column]):
        print(f"Column '{column}' is not numeric (type: {df_result[column].dtype})")
        return df_result
        
    rows_before = len(df_result)
    print(f"Processing {len(conditions)} numeric conditions")
    print(f"Column values sample: {df_result[column].head().tolist()}")
    
    # Process each condition individually
    for i, condition in enumerate(conditions):
        print(f"Condition {i+1}: {condition}")
        
        # Skip invalid conditions
        if not isinstance(condition, dict):
            print(f"Skipping invalid condition format: {type(condition)}")
            continue
            
        condition_type = condition.get('type')
        condition_value = condition.get('value')
        
        if not condition_type or condition_value is None:
            print(f"Skipping condition with missing type or value")
            continue
        
        # Convert value to numeric if needed
        try:
            value = float(condition_value)
        except (ValueError, TypeError):
            print(f"Could not convert condition value '{condition_value}' to numeric")
            continue
        
        # Calculate the condition mask
        rows_count_before = len(df_result)
        
        # Create a mask to identify rows to DROP
        if condition_type == 'greater_than':
            print(f"FILTER: Removing rows where {column} > {value}")
            # Direct filtering approach: drop rows where condition is met
            drop_mask = df_result[column] > value
            rows_to_drop = drop_mask.sum()
            print(f"Found {rows_to_drop} rows to drop where {column} > {value}")
            
            if rows_to_drop > 0:
                # Show samples of rows that will be dropped
                drop_samples = df_result[drop_mask].head(3)
                print(f"Sample rows to drop:\n{drop_samples}")
                
                # Actually drop the rows
                df_result = df_result[~drop_mask]
                print(f"After dropping: {len(df_result)} rows remain (dropped {rows_count_before - len(df_result)} rows)")
                
        elif condition_type == 'less_than':
            print(f"FILTER: Removing rows where {column} < {value}")
            drop_mask = df_result[column] < value
            rows_to_drop = drop_mask.sum()
            print(f"Found {rows_to_drop} rows to drop where {column} < {value}")
            
            if rows_to_drop > 0:
                drop_samples = df_result[drop_mask].head(3)
                print(f"Sample rows to drop:\n{drop_samples}")
                df_result = df_result[~drop_mask]
                print(f"After dropping: {len(df_result)} rows remain (dropped {rows_count_before - len(df_result)} rows)")
                
        elif condition_type == 'equal_to':
            print(f"FILTER: Removing rows where {column} == {value}")
            drop_mask = df_result[column] == value
            rows_to_drop = drop_mask.sum()
            print(f"Found {rows_to_drop} rows to drop where {column} == {value}")
            
            if rows_to_drop > 0:
                drop_samples = df_result[drop_mask].head(3)
                print(f"Sample rows to drop:\n{drop_samples}")
                df_result = df_result[~drop_mask]
                print(f"After dropping: {len(df_result)} rows remain (dropped {rows_count_before - len(df_result)} rows)")
                
        elif condition_type == 'not_equal_to':
            print(f"FILTER: Removing rows where {column} != {value}")
            drop_mask = df_result[column] != value
            rows_to_drop = drop_mask.sum()
            print(f"Found {rows_to_drop} rows to drop where {column} != {value}")
            
            if rows_to_drop > 0:
                drop_samples = df_result[drop_mask].head(3)
                print(f"Sample rows to drop:\n{drop_samples}")
                df_result = df_result[~drop_mask]
                print(f"After dropping: {len(df_result)} rows remain (dropped {rows_count_before - len(df_result)} rows)")
                
        elif condition_type == 'between':
            min_val = condition.get('min')
            max_val = condition.get('max')
            if min_val is not None and max_val is not None:
                try:
                    min_val = float(min_val)
                    max_val = float(max_val)
                    print(f"FILTER: Removing rows where {min_val} <= {column} <= {max_val}")
                    drop_mask = (df_result[column] >= min_val) & (df_result[column] <= max_val)
                    rows_to_drop = drop_mask.sum()
                    print(f"Found {rows_to_drop} rows to drop where {min_val} <= {column} <= {max_val}")
                    
                    if rows_to_drop > 0:
                        drop_samples = df_result[drop_mask].head(3)
                        print(f"Sample rows to drop:\n{drop_samples}")
                        df_result = df_result[~drop_mask]
                        print(f"After dropping: {len(df_result)} rows remain (dropped {rows_count_before - len(df_result)} rows)")
                except (ValueError, TypeError):
                    print(f"Could not convert min/max values to numeric")
    
    # Print final results
    rows_removed = rows_before - len(df_result)
    print(f"FINAL RESULT: Removed {rows_removed} rows out of {rows_before} rows (remaining: {len(df_result)})")
    if len(df_result) > 0:
        print(f"First 5 rows of filtered dataframe:\n{df_result.head()}")
    
    # Return the filtered DataFrame
    return df_result

def apply_categorical_conditions(df, column, conditions):
    """
    Filter data based on categorical conditions.
    
    Args:
        df: The dataframe to process
        column: The column to apply conditions to
        conditions: Dict containing 'include' and 'exclude' lists of values
        
    Returns:
        Filtered dataframe
    """
    if not conditions or not isinstance(conditions, dict):
        print(f"No valid categorical conditions provided for {column}")
        return df
    
    include_values = conditions.get('include', [])
    exclude_values = conditions.get('exclude', [])
    
    if not include_values and not exclude_values:
        print(f"No values provided for categorical filtering on {column}")
        return df
    
    before_count = len(df)
    print(f"Applying categorical filter to column '{column}'")
    print(f"Include values: {include_values}")
    print(f"Exclude values: {exclude_values}")
    
    # First apply include filter if specified
    if include_values:
        df = df[~df[column].isin(include_values)]
        after_include = len(df)
        print(f"After include filter: {after_include} rows (removed {before_count - after_include} rows)")
    
    # Then apply exclude filter if specified
    if exclude_values:
        df = df[df[column].isin(exclude_values)]
        after_exclude = len(df)
        print(f"After exclude filter: {after_exclude} rows (removed {len(df) - after_exclude} rows)")
    
    after_count = len(df)
    rows_removed = before_count - after_count
    print(f"Total rows removed by categorical filter: {rows_removed}")
    
    return df

def apply_value_replacements(df, column, replacements):
    """
    Replace specific values in a column.
    
    Args:
        df: The dataframe to process
        column: The column to apply replacements to
        replacements: List of dicts with 'original' and 'replacement' keys
        
    Returns:
        DataFrame with replaced values
    """
    if not replacements or not isinstance(replacements, list) or column not in df.columns:
        print(f"No valid replacements provided for {column} or column doesn't exist")
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    print(f"Applying {len(replacements)} value replacements to column '{column}'")
    
    for replacement in replacements:
        original = replacement.get('original')
        replacement_value = replacement.get('replacement', '')
        
        if original is None:
            continue
            
        # Count occurrences before replacement
        if pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns, convert the original value to the appropriate numeric type
            try:
                if '.' in original:
                    original = float(original)
                else:
                    original = int(original)
                    
                # Also convert replacement value if it's not empty
                if replacement_value:
                    if '.' in replacement_value:
                        replacement_value = float(replacement_value)
                    else:
                        replacement_value = int(replacement_value)
            except ValueError:
                # If conversion fails, skip this replacement
                print(f"Warning: Could not convert '{original}' to numeric for column '{column}'")
                continue
                
        # Count occurrences and replace
        occurrences = (df[column] == original).sum()
        if occurrences > 0:
            df[column] = df[column].replace(original, replacement_value)
            print(f"  Replaced {occurrences} instances of '{original}' with '{replacement_value}'")
        else:
            print(f"  No instances of '{original}' found in column '{column}'")
    
    return df

def apply_datetime_conditions(df, column, conditions):
    """
    Apply datetime conditions to filter rows.
    Rows that match the conditions will be REMOVED from the dataset.
    
    Args:
        df: DataFrame to filter
        column: Column to apply conditions to
        conditions: List of condition dictionaries with 'type' and 'value' keys
        
    Returns:
        Filtered DataFrame with rows removed that match the conditions
    """
    print(f"\n===== APPLYING DATETIME CONDITIONS FOR COLUMN '{column}' =====")
    print(f"Original dataframe shape: {df.shape}")
    
    # Make a copy to avoid modifying the original
    df_result = df.copy()
    
    # Handle string JSON representation (might be coming from DB)
    if isinstance(conditions, str):
        import json
        try:
            conditions = json.loads(conditions)
            print(f"Parsed JSON conditions: {conditions}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse conditions: {e}")
            return df_result
    
    # Verify conditions is a list or has expected format
    if not conditions or not isinstance(conditions, list):
        print(f"No valid datetime conditions for column '{column}'")
        return df_result
    
    # Verify column exists
    if column not in df_result.columns:
        print(f"Column '{column}' not found in DataFrame")
        return df_result
    
    # Verify column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_result[column]):
        # Try to convert to datetime if it's not already
        try:
            df_result[column] = pd.to_datetime(df_result[column])
        except Exception as e:
            print(f"Column '{column}' cannot be converted to datetime (error: {e})")
            return df_result
    
    rows_before = len(df_result)
    print(f"Processing {len(conditions)} datetime conditions")
    
    # Process each condition
    for condition in conditions:
        try:
            condition_type = condition['type']
            condition_value = pd.to_datetime(condition['value'])
            
            if condition_type == 'greater_than':
                df_result = df_result[~(df_result[column] > condition_value)]
            elif condition_type == 'less_than':
                df_result = df_result[~(df_result[column] < condition_value)]
            elif condition_type == 'equal_to':
                df_result = df_result[~(df_result[column] == condition_value)]
            elif condition_type == 'not_equal_to':
                df_result = df_result[~(df_result[column] != condition_value)]
            elif condition_type == 'between':
                dates = condition_value.split(',')
                start_date = pd.to_datetime(dates[0])
                end_date = pd.to_datetime(dates[1])
                df_result = df_result[~((df_result[column] >= start_date) & (df_result[column] <= end_date))]
            else:
                print(f"Unknown condition type: {condition_type}")
                continue
                
        except Exception as e:
            print(f"Error processing datetime condition: {e}")
            continue
    
    rows_after = len(df_result)
    rows_removed = rows_before - rows_after
    print(f"Removed {rows_removed} rows ({(rows_removed/rows_before)*100:.2f}%) based on datetime conditions")
    
    return df_result

def apply_preprocessing(df, preprocessing_config, dataset_id=None):
    """Apply preprocessing steps to the dataset."""
    print("\n===== APPLYING PREPROCESSING =====")
    print(f"Initial DataFrame shape: {df.shape}")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Extract configuration
    config = preprocessing_config
    column_settings = config.get('column_settings', {})
    column_strategies = {}
    
    # Debug print the column settings
    if column_settings:
        print(f"Found column-specific settings for {len(column_settings)} columns")
        for col, settings in column_settings.items():
            if isinstance(settings, str):
                try:
                    settings = json.loads(settings)
                    column_settings[col] = settings
                    print(f"Parsed column settings for '{col}' from string: {settings}")
                except json.JSONDecodeError:
                    print(f"Error parsing column settings for '{col}': {settings}")
                    # Don't stop processing, continue with what we can parse
            
            # Now check if we have missing values strategy
            if isinstance(settings, dict) and 'missing_values_strategy' in settings:
                print(f"Column '{col}' has missing value strategy: {settings['missing_values_strategy']}")
                column_strategies[col] = {
                    'missing_values_strategy': settings['missing_values_strategy'],
                    'fill_value': settings.get('fill_value')
                }
            
            # Print all conditions for debugging
            if isinstance(settings, dict):
                if 'numeric_conditions' in settings:
                    print(f"Column '{col}' has numeric conditions: {settings['numeric_conditions']}")
                if 'categorical_conditions' in settings:
                    print(f"Column '{col}' has categorical conditions: {settings['categorical_conditions']}")
                if 'datetime_conditions' in settings:
                    print(f"Column '{col}' has datetime conditions: {settings['datetime_conditions']}")
                if 'value_replacements' in settings:
                    print(f"Column '{col}' has value replacements: {settings['value_replacements']}")
    else:
        print("No column-specific settings found")
    
    # Apply numeric, categorical, and datetime conditions
    if column_settings:
        print("\n=== Applying Column Conditions ===")
        for col, settings in column_settings.items():
            if col not in df.columns:
                print(f"Column '{col}' not found, skipping conditions")
                continue

            # Handle numeric conditions
            numeric_conditions = settings.get('numeric_conditions')
            if numeric_conditions:
                print(f"Applying numeric conditions to column '{col}'")
                df = apply_numeric_conditions(df, col, numeric_conditions)
                print(f"DataFrame shape after numeric conditions: {df.shape}")

            # Handle categorical conditions
            categorical_conditions = settings.get('categorical_conditions')
            if categorical_conditions:
                print(f"Applying categorical conditions to column '{col}'")
                df = apply_categorical_conditions(df, col, categorical_conditions)
                print(f"DataFrame shape after categorical conditions: {df.shape}")
            
            # Handle datetime conditions
            datetime_conditions = settings.get('datetime_conditions')
            if datetime_conditions:
                print(f"Applying datetime conditions to column '{col}'")
                df = apply_datetime_conditions(df, col, datetime_conditions)
                print(f"DataFrame shape after datetime conditions: {df.shape}")
    
    # Apply value replacements (do this early before other operations)
    if column_settings:
        print("\n=== Applying Value Replacements ===")
        for col, settings in column_settings.items():
            # Skip columns that don't exist or don't have replacements
            if col not in df.columns or 'value_replacements' not in settings:
                continue
                
            value_replacements = settings.get('value_replacements')
            
            # Convert from string to list if needed
            if isinstance(value_replacements, str):
                try:
                    value_replacements = json.loads(value_replacements)
                    print(f"Parsed value replacements for column '{col}': {value_replacements}")
                except json.JSONDecodeError:
                    print(f"Error parsing value replacements for column '{col}': {value_replacements}")
                    continue
                except Exception as e:
                    print(f"Unexpected error parsing value replacements: {str(e)}")
                    continue
            
            # Apply the replacements if we have them
            if value_replacements:
                print(f"Applying {len(value_replacements)} value replacements to column '{col}'")
                df = apply_value_replacements(df, col, value_replacements)
                print(f"Finished value replacements for column '{col}'")
    
    # Missing values handling
    missing_strategy = config.get('missing_values_strategy')
    fill_value = config.get('fill_value')
    
    if missing_strategy or column_strategies:
        print("\n=== Handling Missing Values ===")
        df = handle_missing_values(
            df, 
            strategy=missing_strategy, 
            fill_value=fill_value, 
            column_strategies=column_strategies
        )
        print(f"DataFrame shape after missing values handling: {df.shape}")
    
    # Handle outliers
    outlier_strategy = config.get('outlier_strategy')
    outlier_params = config.get('outlier_params', {})
    if outlier_strategy and outlier_strategy != 'none':
        df = handle_outliers(df, outlier_strategy, outlier_params)
    
    # Handle global outliers
    handle_outliers_global = config.get('handle_outliers')
    outlier_strategy = config.get('outlier_strategy')
    
    if handle_outliers_global:
        print("\n=== Handling Global Outliers ===")
        if not outlier_strategy or outlier_strategy == 'none':
            print("No outlier strategy specified, skipping global outlier handling")
        else:
            # Get numeric columns for outlier handling
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            print(f"Found {len(numeric_cols)} numeric columns for outlier handling")
            
            for col in numeric_cols:
                print(f"Applying global outlier strategy '{outlier_strategy}' to column '{col}'")
            
            # Calculate IQR and bounds
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if outlier_strategy == 'trim':
                # Create a mask for rows within bounds for this column
                rows_before = len(df)
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                outliers = (~col_mask).sum()
                
                if outliers > 0:
                    print(f"  Removing {outliers} rows with outliers in column '{col}'")
                    df = df[col_mask]
                    print(f"  Removed {rows_before - len(df)} rows")
                    
            elif outlier_strategy == 'cap':
                # Count outliers before capping
                outliers_below = (df[col] < lower_bound).sum()
                outliers_above = (df[col] > upper_bound).sum()
                total_outliers = outliers_below + outliers_above
                
                if total_outliers > 0:
                    print(f"  Capping {total_outliers} outliers in column '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                    print(f"    - {outliers_below} below lower bound, {outliers_above} above upper bound")
                    # Cap the outliers
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Apply column-specific value replacements
    if column_settings:
        print("\n=== Handling Column-Specific Value Replacements ===")
        for col, settings in column_settings.items():
            if col not in df.columns:
                print(f"Column '{col}' not found, skipping value replacements")
                continue
                
            # Check if the column has value replacements
            value_replacements = settings.get('value_replacements')
            
            # Convert from string to list if needed
            if isinstance(value_replacements, str):
                try:
                    value_replacements = json.loads(value_replacements)
                    print(f"Parsed value replacements for '{col}': {value_replacements}")
                except json.JSONDecodeError:
                    print(f"Error parsing value replacements for '{col}': {value_replacements}")
                    continue
                except Exception as e:
                    print(f"Unexpected error parsing value replacements: {str(e)}")
                    continue
            
            # Apply value replacements if they exist
            if value_replacements:
                print(f"Applying {len(value_replacements)} value replacements to column '{col}'")
                df = apply_value_replacements(df, col, value_replacements)
                print(f"Column '{col}' value replacements completed")
    
    # Apply column-specific encoding
    if column_settings:
        print("\n=== Handling Column-Specific Encoding ===")
        
        # Get column types for proper encoding
        column_types = get_column_types(df)
        categorical_cols = [col for col, type_ in column_types.items() 
                           if type_ in ['categorical', 'binary'] and col in df.columns]
        
        print(f"Found {len(categorical_cols)} categorical columns that might be encoded")
        
        # Create column-specific encoding strategies dictionary
        encoding_column_strategies = {}
        for col, settings in column_settings.items():
            if col in df.columns and col in categorical_cols:
                col_encoding_strategy = settings.get('encoding_strategy')
                if col_encoding_strategy and col_encoding_strategy != 'global':
                    encoding_column_strategies[col] = {'encoding_strategy': col_encoding_strategy}
                    print(f"Column '{col}': Will apply encoding strategy '{col_encoding_strategy}'")
        
        # Apply column-specific encoding using the existing function
        if encoding_column_strategies:
            print(f"Applying column-specific encoding for {len(encoding_column_strategies)} columns")
            df = encode_categorical(df, 'none', column_types, encoding_column_strategies)
            print(f"DataFrame shape after column-specific encoding: {df.shape}")
        else:
            print("No column-specific encoding to apply")
    
    # Apply column-specific scaling
    if column_settings:
        print("\n=== Handling Column-Specific Scaling ===")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"Found {len(numeric_cols)} numeric columns that might be scaled")
        
        # Create column-specific scaling strategies dictionary
        scaling_column_strategies = {}
        for col, settings in column_settings.items():
            if col in df.columns and col in numeric_cols:
                col_scaling_strategy = settings.get('scaling_strategy')
                if col_scaling_strategy and col_scaling_strategy != 'global':
                    scaling_column_strategies[col] = {'scaling_strategy': col_scaling_strategy}
                    print(f"Column '{col}': Will apply scaling strategy '{col_scaling_strategy}'")
                    
        # Apply column-specific scaling using the existing function
        if scaling_column_strategies:
            print(f"Applying column-specific scaling for {len(scaling_column_strategies)} columns")
            df = scale_features(df, 'none', scaling_column_strategies)
            print(f"DataFrame shape after column-specific scaling: {df.shape}")
        else:
            print("No column-specific scaling to apply")
            
    # Apply column-specific outlier handling
    if column_settings:
        print("\n=== Handling Column-Specific Outliers ===")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Track columns that will have outlier handling applied
        outlier_cols_processed = []
        
        # Process each column's outlier settings
        for col, settings in column_settings.items():
            if col in df.columns and col in numeric_cols:
                handle_outliers_col = settings.get('handle_outliers', False)
                outlier_strategy_col = settings.get('outlier_strategy')
                
                if handle_outliers_col and outlier_strategy_col and outlier_strategy_col != 'global':
                    print(f"Column '{col}': Applying outlier strategy '{outlier_strategy_col}'")
                    
                    # Get column stats before processing
                    before_min = df[col].min()
                    before_max = df[col].max()
                    before_count = len(df)
                    
                    # Apply outlier handling to this column
                    df = handle_outliers(df, col, strategy=outlier_strategy_col)
                    
                    # Get column stats after processing
                    after_min = df[col].min()
                    after_max = df[col].max()
                    after_count = len(df)
                    
                    # Print results of outlier handling
                    if outlier_strategy_col in ['cap', 'clip']:
                        print(f"  Clipped outliers in '{col}': before range [{before_min:.2f}, {before_max:.2f}], after range [{after_min:.2f}, {after_max:.2f}]")
                    elif outlier_strategy_col in ['trim', 'remove']:
                        print(f"  Removed outliers in '{col}': before {before_count} rows, after {after_count} rows")
                    
                    outlier_cols_processed.append(col)
        
        if outlier_cols_processed:
            print(f"Applied column-specific outlier handling to {len(outlier_cols_processed)} columns: {outlier_cols_processed}")
            print(f"DataFrame shape after column-specific outlier handling: {df.shape}")
        else:
            print("No column-specific outlier handling to apply")

    # Apply global encoding for categorical columns
    encoding_strategy = config.get('encoding_strategy')
    if encoding_strategy and encoding_strategy != 'none':
        print("\n=== Applying Global Encoding Strategy ===")
        print(f"Global encoding strategy: {encoding_strategy}")
        
        # Get column types for all columns
        column_types = get_column_types(df)
        
        # Apply encoding using the encode_categorical function
        df = encode_categorical(df, encoding_strategy, column_types, column_strategies)
        print(f"DataFrame shape after encoding: {df.shape}")
    
    # Scaling features
    scaling_strategy = config.get('scaling_strategy')
    if scaling_strategy and scaling_strategy != 'none':
        print("\n=== Scaling Features ===")
        print(f"Applying global scaling strategy: {scaling_strategy}")
        
        # Apply scaling using the scale_features function
        df = scale_features(df, scaling_strategy, column_strategies)
        print(f"DataFrame shape after scaling: {df.shape}")
    
    # Apply feature selection if requested
    feature_selection = config.get('feature_selection')
    if feature_selection and feature_selection != 'none':
        print("\n=== Applying Feature Selection ===")
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            # Get the threshold or number of features
            n_features = config.get('n_features', min(10, len(numeric_cols)))
            try:
                n_features = int(n_features)
            except:
                n_features = min(10, len(numeric_cols))
            
            print(f"Using feature selection method: {feature_selection} to select {n_features} features")
            
            if feature_selection == 'variance':
                # Select features based on variance threshold
                from sklearn.feature_selection import VarianceThreshold
                
                # Compute variances
                variances = df[numeric_cols].var()
                # Sort by variance
                sorted_vars = variances.sort_values(ascending=False)
                
                # Select top n_features based on variance
                selected_features = sorted_vars.index[:n_features].tolist()
                
                # Keep non-numeric columns
                non_numeric = [col for col in df.columns if col not in numeric_cols]
                df = df[non_numeric + selected_features]
                
                print(f"Selected {len(selected_features)} features based on variance: {selected_features}")
                
            elif feature_selection == 'kbest':
                # This requires a target variable, which we don't have here
                # For now, let's just select top n_features based on variance
                print("K-Best feature selection requires a target variable. Using variance method instead.")
                
                # Compute variances
                variances = df[numeric_cols].var()
                # Sort by variance
                sorted_vars = variances.sort_values(ascending=False)
                
                # Select top n_features based on variance
                selected_features = sorted_vars.index[:n_features].tolist()
                
                # Keep non-numeric columns
                non_numeric = [col for col in df.columns if col not in numeric_cols]
                df = df[non_numeric + selected_features]
                
                print(f"Selected {len(selected_features)} features based on variance: {selected_features}")
                
            print(f"DataFrame shape after feature selection: {df.shape}")
        else:
            print("No numeric columns found for feature selection")
    
    # Apply PCA if requested
    pca_components = config.get('pca_components')
    if pca_components and pca_components != 'none':
        print("\n=== Applying PCA ===")
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            try:
                n_components = int(pca_components)
            except:
                n_components = min(10, len(numeric_cols))
                
            n_components = min(n_components, len(numeric_cols))
            
            if n_components > 0:
                print(f"Applying PCA with {n_components} components")
                from sklearn.decomposition import PCA
                
                # Create a PCA instance
                pca = PCA(n_components=n_components)
                
                # Get non-numeric columns
                non_numeric = [col for col in df.columns if col not in numeric_cols]
                
                # Apply PCA to numeric columns
                pca_result = pca.fit_transform(df[numeric_cols])
                
                # Create a dataframe with PCA components
                pca_df = pd.DataFrame(
                    data=pca_result,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
                
                # Combine non-numeric columns with PCA components
                if non_numeric:
                    df = pd.concat([df[non_numeric].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
                else:
                    df = pca_df
                
                # Print variance explained
                explained_var = pca.explained_variance_ratio_
                total_var = explained_var.sum() * 100
                print(f"Total variance explained: {total_var:.2f}%")
                for i, var in enumerate(explained_var):
                    print(f"  PC{i+1}: {var*100:.2f}% variance explained")
                
                print(f"DataFrame shape after PCA: {df.shape}")
            else:
                print("Invalid number of PCA components, skipping")
        else:
            print("No numeric columns found for PCA")
    
    # If dataset_id is provided, save the preprocessed dataset
    if dataset_id:
        try:
            from django.conf import settings as django_settings
            import os
            
            # Ensure the datasets directory exists
            datasets_dir = os.path.join(django_settings.MEDIA_ROOT, 'datasets')
            os.makedirs(datasets_dir, exist_ok=True)
            
            # Save the preprocessed file
            preprocessed_path = os.path.join(datasets_dir, f'preprocessed_{dataset_id}.csv')
            df.to_csv(preprocessed_path, index=False)
            print(f"\nPreprocessed dataset saved to: {preprocessed_path}")
        except Exception as e:
            print(f"Warning: Could not save preprocessed dataset: {str(e)}")
    
    print(f"\n=========== PREPROCESSING COMPLETED ===========")
    print(f"Final DataFrame shape: {df.shape}")
    return df

def visualize_single_column(df, column, column_type):
    """
    Generate visualization for a single column.
    
    Args:
        df: DataFrame containing the data
        column: Name of the column to visualize
        column_type: Type of the column ('numeric', 'categorical', etc.)
    
    Returns:
        Dictionary containing visualization data
    """
    result = {}
    
    try:
        print(f"Generating visualization for column: {column} of type: {column_type}")
        
        # Don't visualize text columns
        if column_type == 'text':
            return {'error': 'Text columns cannot be visualized', 'message': 'Text columns are not supported for visualization'}
        
        if column_type == 'numeric':
            # 1. Create histogram for numeric columns
            hist_fig = go.Figure()
            
            # Add histogram
            hist_fig.add_trace(go.Histogram(
                x=df[column].dropna(),
                name=column,
                nbinsx=30
            ))
            
            # Add KDE if enough unique values
            if df[column].nunique() > 5:
                # Only calculate KDE if we have enough non-null values
                non_null_data = df[column].dropna()
                if len(non_null_data) > 10:
                    try:
                        kde = gaussian_kde(non_null_data)
                        x_range = np.linspace(non_null_data.min(), non_null_data.max(), 100)
                        hist_fig.add_trace(go.Scatter(
                            x=x_range,
                            y=kde(x_range) * len(df) * (non_null_data.max() - non_null_data.min()) / 30,
                            name='Density',
                            line=dict(color='red', width=2)
                        ))
                    except Exception as e:
                        print(f"Error calculating KDE: {str(e)}")
            
            # Update layout
            hist_fig.update_layout(
                title=f'Distribution of {column}',
                xaxis_title=column,
                yaxis_title='Count',
                showlegend=True,
                bargap=0.1,
                template='plotly_white'
            )
            
            # Add histogram to result
            result['histogram'] = json.dumps(hist_fig.to_dict(), cls=NumpyEncoder)
            
            # 2. Create scatter plot
            scatter_fig = go.Figure()
            
            # Add scatter plot with jitter for better visualization
            scatter_fig.add_trace(go.Scatter(
                x=np.random.normal(0, 0.01, size=len(df)),  # Add jitter to x
                y=df[column],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color=df[column],
                    colorscale='Viridis'
                ),
                name=column
            ))
            
            # Update layout
            scatter_fig.update_layout(
                title=f'Scatter Plot of {column}',
                xaxis_title='Index',
                yaxis_title=column,
                showlegend=False,
                template='plotly_white'
            )
            
            # Add scatter to result
            result['scatter'] = json.dumps(scatter_fig.to_dict(), cls=NumpyEncoder)
            
            # 3. Create box plot
            box_fig = go.Figure()
            
            # Add box plot
            box_fig.add_trace(go.Box(
                y=df[column].dropna(),
                name=column,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=True,  # Show mean
                marker=dict(
                    color='rgb(8,81,156)',
                    outliercolor='rgba(219, 64, 82, 0.6)',
                    line=dict(
                        outliercolor='rgba(219, 64, 82, 0.6)',
                        outlierwidth=2
                    )
                ),
                line_color='rgb(8,81,156)'
            ))
            
            # Update layout
            box_fig.update_layout(
                title=f'Box Plot of {column}',
                yaxis_title=column,
                showlegend=False,
                template='plotly_white'
            )
            
            # Add box plot to result
            result['boxplot'] = json.dumps(box_fig.to_dict(), cls=NumpyEncoder)
            
        elif column_type in ['categorical', 'binary']:
            # Create bar chart for categorical columns
            value_counts = df[column].value_counts()
            
            # Sort values by frequency
            value_counts = value_counts.sort_values(ascending=False)
            
            # Limit to top 30 categories for visualization clarity
            if len(value_counts) > 30:
                print(f"Limiting visualization to top 30 categories out of {len(value_counts)}")
                other_sum = value_counts[30:].sum()
                value_counts = value_counts[:30]
                value_counts['Other'] = other_sum
            
            fig = go.Figure()
            
            # Add trace with better colors
            fig.add_trace(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=column,
                marker=dict(
                    color='rgb(64, 83, 196)',
                    line=dict(color='rgb(8, 48, 107)', width=1.5)
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Distribution of {column}',
                xaxis_title=column,
                yaxis_title='Count',
                xaxis={'categoryorder':'total descending'},
                showlegend=False,
                bargap=0.1,
                template='plotly_white'
            )
            
            # Add bar chart to result
            result['bar'] = json.dumps(fig.to_dict(), cls=NumpyEncoder)
            
        elif column_type == 'datetime':
            # Create time series plot - IMPROVED IMPLEMENTATION
            try:
                # Ensure column is datetime type
                datetime_col = pd.to_datetime(df[column], errors='coerce')
                
                # Create time-based aggregation of counts
                time_counts = datetime_col.dt.date.value_counts().sort_index()
                
                # Convert index to datetime for proper x-axis formatting
                time_counts.index = pd.to_datetime(time_counts.index)
                
                # Create time series plot
                ts_fig = go.Figure()
                
                # Add line plot
                ts_fig.add_trace(go.Scatter(
                    x=time_counts.index,
                    y=time_counts.values,
                    mode='lines+markers',
                    name=column,
                    line=dict(
                        color='rgb(64, 83, 196)',
                        width=2
                    ),
                    marker=dict(
                        size=6,
                        color='rgb(8, 48, 107)'
                    )
                ))
                
                # Update layout with better time formatting
                ts_fig.update_layout(
                    title=f'Time Series of {column}',
                    xaxis_title='Date',
                    yaxis_title='Count',
                    xaxis=dict(
                        type='date',
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        tickformat='%Y-%m-%d'
                    ),
                    showlegend=False,
                    template='plotly_white'
                )
                
                # Add time series to result
                result['timeseries'] = json.dumps(ts_fig.to_dict(), cls=NumpyEncoder)
                
                # Add calendar heatmap for date distribution
                calendar_data = datetime_col.dt.date.value_counts().reset_index()
                calendar_data.columns = ['date', 'count']
                
                if not calendar_data.empty:
                    # Create calendar heatmap
                    calendar_fig = go.Figure()
                    
                    # Convert to datetime
                    calendar_data['date'] = pd.to_datetime(calendar_data['date'])
                    
                    # Extract components
                    calendar_data['year'] = calendar_data['date'].dt.year
                    calendar_data['month'] = calendar_data['date'].dt.month
                    calendar_data['day'] = calendar_data['date'].dt.day
                    calendar_data['weekday'] = calendar_data['date'].dt.weekday
                    
                    # Add heatmap
                    calendar_fig.add_trace(go.Heatmap(
                        z=calendar_data['count'],
                        x=calendar_data['day'],
                        y=calendar_data['month'].astype(str) + '-' + calendar_data['year'].astype(str),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Count')
                    ))
                    
                    # Update layout
                    calendar_fig.update_layout(
                        title=f'Calendar Heatmap of {column}',
                        xaxis_title='Day of Month',
                        yaxis_title='Month-Year',
                        showlegend=False,
                        template='plotly_white'
                    )
                    
                    # Add calendar heatmap to result
                    result['calendar'] = json.dumps(calendar_fig.to_dict(), cls=NumpyEncoder)
                
            except Exception as e:
                print(f"Error in time series visualization: {str(e)}")
                result['message'] = f"Error generating time series visualization: {str(e)}"
        
        # Add column type to result
        result['column_type'] = column_type
        
        return result
        
    except Exception as e:
        print(f"Error in visualize_single_column: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'message': f'Error generating visualization: {str(e)}',
            'column_type': column_type
        }

def visualize_comparison(df, columns, column_types):
    """
    Generate visualization comparing multiple columns.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to compare
        column_types: Dictionary mapping column names to their types
    
    Returns:
        Dictionary containing visualization data
    """
    result = {}
    
    try:
        print(f"Generating comparison visualization for columns: {columns}")
        
        # Check if we have exactly two columns
        if len(columns) != 2:
            return {'message': 'Exactly two columns are required for comparison'}
        
        col1, col2 = columns
        type1 = column_types[col1]
        type2 = column_types[col2]
        
        # Both columns are numeric - create scatter plot
        if type1 == 'numeric' and type2 == 'numeric':
            print(f"Creating scatter plot for numeric columns: {col1} vs {col2}")
            
            # Create scatter plot
            scatter_fig = go.Figure()
            
            # Add scatter plot
            scatter_fig.add_trace(go.Scatter(
                x=df[col1],
                y=df[col2],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color=df[col1],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=col1
                    )
                ),
                name=f'{col1} vs {col2}'
            ))
            
            # Add trendline
            try:
                # Remove missing values
                clean_df = df[[col1, col2]].dropna()
                
                if len(clean_df) > 1:  # Need at least 2 points for regression
                    from scipy import stats
                    
                    # Calculate regression line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_df[col1], clean_df[col2])
                    
                    # Create regression line points
                    x_range = np.linspace(df[col1].min(), df[col1].max(), 100)
                    y_range = slope * x_range + intercept
                    
                    # Add regression line
                    scatter_fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        name=f'Trend (r={r_value:.2f})',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Calculate correlation
                    correlation = clean_df[col1].corr(clean_df[col2])
                    result['correlation'] = correlation
                    
                    # Add correlation annotation
                    scatter_fig.add_annotation(
                        x=0.95,
                        y=0.05,
                        xref="paper",
                        yref="paper",
                        text=f"Correlation: {correlation:.4f}",
                        showarrow=False,
                        font=dict(
                            family="Arial",
                            size=14,
                            color="black"
                        ),
                        align="right",
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4
                    )
            except Exception as e:
                print(f"Error adding trendline: {str(e)}")
            
            # Update layout
            scatter_fig.update_layout(
                title=f'Scatter Plot: {col1} vs {col2}',
                xaxis_title=col1,
                yaxis_title=col2,
                showlegend=True,
                template='plotly_white',
                height=600
            )
            
            # Add histogram subplots for marginal distributions
            scatter_fig.update_layout(
                margin=dict(t=100),
                xaxis=dict(
                    domain=[0, 0.85],
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    domain=[0, 0.85],
                    showgrid=True,
                    zeroline=False
                ),
                showlegend=False
            )
            
            # Create x histogram
            scatter_fig.add_trace(go.Histogram(
                x=df[col1],
                name=col1,
                marker=dict(color='rgba(64, 83, 196, 0.7)'),
                yaxis="y2",
                nbinsx=30
            ))
            
            # Create y histogram
            scatter_fig.add_trace(go.Histogram(
                y=df[col2],
                name=col2,
                marker=dict(color='rgba(64, 83, 196, 0.7)'),
                xaxis="x2",
                nbinsy=30
            ))
            
            # Add marginal subplot definitions
            scatter_fig.update_layout(
                xaxis2=dict(
                    domain=[0.85, 1],
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                yaxis2=dict(
                    domain=[0.85, 1],
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                )
            )
            
            # Add scatter plot to result
            result['scatter'] = json.dumps(scatter_fig.to_dict(), cls=NumpyEncoder)
            
        # One numeric, one categorical - create box plot and violin plot
        elif (type1 == 'numeric' and type2 in ['categorical', 'binary']) or \
             (type2 == 'numeric' and type1 in ['categorical', 'binary']):
            
            # Make sure col1 is categorical and col2 is numeric
            if type1 == 'numeric':
                col1, col2 = col2, col1
                type1, type2 = type2, type1
            
            print(f"Creating box plot for categorical {col1} vs numeric {col2}")
            
            # Filter out categories with too few samples for meaningful visualization
            value_counts = df[col1].value_counts()
            min_samples = 3
            common_categories = value_counts[value_counts >= min_samples].index
            
            # If we have too many categories, limit to the top ones
            max_categories = 10
            if len(common_categories) > max_categories:
                common_categories = value_counts.nlargest(max_categories).index
                print(f"Limited to top {max_categories} categories for visualization clarity")
            
            # Filter dataframe
            filtered_df = df[df[col1].isin(common_categories)]
            
            if len(filtered_df) == 0:
                return {'message': 'Not enough data for box plot visualization'}
                
            # Debug info
            print(f"Number of categories after filtering: {len(common_categories)}")
            print(f"Categories: {common_categories.tolist()}")
            print(f"Filtered data shape: {filtered_df.shape}")
            
            try:
                # Create box plot
                box_fig = go.Figure()
                
                # Add box plot traces for each category
                box_traces = []
                for category in common_categories:
                    category_data = filtered_df[filtered_df[col1] == category][col2].dropna()
                    
                    if len(category_data) >= min_samples:
                        print(f"Adding category {category} with {len(category_data)} data points")
                        # Convert category name to string and truncate if too long
                        category_name = str(category)
                        if len(category_name) > 20:
                            # Truncate long category names to improve display
                            category_name = category_name[:17] + "..."
                            
                        box_trace = go.Box(
                            y=category_data.tolist(),  # Convert to list for serialization
                            name=category_name,
                            boxpoints='outliers',
                            jitter=0.3,
                            pointpos=-1.8,
                            boxmean=True,  # Show mean
                            marker_color='rgb(8,81,156)',
                            line_color='rgb(8,81,156)'
                        )
                        box_traces.append(box_trace)
                
                if not box_traces:
                    print("No valid box traces created")
                    return {'message': 'Not enough data in any category for box plot visualization'}
                
                # Add all traces to the figure
                for trace in box_traces:
                    box_fig.add_trace(trace)
                
                # Update layout with better formatting for x-axis labels
                box_fig.update_layout(
                    title=f'Box Plot: {col2} by {col1}',
                    xaxis_title=col1,
                    yaxis_title=col2,
                    showlegend=False,
                    template='plotly_white',
                    height=600,
                    xaxis=dict(
                        tickangle=-45 if len(common_categories) > 5 else 0,
                        tickfont=dict(size=10 if len(common_categories) > 5 else 12)
                    )
                )
                
                # Add box plot to result
                box_fig_dict = box_fig.to_dict()
                print(f"Box figure data count: {len(box_fig_dict.get('data', []))}")
                result['boxplot'] = json.dumps(box_fig_dict, cls=NumpyEncoder)
                
                # Create violin plot
                violin_fig = go.Figure()
                
                # Add violin plot traces for each category
                violin_traces = []
                for category in common_categories:
                    category_data = filtered_df[filtered_df[col1] == category][col2].dropna()
                    
                    if len(category_data) >= min_samples:
                        # Use the same category name formatting as box plot
                        category_name = str(category)
                        if len(category_name) > 20:
                            category_name = category_name[:17] + "..."
                            
                        violin_trace = go.Violin(
                            y=category_data.tolist(),  # Convert to list for serialization
                            name=category_name,
                            box_visible=True,
                            meanline_visible=True,
                            points='outliers'
                        )
                        violin_traces.append(violin_trace)
                
                # Add all traces to the figure
                for trace in violin_traces:
                    violin_fig.add_trace(trace)
                
                # Update layout with matching formatting to box plot
                violin_fig.update_layout(
                    title=f'Violin Plot: {col2} by {col1}',
                    xaxis_title=col1,
                    yaxis_title=col2,
                    showlegend=False,
                    template='plotly_white',
                    height=600,
                    xaxis=dict(
                        tickangle=-45 if len(common_categories) > 5 else 0,
                        tickfont=dict(size=10 if len(common_categories) > 5 else 12)
                    )
                )
                
                # Add violin plot to result
                violin_fig_dict = violin_fig.to_dict()
                print(f"Violin figure data count: {len(violin_fig_dict.get('data', []))}")
                result['violin'] = json.dumps(violin_fig_dict, cls=NumpyEncoder)
                
                # Calculate and display statistics
                category_stats = {}
            except Exception as e:
                print(f"Error creating box/violin plots: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'message': f'Error generating visualizations: {str(e)}'}
        
        # Categorical and datetime - create a timeline by category
        elif (type1 == 'datetime' and type2 in ['categorical', 'binary']) or \
             (type2 == 'datetime' and type1 in ['categorical', 'binary']):
            
            # Make sure col1 is categorical and col2 is datetime
            if type1 == 'datetime':
                col1, col2 = col2, col1
                type1, type2 = type2, type1
            
            print(f"Creating timeline plot for categorical {col1} vs datetime {col2}")
            
            try:
                # Convert datetime column to proper datetime format
                df[col2] = pd.to_datetime(df[col2], errors='coerce')
                
                # Filter out rows with invalid datetime values
                filtered_df = df.dropna(subset=[col2])
                
                if len(filtered_df) == 0:
                    return {'message': 'No valid datetime values for visualization'}
                
                # Filter categories with too few samples
                value_counts = filtered_df[col1].value_counts()
                min_samples = 3
                common_categories = value_counts[value_counts >= min_samples].index
                
                # Limit to top categories if too many
                max_categories = 10
                if len(common_categories) > max_categories:
                    common_categories = value_counts.nlargest(max_categories).index
                    print(f"Limited to top {max_categories} categories for visualization clarity")
                
                # Filter dataframe to include only common categories
                filtered_df = filtered_df[filtered_df[col1].isin(common_categories)]
                
                if len(filtered_df) == 0:
                    return {'message': 'Not enough data for categorical-datetime visualization'}
                
                # Create timeline plot
                timeline_fig = go.Figure()
                
                # Add a trace for each category showing date distribution
                for category in common_categories:
                    category_data = filtered_df[filtered_df[col1] == category][col2].dropna()
                    
                    if len(category_data) >= min_samples:
                        # Convert category name to string and truncate if too long
                        category_name = str(category)
                        if len(category_name) > 20:
                            category_name = category_name[:17] + "..."
                        
                        # Add a trace for this category
                        timeline_fig.add_trace(go.Box(
                            x=category_data.tolist(),  # Dates on x-axis
                            name=category_name,
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=0,
                            marker=dict(
                                size=6,
                                opacity=0.7
                            )
                        ))
                
                # Update layout
                timeline_fig.update_layout(
                    title=f'Date Distribution by {col1}',
                    xaxis_title=f'{col2} (Date)',
                    yaxis_title=col1,
                    showlegend=False,
                    template='plotly_white',
                    height=600,
                    xaxis=dict(
                        type='date',
                        tickformat='%Y-%m-%d'
                    )
                )
                
                # Add box plot to result
                result['boxplot'] = json.dumps(timeline_fig.to_dict(), cls=NumpyEncoder)
                
                # Create a heatmap showing frequency by category and time period
                # Group by category and month-year to create a heatmap
                filtered_df['month_year'] = filtered_df[col2].dt.strftime('%Y-%m')
                pivot_data = filtered_df.pivot_table(
                    index=col1, 
                    columns='month_year', 
                    aggfunc='size', 
                    fill_value=0
                )
                
                # Create heatmap if we have enough data
                if not pivot_data.empty and pivot_data.shape[1] > 1:
                    heatmap_fig = go.Figure(data=go.Heatmap(
                        z=pivot_data.values,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        colorscale='Viridis',
                        colorbar=dict(title='Count')
                    ))
                    
                    # Update layout
                    heatmap_fig.update_layout(
                        title=f'Frequency Heatmap: {col1} by Month-Year',
                        xaxis_title='Month-Year',
                        yaxis_title=col1,
                        template='plotly_white',
                        height=600
                    )
                    
                    # Add heatmap to result
                    result['heatmap'] = json.dumps(heatmap_fig.to_dict(), cls=NumpyEncoder)
                
            except Exception as e:
                print(f"Error creating datetime-categorical visualizations: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'message': f'Error generating visualizations: {str(e)}'}
            
        # Both categorical - create heatmap of co-occurrence
        elif type1 in ['categorical', 'binary'] and type2 in ['categorical', 'binary']:
            print(f"Creating co-occurrence map for categorical columns: {col1} vs {col2}")
            
            # Create contingency table (crosstab)
            contingency = pd.crosstab(df[col1], df[col2], normalize='all') * 100
            
            # Create heatmap
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=contingency.values,
                x=contingency.columns,
                y=contingency.index,
                colorscale='Viridis',
                colorbar=dict(title='% of Total'),
                hovertemplate='%{y} & %{x}: %{z:.2f}%<extra></extra>'
            ))
            
            # Update layout
            heatmap_fig.update_layout(
                title=f'Co-occurrence: {col1} vs {col2}',
                xaxis_title=col2,
                yaxis_title=col1,
                template='plotly_white',
                height=600,
                xaxis=dict(tickangle=-45)
            )
            
            # Add heatmap to result
            result['heatmap'] = json.dumps(heatmap_fig.to_dict(), cls=NumpyEncoder)
            
            # Create stacked bar chart for visualization of proportions
            bar_fig = go.Figure()
            
            # Calculate proportions for each category in col1
            proportions = pd.crosstab(df[col1], df[col2], normalize='index') * 100
            
            # Create stacked bar chart
            categories = proportions.columns
            for i, category in enumerate(categories):
                bar_fig.add_trace(go.Bar(
                    y=proportions.index,
                    x=proportions[category],
                    name=str(category),
                    orientation='h',
                    marker=dict(
                        line=dict(width=0)
                    )
                ))
            
            # Update layout
            bar_fig.update_layout(
                title=f'Proportion of {col2} within each {col1} Category',
                xaxis_title='Percentage (%)',
                yaxis_title=col1,
                barmode='stack',
                showlegend=True,
                legend_title=col2,
                template='plotly_white',
                height=600
            )
            
            # Add bar to result
            result['stacked_bar'] = json.dumps(bar_fig.to_dict(), cls=NumpyEncoder)
            
        else:
            return {'message': f'Cannot compare columns of types {type1} and {type2}'}
        
        return result
        
    except Exception as e:
        print(f"Error in visualize_comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'message': f'Error generating comparison: {str(e)}'
        }

def visualize_advanced(df, column_types, plot_type, **kwargs):
    """
    Generate advanced visualizations.
    
    Args:
        df: DataFrame containing the data
        column_types: Dictionary mapping column names to their types
        plot_type: Type of visualization to generate
        **kwargs: Additional parameters (e.g., column for anomaly detection)
    
    Returns:
        Dictionary containing visualization data
    """
    result = {}
    
    try:
        print(f"Generating advanced visualization: {plot_type}")
        
        # Get numeric columns for analysis
        numeric_cols = [col for col, type_ in column_types.items() if type_ == 'numeric']
        print(f"Found {len(numeric_cols)} numeric columns for advanced visualization")
        
        if plot_type == 'correlation':
            # Require at least 2 numeric columns for correlation analysis
            if len(numeric_cols) < 2:
                print("Not enough numeric columns for correlation analysis")
                return {'message': 'At least two numeric columns are required for correlation analysis'}
            
            # Limit to top 20 columns for visualization clarity
            if len(numeric_cols) > 20:
                print(f"Limiting correlation matrix to 20 columns out of {len(numeric_cols)}")
                # Use columns with highest variance
                variances = df[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = variances.index[:20].tolist()
            
            print(f"Creating correlation matrix for columns: {numeric_cols}")
            try:
                # Sample the data if it's very large to prevent performance issues
                max_rows = 10000  # Maximum rows for correlation analysis
                if len(df) > max_rows:
                    print(f"Sampling {max_rows} rows from {len(df)} total rows for correlation analysis")
                    sampled_df = df.sample(max_rows, random_state=42)
                else:
                    sampled_df = df
                
                # Handle any missing values before computing correlation
                numeric_data = sampled_df[numeric_cols].copy()
                
                # Check for non-finite values
                non_finite_counts = np.sum(~np.isfinite(numeric_data.values))
                if non_finite_counts > 0:
                    print(f"Warning: Found {non_finite_counts} non-finite values in data")
                    # Replace non-finite values with NaN
                    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
                
                # Print summary of missing values
                na_count = numeric_data.isna().sum().sum()
                if na_count > 0:
                    print(f"Warning: Found {na_count} NaN values in data")
                    # Fill NaN values with column means for correlation calculation
                    numeric_data = numeric_data.fillna(numeric_data.mean())
                
                # Create correlation matrix
                print("Computing correlation matrix")
                corr_matrix = numeric_data.corr()
                print(f"Correlation matrix shape: {corr_matrix.shape}")
                
                # Debug output of the matrix summary
                print("Sample of correlation matrix:")
                print(corr_matrix.iloc[:3, :3])
                
                # Create mask for upper triangle (to avoid duplicating information)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Convert masked values to NaN for plotting
                masked_corr = corr_matrix.mask(mask)
                
                # Create heatmap
                print("Creating heatmap figure")
                heatmap_fig = go.Figure()
                
                # Add heatmap trace with explicit handling of data types
                z_values = masked_corr.values.tolist()  # Convert to list for safe serialization
                
                # Ensure all values are serializable
                for i in range(len(z_values)):
                    for j in range(len(z_values[i])):
                        if z_values[i][j] is not None and not np.isfinite(z_values[i][j]):
                            z_values[i][j] = None
                
                # Create text matrix for hover information
                text_matrix = np.round(masked_corr.values, 2).tolist()
                for i in range(len(text_matrix)):
                    for j in range(len(text_matrix[i])):
                        if text_matrix[i][j] is not None and not np.isfinite(text_matrix[i][j]):
                            text_matrix[i][j] = None
                
                # Add heatmap trace
                heatmap_fig.add_trace(go.Heatmap(
                    z=z_values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale='RdBu_r',  # Red-Blue diverging colorscale
                    zmid=0,  # Center the colorscale at 0
                    colorbar=dict(
                        title='Correlation',
                        titleside='right'
                    ),
                    text=text_matrix,
                    hovertemplate='%{y} & %{x}<br>Correlation: %{text:.4f}<extra></extra>'
                ))
                
                # Update layout
                heatmap_fig.update_layout(
                    title='Correlation Matrix',
                    height=800,
                    width=800,
                    template='plotly_white',
                    margin=dict(b=150, l=150),  # Increase bottom and left margins
                    xaxis=dict(
                        tickangle=-45,
                        side='bottom',
                        tickfont=dict(size=10),
                        showticklabels=True,
                        automargin=True  # Allow margin to adjust based on label length
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10),
                        showticklabels=True,
                        automargin=True  # Allow margin to adjust based on label length
                    )
                )
                
                # Add heatmap to result
                print("Serializing correlation matrix to JSON")
                try:
                    heatmap_dict = heatmap_fig.to_dict()
                    result['heatmap'] = json.dumps(heatmap_dict, cls=NumpyEncoder)
                    print(f"JSON serialization complete, length: {len(result['heatmap'])}")
                except Exception as json_err:
                    print(f"Error serializing heatmap to JSON: {str(json_err)}")
                    
                    # Create simplified version with minimal data
                    print("Creating simplified heatmap version")
                    simple_fig = go.Figure()
                    
                    # Create a simpler version of the heatmap
                    # Use only the lower triangle values and simplify data
                    tril_indices = np.tril_indices(len(corr_matrix.columns), k=-1)
                    col_indices, row_indices = tril_indices
                    
                    values = []
                    text = []
                    x = []
                    y = []
                    
                    for i, j in zip(row_indices, col_indices):
                        if i < len(corr_matrix.index) and j < len(corr_matrix.columns):
                            val = corr_matrix.iloc[i, j]
                            if np.isfinite(val):
                                values.append(val)
                                text.append(f"{val:.4f}")
                                x.append(corr_matrix.columns[j])
                                y.append(corr_matrix.index[i])
                    
                    simple_fig.add_trace(go.Heatmap(
                        z=values,
                        x=x,
                        y=y,
                        colorscale='RdBu_r',
                        zmid=0,
                        colorbar=dict(title='Correlation'),
                        text=text
                    ))
                    
                    simple_fig.update_layout(
                        title='Correlation Matrix (Simplified)',
                        height=700,
                        width=700,
                        template='plotly_white',
                        margin=dict(b=150, l=150),  # Increase bottom and left margins
                        xaxis=dict(
                            tickangle=-45,
                            side='bottom',
                            tickfont=dict(size=10),
                            showticklabels=True,
                            automargin=True
                        ),
                        yaxis=dict(
                            tickfont=dict(size=10),
                            showticklabels=True,
                            automargin=True
                        )
                    )
                    
                    # Try to serialize the simplified version
                    try:
                        result['heatmap'] = json.dumps(simple_fig.to_dict(), cls=NumpyEncoder)
                        print(f"Simplified JSON serialization complete, length: {len(result['heatmap'])}")
                    except Exception as simple_err:
                        print(f"Error serializing simplified heatmap: {str(simple_err)}")
                        
                        # Last resort - create a fallback message figure
                        fallback_fig = go.Figure()
                        fallback_fig.add_annotation(
                            text="Unable to create correlation matrix - data too complex",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False,
                            font=dict(size=18)
                        )
                        result['heatmap'] = json.dumps(fallback_fig.to_dict())
                        print("Created fallback annotation figure")
                
                # Find highly correlated columns
                print("Finding highly correlated pairs")
                try:
                    # Find pairs with high correlation (absolute value > 0.7)
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):  # Only upper triangle to avoid duplicates
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            corr_value = corr_matrix.iloc[i, j]
                            
                            if np.isfinite(corr_value) and abs(corr_value) > 0.7:
                                high_corr_pairs.append((col1, col2, corr_value))
                    
                    # Sort by absolute correlation value
                    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    # Extract top pairs
                    top_pairs = high_corr_pairs[:10]  # Limit to top 10 pairs
                    
                    # Add high correlation pairs to result
                    result['high_corr_pairs'] = [{'col1': col1, 'col2': col2, 'corr': float(corr)} for col1, col2, corr in top_pairs]
                    print(f"Found {len(high_corr_pairs)} highly correlated pairs, using top {len(top_pairs)}")
                except Exception as corr_pair_err:
                    print(f"Error finding correlation pairs: {str(corr_pair_err)}")
                    # If there's an error, still return an empty list so the frontend doesn't break
                    result['high_corr_pairs'] = []
                
                print(f"Final result keys for correlation: {result.keys()}")
                return result
            except Exception as e:
                print(f"Error creating correlation matrix: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'message': f'Error generating correlation matrix: {str(e)}'}
            
        elif plot_type == 'pairplot':
            # Require at least 2 numeric columns for pair plot
            if len(numeric_cols) < 2:
                print("Not enough numeric columns for pair plot")
                return {'message': 'At least two numeric columns are required for pair plot'}
            
            # Limit to top 5 columns for readability (pair plots get very crowded)
            if len(numeric_cols) > 5:
                print(f"Limiting pair plot to 5 columns out of {len(numeric_cols)}")
                # Use columns with highest variance
                variances = df[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = variances.index[:5].tolist()
            
            print(f"Creating pair plot with columns: {numeric_cols}")
            
            try:
                # Sample the data if it's too large to prevent performance issues
                max_rows = 1000  # Limit for pair plot performance
                if len(df) > max_rows:
                    print(f"Sampling {max_rows} rows from {len(df)} total rows for pair plot")
                    # Use stratified sampling if categorical column exists
                    categorical_cols = [col for col, type_ in column_types.items() if type_ in ['categorical', 'binary']]
                    if categorical_cols:
                        # Try to use stratified sampling on first categorical column
                        try:
                            sampled_df = df.groupby(categorical_cols[0], group_keys=False).apply(
                                lambda x: x.sample(min(len(x), max(1, int(max_rows * len(x) / len(df)))))
                            )
                            # If sampled data is still too large, take a random sample
                            if len(sampled_df) > max_rows:
                                sampled_df = sampled_df.sample(max_rows, random_state=42)
                        except Exception as e:
                            print(f"Error in stratified sampling: {str(e)}, using random sampling instead")
                            sampled_df = df.sample(max_rows, random_state=42)
                    else:
                        # Random sampling if no categorical columns
                        sampled_df = df.sample(max_rows, random_state=42)
                else:
                    sampled_df = df
                
                # Create pair plot using subplots
                fig = make_subplots(
                    rows=len(numeric_cols),
                    cols=len(numeric_cols),
                    shared_xaxes=True,
                    shared_yaxes=True,
                    column_titles=numeric_cols,
                    row_titles=numeric_cols,
                    horizontal_spacing=0.02,
                    vertical_spacing=0.02
                )
                
                # Calculate bounds for axis ranges to limit the effect of outliers
                ranges = {}
                for col in numeric_cols:
                    data = sampled_df[col].dropna()
                    if len(data) > 0:
                        q1, q3 = data.quantile([0.01, 0.99])
                        iqr = q3 - q1
                        lower = max(q1 - 1.5 * iqr, data.min())
                        upper = min(q3 + 1.5 * iqr, data.max())
                        ranges[col] = (lower, upper)
                        print(f"Ranges for {col}: {ranges[col]}")
                
                # Fill in the subplots with appropriate charts
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        row_index = i + 1
                        col_index = j + 1
                        
                        # Handle potential missing data
                        valid_data1 = sampled_df[col1].dropna()
                        valid_data2 = sampled_df[col2].dropna()
                        
                        if i == j:  # Diagonal: Show histogram
                            if len(valid_data1) > 0:
                                try:
                                    fig.add_trace(
                                        go.Histogram(
                                            x=valid_data1.tolist(),  # Use valid data only
                                            nbinsx=min(30, len(valid_data1.unique())),  # Adjust bins to data
                                            marker=dict(color='rgba(64, 83, 196, 0.7)'),
                                            showlegend=False
                                        ),
                                        row=row_index, col=col_index
                                    )
                                except Exception as hist_err:
                                    print(f"Error adding histogram for {col1}: {str(hist_err)}")
                        else:  # Off-diagonal: Show scatter plot
                            # Get valid data for both columns
                            common_data = sampled_df[[col1, col2]].dropna()
                            
                            if len(common_data) > 0:
                                try:
                                    # Use a smaller subset for scatter plots if needed
                                    if len(common_data) > 500:
                                        plot_data = common_data.sample(500, random_state=42)
                                    else:
                                        plot_data = common_data
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=plot_data[col2].tolist(),
                                            y=plot_data[col1].tolist(),
                                            mode='markers',
                                            marker=dict(
                                                size=4,
                                                opacity=0.5,
                                                color='rgba(64, 83, 196, 0.7)'
                                            ),
                                            showlegend=False
                                        ),
                                        row=row_index, col=col_index
                                    )
                                    
                                    # Set axis ranges if defined to exclude outliers
                                    if col2 in ranges:
                                        fig.update_xaxes(range=ranges[col2], row=row_index, col=col_index)
                                    if col1 in ranges:
                                        fig.update_yaxes(range=ranges[col1], row=row_index, col=col_index)
                                except Exception as scatter_err:
                                    print(f"Error adding scatter for {col1} vs {col2}: {str(scatter_err)}")
                
                # Update layout
                fig.update_layout(
                    title='Pair Plot of Numeric Columns',
                    height=900,
                    width=900,
                    template='plotly_white',
                    showlegend=False
                )
                
                # Add pair plot to result
                print("Serializing pair plot to JSON")
                try:
                    pair_plot_dict = fig.to_dict()
                    result['pairplot'] = json.dumps(pair_plot_dict, cls=NumpyEncoder)
                    json_size = len(result['pairplot'])
                    print(f"JSON serialization complete, length: {json_size}")
                    
                    # Check if the JSON is too large, if so, simplify
                    max_size = 5 * 1024 * 1024  # 5MB threshold
                    if json_size > max_size:
                        print(f"Pair plot JSON is too large ({json_size} bytes), creating simplified version")
                        
                        # Create a simplified version with fewer plots
                        simple_fig = make_subplots(
                            rows=min(3, len(numeric_cols)),
                            cols=min(3, len(numeric_cols)),
                            shared_xaxes=True,
                            shared_yaxes=True
                        )
                        
                        # Add just a few plots
                        for i in range(min(3, len(numeric_cols))):
                            for j in range(min(3, len(numeric_cols))):
                                if i < len(numeric_cols) and j < len(numeric_cols):
                                    col1 = numeric_cols[i]
                                    col2 = numeric_cols[j]
                                    
                                    if i == j:
                                        simple_fig.add_trace(
                                            go.Histogram(
                                                x=sampled_df[col1].dropna().tolist()[:500],
                                                nbinsx=20,
                                                marker=dict(color='rgba(64, 83, 196, 0.7)'),
                                                name=col1
                                            ),
                                            row=i+1, col=j+1
                                        )
                                    else:
                                        common_data = sampled_df[[col1, col2]].dropna()
                                        if len(common_data) > 200:
                                            common_data = common_data.sample(200, random_state=42)
                                            
                                        simple_fig.add_trace(
                                            go.Scatter(
                                                x=common_data[col2].tolist(),
                                                y=common_data[col1].tolist(),
                                                mode='markers',
                                                marker=dict(size=4),
                                                name=f"{col1} vs {col2}"
                                            ),
                                            row=i+1, col=j+1
                                        )
                    
                    simple_fig.update_layout(
                            title='Simplified Pair Plot (showing subset of data)',
                        height=700,
                            width=700
                        )
                        
                    # Replace the result with simplified version
                    result['pairplot'] = json.dumps(simple_fig.to_dict(), cls=NumpyEncoder)
                    print(f"Simplified JSON size: {len(result['pairplot'])} bytes")
                    
                except Exception as json_err:
                    print(f"Error serializing pair plot to JSON: {str(json_err)}")
                    # Create a minimal fallback plot
                    fallback_fig = go.Figure()
                    fallback_fig.add_annotation(
                        text="Unable to create pair plot - data too complex or large",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=20)
                    )
                    result['pairplot'] = json.dumps(fallback_fig.to_dict())
                    print("Created fallback plot")
                    
            except Exception as e:
                print(f"Error creating pair plot: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'message': f'Error generating pair plot: {str(e)}'}
            
        elif plot_type == 'distribution':
            # Require at least 1 numeric column for distribution analysis
            if not numeric_cols:
                return {'message': 'No numeric columns available for distribution analysis'}
            
            # Limit to top 10 columns for visualization clarity
            if len(numeric_cols) > 10:
                print(f"Limiting distribution analysis to 10 columns out of {len(numeric_cols)}")
                # Use columns with highest variance
                variances = df[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = variances.index[:10].tolist()
            
            # Create subplots for distribution analysis
            fig = make_subplots(
                rows=len(numeric_cols),
                cols=1,
                subplot_titles=[f'Distribution of {col}' for col in numeric_cols],
                vertical_spacing=0.05
            )
            
            # Add distribution plots for each column
            for i, col in enumerate(numeric_cols):
                # Get data without NaN values
                data = df[col].dropna()
                
                if len(data) > 0:
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=data,
                            name=col,
                            nbinsx=30,
                            marker=dict(color='rgba(64, 83, 196, 0.7)'),
                            opacity=0.7,
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add density curve if enough data points
                    if len(data) > 30 and data.nunique() > 5:
                        try:
                            kde = gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 100)
                            density = kde(x_range) * len(data) * (data.max() - data.min()) / 30  # Scale to match histogram
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=density,
                                    name='KDE',
                                    line=dict(color='red', width=2),
                                    showlegend=False
                                ),
                                row=i+1, col=1
                            )
                            
                            # Add Q-Q plot for normality check
                            from scipy import stats
                            
                            # Standardize the data
                            z_scores = (data - data.mean()) / data.std()
                            
                            # Calculate Q-Q plot data
                            qqplot_data = stats.probplot(z_scores, dist="norm", plot=None)
                            x = np.array([point[0] for point in qqplot_data[0]])
                            y = np.array([point[1] for point in qqplot_data[0]])
                            
                            # Add line for perfect normality
                            x_line = np.linspace(min(x), max(x), 100)
                            y_line = x_line
                            
                            # Create Q-Q plot in column 2 (adding dynamically)
                            if i == 0:  # Do this only for the first column to avoid duplicate updates
                                fig.add_trace(
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        mode='markers',
                                        name='Data Points',
                                        marker=dict(color='blue', size=6),
                                        showlegend=False
                                    ),
                                    row=1, col=1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_line,
                                        y=y_line,
                                        mode='lines',
                                        name='Normal Line',
                                        line=dict(color='red', width=2),
                                        showlegend=False
                                    ),
                                    row=1, col=1
                                )
                        except Exception as e:
                            print(f"Error adding density curve: {str(e)}")
            
            # Update layout
            fig.update_layout(
                title='Distribution Analysis of Numeric Columns',
                height=250 * len(numeric_cols),
                width=900,
                template='plotly_white',
                showlegend=False
            )
            
            # Add distribution plot to result
            result['distribution'] = json.dumps(fig.to_dict(), cls=NumpyEncoder)
            print(f"Distribution visualization complete, result keys: {result.keys()}")
            
            # Add descriptive statistics
            stats_df = df[numeric_cols].describe().T
            stats_df['missing'] = df[numeric_cols].isna().sum()
            stats_df['missing_pct'] = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)
            
            # Calculate skewness and kurtosis
            stats_df['skewness'] = df[numeric_cols].skew()
            stats_df['kurtosis'] = df[numeric_cols].kurtosis()
            
            # Convert to JSON-friendly format
            stats_json = {}
            for col in stats_df.index:
                stats_json[col] = {stat: float(stats_df.loc[col, stat]) for stat in stats_df.columns}
            
            result['statistics'] = stats_json
            
        elif plot_type == 'anomaly':
            # Require at least 1 numeric column for anomaly detection
            if not numeric_cols:
                print("No numeric columns found for anomaly detection")
                return {'message': 'No numeric columns available for anomaly detection'}
            
            print(f"Starting anomaly detection")
            
            try:
                # Check if a specific column was requested
                selected_column = kwargs.get('column', None)
                if selected_column:
                    # Check if the selected column exists and is numeric
                    if selected_column in numeric_cols:
                        numeric_cols = [selected_column]  # Only use the selected column
                        print(f"Filtering anomaly detection to requested column: {selected_column}")
                    else:
                        return {'message': f'Column "{selected_column}" is not a numeric column or does not exist'}
                else:
                    # Limit to top 8 columns for visualization clarity
                    if len(numeric_cols) > 8:
                        print(f"Limiting anomaly detection to 8 columns out of {len(numeric_cols)}")
                        # Use columns with highest variance
                        variances = df[numeric_cols].var().sort_values(ascending=False)
                        numeric_cols = variances.index[:8].tolist()
                
                print(f"Creating subplots for anomaly detection with {len(numeric_cols)} columns")
                # Create subplots for anomaly detection
                fig = make_subplots(
                    rows=len(numeric_cols),
                    cols=1,
                    subplot_titles=[f'Anomaly Detection: {col}' for col in numeric_cols],
                    vertical_spacing=0.05  # More space between subplots
                )
                
                # Track overall outliers
                all_outliers = {}
                
                # Add plots for each column
                for i, col in enumerate(numeric_cols):
                    # Get data without NaN values
                    data = df[col].dropna()
                    
                    if len(data) > 0:
                        try:
                            # Calculate outlier bounds using IQR method
                            q1, q3 = data.quantile([0.25, 0.75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            median = data.median()
                            mean = data.mean()
                            std_dev = data.std()
                            
                            # Identify outliers
                            outliers = data[(data < lower_bound) | (data > upper_bound)]
                            outlier_indices = outliers.index.tolist()
                            
                            # Track outliers for this column
                            all_outliers[col] = {
                                'count': len(outliers),
                                'percentage': len(outliers) / len(data) * 100,
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound,
                                'median': median,
                                'mean': mean,
                                'std_dev': std_dev
                            }
                            
                            # Use original data indices instead of sorting
                            x_values = data.index.tolist()
                            
                            # Create hover text for all points
                            hover_template = "Value: %{y:.2f}<br>Index: %{x}"
                            
                            # Add main scatter trace for normal data points
                            normal_data = data[(data >= lower_bound) & (data <= upper_bound)]
                            if len(normal_data) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=normal_data.index.tolist(),
                                        y=normal_data.values,
                                        mode='markers',
                                        marker=dict(
                                            color='rgba(8, 81, 156, 0.6)',
                                            size=8,
                                            line=dict(width=1, color='rgb(8, 81, 156)')
                                        ),
                                        name='Normal Points',
                                        hovertemplate=hover_template,
                                        showlegend=i==0  # Only show legend for first plot
                                    ),
                                    row=i+1, col=1
                                )
                            
                            # Add outlier points with different color
                            if len(outliers) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=outliers.index.tolist(),
                                        y=outliers.values,
                                        mode='markers',
                                        marker=dict(
                                            color='rgba(219, 64, 82, 0.8)',
                                            size=10,
                                            symbol='circle',
                                            line=dict(width=2, color='rgb(219, 64, 82)')
                                        ),
                                        name='Anomalies',
                                        hovertemplate="<b>ANOMALY</b><br>Value: %{y:.2f}<br>Index: %{x}",
                                        showlegend=i==0  # Only show legend for first plot
                                    ),
                                    row=i+1, col=1
                                )
                            
                            # Add median line
                            fig.add_trace(
                                go.Scatter(
                                    x=[min(x_values), max(x_values)],
                                    y=[median, median],
                                    mode='lines',
                                    line=dict(color='blue', width=2, dash='dash'),
                                    name='Median',
                                    showlegend=i==0  # Only show legend for first plot
                                ),
                                row=i+1, col=1
                            )
                            
                            # Add threshold lines
                            fig.add_shape(
                                type='line',
                                x0=min(x_values), x1=max(x_values),
                                y0=upper_bound, y1=upper_bound,
                                line=dict(color='red', width=2, dash='dot'),
                                row=i+1, col=1
                            )
                            
                            fig.add_shape(
                                type='line',
                                x0=min(x_values), x1=max(x_values),
                                y0=lower_bound, y1=lower_bound,
                                line=dict(color='red', width=2, dash='dot'),
                                row=i+1, col=1
                            )
                            
                            # Add shaded area between thresholds
                            fig.add_shape(
                                type='rect',
                                x0=min(x_values), x1=max(x_values),
                                y0=lower_bound, y1=upper_bound,
                                fillcolor='rgba(0, 0, 255, 0.1)',
                                line=dict(width=0),
                                row=i+1, col=1
                            )
                            
                            # Add annotations for threshold values
                            fig.add_annotation(
                                x=0.95,
                                y=upper_bound,
                                xref=f"x{i+1}",
                                yref=f"y{i+1}",
                                text=f"Upper: {upper_bound:.2f}",
                                showarrow=False,
                                font=dict(size=10, color='darkred'),
                                bgcolor='rgba(255, 255, 255, 0.8)',
                                bordercolor='red',
                                borderwidth=1,
                                borderpad=3,
                                align="right"
                            )
                            
                            fig.add_annotation(
                                x=0.95,
                                y=lower_bound,
                                xref=f"x{i+1}",
                                yref=f"y{i+1}",
                                text=f"Lower: {lower_bound:.2f}",
                                showarrow=False,
                                font=dict(size=10, color='darkred'),
                                bgcolor='rgba(255, 255, 255, 0.8)',
                                bordercolor='red',
                                borderwidth=1,
                                borderpad=3,
                                align="right"
                            )
                            
                            # Add summary stats annotation
                            fig.add_annotation(
                                x=0.02,
                                y=0.95,
                                xref=f"x{i+1} domain",
                                yref=f"y{i+1} domain",
                                text=f"Anomalies: {len(outliers)} ({(len(outliers)/len(data)*100):.1f}%)<br>Mean: {mean:.2f}, Median: {median:.2f}",
                                showarrow=False,
                                font=dict(size=10),
                                bgcolor='rgba(255, 255, 255, 0.8)',
                                bordercolor='black',
                                borderwidth=1,
                                borderpad=3,
                                align="left"
                            )
                            
                            # Update axes
                            fig.update_xaxes(
                                title_text="Index (original data points)",
                                row=i+1, col=1
                            )
                            
                            fig.update_yaxes(
                                title_text=col,
                                row=i+1, col=1,
                                side='left'
                            )
                            
                            # Remove secondary y-axis configuration since histogram is removed
                        
                        except Exception as e:
                            print(f"Error processing column {col} for anomaly detection: {str(e)}")
                            # Continue with other columns
                            import traceback
                            traceback.print_exc()
                
                # Calculate appropriate subplot heights
                each_plot_height = 350  # Height per subplot in pixels
                total_height = max(800, min(2000, each_plot_height * len(numeric_cols)))
                
                # Update layout
                fig.update_layout(
                    title={
                        'text': 'Anomaly Detection Analysis',
                        'font': {'size': 24, 'color': 'black'},
                        'x': 0.5,
                        'xanchor': 'center',
                        'y': 0.98
                    },
                    height=total_height,
                    width=900,
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(t=80, l=50, r=50, b=50)
                )
                
                # Add anomaly plot to result
                result['anomaly'] = json.dumps(fig.to_dict(), cls=NumpyEncoder)
                
                # Add outlier statistics with proper conversion to built-in types
                outlier_stats = {}
                for col, stats in all_outliers.items():
                    outlier_stats[col] = {
                        'count': int(stats['count']),
                        'percentage': float(stats['percentage']),
                        'lower_bound': float(stats['lower_bound']),
                        'upper_bound': float(stats['upper_bound']),
                        'median': float(stats['median']),
                        'mean': float(stats['mean']),
                        'std_dev': float(stats['std_dev'])
                    }
                result['outlier_stats'] = outlier_stats
            
            except Exception as e:
                print(f"Error in anomaly detection: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    'message': f'Error in anomaly detection: {str(e)}'
                }
        else:
            return {'message': f'Unsupported plot type: {plot_type}'}
        
        return result
        
    except Exception as e:
        print(f"Error in visualize_advanced: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'message': f'Error generating advanced visualization: {str(e)}'
        }

def export_plotly_html(fig, output_path):
    """
    Export a Plotly figure to an HTML file.
    
    Args:
        fig: Plotly figure object
        output_path: Path where the HTML file should be saved
    
    Returns:
        str: Path to the saved HTML file
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the figure to HTML
        fig.write_html(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error exporting plot to HTML: {str(e)}")
        raise 

def get_column_statistics(df):
    """Get statistics for all columns in the dataset."""
    stats = {}
    column_types = get_column_types(df)
    for column in df.columns:
        col_type = column_types[column]
        if col_type == 'numeric':
            stats[column] = {
                'missing_values': int(df[column].isna().sum()),
                'unique_values': int(df[column].nunique()),
                'mean': float(df[column].mean()) if not pd.isna(df[column].mean()) else 0,
                'median': float(df[column].median()) if not pd.isna(df[column].median()) else 0,
                'std': float(df[column].std()) if not pd.isna(df[column].std()) else 0,
                'min': float(df[column].min()) if not pd.isna(df[column].min()) else 0,
                'max': float(df[column].max()) if not pd.isna(df[column].max()) else 0
            }
        elif col_type in ['categorical', 'binary']:
            value_counts = df[column].value_counts()
            stats[column] = {
                'missing_values': int(df[column].isna().sum()),
                'unique_values': int(df[column].nunique()),
                'most_common_value': str(value_counts.index[0]) if not value_counts.empty else None,
                'most_common_count': int(value_counts.iloc[0]) if not value_counts.empty else 0
            }
        elif col_type == 'datetime':
            stats[column] = {
                'missing_values': int(df[column].isna().sum()),
                'unique_values': int(df[column].nunique()),
                'min_date': str(df[column].min()) if not pd.isna(df[column].min()) else None,
                'max_date': str(df[column].max()) if not pd.isna(df[column].max()) else None
            }
        else:  # text or other types
            stats[column] = {
                'missing_values': int(df[column].isna().sum()),
                'unique_values': int(df[column].nunique())
            }
    return stats

def save_dataset(df, file_path, file_type='csv'):
    """
    Save dataset to file.
    
    Args:
        df: DataFrame to save
        file_path: Path where the file should be saved
        file_type: Type of file to save ('csv' or 'excel')
    
    Returns:
        str: Path to the saved file
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'excel':
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f'Unsupported file type: {file_type}')
        
        return file_path
    except Exception as e:
        print(f"Error saving dataset: {str(e)}")
        raise

# Function to test if conditions are working properly
def test_condition_filtering():
    """
    This function tests if the condition filtering is working properly.
    Run this function to verify that row filtering works as expected.
    """
    import pandas as pd
    print("\n==================================================")
    print("TESTING CONDITIONAL ROW FILTERING FUNCTIONALITY")
    print("==================================================")
    
    # Create a test DataFrame
    test_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'cat': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'd']
    })
    
    print("\nTest DataFrame:")
    print(test_df)
    
    # Test numeric conditions - greater than
    print("\n\nTEST 1: Numeric condition - Remove rows where 'num' > 50")
    numeric_condition = [{'type': 'greater_than', 'value': 50}]
    result1 = apply_numeric_conditions(test_df, 'num', numeric_condition)
    print("\nExpected result: Only rows with num <= 50 remain")
    print("Expected ids remaining: 1, 2, 3, 4, 5")
    
    # Test categorical include condition
    print("\n\nTEST 2: Categorical include condition - Remove rows where 'cat' is 'a' or 'b'")
    cat_include_condition = {'include': ['a', 'b']}
    result2 = apply_categorical_conditions(test_df, 'cat', cat_include_condition)
    print("\nExpected result: Only rows with cat NOT 'a' or 'b' remain")
    print("Expected ids remaining: 3, 6, 9, 10")
    
    # Test categorical exclude condition
    print("\n\nTEST 3: Categorical exclude condition - Remove rows where 'cat' is NOT 'a'")
    cat_exclude_condition = {'exclude': ['a']}
    result3 = apply_categorical_conditions(test_df, 'cat', cat_exclude_condition)
    print("\nExpected result: Only rows with cat = 'a' remain")
    print("Expected ids remaining: 1, 4, 7")
    
    print("\n==================================================")
    print("TESTING COMPLETE - Check if results match expectations")
    print("==================================================")

# Uncomment the line below to run the test when this module is loaded
# test_condition_filtering()