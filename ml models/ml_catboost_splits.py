import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import CatBoost
try:
    from catboost import CatBoostClassifier
except ImportError:
    print("ERROR: CatBoost not installed. Install with: pip install catboost")
    exit(1)

def load_and_clean_data(file_path, dataset_name):
    """Load Excel or CSV file and clean data by removing empty cells."""
    print(f"\nLoading {dataset_name} data from: {file_path}")
    
    try:
        # Load based on file extension
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            try:
                df = pd.read_excel(file_path)
            except ImportError:
                print("ERROR: openpyxl not installed. Install with: pip install openpyxl")
                return None
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"ERROR: Unsupported file format. Use .xlsx, .xls, or .csv files")
            return None
            
        print(f"Original {dataset_name} shape: {df.shape}")
        
        # Check for landcover_class column
        if 'landcover_class' not in df.columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"'landcover_class' column not found in {file_path}")
        
        # Remove rows with empty landcover_class
        initial_rows = len(df)
        df = df.dropna(subset=['landcover_class'])
        df['landcover_class'] = df['landcover_class'].astype(str)
        df = df[df['landcover_class'].str.strip() != '']
        df = df[df['landcover_class'] != 'nan']
        
        rows_removed = initial_rows - len(df)
        print(f"Removed {rows_removed} rows with empty landcover_class")
        
        # Clean feature columns
        df_cleaned = clean_features(df, dataset_name)
        
        print(f"Final {dataset_name} shape: {df_cleaned.shape}")
        print(f"{dataset_name} class distribution: {df_cleaned['landcover_class'].value_counts().to_dict()}")
        
        return df_cleaned
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def select_valid_features(df, dataset_name):
    """Select only valid SAR-derived features to prevent label leakage."""
    print(f"Selecting valid features for {dataset_name}...")
    
    # Define columns to exclude (label leakage prevention)
    exclude_columns = {
        'patch_id', 'image_name', 'patch_row', 'patch_col', 
        'landcover_majority', 'patch_purity', 'tier', 'tier_confidence', 
        'tier_reason', 'landcover_class'
    }
    
    # Define valid SAR and DEM-derived feature patterns
    valid_patterns = [
        'vv_', 'vh_', 'vv_vh_ratio', 'dem_', 'slope_', 'aspect_',
        'texture_', 'glcm_', 'stats_', 'mean', 'std', 'min', 'max',
        'percentile', 'variance', 'skewness', 'kurtosis', 'entropy',
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 
        'alpha', 'gamma', 'cv'
    ]
    
    # Get all columns except landcover_class
    all_columns = set(df.columns) - {'landcover_class'}
    
    # Remove explicitly excluded columns
    remaining_columns = all_columns - exclude_columns
    
    # Filter for valid SAR/DEM features
    valid_features = []
    for col in remaining_columns:
        col_lower = col.lower()
        # Check if column matches any valid pattern
        if any(pattern in col_lower for pattern in valid_patterns):
            valid_features.append(col)
        # Also include columns that are clearly numeric SAR statistics
        elif any(stat in col_lower for stat in ['mean', 'std', 'var', 'min', 'max']):
            valid_features.append(col)
    
    # Add landcover_class back for target
    final_columns = valid_features + ['landcover_class']
    
    # Filter dataframe
    filtered_df = df[final_columns]
    
    excluded_count = len(all_columns) - len(valid_features)
    print(f"Feature selection for {dataset_name}:")
    print(f"  Original features: {len(all_columns)}")
    print(f"  Valid features selected: {len(valid_features)}")
    print(f"  Features excluded: {excluded_count}")
    
    if excluded_count > 0:
        excluded_features = all_columns - set(valid_features)
        print(f"  Excluded features: {sorted(list(excluded_features))}")
    
    return filtered_df

def clean_features(df, dataset_name):
    """Clean feature columns by handling missing values and outliers."""
    print(f"Cleaning {dataset_name} features...")
    
    # First apply feature selection to prevent label leakage
    df = select_valid_features(df, dataset_name)
    
    # Separate features and target
    X = df.drop('landcover_class', axis=1)
    y = df['landcover_class']
    
    # Convert non-numeric columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Count missing values before cleaning
    missing_before = X.isnull().sum().sum()
    print(f"Missing values in {dataset_name}: {missing_before}")
    
    # Aggressive infinity and extreme value cleaning
    print(f"Cleaning infinity and extreme values in {dataset_name}...")
    
    # Replace infinity values with NaN first
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Check for extremely large values that might cause issues
    for col in X.columns:
        # Replace values that are too large for float32
        max_float32 = np.finfo(np.float32).max
        min_float32 = np.finfo(np.float32).min
        
        # Count extreme values
        extreme_count = ((X[col] > max_float32) | (X[col] < min_float32)).sum()
        if extreme_count > 0:
            print(f"  {col}: {extreme_count} extreme values found")
            X[col] = X[col].clip(lower=min_float32, upper=max_float32)
    
    # Fill missing values with column medians (more robust than mean)
    X = X.fillna(X.median())
    
    # Handle remaining NaN values with 0
    X = X.fillna(0)
    
    # Cap outliers using more conservative IQR method
    outliers_capped = 0
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:  # Only if there's variation in the data
            lower_bound = Q1 - 1.5 * IQR  # More conservative than 3*IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before capping
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            outliers_capped += outliers
            
            # Cap extreme values
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Outliers capped in {dataset_name}: {outliers_capped}")
    
    # Final aggressive cleanup
    print(f"Final data validation for {dataset_name}...")
    
    # Replace any remaining infinity values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Replace any remaining NaN values
    X = X.fillna(0)
    
    # Ensure all values are finite
    for col in X.columns:
        if not np.all(np.isfinite(X[col])):
            print(f"  Fixing non-finite values in {col}")
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(0)
            X[col] = X[col].replace([np.inf, -np.inf], 0)
    
    # Final validation
    inf_check = np.isinf(X.values).sum()
    nan_check = np.isnan(X.values).sum()
    
    if inf_check > 0 or nan_check > 0:
        print(f"Warning: Still found {inf_check} inf and {nan_check} nan values")
        X = X.replace([np.inf, -np.inf, np.nan], 0)
    
    print(f"Data cleaning completed for {dataset_name}")
    print(f"Final feature range: min={X.min().min():.2f}, max={X.max().max():.2f}")
    
    # Combine cleaned features with target
    cleaned_df = X.copy()
    cleaned_df['landcover_class'] = y
    
    return cleaned_df
    
def validate_datasets(train_df, test_df):
    """Validate that train and test datasets are compatible after feature selection."""
    print("\nValidating datasets compatibility after feature selection...")
    
    # Check feature columns match
    train_features = set(train_df.columns) - {'landcover_class'}
    test_features = set(test_df.columns) - {'landcover_class'}
    
    print(f"Training features: {len(train_features)}")
    print(f"Test features: {len(test_features)}")
    
    if train_features != test_features:
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features
        
        if missing_in_test:
            print(f"Warning: Features missing in test set: {sorted(list(missing_in_test))}")
        if missing_in_train:
            print(f"Warning: Features missing in train set: {sorted(list(missing_in_train))}")
        
        # Use common features only
        common_features = train_features & test_features
        common_features.add('landcover_class')
        
        train_df = train_df[list(common_features)]
        test_df = test_df[list(common_features)]
        
        print(f"Using {len(common_features)-1} common valid features")
    else:
        print("✓ All features match between train and test sets")
    
    # Validate no label leakage columns remain
    potential_leakage = ['patch_id', 'image_name', 'landcover_majority', 'patch_purity', 'tier']
    remaining_features = train_features
    
    leakage_found = [col for col in potential_leakage if col in remaining_features]
    if leakage_found:
        print(f"ERROR: Potential label leakage columns found: {leakage_found}")
        raise ValueError(f"Label leakage detected: {leakage_found}")
    else:
        print("✓ No label leakage columns detected")
    
    # Check class overlap
    train_classes = set(train_df['landcover_class'].unique())
    test_classes = set(test_df['landcover_class'].unique())
    
    print(f"Training classes: {sorted(list(train_classes))}")
    print(f"Test classes: {sorted(list(test_classes))}")
    
    if test_classes - train_classes:
        print(f"Warning: Test set has classes not in training: {test_classes - train_classes}")
    
    print("✓ Dataset validation completed - ready for ML pipeline")
    return train_df, test_df

def summarize_features(X_train):
    """Summarize the selected features for transparency."""
    print(f"\nFEATURE SUMMARY:")
    print("=" * 50)
    
    feature_categories = {
        'SAR VV': [col for col in X_train.columns if 'vv' in col.lower() and 'vh' not in col.lower()],
        'SAR VH': [col for col in X_train.columns if 'vh' in col.lower() and 'vv' not in col.lower()],
        'SAR Ratio': [col for col in X_train.columns if 'ratio' in col.lower()],
        'DEM/Elevation': [col for col in X_train.columns if any(x in col.lower() for x in ['dem', 'elevation', 'slope', 'aspect'])],
        'Texture': [col for col in X_train.columns if any(x in col.lower() for x in ['texture', 'glcm', 'contrast', 'homogeneity', 'entropy'])],
        'Statistics': [col for col in X_train.columns if any(x in col.lower() for x in ['mean', 'std', 'var', 'min', 'max', 'percentile'])],
        'Other': []
    }
    
    # Categorize remaining features
    categorized = set()
    for category, features in feature_categories.items():
        if category != 'Other':
            categorized.update(features)
    
    feature_categories['Other'] = [col for col in X_train.columns if col not in categorized]
    
    total_features = 0
    for category, features in feature_categories.items():
        if features:
            print(f"{category}: {len(features)} features")
            total_features += len(features)
            # Show first few features as examples
            if len(features) <= 3:
                print(f"  {features}")
            else:
                print(f"  Examples: {features[:3]}...")
    
    print(f"\nTotal valid features: {total_features}")
    print("All features are SAR/DEM-derived - no label leakage risk")
    
def prepare_data(train_df, test_df):
    """Prepare features and targets with final validation."""
    X_train = train_df.drop('landcover_class', axis=1)
    y_train = train_df['landcover_class']
    X_test = test_df.drop('landcover_class', axis=1)
    y_test = test_df['landcover_class']
    
    print(f"\nFinal data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Show feature summary
    summarize_features(X_train)
    
    # Final data validation before training
    print("\nFinal data validation...")
    
    # Check for any remaining problematic values
    for name, X in [("train", X_train), ("test", X_test)]:
        inf_count = np.isinf(X.values).sum()
        nan_count = np.isnan(X.values).sum()
        
        if inf_count > 0 or nan_count > 0:
            print(f"Fixing remaining issues in {name} set: {inf_count} inf, {nan_count} nan")
            X = X.replace([np.inf, -np.inf, np.nan], 0)
            
        # Ensure data types are compatible
        X = X.astype(np.float64)  # Use float64 instead of float32
        
        if name == "train":
            X_train = X
        else:
            X_test = X
    
    print("✓ Data validation completed - ready for training!")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train CatBoost classifier with optimized parameters."""
    print(f"\nTraining CatBoost classifier...")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(y_train.unique())}")
    
    # CatBoost handles categorical features automatically, but we'll use numeric features
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_strength=1,
        one_hot_max_size=2,
        rsm=0.8,  # Random subspace method
        random_seed=42,
        verbose=False,  # Set to True to see training progress
        thread_count=-1,  # Use all available cores
        allow_writing_files=False  # Prevent creating temporary files
    )
    
    print("Training CatBoost model...")
    model.fit(X_train, y_train)
    print("Training completed!")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation."""
    print(f"\n{'='*60}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Test accuracy
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nACCURACY SCORES:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting Check: {train_accuracy - test_accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(y_test, test_pred))
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX:")
    print("-" * 30)
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    
    # Feature importance (top 10)
    print(f"\nTOP 10 FEATURE IMPORTANCES:")
    print("-" * 35)
    feature_names = X_train.columns
    importances = model.get_feature_importance()
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Model parameters
    print(f"\nMODEL PARAMETERS:")
    print("-" * 20)
    print(f"Iterations: {model.get_param('iterations')}")
    print(f"Learning rate: {model.get_param('learning_rate')}")
    print(f"Depth: {model.get_param('depth')}")
    print(f"L2 regularization: {model.get_param('l2_leaf_reg')}")
    print(f"Bootstrap type: {model.get_param('bootstrap_type')}")
    print(f"Subsample: {model.get_param('subsample')}")
    
    return test_accuracy, test_pred

def main():
    """Main function to run custom train/test ML pipeline."""
    # File paths - UPDATE THESE WITH YOUR FILE PATHS
    train_file = "../data_splits/train_data.csv"  # Your training Excel file
    test_file = "../data_splits/test_data.csv"   # Your test Excel file
    
    # Alternative: Use CSV files (no extra dependencies needed)
    # train_file = "data_splits/train_data.csv"
    # test_file = "data_splits/test_data.csv"
    
    print("CUSTOM TRAIN/TEST CATBOOST PIPELINE")
    print("=" * 40)
    
    try:
        # Load and clean datasets
        train_df = load_and_clean_data(train_file, "training")
        if train_df is None:
            return None, None
            
        test_df = load_and_clean_data(test_file, "test")
        if test_df is None:
            return None, None
        
        # Validate compatibility
        train_df, test_df = validate_datasets(train_df, test_df)
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        accuracy, predictions = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"{'='*60}")
        
        return model, accuracy
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return None, None

if __name__ == "__main__":
    # Update the file paths in the main() function before running
    print("Make sure to update train_file and test_file paths in main() function")
    
    # Check for CatBoost dependency
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("\nERROR: Missing required dependency 'catboost'")
        print("Please install it using: pip install catboost")
        print("Then run the script again.")
        exit(1)
    
    result = main()
    if result is not None and result != (None, None):
        model, accuracy = result
        if accuracy is not None:
            print(f"Script completed successfully with accuracy: {accuracy:.4f}")
        else:
            print("Script completed but accuracy is None")
    else:
        print("Script failed - check error messages above")