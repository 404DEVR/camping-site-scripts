import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import TabPFN
try:
    from tabpfn import TabPFNClassifier
except ImportError:
    print("ERROR: TabPFN not installed. Install with: pip install tabpfn")
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

def check_tabpfn_constraints(X_train, X_test, y_train):
    """Check TabPFN constraints and apply necessary preprocessing."""
    print(f"\nChecking TabPFN constraints...")
    
    n_samples_train = len(X_train)
    n_features = X_train.shape[1]
    n_classes = len(y_train.unique())
    
    print(f"Training samples: {n_samples_train}")
    print(f"Features: {n_features}")
    print(f"Classes: {n_classes}")
    
    # TabPFN constraints
    max_samples = 10000  # TabPFN limit
    max_features = 100   # TabPFN limit
    
    # Check sample size constraint
    if n_samples_train > max_samples:
        print(f"WARNING: Training samples ({n_samples_train}) exceed TabPFN limit ({max_samples})")
        print(f"Randomly sampling {max_samples} training samples...")
        
        # Stratified sampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        X_train_sampled, _, y_train_sampled, _ = train_test_split(
            X_train, y_train, 
            train_size=max_samples, 
            stratify=y_train, 
            random_state=42
        )
        X_train = X_train_sampled
        y_train = y_train_sampled
        print(f"Sampled training set: {X_train.shape}")
    
    # Check feature size constraint
    if n_features > max_features:
        print(f"WARNING: Features ({n_features}) exceed TabPFN limit ({max_features})")
        print(f"Selecting top {max_features} features using variance threshold...")
        
        from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
        
        # First remove low variance features
        var_selector = VarianceThreshold(threshold=0.01)
        X_train_var = var_selector.fit_transform(X_train)
        X_test_var = var_selector.transform(X_test)
        
        # Then select top k features using f_classif
        if X_train_var.shape[1] > max_features:
            k_selector = SelectKBest(score_func=f_classif, k=max_features)
            X_train = k_selector.fit_transform(X_train_var, y_train)
            X_test = k_selector.transform(X_test_var)
            
            # Get selected feature names
            selected_features = var_selector.get_support()
            selected_k_features = k_selector.get_support()
            
            original_features = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(len(selected_features))]
            final_features = [feat for i, feat in enumerate(original_features) if selected_features[i]]
            final_features = [feat for i, feat in enumerate(final_features) if selected_k_features[i]]
            
            print(f"Selected features: {final_features[:10]}..." if len(final_features) > 10 else f"Selected features: {final_features}")
        else:
            X_train = X_train_var
            X_test = X_test_var
        
        print(f"Final feature count: {X_train.shape[1]}")
    
    print("✓ TabPFN constraints satisfied")
    return X_train, X_test, y_train
    
def prepare_data(train_df, test_df):
    """Prepare features and targets with final validation."""
    X_train = train_df.drop('landcover_class', axis=1)
    y_train = train_df['landcover_class']
    X_test = test_df.drop('landcover_class', axis=1)
    y_test = test_df['landcover_class']
    
    print(f"\nInitial data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Show feature summary before constraints
    summarize_features(X_train)
    
    # Apply TabPFN constraints
    X_train, X_test, y_train = check_tabpfn_constraints(X_train, X_test, y_train)
    
    # Final data validation before training
    print("\nFinal data validation...")
    
    # Check for any remaining problematic values
    for name, X in [("train", X_train), ("test", X_test)]:
        if hasattr(X, 'values'):
            inf_count = np.isinf(X.values).sum()
            nan_count = np.isnan(X.values).sum()
        else:
            inf_count = np.isinf(X).sum()
            nan_count = np.isnan(X).sum()
        
        if inf_count > 0 or nan_count > 0:
            print(f"Fixing remaining issues in {name} set: {inf_count} inf, {nan_count} nan")
            if hasattr(X, 'replace'):
                X = X.replace([np.inf, -np.inf, np.nan], 0)
            else:
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Ensure data types are compatible
        if hasattr(X, 'astype'):
            X = X.astype(np.float32)  # TabPFN works well with float32
        else:
            X = X.astype(np.float32)
        
        if name == "train":
            X_train = X
        else:
            X_test = X
    
    print("✓ Data validation completed - ready for TabPFN training!")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train TabPFN classifier (no actual training needed - uses pre-trained transformer)."""
    print(f"\nInitializing TabPFN classifier...")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])}")
    print(f"Classes: {len(np.unique(y_train))}")
    
    # Encode string labels to integers if needed
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # TabPFN classifier - uses pre-trained transformer, no training needed
    model = TabPFNClassifier(
        device='cpu',  # Use 'cuda' if you have GPU and want faster inference
        seed=42
    )
    
    print("Fitting TabPFN model (loading pre-trained weights)...")
    # This doesn't actually train, just loads the pre-trained transformer
    model.fit(X_train, y_train_encoded)
    print("TabPFN model ready!")
    
    return model, label_encoder

def evaluate_model(model, label_encoder, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation."""
    print(f"\n{'='*60}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Encode labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Training accuracy
    train_pred_encoded = model.predict(X_train)
    train_pred = label_encoder.inverse_transform(train_pred_encoded)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Test accuracy
    test_pred_encoded = model.predict(X_test)
    test_pred = label_encoder.inverse_transform(test_pred_encoded)
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
    
    # Get prediction probabilities for additional insights
    try:
        print(f"\nPREDICTION CONFIDENCE ANALYSIS:")
        print("-" * 35)
        test_proba = model.predict_proba(X_test)
        max_proba = np.max(test_proba, axis=1)
        
        print(f"Average prediction confidence: {np.mean(max_proba):.4f}")
        print(f"Min prediction confidence: {np.min(max_proba):.4f}")
        print(f"Max prediction confidence: {np.max(max_proba):.4f}")
        
        # Confidence distribution
        high_conf = np.sum(max_proba > 0.8)
        med_conf = np.sum((max_proba > 0.6) & (max_proba <= 0.8))
        low_conf = np.sum(max_proba <= 0.6)
        
        print(f"High confidence predictions (>0.8): {high_conf} ({high_conf/len(test_proba)*100:.1f}%)")
        print(f"Medium confidence predictions (0.6-0.8): {med_conf} ({med_conf/len(test_proba)*100:.1f}%)")
        print(f"Low confidence predictions (≤0.6): {low_conf} ({low_conf/len(test_proba)*100:.1f}%)")
        
    except Exception as e:
        print(f"Could not compute prediction probabilities: {e}")
    
    # Model information
    print(f"\nMODEL INFORMATION:")
    print("-" * 20)
    print(f"Model type: TabPFN (Tabular Prior-Fitted Networks)")
    print(f"Pre-trained: Yes (no training required)")
    print(f"Device: {model.device}")
    print(f"Seed: {model.seed if hasattr(model, 'seed') else 'default'}")
    print(f"Classes encoded: {list(label_encoder.classes_)}")
    
    return test_accuracy, test_pred

def main():
    """Main function to run custom train/test ML pipeline."""
    # File paths - UPDATE THESE WITH YOUR FILE PATHS
    train_file = "../data_splits/train_data.csv"  # Your training Excel file
    test_file = "../data_splits/test_data.csv"   # Your test Excel file
    
    # Alternative: Use CSV files (no extra dependencies needed)
    # train_file = "data_splits/train_data.csv"
    # test_file = "data_splits/test_data.csv"
    
    print("CUSTOM TRAIN/TEST TABPFN PIPELINE")
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
        
        # Prepare data (includes TabPFN constraint handling)
        X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)
        
        # Train model (actually just loads pre-trained weights)
        model, label_encoder = train_model(X_train, y_train)
        
        # Evaluate model
        accuracy, predictions = evaluate_model(model, label_encoder, X_train, X_test, y_train, y_test)
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"{'='*60}")
        
        return model, accuracy
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Update the file paths in the main() function before running
    print("Make sure to update train_file and test_file paths in main() function")
    
    # Check for TabPFN dependency
    try:
        from tabpfn import TabPFNClassifier
        print("✓ TabPFN successfully imported")
    except ImportError:
        print("\nERROR: Missing required dependency 'tabpfn'")
        print("Please install it using: pip install tabpfn")
        print("Note: TabPFN requires PyTorch. Install PyTorch first if needed.")
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