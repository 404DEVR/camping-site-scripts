import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_and_clean_data(file_path):
    """Load and clean the dataset."""
    print(f"Loading data from {file_path}...")
    
    # Load Excel or CSV based on file extension
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Clean landcover_class column
    df = df.dropna(subset=['landcover_class'])
    df['landcover_class'] = df['landcover_class'].astype(str)
    df = df[df['landcover_class'].str.strip() != '']
    df = df[df['landcover_class'] != 'nan']
    
    print(f"Dataset shape after cleaning landcover_class: {df.shape}")
    
    return df

def clean_features(df):
    """Clean and preprocess feature columns."""
    # Separate features and target
    X = df.drop('landcover_class', axis=1)
    y = df['landcover_class']
    
    print("Cleaning feature data...")
    
    # Convert non-numeric columns to numeric where possible
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Replace infinity values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Report data quality
    inf_count = np.isinf(X.values).sum()
    nan_count = X.isnull().sum().sum()
    print(f"Infinity values replaced: {inf_count}")
    print(f"NaN values found: {nan_count}")
    
    # Fill missing values with column means
    X = X.fillna(X.mean())
    
    # Cap extreme outliers using IQR method
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Final cleanup
    if np.any(np.isinf(X.values)) or np.any(np.isnan(X.values)):
        print("Final cleanup of remaining problematic values...")
        X = X.fillna(X.mean())
        for col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], X[col].mean())
    
    # Combine cleaned features with target
    cleaned_df = X.copy()
    cleaned_df['landcover_class'] = y
    
    print(f"Final cleaned dataset shape: {cleaned_df.shape}")
    print(f"Target class distribution: {y.value_counts().to_dict()}")
    
    return cleaned_df

def split_and_save_data(df, output_dir="data_splits", test_size=0.2, random_state=42):
    """Split data into train/test and save as separate CSV files."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate features and target
    X = df.drop('landcover_class', axis=1)
    y = df['landcover_class']
    
    print(f"\nSplitting data into {int((1-test_size)*100)}/{int(test_size*100)} train/test sets...")
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Combine features and target for saving
    train_df = X_train.copy()
    train_df['landcover_class'] = y_train
    
    test_df = X_test.copy()
    test_df['landcover_class'] = y_test
    
    # Save to CSV files
    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training set saved: {train_path} (shape: {train_df.shape})")
    print(f"Test set saved: {test_path} (shape: {test_df.shape})")
    
    # Print class distribution for both sets
    print(f"\nTraining set class distribution:")
    print(y_train.value_counts().to_dict())
    print(f"\nTest set class distribution:")
    print(y_test.value_counts().to_dict())
    
    return train_path, test_path

def main():
    """Main function to process and split the dataset."""
    # Input file path
    input_file = "Combined dataset.xlsx"  # Change this to your file path
    
    try:
        # Load and clean data
        df = load_and_clean_data(input_file)
        
        # Clean features
        cleaned_df = clean_features(df)
        
        # Split and save data
        train_path, test_path = split_and_save_data(cleaned_df)
        
        print(f"\nData splitting completed successfully!")
        print(f"Use these files in your ML pipeline:")
        print(f"- Training data: {train_path}")
        print(f"- Test data: {test_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()