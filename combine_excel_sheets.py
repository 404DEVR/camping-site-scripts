import pandas as pd
import os
import glob
from pathlib import Path

def get_excel_files(input_path):
    """Get list of Excel files from input path or pattern."""
    excel_files = []
    
    if os.path.isfile(input_path):
        # Single file provided
        if input_path.endswith(('.xlsx', '.xls')):
            excel_files = [input_path]
        else:
            print(f"Error: {input_path} is not an Excel file")
            return []
    elif os.path.isdir(input_path):
        # Directory provided - get all Excel files
        excel_files = glob.glob(os.path.join(input_path, "*.xlsx"))
        excel_files.extend(glob.glob(os.path.join(input_path, "*.xls")))
        excel_files.sort()  # Sort for consistent order
    else:
        # Pattern provided (e.g., "data_*.xlsx")
        excel_files = glob.glob(input_path)
        excel_files.sort()
    
    if not excel_files:
        print(f"No Excel files found matching: {input_path}")
        return []
    
    print(f"Found {len(excel_files)} Excel files:")
    for i, file in enumerate(excel_files, 1):
        print(f"  {i}. {file}")
    
    return excel_files

def filter_landcover_classes(df, file_path):
    """Filter DataFrame to include only valid landcover classes."""
    # Valid landcover classes (case-insensitive)
    valid_classes = {'bare_soil', 'forest', 'grassland', 'snow_ice', 'water'}
    
    if 'landcover_class' not in df.columns:
        print(f"  Warning: No 'landcover_class' column found in {file_path}")
        return df
    
    # Get original counts
    original_count = len(df)
    original_classes = df['landcover_class'].value_counts().to_dict()
    
    # Convert to lowercase for comparison
    df['landcover_class_lower'] = df['landcover_class'].astype(str).str.lower().str.strip()
    
    # Filter to valid classes only
    valid_mask = df['landcover_class_lower'].isin(valid_classes)
    df_filtered = df[valid_mask].copy()
    
    # Remove the temporary column
    df_filtered = df_filtered.drop('landcover_class_lower', axis=1)
    
    # Report filtering results
    filtered_count = len(df_filtered)
    excluded_count = original_count - filtered_count
    
    print(f"  Landcover filtering:")
    print(f"    Original rows: {original_count}")
    print(f"    Valid rows: {filtered_count}")
    print(f"    Excluded rows: {excluded_count}")
    
    if excluded_count > 0:
        # Show what was excluded
        excluded_df = df[~valid_mask]
        excluded_classes = excluded_df['landcover_class'].value_counts().to_dict()
        print(f"    Excluded classes: {excluded_classes}")
    
    # Show final class distribution
    if filtered_count > 0:
        final_classes = df_filtered['landcover_class'].value_counts().to_dict()
        print(f"    Final classes: {final_classes}")
    
    return df_filtered

def load_excel_file(file_path):
    """Load Excel file and return DataFrame with metadata."""
    try:
        print(f"\nLoading: {file_path}")
        
        # Try to read the Excel file
        df = pd.read_excel(file_path)
        
        print(f"  Original shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Filter landcover classes
        df = filter_landcover_classes(df, file_path)
        
        if len(df) == 0:
            print(f"  ERROR: No valid rows remaining after filtering")
            return None, False
        
        print(f"  Final shape: {df.shape}")
        
        return df, True
        
    except Exception as e:
        print(f"  ERROR loading {file_path}: {e}")
        return None, False

def validate_column_compatibility(dataframes, file_names):
    """Check if all files have identical column structures (strict matching required)."""
    print("\nValidating column compatibility (strict mode)...")
    
    if not dataframes:
        return False
    
    # Get column lists for each file (preserve order)
    column_lists = []
    for df in dataframes:
        column_lists.append(list(df.columns))
    
    # Check if all column lists are identical (same columns, same order)
    first_columns = column_lists[0]
    all_identical = all(cols == first_columns for cols in column_lists)
    
    if all_identical:
        print("✓ All files have identical column structures")
        print(f"  Columns ({len(first_columns)}): {first_columns}")
        return True
    else:
        print("❌ ERROR: Files have different column structures")
        print("All files must have EXACTLY the same columns in the same order.")
        
        # Show detailed differences
        for i, (cols, file_name) in enumerate(zip(column_lists, file_names)):
            print(f"\n  File {i+1}: {file_name}")
            print(f"    Columns ({len(cols)}): {cols}")
            
            if i > 0:
                # Compare with first file
                missing = set(first_columns) - set(cols)
                extra = set(cols) - set(first_columns)
                order_diff = cols != first_columns
                
                if missing:
                    print(f"    Missing columns: {sorted(list(missing))}")
                if extra:
                    print(f"    Extra columns: {sorted(list(extra))}")
                if order_diff and not missing and not extra:
                    print(f"    Column order differs from first file")
        
        print(f"\nTo fix this:")
        print(f"1. Ensure all Excel files have identical column names")
        print(f"2. Ensure columns are in the same order")
        print(f"3. Remove any extra columns or add missing columns")
        
        return False

def combine_dataframes(dataframes, file_names):
    """Combine all dataframes into a single dataframe."""
    print(f"\nCombining {len(dataframes)} dataframes...")
    
    # Show summary by source file before combining
    print("\nSummary by source file:")
    total_rows = 0
    for i, (df, file_name) in enumerate(zip(dataframes, file_names)):
        rows = len(df)
        total_rows += rows
        print(f"  {i+1}. {file_name}: {rows} rows")
    
    # Combine all dataframes (no additional columns added)
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    print(f"\nCombined shape: {combined_df.shape}")
    print(f"Total rows: {total_rows} -> {len(combined_df)} (should match)")
    
    # Verify landcover class distribution in final result
    if 'landcover_class' in combined_df.columns:
        final_distribution = combined_df['landcover_class'].value_counts().to_dict()
        print(f"Final landcover distribution: {final_distribution}")
    
    return combined_df

def save_combined_file(df, output_path):
    """Save combined dataframe to Excel file."""
    print(f"\nSaving combined data to: {output_path}")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to Excel
        df.to_excel(output_path, index=False)
        
        print(f"✓ Successfully saved combined file")
        print(f"  Final shape: {df.shape}")
        print(f"  Total rows: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR saving file: {e}")
        return False

def main():
    """Main function to combine Excel files with landcover class filtering."""
    print("EXCEL FILES COMBINER WITH LANDCOVER FILTERING")
    print("=" * 50)
    print("Valid landcover classes: bare_soil, forest, grassland, snow_ice, water")
    print("All other rows will be excluded automatically")
    print("=" * 50)
    
    # Get input files
    print("\nInput options:")
    print("1. Single file: path/to/file.xlsx")
    print("2. Directory: path/to/directory/")
    print("3. Pattern: data_*.xlsx")
    print("4. Multiple files: file1.xlsx,file2.xlsx,file3.xlsx")
    
    input_path = input("\nEnter input (file/directory/pattern): ").strip()
    
    if not input_path:
        print("No input provided. Exiting.")
        return
    
    # Handle multiple files separated by commas
    if ',' in input_path:
        excel_files = [f.strip() for f in input_path.split(',')]
        # Validate each file exists
        valid_files = []
        for file in excel_files:
            if os.path.isfile(file) and file.endswith(('.xlsx', '.xls')):
                valid_files.append(file)
            else:
                print(f"Warning: {file} not found or not an Excel file")
        excel_files = valid_files
    else:
        excel_files = get_excel_files(input_path)
    
    if not excel_files:
        return
    
    # Load all Excel files
    dataframes = []
    file_names = []
    
    for file_path in excel_files:
        df, success = load_excel_file(file_path)
        if success and df is not None:
            dataframes.append(df)
            file_names.append(os.path.basename(file_path))
        else:
            print(f"Skipping {file_path} due to loading error")
    
    if not dataframes:
        print("No files loaded successfully. Exiting.")
        return
    
    # Validate column compatibility
    if not validate_column_compatibility(dataframes, file_names):
        return
    
    # Combine dataframes
    combined_df = combine_dataframes(dataframes, file_names)
    
    # Get output file path
    default_output = "combined_data.xlsx"
    output_path = input(f"\nEnter output file path (default: {default_output}): ").strip()
    
    if not output_path:
        output_path = default_output
    
    if not output_path.endswith(('.xlsx', '.xls')):
        output_path += '.xlsx'
    
    # Save combined file
    success = save_combined_file(combined_df, output_path)
    
    if success:
        print(f"\n{'='*50}")
        print("COMBINATION COMPLETED SUCCESSFULLY!")
        print(f"Combined {len(excel_files)} files into {output_path}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")
        print(f"{'='*50}")
    else:
        print("Combination failed. Check error messages above.")

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import openpyxl
    except ImportError:
        print("ERROR: Missing required dependency 'openpyxl'")
        print("Please install it using: pip install openpyxl")
        exit(1)
    
    main()