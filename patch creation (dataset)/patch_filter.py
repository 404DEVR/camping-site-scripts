import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class UniversalPatchFilterProcessor:
    def __init__(self):
        """
        Universal patch filtering processor for CSV or Excel files
        """
        self.landcover_mapping = {
            10: 'Forest', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
            50: 'Urban', 60: 'Bare_Soil', 70: 'Snow_Ice', 80: 'Water',
            90: 'Herbaceous_Wetland', 95: 'Mangroves', 100: 'Moss_Lichen'
        }
        
        print("✅ Universal Patch Filter Processor Initialized")
        print("✅ Supports: CSV (.csv) and Excel (.xlsx, .xls) files")

    def load_file(self, file_path, sheet_name=0):
        """
        Load file - automatically detects CSV or Excel format
        
        Parameters:
        -----------
        file_path : str
            Path to your file (.csv, .xlsx, .xls)
        sheet_name : str or int
            Sheet name (for Excel files only), default is first sheet
        """
        try:
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'csv':
                print(f"📄 Loading CSV file: {file_path}")
                df = pd.read_csv(file_path)
                
            elif file_extension in ['xlsx', 'xls']:
                print(f"📊 Loading Excel file: {file_path}")
                if isinstance(sheet_name, int):
                    print(f"   Using sheet index: {sheet_name} (first sheet)")
                else:
                    print(f"   Using sheet name: {sheet_name}")
                
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                
            else:
                print(f"❌ Unsupported file format: {file_extension}")
                print("   Supported formats: .csv, .xlsx, .xls")
                return None
            
            print(f"✅ Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            if 'openpyxl' in str(e):
                print("💡 Try installing: pip install openpyxl")
            elif 'xlrd' in str(e):
                print("💡 Try installing: pip install xlrd")
            return None

    def save_file(self, df, output_path, file_format='auto'):
        """
        Save file in specified format
        
        Parameters:
        -----------
        df : DataFrame
            Data to save
        output_path : str
            Output file path
        file_format : str
            'auto', 'csv', 'excel'
        """
        try:
            if file_format == 'auto':
                # Auto-detect from file extension
                if output_path.lower().endswith(('.xlsx', '.xls')):
                    file_format = 'excel'
                else:
                    file_format = 'csv'
            
            if file_format == 'excel':
                print(f"💾 Saving as Excel: {output_path}")
                df.to_excel(output_path, index=False, engine='openpyxl')
            else:
                print(f"💾 Saving as CSV: {output_path}")
                df.to_csv(output_path, index=False)
            
            print(f"✅ Successfully saved: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            if 'openpyxl' in str(e):
                print("💡 Try installing: pip install openpyxl")
            return False

    def analyze_file_structure(self, df):
        """Analyze file structure"""
        print(f"\n=== FILE ANALYSIS ===")
        print(f"Total patches: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        # Check key columns
        key_columns = ['patch_id', 'landcover_majority', 'patch_purity']
        for col in key_columns:
            if col in df.columns:
                print(f"✅ {col}: Available")
            else:
                print(f"❌ {col}: Missing")
                # Try to find similar column names
                similar_cols = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                if similar_cols:
                    print(f"   💡 Similar columns found: {similar_cols}")
        
        # Analyze landcover distribution
        if 'landcover_majority' in df.columns:
            print(f"\n=== LANDCOVER DISTRIBUTION ===")
            lc_counts = df['landcover_majority'].value_counts()
            for lc_code, count in lc_counts.items():
                lc_name = self.landcover_mapping.get(lc_code, f'Unknown_{lc_code}')
                percentage = count / len(df) * 100
                print(f"Class {lc_code} ({lc_name}): {count:4d} patches ({percentage:5.1f}%)")
        
        # Analyze purity distribution
        if 'patch_purity' in df.columns:
            print(f"\n=== PURITY DISTRIBUTION ===")
            purity_stats = df['patch_purity'].describe()
            print(f"Mean purity: {purity_stats['mean']:.3f}")
            print(f"Min purity:  {purity_stats['min']:.3f}")
            print(f"Max purity:  {purity_stats['max']:.3f}")
            
            # Purity thresholds analysis
            thresholds = [0.98, 0.95, 0.90, 0.85, 0.75, 0.65]
            print(f"\nPatches meeting purity thresholds:")
            for threshold in thresholds:
                count = len(df[df['patch_purity'] >= threshold])
                percentage = count / len(df) * 100
                print(f"  ≥{threshold:.0%}: {count:4d} patches ({percentage:5.1f}%)")

    def assign_tier_strict(self, row):
        """STRICT tier assignment"""
        purity = row['patch_purity']
        
        if purity >= 0.98:
            return {'tier': 'gold', 'confidence': 1.0, 'reason': f'ultra_pure_{purity:.3f}'}
        elif purity >= 0.95:
            return {'tier': 'silver', 'confidence': 0.95, 'reason': f'excellent_purity_{purity:.3f}'}
        elif purity >= 0.90:
            return {'tier': 'bronze', 'confidence': 0.85, 'reason': f'very_good_purity_{purity:.3f}'}
        else:
            return {'tier': 'rejected', 'confidence': 0.0, 'reason': f'insufficient_purity_{purity:.3f}'}

    def assign_tier_balanced(self, row):
        """BALANCED tier assignment"""
        purity = row['patch_purity']
        
        if purity >= 0.95:
            return {'tier': 'gold', 'confidence': 1.0, 'reason': f'excellent_purity_{purity:.3f}'}
        elif purity >= 0.85:
            return {'tier': 'silver', 'confidence': 0.9, 'reason': f'very_good_purity_{purity:.3f}'}
        elif purity >= 0.75:
            return {'tier': 'bronze', 'confidence': 0.8, 'reason': f'good_purity_{purity:.3f}'}
        elif purity >= 0.65:
            return {'tier': 'acceptable', 'confidence': 0.6, 'reason': f'acceptable_purity_{purity:.3f}'}
        else:
            return {'tier': 'rejected', 'confidence': 0.0, 'reason': f'low_purity_{purity:.3f}'}

    def filter_patches(self, input_file, mode='strict', save_rejected=False, 
                      sheet_name=0, output_format='auto'):
        """
        Main filtering function for CSV or Excel files
        
        Parameters:
        -----------
        input_file : str
            Path to your file (.csv, .xlsx, .xls)
        mode : str
            'strict' or 'balanced' filtering mode
        save_rejected : bool
            Whether to include rejected patches in output
        sheet_name : str or int
            Sheet name for Excel files (default: first sheet)
        output_format : str
            'auto', 'csv', 'excel' - output format
        """
        
        print("=" * 60)
        print("UNIVERSAL PATCH FILTERING PROCESSOR")
        print("=" * 60)
        
        # Load file (CSV or Excel)
        df = self.load_file(input_file, sheet_name=sheet_name)
        if df is None:
            return None
        
        # Analyze the structure
        self.analyze_file_structure(df)
        
        # Check if required columns exist
        if 'patch_purity' not in df.columns:
            print("❌ Error: 'patch_purity' column not found!")
            print("   Available columns:", list(df.columns))
            return None
        
        # Apply tier assignment
        print(f"\n=== APPLYING {mode.upper()} FILTERING ===")
        
        tier_results = []
        for idx, row in df.iterrows():
            if mode == 'strict':
                tier_result = self.assign_tier_strict(row)
            else:
                tier_result = self.assign_tier_balanced(row)
            tier_results.append(tier_result)
        
        # Add tier information to dataframe
        df['tier'] = [result['tier'] for result in tier_results]
        df['tier_confidence'] = [result['confidence'] for result in tier_results]
        df['tier_reason'] = [result['reason'] for result in tier_results]
        
        # Add landcover class names if possible
        if 'landcover_majority' in df.columns:
            df['landcover_class'] = df['landcover_majority'].map(self.landcover_mapping)
        
        # Display tier statistics
        print(f"\n=== TIER ASSIGNMENT RESULTS ===")
        tier_counts = df['tier'].value_counts()
        total_patches = len(df)
        
        tier_order = ['gold', 'silver', 'bronze', 'acceptable', 'rejected']
        for tier in tier_order:
            count = tier_counts.get(tier, 0)
            percentage = count / total_patches * 100
            status = "✅ Keep" if tier != 'rejected' else "❌ Discard"
            print(f"{tier.capitalize():12}: {count:6d} patches ({percentage:5.1f}%) - {status}")
        
        # Filter dataset
        if save_rejected:
            df_filtered = df.copy()
            print(f"\n✅ Complete dataset retained: {len(df_filtered)} patches")
        else:
            df_filtered = df[df['tier'] != 'rejected'].copy()
            rejected_count = len(df) - len(df_filtered)
            print(f"\n✅ Filtered dataset: {len(df_filtered)} patches")
            print(f"❌ Rejected patches: {rejected_count} patches")
            print(f"📊 Retention rate: {len(df_filtered)/len(df)*100:.1f}%")
        
        # Generate output filename
        input_name = input_file.rsplit('.', 1)[0]  # Remove extension
        input_ext = input_file.rsplit('.', 1)[1] if '.' in input_file else 'csv'
        
        if output_format == 'auto':
            output_ext = input_ext  # Keep same format as input
        elif output_format == 'excel':
            output_ext = 'xlsx'
        else:
            output_ext = 'csv'
        
        output_file = f"{input_name}_filtered_{mode}.{output_ext}"
        
        # Save results
        success = self.save_file(df_filtered, output_file, output_format)
        
        if success:
            print(f"\n✅ Filtered dataset saved: {output_file}")
            print(f"   Final size: {len(df_filtered)} patches")
            print(f"   Columns: {len(df_filtered.columns)} (original + 3 new)")
            
            # Save summary
            summary_file = f"{input_name}_filtered_{mode}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("PATCH FILTERING SUMMARY\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Input file: {input_file}\n")
                f.write(f"Output file: {output_file}\n")
                f.write(f"Filtering mode: {mode.upper()}\n")
                f.write(f"Original patches: {total_patches}\n")
                f.write(f"Filtered patches: {len(df_filtered)}\n")
                f.write(f"Retention rate: {len(df_filtered)/total_patches*100:.1f}%\n\n")
                
                f.write("Tier Distribution:\n")
                for tier in tier_order:
                    count = tier_counts.get(tier, 0)
                    percentage = count / total_patches * 100
                    f.write(f"{tier.capitalize():12}: {count:6d} ({percentage:5.1f}%)\n")
            
            print(f"✅ Summary saved: {summary_file}")
        
        return df_filtered


def main():
    """Interactive execution"""
    processor = UniversalPatchFilterProcessor()
    
    # Get file path
    input_file = input("Enter file path (.csv, .xlsx, .xls): ").strip()
    if not input_file:
        input_file = "python_features.csv"  # Default
    
    # Check file type and ask for sheet if Excel
    sheet_name = 0
    if input_file.lower().endswith(('.xlsx', '.xls')):
        sheet_choice = input("Enter sheet name or number (default: 0 for first sheet): ").strip()
        if sheet_choice:
            try:
                sheet_name = int(sheet_choice)
            except:
                sheet_name = sheet_choice
    
    # Get filtering options
    print("\nFiltering modes:")
    print("1. STRICT (≥90% purity)")
    print("2. BALANCED (≥65% purity)")
    mode_choice = input("Choose mode (1 or 2): ").strip()
    mode = 'strict' if mode_choice != '2' else 'balanced'
    
    save_rejected = input("Include rejected patches? (y/n): ").strip().lower() == 'y'
    
    # Output format
    print("\nOutput format:")
    print("1. Same as input")
    print("2. CSV")
    print("3. Excel")
    format_choice = input("Choose format (1, 2, or 3): ").strip()
    if format_choice == '2':
        output_format = 'csv'
    elif format_choice == '3':
        output_format = 'excel'
    else:
        output_format = 'auto'
    
    # Process file
    result = processor.filter_patches(
        input_file=input_file,
        mode=mode,
        save_rejected=save_rejected,
        sheet_name=sheet_name,
        output_format=output_format
    )
    
    if result is not None:
        print("\n🎉 Processing completed successfully!")


if __name__ == "__main__":
    # AUTOMATIC EXECUTION - Modify these settings:
    
    INPUT_FILE = "python_features.xlsx"  # Change to your Excel file path
    SHEET_NAME = 0  # 0 for first sheet, or "Sheet1" for sheet name
    MODE = "strict"  # 'strict' or 'balanced'
    SAVE_REJECTED = False
    OUTPUT_FORMAT = "auto"  # 'auto', 'csv', or 'excel'
    
    processor = UniversalPatchFilterProcessor()
    
    filtered_df = processor.filter_patches(
        input_file=INPUT_FILE,
        mode=MODE,
        save_rejected=SAVE_REJECTED,
        sheet_name=SHEET_NAME,
        output_format=OUTPUT_FORMAT
    )
    
    # Uncomment for interactive mode:
    # main()
