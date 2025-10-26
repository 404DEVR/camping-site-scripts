import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')


class MATLABCompatibleFeatureExtractor:
    def __init__(self, patch_size=32, overlap=0.5):
        """
        Python feature extractor compatible with MATLAB G0 patch system
        Excludes G0 parameters - those come from MATLAB
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = int(patch_size * (1 - overlap))  # stride = 16 for 50% overlap
        
        # Feature names excluding G0 parameters (MATLAB will provide those)
        self.feature_names = [
            # Backscatter features (5)
            'vv_mean', 'vh_mean', 'vv_vh_ratio', 'vv_vh_diff', 'vv_vh_norm_diff',
            
            # VV texture features (5)
            'vv_contrast', 'vv_dissimilarity', 'vv_homogeneity', 'vv_asm', 'vv_correlation',
            
            # VH texture features (5)
            'vh_contrast', 'vh_dissimilarity', 'vh_homogeneity', 'vh_asm', 'vh_correlation',
            
            # VV statistical features (7)
            'vv_stat_mean', 'vv_std', 'vv_var', 'vv_skew', 'vv_kurt', 'vv_min', 'vv_max',
            
            # VH statistical features (7)
            'vh_stat_mean', 'vh_std', 'vh_var', 'vh_skew', 'vh_kurt', 'vh_min', 'vh_max',
            
            # Topographic features (12)
            'dem_mean', 'dem_std', 'dem_min', 'dem_max',
            'slope_mean', 'slope_std', 'slope_min', 'slope_max',
            'aspect_mean', 'aspect_std', 'aspect_min', 'aspect_max'
        ]
        
        print(f"Initialized MATLAB-compatible feature extractor")
        print(f"Patch size: {self.patch_size}×{self.patch_size}, Overlap: {self.overlap*100}%, Stride: {self.stride}")
        print(f"Will extract {len(self.feature_names)} features per patch (excluding G0 - MATLAB provides those)")


    def load_files(self, sar_file, landcover_file, dem_file, slope_file, aspect_file):
        """Load all data files"""
        data = {}
        
        # Load SAR data (dual-band: VV and VH)
        with rasterio.open(sar_file) as src:
            if src.count >= 2:
                data['vv'] = src.read(1).astype(np.float32)
                data['vh'] = src.read(2).astype(np.float32)
                print(f"SAR data loaded: {data['vv'].shape} (dual-band: VV + VH)")
            else:
                print("Warning: SAR file appears to have only 1 band")
                data['vv'] = src.read(1).astype(np.float32)
                data['vh'] = np.zeros_like(data['vv'])
        
        # Load auxiliary data
        with rasterio.open(landcover_file) as src:
            data['landcover'] = src.read(1).astype(np.int16)
        with rasterio.open(dem_file) as src:
            data['dem'] = src.read(1).astype(np.float32)
        with rasterio.open(slope_file) as src:
            data['slope'] = src.read(1).astype(np.float32)
        with rasterio.open(aspect_file) as src:
            data['aspect'] = src.read(1).astype(np.float32)
            
        print(f"All data loaded - Image size: {data['vv'].shape}")
        return data


    def generate_patch_id(self, image_name, row, col):
        """Generate patch ID identical to MATLAB format"""
        return f"{image_name}_r{row:04d}_c{col:04d}"


    def calculate_backscatter_features(self, vv_patch, vh_patch):
        """Calculate SAR backscatter features"""
        try:
            vv_mean = np.nanmean(vv_patch)
            vh_mean = np.nanmean(vh_patch)
            
            # Avoid division by zero
            vv_vh_ratio = vv_mean / (vh_mean + 1e-8) if vh_mean != 0 else np.nan
            vv_vh_diff = vv_mean - vh_mean
            vv_vh_norm_diff = (vv_mean - vh_mean) / (vv_mean + vh_mean + 1e-8)
            
            return [vv_mean, vh_mean, vv_vh_ratio, vv_vh_diff, vv_vh_norm_diff]
        except:
            return [np.nan] * 5


    def calculate_glcm_features(self, patch):
        """Calculate GLCM texture features"""
        try:
            # Remove NaN values
            patch_clean = patch[~np.isnan(patch)]
            if len(patch_clean) < 100:
                return [np.nan] * 5
            
            # Normalize to 0-255 for GLCM
            patch_norm = ((patch - np.nanmin(patch)) / 
                          (np.nanmax(patch) - np.nanmin(patch)) * 255).astype(np.uint8)
            patch_norm = np.nan_to_num(patch_norm, nan=0)
            
            # Calculate GLCM
            glcm = graycomatrix(patch_norm, distances=[1], 
                                angles=[0, 45, 90, 135], symmetric=True, normed=True)
            
            # Extract texture features
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            asm = graycoprops(glcm, 'ASM').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            return [contrast, dissimilarity, homogeneity, asm, correlation]
        except:
            return [np.nan] * 5


    def calculate_statistical_features(self, patch):
        """Calculate statistical features"""
        try:
            flat_patch = patch.flatten()
            flat_patch = flat_patch[~np.isnan(flat_patch)]
            
            if len(flat_patch) < 50:
                return [np.nan] * 7
            
            mean_val = np.mean(flat_patch)
            std_val = np.std(flat_patch)
            var_val = np.var(flat_patch)
            skew_val = stats.skew(flat_patch)
            kurt_val = stats.kurtosis(flat_patch)
            min_val = np.min(flat_patch)
            max_val = np.max(flat_patch)
            
            return [mean_val, std_val, var_val, skew_val, kurt_val, min_val, max_val]
        except:
            return [np.nan] * 7


    def calculate_topographic_features(self, dem_patch, slope_patch, aspect_patch):
        """Calculate topographic features"""
        features = []
        
        # DEM features
        features.extend([
            np.nanmean(dem_patch), np.nanstd(dem_patch), 
            np.nanmin(dem_patch), np.nanmax(dem_patch)
        ])
        
        # Slope features
        features.extend([
            np.nanmean(slope_patch), np.nanstd(slope_patch), 
            np.nanmin(slope_patch), np.nanmax(slope_patch)
        ])
        
        # Aspect features
        features.extend([
            np.nanmean(aspect_patch), np.nanstd(aspect_patch), 
            np.nanmin(aspect_patch), np.nanmax(aspect_patch)
        ])
        
        return features


    def calculate_all_features(self, patch_data):
        """Calculate all features EXCEPT G0 parameters"""
        features = []
        
        # Backscatter features (5)
        features.extend(self.calculate_backscatter_features(patch_data['vv'], patch_data['vh']))
        
        # VV texture features (5)
        features.extend(self.calculate_glcm_features(patch_data['vv']))
        
        # VH texture features (5)
        features.extend(self.calculate_glcm_features(patch_data['vh']))
        
        # VV statistical features (7)
        features.extend(self.calculate_statistical_features(patch_data['vv']))
        
        # VH statistical features (7)
        features.extend(self.calculate_statistical_features(patch_data['vh']))
        
        # Topographic features (12)
        features.extend(self.calculate_topographic_features(
            patch_data['dem'], patch_data['slope'], patch_data['aspect']))
        
        return features


    def extract_patches(self, data, image_name):
        """
        Extract patches using SAME coordinate system as MATLAB
        """
        print(f"=== MATLAB-COMPATIBLE FEATURE EXTRACTION ===")
        
        vv, vh = data['vv'], data['vh']
        landcover, dem = data['landcover'], data['dem']
        slope, aspect = data['slope'], data['aspect']
        
        height, width = vv.shape
        
        # Calculate expected patch count (same as MATLAB)
        num_patches_row = (height - self.patch_size) // self.stride + 1
        num_patches_col = (width - self.patch_size) // self.stride + 1
        expected_patches = num_patches_row * num_patches_col
        
        print(f"Image: {image_name} ({height}×{width} pixels)")
        print(f"Expected patches: {expected_patches} ({num_patches_row}×{num_patches_col})")
        print(f"Patch coordinates will match MATLAB exactly")
        
        patches_data = []
        processed_count = 0
        
        # Use SAME iteration as MATLAB: range(0, height-patch_size+1, stride)
        for i in range(0, height - self.patch_size + 1, self.stride):
            for j in range(0, width - self.patch_size + 1, self.stride):
                
                # Generate SAME patch ID as MATLAB (0-based coordinates)
                patch_id = self.generate_patch_id(image_name, i, j)
                
                # Extract patch data
                patch_data = {
                    'vv': vv[i:i+self.patch_size, j:j+self.patch_size],
                    'vh': vh[i:i+self.patch_size, j:j+self.patch_size],
                    'landcover': landcover[i:i+self.patch_size, j:j+self.patch_size],
                    'dem': dem[i:i+self.patch_size, j:j+self.patch_size],
                    'slope': slope[i:i+self.patch_size, j:j+self.patch_size],
                    'aspect': aspect[i:i+self.patch_size, j:j+self.patch_size]
                }
                
                # Calculate all features (excluding G0)
                features = self.calculate_all_features(patch_data)
                
                # Get landcover information
                lc_values, lc_counts = np.unique(patch_data['landcover'], return_counts=True)
                majority_label = lc_values[np.argmax(lc_counts)]
                patch_purity = np.max(lc_counts) / (self.patch_size * self.patch_size)
                
                # Store patch data
                patch_record = {
                    'patch_id': patch_id,
                    'image_name': image_name,
                    'patch_row': i,  # 0-based coordinates (same as MATLAB output)
                    'patch_col': j,
                    'landcover_majority': majority_label,
                    'patch_purity': patch_purity
                }
                
                # Add all features
                for feat_name, feat_value in zip(self.feature_names, features):
                    patch_record[feat_name] = feat_value
                
                patches_data.append(patch_record)
                processed_count += 1
                
                # Progress update
                if processed_count % 200 == 0:
                    progress = processed_count / expected_patches * 100
                    print(f"Processed {processed_count}/{expected_patches} patches ({progress:.1f}%)")
        
        print(f"✅ Feature extraction complete: {processed_count} patches processed")
        return patches_data


    def create_dataset(self, sar_file, landcover_file, dem_file, slope_file, aspect_file, image_name=None):
        """
        Create feature dataset compatible with MATLAB G0 results
        """
        print("=== MATLAB-COMPATIBLE FEATURE EXTRACTOR ===")
        print("Extracts all features EXCEPT G0 parameters")
        print("G0 parameters will come from MATLAB processing")
        print("=" * 60)
        
        # Extract image name for patch IDs
        if image_name is None:
            import os
            image_name = os.path.splitext(os.path.basename(sar_file))[0]
        print(image_name)
        
        # Load all data
        data = self.load_files(sar_file, landcover_file, dem_file, slope_file, aspect_file)
        
        # Extract patches and features
        patches_data = self.extract_patches(data, image_name)
        
        if len(patches_data) > 0:
            # Create DataFrame
            df_features = pd.DataFrame(patches_data)
            
            print(f"\n=== PYTHON FEATURE DATASET SUMMARY ===")
            print(f"Total patches: {len(patches_data)}")
            print(f"Features per patch: {len(self.feature_names)} (excluding G0)")
            print(f"Patch ID format: {patches_data[0]['patch_id']}")
            print(f"Coordinate system: 0-based (compatible with MATLAB)")
            
            # Feature summary
            feature_cols = [col for col in df_features.columns if col in self.feature_names]
            print(f"\nFeature categories:")
            print(f"  Backscatter features: 5")
            print(f"  VV texture features: 5") 
            print(f"  VH texture features: 5")
            print(f"  VV statistical features: 7")
            print(f"  VH statistical features: 7")
            print(f"  Topographic features: 12")
            print(f"  Total: {len(feature_cols)} features")
            
            # Landcover distribution
            if 'landcover_majority' in df_features.columns:
                print(f"\nLandcover distribution:")
                lc_counts = df_features['landcover_majority'].value_counts()
                for lc_code, count in lc_counts.head(10).items():
                    percentage = count / len(df_features) * 100
                    print(f"  Class {lc_code}: {count} patches ({percentage:.1f}%)")
            
            # Save dataset
            output_filename = 'python_features.csv'
            df_features.to_csv(output_filename, index=False)
            
            print(f"\n✅ PYTHON FEATURES SAVED: {output_filename}")
            print(f"Columns: {len(df_features.columns)}")
            print(f"Ready for merging with MATLAB G0 results!")
            
            # Show sample patch IDs for verification
            print(f"\nSample patch IDs (for verification with MATLAB):")
            for i in range(min(5, len(patches_data))):
                patch = patches_data[i]
                print(f"  {patch['patch_id']} - coords: ({patch['patch_row']}, {patch['patch_col']})")
            
            return df_features
        else:
            print("❌ No patches extracted. Check your data files.")
            return None


class DatasetMerger:
    """Merge MATLAB G0 results with Python features by patch_id"""
    
    def merge_datasets(self, matlab_g0_file, python_features_file, output_file):
        """
        Merge MATLAB G0 results with Python features
        """
        print("=== DATASET MERGER ===")
        
        # Load datasets
        try:
            matlab_df = pd.read_csv(matlab_g0_file)
            python_df = pd.read_csv(python_features_file)
            
            print(f"MATLAB G0 data: {len(matlab_df)} rows")
            print(f"Python features: {len(python_df)} rows")
            
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            return None
        
        # Merge on patch_id
        try:
            merged_df = pd.merge(matlab_df, python_df, on='patch_id', how='inner')
            
            print(f"✅ Merged dataset: {len(merged_df)} rows")
            print(f"Match rate: {len(merged_df)/min(len(matlab_df), len(python_df))*100:.1f}%")
            
            # Save merged dataset
            merged_df.to_csv(output_file, index=False)
            print(f"✅ Final dataset saved: {output_file}")
            
            # Show column summary
            matlab_cols = [col for col in matlab_df.columns if col not in python_df.columns or col == 'patch_id']
            python_cols = [col for col in python_df.columns if col not in matlab_df.columns or col == 'patch_id']
            
            print(f"\nFinal dataset columns ({len(merged_df.columns)} total):")
            print(f"  From MATLAB: {len(matlab_cols)} (including G0 parameters)")
            print(f"  From Python: {len(python_cols)} (texture, stats, topo)")
            
            return merged_df
            
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return None


if __name__ == "__main__":
    # Initialize extractor with SAME parameters as MATLAB
    extractor = MATLABCompatibleFeatureExtractor(patch_size=32, overlap=0.5)
    
    # File paths - CHANGE THESE TO YOUR ACTUAL PATHS
    sar_file = 'data/SAR_VV_VH_5km_snow_class_image.tif'
    landcover_file = 'data/Landcover_ESA_5km_snow_class_image.tif'
    dem_file = 'data/DEM_SRTM_5km_snow_class_image.tif'
    slope_file = 'data/Slope_degrees_5km_snow_class_image.tif'
    aspect_file = 'data/Aspect_degrees_5km_snow_class_image.tif'
    
    # Extract features (no G0 parameters)
    python_features = extractor.create_dataset(
        sar_file, landcover_file, dem_file, slope_file, aspect_file,
        image_name='SAR_VV_VH_5km_snow_class_image'  # Must match MATLAB image name
    )
    
    # Optional: Merge with MATLAB results if available
    if python_features is not None:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run your MATLAB code to generate: g0_results_matlab.csv")
        print("2. Run merger to combine datasets:")
        print("   merger = DatasetMerger()")
        print("   final_dataset = merger.merge_datasets(")
        print("       'g0_results_matlab.csv',")
        print("       'python_features.csv',") 
        print("       'final_combined_dataset.csv')")
        print("="*60)
        
        # Uncomment if you have MATLAB results ready:
        # merger = DatasetMerger()
        # final_dataset = merger.merge_datasets(
        #     'g0_results_matlab.csv',
        #     'python_features.csv', 
        #     'final_combined_dataset.csv'
        # )
