# CNN Patch Dataset Creator

Production-ready script to generate CNN-ready datasets of 32×32 patches for landcover classification using ResNet-50.

## Quick Start

### Basic Usage
```bash
python create_cnn_patch_dataset.py --excel Combined_dataset.xlsx --sar_dir sar_images/ --output_dir cnn_dataset/
```

### Perfect Match Mode (Recommended)
```bash
python create_cnn_patch_dataset.py --excel Combined_dataset.xlsx --sar_dir sar_images/ --output_dir cnn_dataset/ --filter
```

## Features

- ✅ **Exact 1:1 Correspondence**: Every Excel row creates exactly one patch file
- ✅ **Automatic Data Cleaning**: Handles NaN values and invalid coordinates
- ✅ **Perfect Match Mode**: `--filter` option removes problematic data for 100% success rate
- ✅ **Dual-Band SAR Support**: Handles VV and VH polarizations automatically
- ✅ **Class Organization**: Organizes patches into folders by landcover class
- ✅ **Robust Error Handling**: Gracefully handles missing images and out-of-bounds coordinates
- ✅ **Comprehensive Logging**: Detailed progress tracking and statistics

## Input Requirements

### Excel File
Must contain these columns:
- `image_name`: SAR image filename (without extension)
- `patch_row`: Row coordinate (0-based index)
- `patch_col`: Column coordinate (0-based index)  
- `landcover_class`: Class label for organizing output folders

### SAR Images Directory
- Contains dual-polarized SAR images (VV and VH bands)
- Supported formats: `.tif`, `.tiff`
- Images should be named to match the `image_name` column in Excel
- Each image should have 2 bands: Band 1 = VV, Band 2 = VH

## Output Structure

```
cnn_dataset/
├── Forest/
│   ├── SAR_image1_r0000_c0000.tif
│   ├── SAR_image1_r0016_c0032.tif
│   └── ...
├── Water/
│   ├── SAR_image1_r0000_c0016.tif
│   └── ...
└── Bare_Soil/
    ├── SAR_image2_r0032_c0000.tif
    └── ...
```

## Command Line Options

```bash
python create_cnn_patch_dataset.py [OPTIONS]

Required:
  --excel EXCEL_FILE        Path to Excel file containing patch metadata
  --sar_dir SAR_DIRECTORY   Directory containing SAR images
  --output_dir OUTPUT_DIR   Output directory for CNN dataset

Optional:
  --patch_size SIZE         Patch size in pixels (default: 32)
  --stride STRIDE          Stride used in original extraction (default: 16)
  --filter                 Filter out invalid coordinates for perfect match
  --verify                 Verify dataset after creation
```

## Examples

### Standard Extraction
```bash
python create_cnn_patch_dataset.py \
    --excel Combined_dataset.xlsx \
    --sar_dir sar_images \
    --output_dir cnn_dataset
```

### Perfect Match with Filtering
```bash
python create_cnn_patch_dataset.py \
    --excel Combined_dataset.xlsx \
    --sar_dir sar_images \
    --output_dir cnn_dataset_perfect \
    --filter
```

## Output Statistics

The script provides comprehensive statistics:

```
=== FINAL EXTRACTION STATISTICS ===
Total patches requested: 7,562
Total patches created: 7,562
Total patches skipped: 0
Success rate: 100.0%

Per-class statistics:
  Forest: Requested: 3023, Created: 3023, Success rate: 100.0%
  Water: Requested: 1405, Created: 1405, Success rate: 100.0%
  ...
```

## Integration with CNN Training

The output patches are ready for direct use with PyTorch, TensorFlow, or any CNN framework:

### PyTorch Example
```python
from torch.utils.data import Dataset
import rasterio
import torch

class SARPatchDataset(Dataset):
    def __init__(self, dataset_dir):
        self.patches = []
        self.labels = []
        
        for class_dir in Path(dataset_dir).iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for patch_file in class_dir.glob('*.tif'):
                    self.patches.append(patch_file)
                    self.labels.append(class_name)
    
    def __getitem__(self, idx):
        with rasterio.open(self.patches[idx]) as src:
            patch = src.read([1, 2]).astype(np.float32)  # VV, VH
        return torch.tensor(patch), self.labels[idx]
```

## Requirements

```bash
pip install pandas rasterio numpy pathlib
```

## Troubleshooting

### Common Issues

1. **"Missing required columns" error**
   - Check Excel column names: `image_name`, `patch_row`, `patch_col`, `landcover_class`

2. **"SAR image not found" warnings**
   - Verify SAR image filenames match Excel `image_name` values
   - Check file extensions (.tif, .tiff)

3. **"Out of bounds" errors**
   - Use `--filter` option to automatically remove invalid coordinates
   - Verify patch coordinates are valid for image dimensions

### Performance Tips

- Use `--filter` option for guaranteed 100% success rate
- Use SSD storage for faster I/O when processing many patches
- Process one dataset at a time for optimal memory usage

## Files

- `create_cnn_patch_dataset.py` - Main extraction script (production-ready)
- `CNN_PATCH_DATASET_README.md` - Detailed documentation
- `requirements.txt` - Python dependencies

Ready for CNN training with ResNet-50 or any other architecture!