# AMACFA+ (Automatic Macroscopic Fabric and Orientation Analysis - Enhanced)

An enhanced Python package for automated analysis of ceramic sherd fabric, including inclusion detection, void analysis, orientation measurements, and color characterization.

## Features

- **Inclusion Detection**: Robust blob detection for ceramic inclusions with adaptive parameters
- **Void Analysis**: Complete dark blob detection for void identification and measurement  
- **Edge Detection**: Improved automatic thresholding for optimal edge detection
- **Size Analysis**: DPI-aware area calculations with proper unit conversion
- **Color Analysis**: HSV color space analysis for ceramic characterization
- **Orientation Analysis**: Inclusion orientation measurements
- **Interactive Visualization**: Tools for exploring individual inclusions

## Installation

### From Source

```bash
git clone https://github.com/your-username/amacfa-plus.git
cd amacfa-plus
pip install -e .
```

### Dependencies

- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Pandas  
- Matplotlib
- SciPy
- scikit-image
- Pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
import cv2
from amacfa_plus import sherd_mask, super_zorro_cv, sherd_blobs, size_count_summary

# Load ceramic sherd image
image = cv2.imread('ceramic_sherd.jpg')

# Create mask to isolate sherd from background
mask = sherd_mask(image, scan_dpi=600)

# Apply mask
masked_image = super_zorro_cv(image, mask)

# Detect inclusions and voids
blobs_light, blobs_dark = sherd_blobs(masked_image, scan_dpi=600)

# Analyze size distributions
size_summary = size_count_summary(blobs_light, blobs_dark, scan_dpi=600)

print(f"Found {len(blobs_light)} inclusions and {len(blobs_dark)} voids")
```

## Core Functions

### Detection Module (`amacfa_plus.core.detection`)

- `sherd_mask(image, scan_dpi)` - Create binary mask to isolate ceramic sherd
- `super_zorro_cv(image, mask)` - Apply mask with morphological operations
- `sherd_blobs(image, scan_dpi)` - Detect inclusions and voids using blob detection
- `setup_robust_blob_params(image, scan_dpi, blob_type)` - Create adaptive blob detector parameters

### Analysis Module (`amacfa_plus.core.analysis`)

- `size_count_summary(blobs_light, blobs_dark, scan_dpi)` - Analyze size distributions
- `void_counter(image, scan_dpi)` - Count and measure voids
- `contour_counter(image, scan_dpi)` - Contour-based feature counting
- `inclusion_orientation(image)` - Measure inclusion orientations
- `sherd_color_summary(image, mask)` - Analyze overall sherd color characteristics

### Visualization Module (`amacfa_plus.core.visualization`)

- `sacredsquare(image, blobs)` - Extract squares representing inclusions
- `inclusion_colors(image, inclusion_list)` - Extract color information for inclusions
- `inclusion_viewer(sq_lst, img_color)` - Interactive viewer for individual inclusions

## Examples

See the `examples/` directory for complete usage examples:

- `basic_analysis.py` - Complete analysis pipeline with visualization

## Key Improvements in AMACFA+

### Bug Fixes
- Fixed critical variable naming errors that prevented execution
- Corrected mathematical calculation errors in size analysis
- Fixed blob area calculations (were 4x too large due to diameter/radius confusion)
- Standardized DPI conversion throughout codebase

### Enhanced Features
- **Complete Dark Blob Analysis**: Previously disabled void detection now fully functional
- **Improved Edge Detection**: Automatic threshold selection with DPI-aware parameters
- **Robust Blob Parameters**: Adaptive parameter calculation based on image characteristics
- **Better Error Handling**: Input validation and graceful error recovery
- **DPI Standardization**: Consistent DPI-based scaling throughout all functions

### Mathematical Accuracy
- Verified and corrected all area calculation formulas
- Proper pixel-to-physical unit conversion (scan_dpi * 0.3937)
- Fixed blob size calculations: `π * ((diameter/2)/dpcm)²`
- Consistent statistical analysis with proper variable handling

## DPI and Units

AMACFA+ uses DPI (dots per inch) for accurate physical measurements:

- Default scan DPI: 600
- Conversion factor: `dpcm = scan_dpi * 0.3937` (dots per cm)
- Area calculations return values in cm²
- All measurements are DPI-aware and scale appropriately

## Scientific Applications

AMACFA+ is designed for archaeological and materials science applications involving:

- Ceramic fabric analysis
- Inclusion size and distribution studies
- Void analysis in ceramic materials
- Orientation analysis of inclusions
- Color characterization of ceramic sherds
- Quantitative petrographic analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

 https://github.com/your-username/amacfa-plus
```

## Support

For issues and questions:
- Check the examples in `examples/`
- Review function docstrings for detailed parameter information
- Submit issues on GitHub