# Advanced Stereo Vision Pipeline

This is a state-of-the-art volumetric reconstruction system for road anomaly detection using stereo vision. The system achieves sub-millimeter precision through rigorous photogrammetric methods, advanced disparity estimation, and watertight mesh-based volume calculation.

## Features

- **CharuCo-based Calibration**: Sub-pixel accuracy calibration with occlusion robustness
- **Advanced Disparity Estimation**: SGBM with LRC validation and WLS filtering
- **V-Disparity Ground Plane Detection**: Automatic road surface modeling
- **3D Point Cloud Reconstruction**: Metric-accurate 3D reconstruction with outlier removal
- **Alpha Shape Meshing**: Tight-fitting concave hull generation
- **Watertight Volume Calculation**: Precise volume computation using Divergence Theorem
- **Quality Metrics**: Comprehensive validation and diagnostic tools

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Calibration

First, calibrate your stereo camera system using CharuCo board images:

```bash
python pothole_volume_pipeline.py calibrate \
    --left-images "calibration/left/*.png" \
    --right-images "calibration/right/*.png" \
    --output calibration.npz
```

**Calibration Tips:**
- Capture at least 15-20 image pairs
- Include various orientations (45° pitch and yaw variations)
- Ensure good lighting and sharp images
- Cover the entire field of view
- Target reprojection error < 0.5 pixels

### 2. Process a Single Stereo Pair

Process a single stereo image pair:

```bash
python pothole_volume_pipeline.py process \
    --left images/left_001.png \
    --right images/right_001.png \
    --calibration calibration.npz \
    --output results/
```

This will generate:
- `disparity.npy`: Computed disparity map
- `diagnostics.png`: Comprehensive diagnostic visualization panel
- `anomalies_0000.json`: Detected anomalies with volume measurements

### 3. Batch Processing

Process multiple stereo pairs:

```bash
python pothole_volume_pipeline.py batch \
    --left-dir images/left/ \
    --right-dir images/right/ \
    --calibration calibration.npz \
    --output results/
```

## Python API

You can also use the pipeline programmatically:

```python
from stereo_vision.pipeline import create_pipeline
import cv2

# Create pipeline with default configuration
pipeline = create_pipeline()

# Load calibration
pipeline.load_calibration('calibration.npz')

# Load stereo pair
left_image = cv2.imread('left.png')
right_image = cv2.imread('right.png')

# Process
result = pipeline.process_stereo_pair(left_image, right_image)

# Access results
for anomaly in result.anomalies:
    print(f"{anomaly.anomaly_type}: {anomaly.volume_liters:.2f} liters")
    print(f"  Uncertainty: ±{anomaly.uncertainty_cubic_meters * 1000:.2f} liters")
    print(f"  Valid: {anomaly.is_valid}")
```

## Configuration

The pipeline can be configured using JSON configuration files:

```python
from stereo_vision.config import PipelineConfig

# Load custom configuration
config = PipelineConfig.load_from_file('config.json')

# Or create programmatically
config = PipelineConfig(
    camera=CameraConfig(baseline=0.12, focal_length=700.0),
    sgbm=SGBMConfig(num_disparities=128, block_size=7),
    depth_range=DepthRangeConfig(min_depth=1.0, max_depth=50.0)
)

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:", errors)
else:
    config.save_to_file('my_config.json')
```

### Pre-configured Profiles

```python
from stereo_vision.config import (
    create_default_config,
    create_high_accuracy_config,
    create_fast_config
)

# High accuracy (slower, more precise)
config = create_high_accuracy_config()

# Fast processing (faster, less precise)
config = create_fast_config()
```

## Pipeline Stages

The pipeline executes the following stages:

1. **Preprocessing**: Contrast enhancement, brightness normalization, noise filtering
2. **Rectification**: Epipolar alignment using calibration parameters
3. **Disparity Estimation**: SGBM → LRC validation → WLS filtering
4. **Ground Plane Detection**: V-Disparity histogram → Hough Transform → Plane model
5. **Anomaly Segmentation**: Classify pixels as potholes or humps
6. **3D Reconstruction**: Disparity to 3D point cloud with outlier removal
7. **Mesh Generation**: Alpha Shape → Boundary detection → Capping
8. **Volume Calculation**: Signed tetrahedron integration
9. **Quality Metrics**: LRC error rate, planarity RMSE, calibration quality

## Output Format

### Anomaly Results (JSON)

```json
{
  "type": "pothole",
  "bounding_box": [120, 340, 85, 62],
  "volume_cubic_meters": 0.0234,
  "volume_liters": 23.4,
  "volume_cubic_cm": 23400.0,
  "uncertainty_cubic_meters": 0.0012,
  "is_valid": true,
  "validation_message": "Volume is within valid constraints",
  "area_square_meters": 0.527,
  "depth_statistics": {
    "min_depth": 2.3,
    "max_depth": 2.8,
    "mean_depth": 2.55,
    "median_depth": 2.54,
    "std_depth": 0.12,
    "num_points": 1247
  }
}
```

### Diagnostic Panel

The diagnostic panel provides a 2×3 grid visualization:

| Original Image | Anomaly Overlay | Disparity Map |
|----------------|-----------------|---------------|
| Ground Plane Fit | V-Disparity | Anomaly Masks |

## Quality Metrics

The pipeline reports several quality metrics:

- **LRC Error Rate**: Percentage of pixels failing left-right consistency (lower is better)
- **Planarity RMSE**: Root mean square error of ground plane fit (lower is better)
- **Calibration Reprojection Error**: RMS reprojection error from calibration (< 0.5 pixels recommended)
- **Temporal Stability**: Coefficient of variation for volume measurements over time (lower is better)

## Troubleshooting

### High LRC Error Rate (> 20%)

- Check image quality (blur, noise, exposure)
- Verify calibration quality
- Adjust SGBM parameters (increase `uniqueness_ratio`)
- Check for occlusions or reflective surfaces

### Ground Plane Detection Fails

- Ensure sufficient road surface is visible
- Check disparity map quality
- Adjust V-Disparity Hough parameters
- Verify depth range configuration

### Volume Calculation Fails

- Check point cloud density (need at least 4 points)
- Adjust Alpha Shape parameter (try 0.5 to 5.0)
- Enable outlier removal
- Verify anomaly size thresholds

### Poor Calibration (RMS > 0.5 pixels)

- Capture more calibration images (20+ recommended)
- Ensure CharuCo board is flat and well-lit
- Include more geometric diversity
- Check for motion blur or focus issues

## Performance

Typical processing times on a modern CPU:

- Calibration: 30-60 seconds (for 20 image pairs)
- Single pair processing: 2-5 seconds
- Batch processing: ~3 seconds per pair

GPU acceleration is not currently implemented but could significantly improve performance.

## Requirements

See `requirements.txt` for complete dependencies:

- OpenCV (with contrib modules for WLS filter)
- NumPy
- SciPy
- Trimesh
- Hypothesis (for property-based testing)
- pytest

## Architecture

The system follows a modular pipeline architecture:

```
stereo_vision/
├── __init__.py
├── pipeline.py          # Main pipeline controller
├── config.py            # Configuration management
├── calibration.py       # CharuCo calibration
├── preprocessing.py     # Image preprocessing
├── disparity.py         # SGBM, LRC, WLS
├── ground_plane.py      # V-Disparity, Hough, segmentation
├── reconstruction.py    # 3D point cloud generation
├── volumetric.py        # Alpha Shape, meshing, volume
└── quality_metrics.py   # Metrics and diagnostics
```

## References

This implementation is based on state-of-the-art computer vision techniques:

- CharuCo boards for robust calibration
- Semi-Global Block Matching (SGBM) for disparity estimation
- V-Disparity representation for ground plane detection
- Alpha Shapes for concave hull generation
- Divergence Theorem for volume calculation

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting pull requests:

```bash
pytest tests/
```

## Support

For issues, questions, or feature requests, please open an issue on the project repository.
