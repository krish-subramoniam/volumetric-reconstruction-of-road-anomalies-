# Gradio UI for Advanced Stereo Vision Pipeline

Interactive web interface for testing all functionalities of the stereo vision pipeline.

## Features

### 🚀 Full Pipeline Tab
- Run the complete pipeline end-to-end with one click
- Upload stereo image pairs
- Adjust baseline and focal length parameters
- View disparity maps, V-disparity, point clouds, and volume meshes
- Get comprehensive statistics

### 🎨 Preprocessing Tab
- Test CLAHE contrast enhancement
- Brightness normalization between stereo pairs
- Bilateral noise filtering
- Compare before/after results

### 📏 Disparity Estimation Tab
- Compute disparity maps using SGBM algorithm
- Adjust number of disparities, block size, and minimum disparity
- View colored disparity visualization
- Get disparity statistics (min, max, mean, valid pixels)

### 🛣️ Ground Plane Detection Tab
- Generate V-disparity maps
- Detect ground plane parameters
- Visualize V-disparity with heatmap

### 🌐 3D Reconstruction Tab
- Generate 3D point clouds from disparity
- Set depth range filters
- Optional statistical outlier removal
- Interactive 3D visualization

### 📦 Volume Calculation Tab
- Calculate volumes using alpha shape meshing
- Adjust alpha parameter for mesh tightness
- Set ground plane Z coordinate
- Visualize mesh structure

### 📊 Quality Metrics Tab
- Calculate image contrast
- Measure brightness differences
- Compute LRC (Left-Right Consistency) error rate
- Evaluate overall quality

## Installation

1. Install required dependencies:
```bash
pip install gradio matplotlib pillow
```

2. Ensure the stereo vision pipeline is installed:
```bash
pip install -e .
```

## Usage

### Start the Application

```bash
python gradio_app.py
```

The application will start on `http://localhost:7860`

### Using the Interface

1. **Quick Start - Full Pipeline**:
   - Go to the "Full Pipeline" tab
   - Upload left and right stereo images
   - Adjust baseline (default: 0.12m) and focal length (default: 700px)
   - Click "Run Full Pipeline"
   - View all results: disparity, V-disparity, point cloud, and volume mesh

2. **Detailed Testing**:
   - Use individual tabs to test specific modules
   - Upload images in the respective tabs
   - Adjust parameters as needed
   - Click the process button to see results

3. **Sequential Workflow**:
   - Start with "Preprocessing" to enhance images
   - Move to "Disparity Estimation" to compute disparity
   - Use "Ground Plane Detection" to find the ground plane
   - Generate "3D Reconstruction" point cloud
   - Calculate "Volume" from the point cloud
   - Check "Quality Metrics" for evaluation

### Example Workflow

```
1. Upload stereo images → Preprocessing Tab
2. Enhance contrast and normalize brightness
3. Compute disparity → Disparity Estimation Tab
4. Detect ground plane → Ground Plane Detection Tab
5. Generate point cloud → 3D Reconstruction Tab
6. Calculate volume → Volume Calculation Tab
7. Evaluate quality → Quality Metrics Tab
```

## Parameters Guide

### Baseline
- Distance between left and right cameras
- Typical range: 0.05m - 0.5m
- Affects depth accuracy

### Focal Length
- Camera focal length in pixels
- Typical range: 500px - 1500px
- Affects depth scale

### Number of Disparities
- Maximum disparity search range
- Must be divisible by 16
- Higher values = larger depth range but slower

### Block Size
- Window size for matching
- Must be odd number
- Larger = smoother but less detail

### Alpha Parameter
- Controls alpha shape tightness
- Lower values = tighter fit
- Typical range: 0.01 - 0.5

### Depth Range
- Min/Max depth for filtering
- Removes points outside range
- Typical: 0.5m - 20m

## Tips

- **Image Quality**: Use well-lit, textured images for best results
- **Calibration**: Accurate baseline and focal length improve depth accuracy
- **Performance**: Large images may take longer to process
- **Visualization**: Point clouds are sampled to 10,000 points for faster rendering
- **Sequential Processing**: Some tabs use results from previous computations (e.g., ground plane uses disparity)

## Troubleshooting

### "Please compute disparity first"
- Go to "Disparity Estimation" tab and compute disparity before using ground plane or 3D reconstruction

### "Please generate point cloud first"
- Generate a point cloud in "3D Reconstruction" tab before calculating volume

### Slow Performance
- Reduce image resolution
- Decrease number of disparities
- Disable outlier removal for faster processing

### Poor Disparity Quality
- Ensure images are properly calibrated
- Check that images have sufficient texture
- Adjust SGBM parameters (block size, number of disparities)

## Advanced Usage

### Custom Pipeline Configuration

You can modify the pipeline configuration by editing the initialization parameters in the code:

```python
config = PipelineConfig(
    baseline=0.12,
    focal_length=700.0,
    cx=320.0,
    cy=240.0,
    # Add more parameters as needed
)
```

### Saving Results

Results can be saved by right-clicking on images in the interface and selecting "Save image".

### Batch Processing

For batch processing multiple image pairs, consider using the command-line pipeline directly instead of the Gradio interface.

## API Reference

The Gradio app uses the following pipeline modules:

- `stereo_vision.pipeline.StereoVisionPipeline` - Main pipeline
- `stereo_vision.preprocessing.ImagePreprocessor` - Image preprocessing
- `stereo_vision.disparity.SGBMEstimator` - Disparity estimation
- `stereo_vision.ground_plane.VDisparityGenerator` - Ground plane detection
- `stereo_vision.reconstruction.PointCloudGenerator` - 3D reconstruction
- `stereo_vision.volumetric.AlphaShapeGenerator` - Volume calculation
- `stereo_vision.quality_metrics.QualityMetrics` - Quality evaluation

## License

Same as the main stereo vision pipeline project.
