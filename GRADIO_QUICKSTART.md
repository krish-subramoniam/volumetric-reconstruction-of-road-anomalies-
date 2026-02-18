# Gradio UI Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Gradio
```bash
pip install gradio
```

### Step 2: Launch the UI
```bash
python launch_gradio.py
```

Or directly:
```bash
python gradio_app.py
```

### Step 3: Open Your Browser
Navigate to: **http://localhost:7860**

---

## ⚠️ Important Notes

### Calibration
The app now includes **synthetic calibration** for testing purposes:
- Upload any stereo image pair to test
- Adjust baseline (0.01-1.0m) and focal length (100-2000px) parameters
- No calibration files required for basic testing

For production use with real stereo cameras, perform proper calibration using CharuCo boards.

### Known Limitations
- **WLS Filtering**: Disabled by default (requires opencv-contrib-python)
- **Ground Plane Detection**: May fail on images without clear ground planes or non-stereo pairs
- **Synthetic Calibration**: Provides approximate results; use real calibration for accuracy

---

## 📸 Quick Test with Sample Images

### Option 1: Use Your Own Stereo Images
- Prepare left and right stereo image pairs
- Images should be rectified and calibrated
- Supported formats: JPG, PNG, BMP

### Option 2: Generate Test Images
```python
import numpy as np
import cv2

# Create simple test pattern
left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
right = np.roll(left, 10, axis=1)  # Shift for disparity

cv2.imwrite('test_left.png', left)
cv2.imwrite('test_right.png', right)
```

---

## 🎯 5-Minute Tutorial

### Full Pipeline Demo

1. **Click on "Full Pipeline" tab**

2. **Upload Images**
   - Click "Left Image" → Upload your left stereo image
   - Click "Right Image" → Upload your right stereo image

3. **Set Parameters**
   - Baseline: 0.12m (distance between cameras)
   - Focal Length: 700px (camera focal length)

4. **Run Pipeline**
   - Click "🚀 Run Full Pipeline"
   - Wait for processing (5-30 seconds depending on image size)

5. **View Results**
   - Disparity Map: Color-coded depth information
   - V-Disparity: Ground plane visualization
   - Point Cloud: 3D reconstruction
   - Volume Mesh: 3D volume calculation
   - Statistics: Detailed metrics

---

## 🔧 Module-by-Module Testing

### Test Preprocessing
1. Go to "🎨 Preprocessing" tab
2. Upload stereo images
3. Enable/disable:
   - Contrast Enhancement (CLAHE)
   - Brightness Normalization
   - Noise Filtering (Bilateral)
4. Click "Process Images"
5. Compare before/after results

### Test Disparity Estimation
1. Go to "📏 Disparity Estimation" tab
2. Upload stereo images
3. Adjust parameters:
   - Number of Disparities: 128 (higher = more depth range)
   - Block Size: 5 (larger = smoother)
   - Min Disparity: 0
4. Click "Compute Disparity"
5. View colored disparity map and statistics

### Test Ground Plane Detection
1. First compute disparity (previous step)
2. Go to "🛣️ Ground Plane Detection" tab
3. Check "Use Current Disparity Map"
4. Click "Detect Ground Plane"
5. View V-disparity heatmap and plane parameters

### Test 3D Reconstruction
1. Ensure disparity is computed
2. Go to "🌐 3D Reconstruction" tab
3. Set depth range:
   - Min Depth: 0.5m
   - Max Depth: 20.0m
4. Enable "Remove Outliers" for cleaner results
5. Click "Generate Point Cloud"
6. View 3D visualization

### Test Volume Calculation
1. Ensure point cloud is generated
2. Go to "📦 Volume Calculation" tab
3. Set alpha parameter: 0.1 (lower = tighter fit)
4. Set ground plane Z: 0.0m
5. Click "Calculate Volume"
6. View mesh and volume in m³ and liters

### Test Quality Metrics
1. Go to "📊 Quality Metrics" tab
2. Upload stereo images
3. Click "Calculate Metrics"
4. View:
   - Image contrast
   - Brightness difference
   - LRC error rate

---

## 💡 Tips for Best Results

### Image Quality
- ✅ Use well-lit images
- ✅ Ensure good texture (avoid plain surfaces)
- ✅ Use calibrated cameras
- ❌ Avoid motion blur
- ❌ Avoid overexposed/underexposed regions

### Parameter Tuning

**Baseline**
- Smaller (0.05-0.1m): Better for close objects
- Larger (0.2-0.5m): Better for distant objects

**Number of Disparities**
- 64: Fast, limited depth range
- 128: Balanced (recommended)
- 256: Slow, maximum depth range

**Block Size**
- 3-5: More detail, more noise
- 7-11: Smoother, less detail
- Must be odd number

**Alpha Parameter**
- 0.01-0.05: Very tight fit (may miss data)
- 0.1-0.2: Balanced (recommended)
- 0.3-0.5: Loose fit (includes more points)

---

## 🐛 Troubleshooting

### "Please compute disparity first"
**Solution**: Go to "Disparity Estimation" tab and compute disparity before using other modules

### "Please generate point cloud first"
**Solution**: Generate point cloud in "3D Reconstruction" tab before calculating volume

### Poor Disparity Quality
**Solutions**:
- Increase image contrast (use preprocessing)
- Adjust SGBM parameters
- Ensure images are properly aligned
- Check camera calibration

### Slow Performance
**Solutions**:
- Reduce image resolution (resize before upload)
- Decrease number of disparities
- Disable outlier removal
- Use smaller alpha parameter

### Volume Calculation Fails
**Solutions**:
- Ensure sufficient points in point cloud (>100)
- Adjust alpha parameter
- Check ground plane Z coordinate
- Verify point cloud quality

---

## 📊 Understanding the Results

### Disparity Map
- **Brighter colors** = Closer objects
- **Darker colors** = Farther objects
- **Black regions** = No disparity (occlusions)

### V-Disparity
- **Bright diagonal line** = Ground plane
- **Horizontal lines** = Obstacles
- **Vertical axis** = Image row
- **Horizontal axis** = Disparity value

### Point Cloud
- **X-axis** = Left-right position
- **Y-axis** = Up-down position
- **Z-axis** = Depth (distance from camera)
- **Color** = Depth value

### Volume Mesh
- **Vertices** = Mesh points
- **Faces** = Triangular surfaces
- **Volume** = Enclosed space in m³

---

## 🎓 Example Workflows

### Workflow 1: Quick Volume Measurement
```
1. Upload images → Full Pipeline tab
2. Set baseline and focal length
3. Click "Run Full Pipeline"
4. Read volume from statistics
```

### Workflow 2: Quality Analysis
```
1. Upload images → Quality Metrics tab
2. Calculate metrics
3. If quality is poor → Preprocessing tab
4. Enhance and reprocess
```

### Workflow 3: Parameter Optimization
```
1. Upload images → Disparity Estimation tab
2. Try different parameters
3. Find best disparity quality
4. Use those parameters in Full Pipeline
```

---

## 🔗 Next Steps

- Read the full documentation: `GRADIO_APP_README.md`
- Explore the pipeline code: `stereo_vision/` directory
- Run automated tests: `python test_gradio_app.py`
- Check benchmarking guide: `BENCHMARKING_GUIDE.md`

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages in the status boxes
3. Verify all dependencies are installed
4. Check that images are valid stereo pairs

---

**Happy Testing! 🎉**
