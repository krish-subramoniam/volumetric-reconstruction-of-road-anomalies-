# Gradio App Calibration Fix - Summary

## Problem
The Gradio app was failing when users tried to process stereo images with the error:
```
Pipeline not calibrated. Run calibrate() or load_calibration() first.
```

This happened because the `StereoVisionPipeline.process_stereo_pair()` method requires the pipeline to be calibrated before it can process images.

## Solution
Added synthetic calibration data initialization in the `run_full_pipeline()` function to bypass the calibration requirement for testing purposes.

### Changes Made

#### 1. Synthetic Calibration Creation (`gradio_app.py` lines 30-115)
Created synthetic camera calibration parameters:
- Identity camera matrices with user-specified focal length
- Zero distortion coefficients
- Identity rectification maps (no rectification needed)
- Q matrix for 3D reprojection
- Stereo parameters with baseline and rotation/translation

#### 2. WLS Filter Disabled
Disabled WLS filtering by default since it requires `opencv-contrib`:
```python
config.wls.enabled = False
```

#### 3. Error Handling
Added graceful error handling for ground plane detection failures:
- Provides helpful error message when ground plane detection fails
- Explains common causes (no ground plane, not stereo pairs, insufficient features)
- Suggests using real stereo images

#### 4. Result Handling
Updated result handling to work with `PipelineResult` dataclass:
- Access `result.disparity_map` instead of `result['disparity']`
- Access `result.anomalies` list for detected anomalies
- Handle cases where no anomalies are detected

## Testing
Created `test_pipeline_fix.py` to verify the fix:
- Creates synthetic stereo images
- Initializes pipeline with synthetic calibration
- Verifies pipeline can process images without calibration files
- Handles expected ground plane detection failures gracefully

Test result: ✓ All tests passed!

## Usage Notes

### For Testing Without Real Calibration
The Gradio app now works without requiring actual calibration files. Users can:
1. Upload any stereo image pair
2. Adjust baseline and focal length parameters
3. Run the pipeline

### For Production Use
For accurate results with real stereo cameras:
1. Perform proper calibration using CharuCo boards
2. Save calibration with `pipeline.save_calibration('calibration.npz')`
3. Load calibration with `pipeline.load_calibration('calibration.npz')`

### Limitations of Synthetic Calibration
- No lens distortion correction
- No proper stereo rectification
- Assumes ideal pinhole camera model
- May produce less accurate depth measurements
- Ground plane detection may fail on non-ideal images

### When Ground Plane Detection Fails
The app will show a helpful error message:
```
Ground plane detection failed.

This typically happens when:
- Images don't contain a clear ground plane
- Images are not proper stereo pairs
- Images have insufficient texture/features

Please try with real stereo images that show a road or ground surface.
```

## Files Modified
1. `gradio_app.py` - Added synthetic calibration and error handling
2. `test_pipeline_fix.py` - Created test script to verify the fix

## Dependencies
The fix works with the standard `opencv-python` package and does not require `opencv-contrib-python`.

## Running the App
```bash
.\venv\Scripts\python.exe gradio_app.py
```

The app will be available at: http://localhost:7860

## Next Steps
For users who want full pipeline functionality:
1. Obtain proper stereo camera calibration
2. Use real stereo image pairs with visible ground planes
3. Consider installing `opencv-contrib-python` for WLS filtering
