"""
Test script to verify volume calculation fix with synthetic stereo images.
"""

import numpy as np
import cv2
from stereo_vision.pipeline import StereoVisionPipeline
from stereo_vision.config import PipelineConfig, CameraConfig
from stereo_vision.calibration import CameraParameters, StereoParameters

def create_synthetic_stereo_pair_with_anomaly():
    """Create synthetic stereo images with a clear anomaly (depression)."""
    height, width = 480, 640
    baseline = 0.12
    focal_length = 700.0
    
    # Create left image with a dark depression (pothole)
    left_img = np.ones((height, width), dtype=np.uint8) * 128
    
    # Add texture for better matching
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            left_img[i:i+2, j:j+2] = 200
    
    # Create a depression (darker region) in the center
    center_y, center_x = height // 2, width // 2
    depression_size = 80
    y1, y2 = center_y - depression_size//2, center_y + depression_size//2
    x1, x2 = center_x - depression_size//2, center_x + depression_size//2
    
    # Make depression darker (simulating depth change)
    left_img[y1:y2, x1:x2] = 80
    
    # Create right image with horizontal shift (disparity)
    right_img = np.ones((height, width), dtype=np.uint8) * 128
    
    # Add same texture pattern
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            right_img[i:i+2, j:j+2] = 200
    
    # Shift the depression to create disparity
    # Larger disparity = closer object, smaller disparity = farther
    # For depression, we want it to appear farther (smaller disparity)
    disparity_shift = 15  # pixels - depression is farther
    background_shift = 20  # pixels - background is closer
    
    # Shift background
    right_img[:, background_shift:] = left_img[:, :-background_shift]
    
    # Shift depression with different disparity
    depression_right = left_img[y1:y2, x1:x2]
    if x2 + disparity_shift < width:
        right_img[y1:y2, x1+disparity_shift:x2+disparity_shift] = depression_right
    
    return left_img, right_img, baseline, focal_length


def test_volume_calculation():
    """Test the volume calculation with synthetic data."""
    print("Creating synthetic stereo pair with anomaly...")
    left_img, right_img, baseline, focal_length = create_synthetic_stereo_pair_with_anomaly()
    
    print(f"Image size: {left_img.shape}")
    print(f"Baseline: {baseline}m, Focal length: {focal_length}px")
    
    # Save images for inspection
    cv2.imwrite("test_left_synthetic.png", left_img)
    cv2.imwrite("test_right_synthetic.png", right_img)
    print("Saved test images: test_left_synthetic.png, test_right_synthetic.png")
    
    # Create pipeline
    camera_config = CameraConfig(baseline=baseline, focal_length=focal_length)
    config = PipelineConfig(camera=camera_config)
    config.wls.enabled = False  # Disable WLS
    
    # Adjust depth range for synthetic data
    config.depth_range.min_depth = 1.0  # Keep default
    config.depth_range.max_depth = 100.0
    
    pipeline = StereoVisionPipeline(config)
    
    # Create synthetic calibration
    h, w = left_img.shape
    camera_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    distortion = np.zeros(5, dtype=np.float32)
    
    left_cam = CameraParameters(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion,
        reprojection_error=0.0,
        image_size=(w, h)
    )
    
    right_cam = CameraParameters(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion,
        reprojection_error=0.0,
        image_size=(w, h)
    )
    
    Q = np.array([
        [1, 0, 0, -w/2],
        [0, 1, 0, -h/2],
        [0, 0, 0, focal_length],
        [0, 0, 1/baseline, 0]
    ], dtype=np.float32)
    
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            map_x[y, x] = x
            map_y[y, x] = y
    
    stereo_params = StereoParameters(
        left_camera=left_cam,
        right_camera=right_cam,
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation_vector=np.array([[baseline], [0], [0]], dtype=np.float32),
        baseline=baseline,
        Q_matrix=Q,
        rectification_maps_left=(map_x, map_y),
        rectification_maps_right=(map_x, map_y)
    )
    
    pipeline.stereo_params = stereo_params
    pipeline.is_calibrated = True
    
    from stereo_vision.reconstruction import PointCloudGenerator
    pipeline.point_cloud_generator = PointCloudGenerator(
        Q_matrix=Q,
        min_depth=config.depth_range.min_depth,
        max_depth=config.depth_range.max_depth
    )
    
    print("\nProcessing stereo pair...")
    try:
        result = pipeline.process_stereo_pair(left_img, right_img, generate_diagnostics=False)
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Anomalies detected: {len(result.anomalies)}")
        
        if result.anomalies:
            total_volume = sum(a.volume_cubic_meters for a in result.anomalies if a.is_valid)
            print(f"Total volume: {total_volume:.6f} m³ ({total_volume*1000:.2f} liters)")
            
            for i, anomaly in enumerate(result.anomalies):
                print(f"\nAnomaly {i+1} ({anomaly.anomaly_type}):")
                print(f"  Bounding box: {anomaly.bounding_box}")
                print(f"  Point cloud size: {anomaly.point_cloud.shape[0]} points")
                print(f"  Volume: {anomaly.volume_cubic_meters:.6f} m³ ({anomaly.volume_liters:.2f} L)")
                print(f"  Valid: {anomaly.is_valid}")
                print(f"  Message: {anomaly.validation_message}")
                if anomaly.point_cloud.shape[0] > 0:
                    print(f"  Depth range: {anomaly.point_cloud[:, 2].min():.3f}m - {anomaly.point_cloud[:, 2].max():.3f}m")
        else:
            print("No anomalies detected!")
            
        # Check disparity map
        valid_disp = result.disparity_map[result.disparity_map > 0]
        if len(valid_disp) > 0:
            print(f"\nDisparity map stats:")
            print(f"  Valid pixels: {len(valid_disp)} ({100*len(valid_disp)/result.disparity_map.size:.1f}%)")
            print(f"  Range: {valid_disp.min():.2f} - {valid_disp.max():.2f}")
        else:
            print("\nWARNING: No valid disparity values!")
        
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_volume_calculation()
    
    if result and result.anomalies:
        print("✓ Volume calculation test PASSED - anomalies detected with volumes")
    else:
        print("✗ Volume calculation test FAILED - no valid volumes calculated")
