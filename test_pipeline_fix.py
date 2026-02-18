"""Test script to verify the pipeline calibration fix."""

import numpy as np
import cv2
from stereo_vision.pipeline import StereoVisionPipeline
from stereo_vision.config import PipelineConfig, CameraConfig
from stereo_vision.calibration import CameraParameters, StereoParameters
from stereo_vision.reconstruction import PointCloudGenerator

def test_synthetic_calibration():
    """Test that synthetic calibration allows pipeline to run."""
    
    # Create test images
    h, w = 480, 640
    left_image = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    right_image = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    
    # Create pipeline
    baseline = 0.12
    focal_length = 700.0
    
    camera_config = CameraConfig(baseline=baseline, focal_length=focal_length)
    config = PipelineConfig(camera=camera_config)
    # Disable WLS filtering as it requires opencv-contrib
    config.wls.enabled = False
    pipeline = StereoVisionPipeline(config)
    
    # Create synthetic calibration
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
    
    # Create Q matrix
    Q = np.array([
        [1, 0, 0, -w/2],
        [0, 1, 0, -h/2],
        [0, 0, 0, focal_length],
        [0, 0, 1/baseline, 0]
    ], dtype=np.float32)
    
    # Create identity rectification maps
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            map_x[y, x] = x
            map_y[y, x] = y
    
    # Create stereo parameters
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
    
    # Set calibration
    pipeline.stereo_params = stereo_params
    pipeline.is_calibrated = True
    
    # Initialize point cloud generator
    pipeline.point_cloud_generator = PointCloudGenerator(
        Q_matrix=Q,
        min_depth=config.depth_range.min_depth,
        max_depth=config.depth_range.max_depth
    )
    
    # Test that pipeline can now process images
    try:
        result = pipeline.process_stereo_pair(left_image, right_image, generate_diagnostics=False)
        print("✓ Pipeline successfully processed stereo pair with synthetic calibration")
        print(f"  - Disparity map shape: {result.disparity_map.shape}")
        print(f"  - V-disparity shape: {result.v_disparity.shape}")
        print(f"  - Anomalies detected: {len(result.anomalies)}")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        return True
    except RuntimeError as e:
        if "Ground plane detection failed" in str(e):
            print("✓ Pipeline runs but ground plane detection failed (expected with random images)")
            print("  This is normal - real stereo images are needed for ground plane detection")
            return True
        else:
            print(f"✗ Pipeline failed: {str(e)}")
            return False
    except Exception as e:
        print(f"✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing synthetic calibration fix...")
    success = test_synthetic_calibration()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
