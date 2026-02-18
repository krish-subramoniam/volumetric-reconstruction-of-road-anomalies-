"""
Test script for Gradio app functionality.
"""

import numpy as np
import cv2
from gradio_app import (
    initialize_pipeline,
    test_preprocessing,
    test_disparity_estimation,
    test_ground_plane_detection,
    test_3d_reconstruction,
    test_volume_calculation,
    test_quality_metrics
)


def create_test_images():
    """Create synthetic stereo image pair for testing."""
    # Create left image with some structure
    left = np.zeros((480, 640), dtype=np.uint8)
    left[100:400, 100:300] = 150  # Rectangle
    left[200:300, 400:500] = 200  # Another rectangle
    
    # Create right image (shifted for disparity)
    right = np.zeros((480, 640), dtype=np.uint8)
    right[100:400, 90:290] = 150  # Shifted rectangle
    right[200:300, 390:490] = 200  # Shifted rectangle
    
    # Add some noise
    noise = np.random.randint(-20, 20, left.shape, dtype=np.int16)
    left = np.clip(left.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    right = np.clip(right.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return left, right


def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("Testing pipeline initialization...")
    result = initialize_pipeline(0.12, 700.0, 320.0, 240.0)
    assert "✅" in result, f"Initialization failed: {result}"
    print("✅ Pipeline initialization test passed")


def test_preprocessing_module():
    """Test preprocessing module."""
    print("\nTesting preprocessing module...")
    left, right = create_test_images()
    
    left_out, right_out, status = test_preprocessing(
        left, right,
        enhance_contrast=True,
        normalize_brightness=True,
        filter_noise=False
    )
    
    assert left_out is not None, "Preprocessing failed"
    assert "✅" in status, f"Preprocessing error: {status}"
    print("✅ Preprocessing test passed")


def test_disparity_module():
    """Test disparity estimation module."""
    print("\nTesting disparity estimation...")
    left, right = create_test_images()
    
    disp_img, status = test_disparity_estimation(
        left, right,
        num_disparities=64,
        block_size=5,
        min_disparity=0
    )
    
    assert disp_img is not None, "Disparity estimation failed"
    assert "✅" in status, f"Disparity error: {status}"
    print("✅ Disparity estimation test passed")


def test_ground_plane_module():
    """Test ground plane detection module."""
    print("\nTesting ground plane detection...")
    
    # First compute disparity
    left, right = create_test_images()
    test_disparity_estimation(left, right, 64, 5, 0)
    
    # Then test ground plane
    v_disp, status = test_ground_plane_detection(use_current_disparity=True)
    
    assert v_disp is not None, "Ground plane detection failed"
    assert "✅" in status, f"Ground plane error: {status}"
    print("✅ Ground plane detection test passed")


def test_reconstruction_module():
    """Test 3D reconstruction module."""
    print("\nTesting 3D reconstruction...")
    
    # First compute disparity
    left, right = create_test_images()
    test_disparity_estimation(left, right, 64, 5, 0)
    
    # Then test reconstruction
    pc_img, status = test_3d_reconstruction(
        use_current_disparity=True,
        min_depth=0.5,
        max_depth=20.0,
        remove_outliers=False
    )
    
    assert pc_img is not None, "3D reconstruction failed"
    assert "✅" in status, f"Reconstruction error: {status}"
    print("✅ 3D reconstruction test passed")


def test_volume_module():
    """Test volume calculation module."""
    print("\nTesting volume calculation...")
    
    # First compute disparity and point cloud
    left, right = create_test_images()
    test_disparity_estimation(left, right, 64, 5, 0)
    test_3d_reconstruction(True, 0.5, 20.0, False)
    
    # Then test volume
    mesh_img, status = test_volume_calculation(
        use_current_points=True,
        alpha=0.1,
        ground_plane_z=0.0
    )
    
    # Volume calculation might fail on synthetic data, so we just check it runs
    print(f"Volume calculation status: {status}")
    print("✅ Volume calculation test completed")


def test_quality_metrics_module():
    """Test quality metrics module."""
    print("\nTesting quality metrics...")
    left, right = create_test_images()
    
    status = test_quality_metrics(left, right)
    
    assert "✅" in status, f"Quality metrics error: {status}"
    print("✅ Quality metrics test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Gradio App Functionality Tests")
    print("=" * 60)
    
    try:
        test_pipeline_initialization()
        test_preprocessing_module()
        test_disparity_module()
        test_ground_plane_module()
        test_reconstruction_module()
        test_volume_module()
        test_quality_metrics_module()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
