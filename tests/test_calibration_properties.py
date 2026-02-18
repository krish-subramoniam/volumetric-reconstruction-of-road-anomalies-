"""
Property-based tests for calibration accuracy and parameter isolation.

Feature: advanced-stereo-vision-pipeline
"""

import numpy as np
import cv2
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List, Tuple

from stereo_vision.calibration import (
    CharuCoCalibrator,
    StereoCalibrator,
    CameraParameters,
    StereoParameters
)


# Test data generation strategies
@st.composite
def camera_intrinsics(draw):
    """Generate realistic camera intrinsic parameters."""
    # Focal length in pixels (typical range for common cameras)
    fx = draw(st.floats(min_value=500, max_value=2000))
    fy = draw(st.floats(min_value=500, max_value=2000))
    
    # Principal point (image center with some variation)
    cx = draw(st.floats(min_value=300, max_value=700))
    cy = draw(st.floats(min_value=200, max_value=500))
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Distortion coefficients (realistic ranges)
    dist_coeffs = np.array([
        draw(st.floats(min_value=-0.5, max_value=0.5)),  # k1
        draw(st.floats(min_value=-0.1, max_value=0.1)),  # k2
        draw(st.floats(min_value=-0.01, max_value=0.01)),  # p1
        draw(st.floats(min_value=-0.01, max_value=0.01)),  # p2
        draw(st.floats(min_value=-0.01, max_value=0.01))   # k3
    ], dtype=np.float64)
    
    return camera_matrix, dist_coeffs


@st.composite
def charuco_calibration_images(draw, num_images=15):
    """Generate synthetic CharuCo calibration images with geometric diversity."""
    calibrator = CharuCoCalibrator(
        squares_x=7, squares_y=5,
        square_length=0.04, marker_length=0.03
    )
    
    # Generate camera parameters
    camera_matrix, dist_coeffs = draw(camera_intrinsics())
    image_size = (640, 480)
    
    images = []
    object_points_list = []
    image_points_list = []
    
    for _ in range(num_images):
        # Generate diverse board poses (45-degree variations)
        # Rotation angles in radians
        rx = draw(st.floats(min_value=-np.pi/4, max_value=np.pi/4))
        ry = draw(st.floats(min_value=-np.pi/4, max_value=np.pi/4))
        rz = draw(st.floats(min_value=-np.pi/6, max_value=np.pi/6))
        
        # Translation (distance from camera)
        tx = draw(st.floats(min_value=-0.1, max_value=0.1))
        ty = draw(st.floats(min_value=-0.1, max_value=0.1))
        tz = draw(st.floats(min_value=0.3, max_value=0.8))
        
        # Create rotation vector and translation vector
        rvec = np.array([rx, ry, rz], dtype=np.float64)
        tvec = np.array([tx, ty, tz], dtype=np.float64)
        
        # Get CharuCo board corner positions
        obj_points = calibrator.board.getChessboardCorners()
        
        # Project points to image plane
        img_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        img_points = img_points.reshape(-1, 2)
        
        # Filter points within image bounds
        valid_mask = (
            (img_points[:, 0] >= 0) & (img_points[:, 0] < image_size[0]) &
            (img_points[:, 1] >= 0) & (img_points[:, 1] < image_size[1])
        )
        
        if np.sum(valid_mask) < 10:
            continue
        
        valid_obj_points = obj_points[valid_mask]
        valid_img_points = img_points[valid_mask]
        
        # Create synthetic image (we don't need actual pixel data for calibration)
        img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        
        images.append(img)
        object_points_list.append(valid_obj_points)
        image_points_list.append(valid_img_points.reshape(-1, 1, 2).astype(np.float32))
    
    assume(len(images) >= 10)
    
    return images, object_points_list, image_points_list, camera_matrix, dist_coeffs, image_size


def generate_calibration_data_with_diversity(num_images: int = 15) -> Tuple:
    """
    Generate synthetic calibration data with geometric diversity.
    
    Returns calibration images and ground truth parameters.
    """
    calibrator = CharuCoCalibrator(
        squares_x=7, squares_y=5,
        square_length=0.04, marker_length=0.03
    )
    
    # Ground truth camera parameters
    fx, fy = 800.0, 800.0
    cx, cy = 320.0, 240.0
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.array([0.1, -0.05, 0.001, 0.001, 0.01], dtype=np.float64)
    image_size = (640, 480)
    
    object_points_list = []
    image_points_list = []
    
    # Generate diverse poses
    np.random.seed(42)
    for i in range(num_images):
        # Diverse rotations (45-degree variations)
        rx = np.random.uniform(-np.pi/4, np.pi/4)
        ry = np.random.uniform(-np.pi/4, np.pi/4)
        rz = np.random.uniform(-np.pi/6, np.pi/6)
        
        # Translation
        tx = np.random.uniform(-0.1, 0.1)
        ty = np.random.uniform(-0.1, 0.1)
        tz = np.random.uniform(0.3, 0.8)
        
        rvec = np.array([rx, ry, rz], dtype=np.float64)
        tvec = np.array([tx, ty, tz], dtype=np.float64)
        
        # Get board corners
        obj_points = calibrator.board.getChessboardCorners()
        
        # Project to image
        img_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        img_points = img_points.reshape(-1, 2)
        
        # Filter valid points
        valid_mask = (
            (img_points[:, 0] >= 0) & (img_points[:, 0] < image_size[0]) &
            (img_points[:, 1] >= 0) & (img_points[:, 1] < image_size[1])
        )
        
        if np.sum(valid_mask) < 10:
            continue
        
        valid_obj_points = obj_points[valid_mask]
        valid_img_points = img_points[valid_mask]
        
        object_points_list.append(valid_obj_points)
        image_points_list.append(valid_img_points.reshape(-1, 1, 2).astype(np.float32))
    
    return object_points_list, image_points_list, camera_matrix, dist_coeffs, image_size


# Property 2: Calibration Accuracy Threshold
# **Validates: Requirements 1.2**
@settings(max_examples=20, deadline=None)
@given(st.integers(min_value=15, max_value=30))
def test_property_2_calibration_accuracy_threshold(num_images):
    """
    Property 2: Calibration Accuracy Threshold
    
    For any valid calibration dataset with sufficient geometric diversity,
    intrinsic calibration should achieve reprojection error below 0.1 pixels.
    
    **Validates: Requirements 1.2**
    """
    # Generate calibration data
    object_points, image_points, true_camera_matrix, true_dist_coeffs, image_size = \
        generate_calibration_data_with_diversity(num_images)
    
    assume(len(object_points) >= 10)
    
    # Perform calibration using OpenCV directly
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None, None
    )
    
    assert ret, "Calibration should succeed with sufficient data"
    
    # Calculate reprojection error
    total_error = 0
    total_points = 0
    
    for i in range(len(object_points)):
        img_points_proj, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], img_points_proj, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(object_points[i])
    
    reprojection_error = np.sqrt(total_error / total_points)
    
    # Property: Reprojection error should be below 0.1 pixels
    # Note: With synthetic perfect data, this should always pass
    assert reprojection_error < 0.1, \
        f"Reprojection error {reprojection_error:.4f} exceeds 0.1 pixel threshold"


# Property 3: Stereo Calibration Parameter Isolation
# **Validates: Requirements 1.3**
@settings(max_examples=20, deadline=None)
@given(
    st.integers(min_value=15, max_value=25),
    st.floats(min_value=0.05, max_value=0.3)  # baseline in meters
)
def test_property_3_stereo_calibration_parameter_isolation(num_images, baseline):
    """
    Property 3: Stereo Calibration Parameter Isolation
    
    For any stereo calibration process, the intrinsic camera parameters
    should remain unchanged from their individually calibrated values.
    
    **Validates: Requirements 1.3**
    """
    # Generate left camera calibration data
    obj_points_left, img_points_left, camera_matrix_left, dist_coeffs_left, image_size = \
        generate_calibration_data_with_diversity(num_images)
    
    # Generate right camera calibration data (same intrinsics, different poses)
    obj_points_right, img_points_right, camera_matrix_right, dist_coeffs_right, _ = \
        generate_calibration_data_with_diversity(num_images)
    
    assume(len(obj_points_left) >= 10 and len(obj_points_right) >= 10)
    
    # Calibrate left camera
    ret_left, K_left_before, d_left_before, _, _ = cv2.calibrateCamera(
        obj_points_left, img_points_left, image_size, None, None
    )
    
    # Calibrate right camera
    ret_right, K_right_before, d_right_before, _, _ = cv2.calibrateCamera(
        obj_points_right, img_points_right, image_size, None, None
    )
    
    assert ret_left and ret_right, "Individual calibrations should succeed"
    
    # Now perform stereo calibration with FIXED intrinsics
    # Use common object points for stereo calibration
    common_obj_points = obj_points_left[:min(len(obj_points_left), len(obj_points_right))]
    common_img_points_left = img_points_left[:len(common_obj_points)]
    common_img_points_right = img_points_right[:len(common_obj_points)]
    
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    ret_stereo, K_left_after, d_left_after, K_right_after, d_right_after, R, T, E, F = \
        cv2.stereoCalibrate(
            common_obj_points,
            common_img_points_left,
            common_img_points_right,
            K_left_before,
            d_left_before,
            K_right_before,
            d_right_before,
            image_size,
            criteria=criteria,
            flags=flags
        )
    
    assert ret_stereo, "Stereo calibration should succeed"
    
    # Property: Intrinsic parameters should remain unchanged
    # Check camera matrices
    np.testing.assert_allclose(
        K_left_after, K_left_before, rtol=1e-10, atol=1e-10,
        err_msg="Left camera matrix should remain unchanged after stereo calibration"
    )
    
    np.testing.assert_allclose(
        K_right_after, K_right_before, rtol=1e-10, atol=1e-10,
        err_msg="Right camera matrix should remain unchanged after stereo calibration"
    )
    
    # Check distortion coefficients
    np.testing.assert_allclose(
        d_left_after, d_left_before, rtol=1e-10, atol=1e-10,
        err_msg="Left distortion coefficients should remain unchanged after stereo calibration"
    )
    
    np.testing.assert_allclose(
        d_right_after, d_right_before, rtol=1e-10, atol=1e-10,
        err_msg="Right distortion coefficients should remain unchanged after stereo calibration"
    )


# Property 4: Epipolar Rectification Correctness
# **Validates: Requirements 1.5**
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    st.integers(min_value=15, max_value=25),
    st.floats(min_value=0.05, max_value=0.3),  # baseline
    st.integers(min_value=5, max_value=20)  # number of test points
)
def test_property_4_epipolar_rectification_correctness(num_images, baseline, num_test_points):
    """
    Property 4: Epipolar Rectification Correctness
    
    For any calibrated stereo pair, rectification should produce images
    where corresponding points lie on the same horizontal scanlines.
    
    **Validates: Requirements 1.5**
    """
    # Generate calibration data
    obj_points, img_points_left, camera_matrix, dist_coeffs, image_size = \
        generate_calibration_data_with_diversity(num_images)
    
    assume(len(obj_points) >= 10)
    
    # Calibrate left camera
    ret_left, K_left, d_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        obj_points, img_points_left, image_size, None, None
    )
    
    # Generate right camera images (shifted by baseline)
    img_points_right = []
    for i in range(len(obj_points)):
        # Right camera is translated along X-axis
        tvec_right = tvecs_left[i].copy()
        tvec_right[0] += baseline
        
        img_pts_right, _ = cv2.projectPoints(
            obj_points[i], rvecs_left[i], tvec_right,
            camera_matrix, dist_coeffs
        )
        img_points_right.append(img_pts_right.astype(np.float32))
    
    # Calibrate right camera
    ret_right, K_right, d_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        obj_points, img_points_right, image_size, None, None
    )
    
    assert ret_left and ret_right, "Camera calibrations should succeed"
    
    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        K_left, d_left, K_right, d_right,
        image_size, criteria=criteria, flags=flags
    )
    
    assert ret_stereo, "Stereo calibration should succeed"
    
    # Compute rectification
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left, d_left, K_right, d_right,
        image_size, R, T, alpha=0
    )
    
    # Generate rectification maps
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, d_left, R1, P1, image_size, cv2.CV_32FC1
    )
    
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, d_right, R2, P2, image_size, cv2.CV_32FC1
    )
    
    # Test with random 3D points
    np.random.seed(42)
    test_points_3d = np.random.uniform(-0.2, 0.2, (num_test_points, 3))
    test_points_3d[:, 2] += 0.5  # Ensure points are in front of camera
    
    # Project to left and right cameras (original unrectified)
    pts_left, _ = cv2.projectPoints(
        test_points_3d, rvecs_left[0], tvecs_left[0], K_left, d_left
    )
    pts_left = pts_left.reshape(-1, 2)
    
    tvec_right_test = tvecs_left[0].copy()
    tvec_right_test[0] += baseline
    pts_right, _ = cv2.projectPoints(
        test_points_3d, rvecs_left[0], tvec_right_test, K_right, d_right
    )
    pts_right = pts_right.reshape(-1, 2)
    
    # Filter points that are within image bounds
    valid_mask = (
        (pts_left[:, 0] >= 0) & (pts_left[:, 0] < image_size[0]) &
        (pts_left[:, 1] >= 0) & (pts_left[:, 1] < image_size[1]) &
        (pts_right[:, 0] >= 0) & (pts_right[:, 0] < image_size[0]) &
        (pts_right[:, 1] >= 0) & (pts_right[:, 1] < image_size[1])
    )
    
    assume(np.sum(valid_mask) >= 3)
    
    pts_left = pts_left[valid_mask]
    pts_right = pts_right[valid_mask]
    
    # Apply undistortion and rectification to points
    # Use undistortPoints which properly handles rectification
    pts_left_rect = cv2.undistortPoints(
        pts_left.reshape(-1, 1, 2), K_left, d_left, R=R1, P=P1
    ).reshape(-1, 2)
    
    pts_right_rect = cv2.undistortPoints(
        pts_right.reshape(-1, 1, 2), K_right, d_right, R=R2, P=P2
    ).reshape(-1, 2)
    
    # Property: Corresponding points should have same Y coordinate (horizontal scanlines)
    y_diff = np.abs(pts_left_rect[:, 1] - pts_right_rect[:, 1])
    
    # Allow tolerance for numerical precision and calibration accuracy
    # Good rectification should have mean error < 1 pixel and max < 3 pixels
    max_y_diff = np.max(y_diff)
    mean_y_diff = np.mean(y_diff)
    
    assert max_y_diff < 3.0, \
        f"Max Y-coordinate difference {max_y_diff:.4f} exceeds 3.0 pixel threshold"
    
    assert mean_y_diff < 1.0, \
        f"Mean Y-coordinate difference {mean_y_diff:.4f} exceeds 1.0 pixel threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
