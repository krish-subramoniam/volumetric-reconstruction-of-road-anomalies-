"""
Property-based tests for quality metrics module.

These tests validate the correctness properties for LRC error rate calculation,
planarity residual computation, temporal stability measurement, and calibration
quality reporting.

Feature: advanced-stereo-vision-pipeline
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from stereo_vision.quality_metrics import (
    LRCErrorCalculator,
    PlanarityCalculator,
    TemporalStabilityCalculator,
    CalibrationQualityReporter
)
from stereo_vision.calibration import CameraParameters, StereoParameters
from stereo_vision.ground_plane import GroundPlaneModel


# Strategy for generating valid disparity maps
@st.composite
def disparity_map_strategy(draw, height=50, width=75, max_disparity=32):
    """Generate valid disparity maps for testing."""
    # Create disparity map with some valid disparities
    disparity = np.zeros((height, width), dtype=np.float32)
    
    # Add valid disparities in a region (reduced number for faster generation)
    num_valid = draw(st.integers(min_value=100, max_value=500))
    
    for _ in range(num_valid):
        y = draw(st.integers(min_value=0, max_value=height - 1))
        x = draw(st.integers(min_value=0, max_value=width - 1))
        d = draw(st.floats(min_value=1.0, max_value=float(max_disparity)))
        disparity[y, x] = d
    
    return disparity


@st.composite
def stereo_disparity_pair_strategy(draw, height=50, width=75):
    """Generate consistent left-right disparity pair."""
    disp_left = draw(disparity_map_strategy(height=height, width=width))
    
    # Create right disparity by shifting left disparity
    disp_right = np.zeros_like(disp_left)
    
    for y in range(height):
        for x in range(width):
            if disp_left[y, x] > 0:
                d = disp_left[y, x]
                x_right = int(x - d)
                if 0 <= x_right < width:
                    # Add small noise to simulate real matching
                    noise = draw(st.floats(min_value=-0.5, max_value=0.5))
                    disp_right[y, x_right] = d + noise
    
    return disp_left, disp_right


# Property 24: LRC Error Rate Calculation
@given(stereo_disparity_pair_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_property_24_lrc_error_rate_calculation(disparity_pair):
    """
    **Feature: advanced-stereo-vision-pipeline, Property 24: LRC Error Rate Calculation**
    
    For any stereo pair processing, left-right consistency error rate calculation
    should accurately reflect the percentage of pixels failing validation.
    
    **Validates: Requirements 7.1**
    """
    disp_left, disp_right = disparity_pair
    
    # Skip if no valid disparities
    assume(np.sum(disp_left > 0) > 0)
    
    calculator = LRCErrorCalculator(max_diff=1)
    error_rate = calculator.calculate_error_rate(disp_left, disp_right)
    
    # Property: Error rate should be between 0 and 100
    assert 0.0 <= error_rate <= 100.0, \
        f"Error rate {error_rate} is outside valid range [0, 100]"
    
    # Property: If all disparities are consistent, error rate should be low
    # Count manually to verify
    height, width = disp_left.shape
    valid_count = 0
    failed_count = 0
    
    for y in range(height):
        for x in range(width):
            if disp_left[y, x] > 0:
                valid_count += 1
                d_left = disp_left[y, x]
                x_right = int(x - d_left)
                
                if x_right < 0 or x_right >= width:
                    failed_count += 1
                elif abs(d_left - disp_right[y, x_right]) > 1:
                    failed_count += 1
    
    if valid_count > 0:
        expected_error_rate = (failed_count / valid_count) * 100.0
        
        # Allow small numerical tolerance
        assert abs(error_rate - expected_error_rate) < 0.1, \
            f"Calculated error rate {error_rate} doesn't match expected {expected_error_rate}"


# Property 25: Planarity Residual Computation
@given(
    disparity_map_strategy(height=50, width=75),
    st.floats(min_value=0.1, max_value=2.0),  # slope
    st.floats(min_value=5.0, max_value=30.0)  # intercept
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_property_25_planarity_residual_computation(disparity_map, slope, intercept):
    """
    **Feature: advanced-stereo-vision-pipeline, Property 25: Planarity Residual Computation**
    
    For any ground plane fit, RMSE calculation should correctly measure the
    deviation of inlier points from the fitted plane.
    
    **Validates: Requirements 7.2**
    """
    # Skip if no valid disparities
    assume(np.sum(disparity_map > 0) > 10)
    
    # Create ground plane model
    ground_plane = GroundPlaneModel()
    ground_plane.fit_from_line_params(slope, intercept)
    
    calculator = PlanarityCalculator()
    rmse = calculator.calculate_planarity_rmse(disparity_map, ground_plane, inlier_threshold=2.0)
    
    # Property: RMSE should be non-negative
    assert rmse >= 0.0, f"RMSE {rmse} is negative"
    
    # Property: RMSE should be finite
    assert np.isfinite(rmse), f"RMSE {rmse} is not finite"
    
    # Property: Manually calculate RMSE to verify
    height, width = disparity_map.shape
    residuals = []
    
    for row in range(height):
        expected_disp = ground_plane.get_expected_disparity(row)
        for col in range(width):
            actual_disp = disparity_map[row, col]
            if actual_disp > 0:
                residual = actual_disp - expected_disp
                if abs(residual) <= 2.0:  # inlier threshold
                    residuals.append(residual)
    
    if len(residuals) > 0:
        expected_rmse = np.sqrt(np.mean(np.array(residuals) ** 2))
        
        # Allow small numerical tolerance
        assert abs(rmse - expected_rmse) < 0.01, \
            f"Calculated RMSE {rmse} doesn't match expected {expected_rmse}"


# Property 26: Temporal Stability Measurement
@given(
    st.lists(
        st.floats(min_value=0.001, max_value=10.0),
        min_size=2,
        max_size=20
    )
)
@settings(max_examples=100, deadline=None)
def test_property_26_temporal_stability_measurement(volume_sequence):
    """
    **Feature: advanced-stereo-vision-pipeline, Property 26: Temporal Stability Measurement**
    
    For any sequence of volume measurements on the same static anomaly, temporal
    stability metrics should reflect the consistency of the estimates.
    
    **Validates: Requirements 7.3**
    """
    calculator = TemporalStabilityCalculator()
    cv = calculator.calculate_temporal_stability(volume_sequence)
    
    # Property: CV should be non-negative
    assert cv >= 0.0, f"Coefficient of variation {cv} is negative"
    
    # Property: CV should be finite
    assert np.isfinite(cv), f"CV {cv} is not finite"
    
    # Property: CV should be a percentage (typically 0-100, but can be higher)
    assert cv >= 0.0, f"CV {cv} is negative"
    
    # Property: Manually calculate CV to verify
    volumes = np.array(volume_sequence)
    valid_volumes = volumes[volumes > 0]
    
    if len(valid_volumes) >= 2:
        mean_vol = np.mean(valid_volumes)
        std_vol = np.std(valid_volumes)
        
        if mean_vol > 0:
            expected_cv = (std_vol / mean_vol) * 100.0
            
            # Allow small numerical tolerance
            assert abs(cv - expected_cv) < 0.01, \
                f"Calculated CV {cv} doesn't match expected {expected_cv}"
    
    # Property: If all volumes are identical, CV should be 0 (within floating point precision)
    if len(set(volume_sequence)) == 1:
        assert abs(cv) < 1e-10, f"CV for identical volumes should be ~0, got {cv}"


# Property 27: Calibration Quality Reporting
@given(
    st.floats(min_value=0.01, max_value=0.5),  # left reprojection error
    st.floats(min_value=0.01, max_value=0.5),  # right reprojection error
    st.floats(min_value=0.05, max_value=0.5)   # baseline
)
@settings(max_examples=100, deadline=None)
def test_property_27_calibration_quality_reporting(left_error, right_error, baseline):
    """
    **Feature: advanced-stereo-vision-pipeline, Property 27: Calibration Quality Reporting**
    
    For any completed calibration, quality metrics including reprojection errors
    should be calculated and reported accurately.
    
    **Validates: Requirements 7.4**
    """
    # Create mock camera parameters
    left_camera = CameraParameters(
        camera_matrix=np.eye(3),
        distortion_coeffs=np.zeros(5),
        reprojection_error=left_error,
        image_size=(640, 480)
    )
    
    right_camera = CameraParameters(
        camera_matrix=np.eye(3),
        distortion_coeffs=np.zeros(5),
        reprojection_error=right_error,
        image_size=(640, 480)
    )
    
    stereo_params = StereoParameters(
        left_camera=left_camera,
        right_camera=right_camera,
        rotation_matrix=np.eye(3),
        translation_vector=np.array([baseline, 0, 0]),
        baseline=baseline,
        Q_matrix=np.eye(4),
        rectification_maps_left=(np.zeros((480, 640)), np.zeros((480, 640))),
        rectification_maps_right=(np.zeros((480, 640)), np.zeros((480, 640)))
    )
    
    reporter = CalibrationQualityReporter()
    quality = reporter.report_stereo_quality(stereo_params)
    
    # Property: Reported errors should match input errors
    assert abs(quality['left_reprojection_error'] - left_error) < 1e-6, \
        f"Left error mismatch: {quality['left_reprojection_error']} vs {left_error}"
    
    assert abs(quality['right_reprojection_error'] - right_error) < 1e-6, \
        f"Right error mismatch: {quality['right_reprojection_error']} vs {right_error}"
    
    # Property: Baseline should match
    assert abs(quality['baseline'] - baseline) < 1e-6, \
        f"Baseline mismatch: {quality['baseline']} vs {baseline}"
    
    # Property: Mean error should be average of left and right
    expected_mean = (left_error + right_error) / 2.0
    assert abs(quality['mean_reprojection_error'] - expected_mean) < 1e-6, \
        f"Mean error mismatch: {quality['mean_reprojection_error']} vs {expected_mean}"
    
    # Property: All reported values should be positive
    assert quality['left_reprojection_error'] > 0
    assert quality['right_reprojection_error'] > 0
    assert quality['baseline'] > 0
    assert quality['mean_reprojection_error'] > 0


# Additional unit tests for edge cases
def test_lrc_error_rate_empty_disparity():
    """Test LRC error rate with empty disparity maps."""
    calculator = LRCErrorCalculator()
    
    disp_left = np.zeros((100, 150), dtype=np.float32)
    disp_right = np.zeros((100, 150), dtype=np.float32)
    
    error_rate = calculator.calculate_error_rate(disp_left, disp_right)
    
    # Should return 0 for empty maps
    assert error_rate == 0.0


def test_planarity_rmse_unfitted_plane():
    """Test planarity RMSE with unfitted ground plane."""
    calculator = PlanarityCalculator()
    ground_plane = GroundPlaneModel()
    
    disparity = np.random.rand(100, 150).astype(np.float32) * 50
    
    with pytest.raises(ValueError, match="Ground plane model has not been fitted"):
        calculator.calculate_planarity_rmse(disparity, ground_plane)


def test_temporal_stability_single_sample():
    """Test temporal stability with single sample."""
    calculator = TemporalStabilityCalculator()
    
    cv = calculator.calculate_temporal_stability([5.0])
    
    # Should return 0 for single sample
    assert cv == 0.0


def test_temporal_stability_identical_volumes():
    """Test temporal stability with identical volumes."""
    calculator = TemporalStabilityCalculator()
    
    cv = calculator.calculate_temporal_stability([5.0, 5.0, 5.0, 5.0])
    
    # Should return 0 for identical volumes
    assert cv == 0.0


def test_calibration_quality_validation():
    """Test calibration quality validation."""
    left_camera = CameraParameters(
        camera_matrix=np.eye(3),
        distortion_coeffs=np.zeros(5),
        reprojection_error=0.3,
        image_size=(640, 480)
    )
    
    right_camera = CameraParameters(
        camera_matrix=np.eye(3),
        distortion_coeffs=np.zeros(5),
        reprojection_error=0.4,
        image_size=(640, 480)
    )
    
    stereo_params = StereoParameters(
        left_camera=left_camera,
        right_camera=right_camera,
        rotation_matrix=np.eye(3),
        translation_vector=np.array([0.1, 0, 0]),
        baseline=0.1,
        Q_matrix=np.eye(4),
        rectification_maps_left=(np.zeros((480, 640)), np.zeros((480, 640))),
        rectification_maps_right=(np.zeros((480, 640)), np.zeros((480, 640)))
    )
    
    reporter = CalibrationQualityReporter()
    
    # Should pass with default threshold (0.5)
    is_valid, message = reporter.validate_calibration_quality(stereo_params)
    assert is_valid
    
    # Should fail with stricter threshold
    is_valid, message = reporter.validate_calibration_quality(stereo_params, max_reprojection_error=0.2)
    assert not is_valid
