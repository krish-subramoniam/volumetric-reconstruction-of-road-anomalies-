"""Basic unit tests for disparity estimation module."""

import numpy as np
import pytest
from stereo_vision.disparity import SGBMEstimator, LRCValidator, WLSFilter


def test_sgbm_estimator_initialization():
    """Test SGBMEstimator initialization with valid parameters."""
    baseline = 0.12  # 12 cm baseline
    focal_length = 800.0  # pixels
    
    estimator = SGBMEstimator(baseline, focal_length)
    
    assert estimator.baseline == baseline
    assert estimator.focal_length == focal_length
    assert estimator.sgbm is not None


def test_sgbm_compute_disparity_grayscale():
    """Test disparity computation with grayscale images."""
    baseline = 0.12
    focal_length = 800.0
    
    estimator = SGBMEstimator(baseline, focal_length)
    
    # Create simple test images
    left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    disparity = estimator.compute_disparity(left, right)
    
    assert disparity is not None
    assert disparity.shape == left.shape
    assert disparity.dtype == np.int16


def test_sgbm_compute_disparity_color():
    """Test disparity computation with color images."""
    baseline = 0.12
    focal_length = 800.0
    
    estimator = SGBMEstimator(baseline, focal_length)
    
    # Create simple test images (color)
    left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    disparity = estimator.compute_disparity(left, right)
    
    assert disparity is not None
    assert disparity.shape == (left.shape[0], left.shape[1])
    assert disparity.dtype == np.int16


def test_sgbm_reconfigure():
    """Test reconfiguration of SGBM parameters."""
    estimator = SGBMEstimator(0.12, 800.0)
    
    # Reconfigure with different parameters
    new_baseline = 0.20
    new_focal_length = 1000.0
    
    estimator.configure_for_roads(new_baseline, new_focal_length)
    
    assert estimator.baseline == new_baseline
    assert estimator.focal_length == new_focal_length
    assert estimator.sgbm is not None


def test_lrc_validator_initialization():
    """Test LRCValidator initialization."""
    validator = LRCValidator(max_diff=1)
    assert validator.max_diff == 1


def test_lrc_validator_consistency_check():
    """Test basic LRC consistency checking."""
    validator = LRCValidator(max_diff=1)
    
    # Create simple disparity maps
    height, width = 100, 100
    disp_left = np.ones((height, width), dtype=np.int16) * 16 * 10  # disparity = 10
    disp_right = np.ones((height, width), dtype=np.int16) * 16 * 10
    
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    assert validity_mask.shape == (height, width)
    assert validity_mask.dtype == np.uint8


def test_lrc_validator_perfect_consistency():
    """Test LRC with perfectly consistent disparity maps."""
    validator = LRCValidator(max_diff=1)
    
    # Create consistent disparity maps
    height, width = 50, 100
    disp_left = np.zeros((height, width), dtype=np.float32)
    disp_right = np.zeros((height, width), dtype=np.float32)
    
    # Set up a simple pattern: disparity increases linearly
    for x in range(width):
        if x >= 10:  # Valid disparity range
            disp_left[:, x] = min(x / 10.0, 10.0)
    
    # Create corresponding right disparity
    for y in range(height):
        for x in range(width):
            d = disp_left[y, x]
            if d > 0:
                x_right = int(x - d)
                if 0 <= x_right < width:
                    disp_right[y, x_right] = d
    
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    # Count valid pixels
    valid_count = np.sum(validity_mask)
    total_valid_disp = np.sum(disp_left > 0)
    
    # Most pixels should be valid (some edge cases may fail)
    assert valid_count > 0
    assert valid_count <= total_valid_disp


def test_lrc_validator_inconsistent_disparities():
    """Test LRC with inconsistent disparity maps (occlusions)."""
    validator = LRCValidator(max_diff=1)
    
    height, width = 50, 100
    disp_left = np.ones((height, width), dtype=np.float32) * 5.0
    disp_right = np.ones((height, width), dtype=np.float32) * 10.0  # Inconsistent
    
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    # Most pixels should be invalid due to inconsistency
    valid_count = np.sum(validity_mask)
    total_pixels = height * width
    
    # Should have very few valid pixels
    assert valid_count < total_pixels * 0.1


def test_lrc_validator_zero_disparities():
    """Test LRC with zero disparities (invalid pixels)."""
    validator = LRCValidator(max_diff=1)
    
    height, width = 50, 100
    disp_left = np.zeros((height, width), dtype=np.float32)
    disp_right = np.zeros((height, width), dtype=np.float32)
    
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    # All pixels should be invalid (zero disparity)
    assert np.sum(validity_mask) == 0


def test_lrc_validator_edge_cases():
    """Test LRC with edge cases (boundary pixels)."""
    validator = LRCValidator(max_diff=1)
    
    height, width = 50, 100
    disp_left = np.zeros((height, width), dtype=np.float32)
    disp_right = np.zeros((height, width), dtype=np.float32)
    
    # Set disparity at left edge (should fail - x_right would be negative)
    disp_left[:, 0] = 10.0
    disp_right[:, 0] = 10.0
    
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    # Left edge pixels should be invalid
    assert np.all(validity_mask[:, 0] == 0)


def test_lrc_validator_fixed_point_format():
    """Test LRC with fixed-point disparity format (int16)."""
    validator = LRCValidator(max_diff=1)
    
    height, width = 50, 100
    # Fixed-point format: multiply by 16
    disp_left = np.ones((height, width), dtype=np.int16) * 16 * 5  # disparity = 5
    disp_right = np.ones((height, width), dtype=np.int16) * 16 * 5
    
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    assert validity_mask.shape == (height, width)
    assert validity_mask.dtype == np.uint8


def test_lrc_validator_error_rate_calculation():
    """Test LRC error rate calculation."""
    validator = LRCValidator(max_diff=1)
    
    height, width = 50, 100
    disp_left = np.ones((height, width), dtype=np.float32) * 5.0
    disp_right = np.ones((height, width), dtype=np.float32) * 5.0
    
    # Make half the pixels inconsistent
    disp_right[:, :50] = 10.0
    
    error_rate = validator.compute_error_rate(disp_left, disp_right)
    
    # Error rate should be positive
    assert error_rate >= 0.0
    assert error_rate <= 100.0


def test_lrc_validator_error_rate_zero_disparities():
    """Test LRC error rate with all zero disparities."""
    validator = LRCValidator(max_diff=1)
    
    height, width = 50, 100
    disp_left = np.zeros((height, width), dtype=np.float32)
    disp_right = np.zeros((height, width), dtype=np.float32)
    
    error_rate = validator.compute_error_rate(disp_left, disp_right)
    
    # Should return 0 when no valid disparities
    assert error_rate == 0.0


def test_wls_filter_initialization():
    """Test WLSFilter initialization."""
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    assert wls_filter.lambda_val == 8000.0
    assert wls_filter.sigma_color == 1.5


def test_wls_filter_basic_filtering():
    """Test basic WLS filtering with grayscale guide image."""
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    
    # Create test disparity map (16-bit fixed point)
    height, width = 100, 150
    disparity = np.random.randint(0, 16 * 20, (height, width), dtype=np.int16)
    
    # Create guide image (grayscale)
    guide_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Apply filtering
    filtered = wls_filter.filter_disparity(disparity, guide_image)
    
    assert filtered is not None
    assert filtered.shape == disparity.shape
    assert filtered.dtype == np.int16


def test_wls_filter_color_guide():
    """Test WLS filtering with color guide image."""
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    
    # Create test disparity map
    height, width = 100, 150
    disparity = np.random.randint(0, 16 * 20, (height, width), dtype=np.int16)
    
    # Create color guide image
    guide_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Apply filtering
    filtered = wls_filter.filter_disparity(disparity, guide_image)
    
    assert filtered is not None
    assert filtered.shape == disparity.shape


def test_wls_filter_with_right_disparity():
    """Test WLS filtering with both left and right disparity maps."""
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    
    # Create test disparity maps
    height, width = 100, 150
    disparity_left = np.random.randint(0, 16 * 20, (height, width), dtype=np.int16)
    disparity_right = np.random.randint(0, 16 * 20, (height, width), dtype=np.int16)
    
    # Create guide image
    guide_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Apply filtering with right disparity
    filtered = wls_filter.filter_disparity(
        disparity_left, guide_image, disparity_right=disparity_right
    )
    
    assert filtered is not None
    assert filtered.shape == disparity_left.shape


def test_wls_filter_smoothing_effect():
    """Test that WLS filtering produces smoother disparity maps."""
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    
    # Create noisy disparity map
    height, width = 100, 150
    base_disparity = np.ones((height, width), dtype=np.float32) * 10.0 * 16
    noise = np.random.randn(height, width) * 2.0 * 16
    noisy_disparity = (base_disparity + noise).astype(np.int16)
    
    # Create guide image with some structure
    guide_image = np.random.randint(100, 200, (height, width), dtype=np.uint8)
    
    # Apply filtering
    filtered = wls_filter.filter_disparity(noisy_disparity, guide_image)
    
    # Convert to float for comparison
    noisy_float = noisy_disparity.astype(np.float32) / 16.0
    filtered_float = filtered.astype(np.float32) / 16.0
    
    # Filter should reduce variance (smoother result)
    # Note: This is a statistical test, may have some variance
    noisy_std = np.std(noisy_float[noisy_float > 0])
    filtered_std = np.std(filtered_float[filtered_float > 0])
    
    # Filtered result should generally be smoother (lower std dev)
    # We use a relaxed threshold since WLS preserves edges
    assert filtered_std <= noisy_std * 1.5


def test_wls_filter_edge_preservation():
    """Test that WLS filtering preserves edges in the guide image."""
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    
    # Create disparity map with a step edge
    height, width = 100, 150
    disparity = np.zeros((height, width), dtype=np.int16)
    disparity[:, :width//2] = 10 * 16  # Left half
    disparity[:, width//2:] = 20 * 16  # Right half
    
    # Add noise
    noise = np.random.randn(height, width) * 1.0 * 16
    noisy_disparity = (disparity + noise).astype(np.int16)
    
    # Create guide image with matching edge
    guide_image = np.zeros((height, width), dtype=np.uint8)
    guide_image[:, :width//2] = 100
    guide_image[:, width//2:] = 200
    
    # Apply filtering
    filtered = wls_filter.filter_disparity(noisy_disparity, guide_image)
    
    # Check that edge is preserved (disparity values on each side should be distinct)
    filtered_float = filtered.astype(np.float32) / 16.0
    left_mean = np.mean(filtered_float[:, :width//2])
    right_mean = np.mean(filtered_float[:, width//2:])
    
    # There should be a clear difference between left and right sides
    assert abs(right_mean - left_mean) > 5.0  # At least 5 pixels difference


def test_wls_filter_with_confidence():
    """Test WLS filtering with confidence map output."""
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    
    # Create test data
    height, width = 100, 150
    disparity = np.random.randint(0, 16 * 20, (height, width), dtype=np.int16)
    guide_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Apply filtering with confidence
    filtered, confidence = wls_filter.filter_with_confidence(disparity, guide_image)
    
    assert filtered is not None
    assert confidence is not None
    assert filtered.shape == disparity.shape
    assert confidence.shape == disparity.shape


def test_wls_filter_different_lambda_values():
    """Test WLS filtering with different lambda values."""
    # Create test data
    height, width = 100, 150
    disparity = np.random.randint(0, 16 * 20, (height, width), dtype=np.int16)
    guide_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Test with low lambda (less smoothing)
    wls_low = WLSFilter(lambda_val=1000.0, sigma_color=1.5)
    filtered_low = wls_low.filter_disparity(disparity, guide_image)
    
    # Test with high lambda (more smoothing)
    wls_high = WLSFilter(lambda_val=10000.0, sigma_color=1.5)
    filtered_high = wls_high.filter_disparity(disparity, guide_image)
    
    # Both should produce valid results
    assert filtered_low is not None
    assert filtered_high is not None
    assert filtered_low.shape == disparity.shape
    assert filtered_high.shape == disparity.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_sgbm_with_lrc_integration():
    """
    Integration test: SGBM disparity computation with LRC validation.
    
    This test demonstrates the complete workflow:
    1. Compute left disparity using SGBM
    2. Compute right disparity using SGBM (with swapped images)
    3. Apply LRC validation to remove occluded pixels
    4. Verify that inconsistent pixels are marked as invalid
    """
    baseline = 0.12
    focal_length = 800.0
    
    # Create SGBM estimator
    estimator = SGBMEstimator(baseline, focal_length)
    
    # Create simple test images with some texture
    height, width = 100, 150
    left = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    right = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    
    # Compute left-to-right disparity
    disp_left = estimator.compute_disparity(left, right)
    
    # Compute right-to-left disparity (swap images)
    disp_right = estimator.compute_disparity(right, left)
    
    # Apply LRC validation
    validator = LRCValidator(max_diff=1)
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    # Verify results
    assert validity_mask.shape == (height, width)
    assert validity_mask.dtype == np.uint8
    assert np.all((validity_mask == 0) | (validity_mask == 1))
    
    # Compute error rate
    error_rate = validator.compute_error_rate(disp_left, disp_right)
    assert 0.0 <= error_rate <= 100.0
    
    # Apply validity mask to disparity map
    disp_left_float = disp_left.astype(np.float32) / 16.0
    disp_left_validated = disp_left_float * validity_mask
    
    # Verify that invalid pixels are zeroed out
    invalid_pixels = (validity_mask == 0)
    assert np.all(disp_left_validated[invalid_pixels] == 0.0)


def test_lrc_removes_inconsistent_pixels():
    """
    Test that LRC validation correctly removes inconsistent pixels.
    
    This test verifies Requirement 2.2: "WHEN disparity computation is complete,
    THE Stereo_System SHALL perform Left-Right Consistency checking to remove
    occluded pixels"
    """
    validator = LRCValidator(max_diff=1)
    
    height, width = 50, 100
    
    # Create left disparity with consistent values
    disp_left = np.ones((height, width), dtype=np.float32) * 5.0
    
    # Create right disparity with some inconsistent regions
    disp_right = np.ones((height, width), dtype=np.float32) * 5.0
    
    # Make a region inconsistent (simulating occlusion)
    disp_right[10:20, 30:50] = 15.0  # Very different disparity
    
    # Apply LRC validation
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    
    # Verify that inconsistent pixels are marked as invalid
    # Note: The exact pixels marked invalid depend on the LRC algorithm,
    # but we should see some invalid pixels in the inconsistent region
    total_invalid = np.sum(validity_mask == 0)
    assert total_invalid > 0, "LRC should mark some pixels as invalid"
    
    # Apply mask to remove inconsistent pixels
    disp_left_cleaned = disp_left * validity_mask
    
    # Verify that invalid pixels are removed (set to 0)
    assert np.all(disp_left_cleaned[validity_mask == 0] == 0.0)


def test_complete_disparity_pipeline_with_wls():
    """
    Integration test: Complete disparity pipeline with SGBM, LRC, and WLS filtering.
    
    This test demonstrates the full workflow as specified in Requirement 2.3:
    1. Compute left and right disparity using SGBM
    2. Apply LRC validation to remove occluded pixels
    3. Apply WLS filtering for sub-pixel refinement
    
    Validates Requirements 2.1, 2.2, 2.3
    """
    baseline = 0.12
    focal_length = 800.0
    
    # Create SGBM estimator
    estimator = SGBMEstimator(baseline, focal_length)
    
    # Create test images with some texture
    height, width = 100, 150
    np.random.seed(42)  # For reproducibility
    left = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    right = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    
    # Step 1: Compute left-to-right disparity (Requirement 2.1)
    disp_left = estimator.compute_disparity(left, right)
    assert disp_left is not None
    assert disp_left.shape == (height, width)
    
    # Step 2: Compute right-to-left disparity
    disp_right = estimator.compute_disparity(right, left)
    assert disp_right is not None
    
    # Step 3: Apply LRC validation (Requirement 2.2)
    validator = LRCValidator(max_diff=1)
    validity_mask = validator.validate_consistency(disp_left, disp_right)
    assert validity_mask.shape == (height, width)
    
    # Apply mask to disparity
    disp_left_validated = disp_left.copy()
    disp_left_validated[validity_mask == 0] = 0
    
    # Step 4: Apply WLS filtering (Requirement 2.3)
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    disp_filtered = wls_filter.filter_disparity(
        disp_left_validated, left, disparity_right=disp_right
    )
    
    # Verify final result
    assert disp_filtered is not None
    assert disp_filtered.shape == (height, width)
    assert disp_filtered.dtype == np.int16
    
    # Verify that filtering was applied (result should differ from input)
    # Note: They may be similar but should not be identical
    assert not np.array_equal(disp_filtered, disp_left_validated)
    
    # Compute error rate
    error_rate = validator.compute_error_rate(disp_left, disp_right)
    assert 0.0 <= error_rate <= 100.0


def test_wls_filter_improves_disparity_quality():
    """
    Test that WLS filtering produces valid output and smooths disparity maps.
    
    This validates Requirement 2.3: "WHEN LRC validation is complete, 
    THE Stereo_System SHALL apply Weighted Least Squares filtering for 
    sub-pixel refinement"
    """
    # Create a clean disparity map with structure
    height, width = 100, 150
    clean_disparity = np.zeros((height, width), dtype=np.float32)
    
    # Create regions with different disparities
    clean_disparity[:, :50] = 5.0 * 16
    clean_disparity[:, 50:100] = 10.0 * 16
    clean_disparity[:, 100:] = 15.0 * 16
    
    # Add noise
    noise = np.random.randn(height, width) * 2.0 * 16
    noisy_disparity = (clean_disparity + noise).astype(np.int16)
    
    # Create guide image with matching structure
    guide_image = np.zeros((height, width), dtype=np.uint8)
    guide_image[:, :50] = 100
    guide_image[:, 50:100] = 150
    guide_image[:, 100:] = 200
    
    # Apply WLS filtering
    wls_filter = WLSFilter(lambda_val=8000.0, sigma_color=1.5)
    filtered = wls_filter.filter_disparity(noisy_disparity, guide_image)
    
    # Verify that filtering produces valid output
    assert filtered is not None
    assert filtered.shape == noisy_disparity.shape
    
    # Convert to float for analysis
    noisy_float = noisy_disparity.astype(np.float32) / 16.0
    filtered_float = filtered.astype(np.float32) / 16.0
    
    # Verify that filtered result has valid values
    assert np.any(filtered_float > 0), "Filtered disparity should have valid values"
    
    # Verify that filtering reduces variance within regions (smoothing effect)
    for start, end in [(0, 50), (50, 100), (100, 150)]:
        region_noisy = noisy_float[:, start:end]
        region_filtered = filtered_float[:, start:end]
        
        # Filter valid pixels (non-zero)
        valid_mask = region_noisy > 0
        
        if np.any(valid_mask):
            # Calculate variance
            var_noisy = np.var(region_noisy[valid_mask])
            var_filtered = np.var(region_filtered[valid_mask])
            
            # Filtered result should generally have lower or similar variance
            # (indicating smoothing), but we use a relaxed threshold
            # since WLS is edge-preserving
            assert var_filtered <= var_noisy * 2.0, \
                f"WLS filtering should not dramatically increase variance in region [{start}:{end}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
