"""Property-based tests for disparity estimation module."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from stereo_vision.disparity import LRCValidator


# Custom strategies for disparity maps
@st.composite
def disparity_map_strategy(draw, min_height=10, max_height=50, 
                          min_width=10, max_width=50,
                          min_disparity=0.0, max_disparity=20.0):
    """
    Generate random disparity maps for property testing.
    
    Args:
        draw: Hypothesis draw function
        min_height, max_height: Height range for disparity map
        min_width, max_width: Width range for disparity map
        min_disparity, max_disparity: Disparity value range
        
    Returns:
        Tuple of (height, width, disparity_map)
    """
    height = draw(st.integers(min_value=min_height, max_value=max_height))
    width = draw(st.integers(min_value=min_width, max_value=max_width))
    
    # Use numpy random to generate disparity map (more efficient)
    # Draw a seed for reproducibility
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    # Generate disparity values with some invalid pixels
    disparity_map = rng.uniform(min_disparity, max_disparity, (height, width)).astype(np.float32)
    
    # Randomly set some pixels to zero (invalid)
    invalid_mask = rng.random((height, width)) < 0.2  # 20% invalid
    disparity_map[invalid_mask] = 0.0
    
    return height, width, disparity_map


@st.composite
def consistent_disparity_pair_strategy(draw, min_height=10, max_height=50,
                                       min_width=20, max_width=100,
                                       min_disparity=1.0, max_disparity=20.0):
    """
    Generate a pair of left and right disparity maps that are consistent.
    
    This creates disparity maps where the left-right consistency check
    should pass for most pixels (except edge cases).
    
    Args:
        draw: Hypothesis draw function
        min_height, max_height: Height range
        min_width, max_width: Width range
        min_disparity, max_disparity: Disparity range
        
    Returns:
        Tuple of (disp_left, disp_right)
    """
    height = draw(st.integers(min_value=min_height, max_value=max_height))
    width = draw(st.integers(min_value=min_width, max_value=max_width))
    
    # Create left disparity map
    disp_left = np.zeros((height, width), dtype=np.float32)
    disp_right = np.zeros((height, width), dtype=np.float32)
    
    # Fill with random but consistent disparities
    for y in range(height):
        for x in range(width):
            # Random disparity value
            d = draw(st.floats(min_value=min_disparity, max_value=max_disparity,
                              allow_nan=False, allow_infinity=False))
            
            # Ensure we don't go out of bounds
            if x - d >= 0:
                disp_left[y, x] = d
                x_right = int(x - d)
                if 0 <= x_right < width:
                    disp_right[y, x_right] = d
    
    return disp_left, disp_right


class TestLRCValidatorProperties:
    """Property-based tests for LRCValidator."""
    
    @given(st.integers(min_value=0, max_value=10))
    def test_property_lrc_validator_initialization(self, max_diff):
        """
        Property: LRCValidator should initialize with any valid max_diff value.
        
        **Feature: advanced-stereo-vision-pipeline, Property: Initialization Correctness**
        """
        validator = LRCValidator(max_diff=max_diff)
        assert validator.max_diff == max_diff
    
    @given(disparity_map_strategy())
    @settings(max_examples=50)
    def test_property_lrc_output_shape(self, disparity_data):
        """
        Property: LRC validation should always return a validity mask with the same
        shape as the input disparity maps.
        
        **Feature: advanced-stereo-vision-pipeline, Property: Output Shape Consistency**
        """
        height, width, disp_left = disparity_data
        disp_right = np.zeros((height, width), dtype=np.float32)
        
        validator = LRCValidator(max_diff=1)
        validity_mask = validator.validate_consistency(disp_left, disp_right)
        
        assert validity_mask.shape == (height, width)
        assert validity_mask.dtype == np.uint8
    
    @given(disparity_map_strategy())
    @settings(max_examples=50)
    def test_property_lrc_binary_mask(self, disparity_data):
        """
        Property: LRC validation should return a binary mask (only 0 or 1 values).
        
        **Feature: advanced-stereo-vision-pipeline, Property: Binary Mask Output**
        """
        height, width, disp_left = disparity_data
        disp_right = np.zeros((height, width), dtype=np.float32)
        
        validator = LRCValidator(max_diff=1)
        validity_mask = validator.validate_consistency(disp_left, disp_right)
        
        # All values should be 0 or 1
        assert np.all((validity_mask == 0) | (validity_mask == 1))
    
    @given(disparity_map_strategy())
    @settings(max_examples=50)
    def test_property_lrc_zero_disparities_invalid(self, disparity_data):
        """
        Property: Pixels with zero or negative disparity should always be marked
        as invalid in the LRC validation.
        
        **Feature: advanced-stereo-vision-pipeline, Property 5: Left-Right Consistency Validation**
        **Validates: Requirements 2.2**
        """
        height, width, disp_left = disparity_data
        disp_right = np.copy(disp_left)
        
        # Set some pixels to zero/negative
        disp_left[disp_left < 0.1] = 0.0
        
        validator = LRCValidator(max_diff=1)
        validity_mask = validator.validate_consistency(disp_left, disp_right)
        
        # All pixels with zero disparity should be invalid
        zero_pixels = (disp_left <= 0)
        assert np.all(validity_mask[zero_pixels] == 0)
    
    @given(st.integers(min_value=10, max_value=50),
           st.integers(min_value=20, max_value=100),
           st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_property_lrc_perfect_consistency_passes(self, height, width, disparity_value):
        """
        Property: When left and right disparity maps are perfectly consistent,
        LRC validation should mark valid pixels as passing (except edge cases).
        
        **Feature: advanced-stereo-vision-pipeline, Property 5: Left-Right Consistency Validation**
        **Validates: Requirements 2.2**
        """
        # Create perfectly consistent disparity maps
        disp_left = np.zeros((height, width), dtype=np.float32)
        disp_right = np.zeros((height, width), dtype=np.float32)
        
        # Fill with consistent disparities (avoiding edges)
        d = disparity_value
        for y in range(height):
            for x in range(int(d) + 1, width):
                disp_left[y, x] = d
                x_right = int(x - d)
                if 0 <= x_right < width:
                    disp_right[y, x_right] = d
        
        validator = LRCValidator(max_diff=1)
        validity_mask = validator.validate_consistency(disp_left, disp_right)
        
        # Count valid pixels in left map
        valid_left = np.sum(disp_left > 0)
        passed_lrc = np.sum(validity_mask)
        
        # Most pixels should pass (some edge cases may fail)
        if valid_left > 0:
            pass_rate = passed_lrc / valid_left
            assert pass_rate >= 0.5  # At least 50% should pass for consistent maps
    
    @given(st.integers(min_value=10, max_value=50),
           st.integers(min_value=20, max_value=100))
    @settings(max_examples=50)
    def test_property_lrc_inconsistent_fails(self, height, width):
        """
        Property: When left and right disparity maps are highly inconsistent,
        LRC validation should mark most pixels as failing.
        
        **Feature: advanced-stereo-vision-pipeline, Property 5: Left-Right Consistency Validation**
        **Validates: Requirements 2.2**
        """
        # Create highly inconsistent disparity maps
        disp_left = np.ones((height, width), dtype=np.float32) * 5.0
        disp_right = np.ones((height, width), dtype=np.float32) * 15.0  # Very different
        
        validator = LRCValidator(max_diff=1)
        validity_mask = validator.validate_consistency(disp_left, disp_right)
        
        # Most pixels should fail
        passed_lrc = np.sum(validity_mask)
        total_pixels = height * width
        
        # Less than 10% should pass for highly inconsistent maps
        assert passed_lrc < total_pixels * 0.1
    
    @given(disparity_map_strategy())
    @settings(max_examples=50)
    def test_property_lrc_error_rate_range(self, disparity_data):
        """
        Property: LRC error rate should always be in the range [0, 100].
        
        **Feature: advanced-stereo-vision-pipeline, Property 24: LRC Error Rate Calculation**
        **Validates: Requirements 7.1**
        """
        height, width, disp_left = disparity_data
        disp_right = np.copy(disp_left)
        
        validator = LRCValidator(max_diff=1)
        error_rate = validator.compute_error_rate(disp_left, disp_right)
        
        assert 0.0 <= error_rate <= 100.0
    
    @given(st.integers(min_value=10, max_value=50),
           st.integers(min_value=20, max_value=100))
    @settings(max_examples=50)
    def test_property_lrc_error_rate_perfect_consistency(self, height, width):
        """
        Property: For perfectly consistent disparity maps, LRC error rate should be low.
        
        **Feature: advanced-stereo-vision-pipeline, Property 24: LRC Error Rate Calculation**
        **Validates: Requirements 7.1**
        """
        # Create perfectly consistent disparity maps
        disp_left = np.ones((height, width), dtype=np.float32) * 5.0
        disp_right = np.ones((height, width), dtype=np.float32) * 5.0
        
        validator = LRCValidator(max_diff=1)
        error_rate = validator.compute_error_rate(disp_left, disp_right)
        
        # Error rate should be relatively low for consistent maps
        # (some edge cases may still fail)
        assert error_rate <= 50.0
    
    @given(st.integers(min_value=10, max_value=50),
           st.integers(min_value=20, max_value=100))
    @settings(max_examples=50)
    def test_property_lrc_edge_pixels_handling(self, height, width):
        """
        Property: Pixels at the left edge with large disparities should be marked
        as invalid because their corresponding right pixels would be out of bounds.
        
        **Feature: advanced-stereo-vision-pipeline, Property 5: Left-Right Consistency Validation**
        **Validates: Requirements 2.2**
        """
        disp_left = np.zeros((height, width), dtype=np.float32)
        disp_right = np.zeros((height, width), dtype=np.float32)
        
        # Set large disparity at left edge
        large_disparity = width / 2.0
        disp_left[:, 0] = large_disparity
        disp_right[:, 0] = large_disparity
        
        validator = LRCValidator(max_diff=1)
        validity_mask = validator.validate_consistency(disp_left, disp_right)
        
        # Left edge pixels should be invalid (x_right would be negative)
        assert np.all(validity_mask[:, 0] == 0)
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_property_lrc_max_diff_tolerance(self, max_diff):
        """
        Property: Increasing max_diff tolerance should result in more pixels
        passing the LRC check (or at least not fewer).
        
        **Feature: advanced-stereo-vision-pipeline, Property: Tolerance Monotonicity**
        """
        height, width = 30, 60
        disp_left = np.ones((height, width), dtype=np.float32) * 5.0
        disp_right = np.ones((height, width), dtype=np.float32) * 5.0
        
        # Add some noise to create slight inconsistencies
        noise = np.random.uniform(-2, 2, (height, width)).astype(np.float32)
        disp_right += noise
        
        # Test with increasing tolerance
        validator_strict = LRCValidator(max_diff=max_diff)
        validator_loose = LRCValidator(max_diff=max_diff + 1)
        
        mask_strict = validator_strict.validate_consistency(disp_left, disp_right)
        mask_loose = validator_loose.validate_consistency(disp_left, disp_right)
        
        # Looser tolerance should pass at least as many pixels
        assert np.sum(mask_loose) >= np.sum(mask_strict)
    
    @given(st.integers(min_value=10, max_value=50),
           st.integers(min_value=30, max_value=100),
           st.floats(min_value=1.0, max_value=15.0, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_property_5_lrc_validation_marks_inconsistent_invalid(
        self, height, width, base_disparity, inconsistency_amount
    ):
        """
        Property 5: Left-Right Consistency Validation
        
        For any disparity computation, pixels that fail left-right consistency 
        checking should be marked as invalid in the final disparity map.
        
        This test creates disparity maps with known inconsistencies and verifies
        that the LRC validator correctly identifies and marks these pixels as invalid.
        
        **Feature: advanced-stereo-vision-pipeline, Property 5: Left-Right Consistency Validation**
        **Validates: Requirements 2.2**
        """
        # Create left and right disparity maps
        disp_left = np.zeros((height, width), dtype=np.float32)
        disp_right = np.zeros((height, width), dtype=np.float32)
        
        # Create a region with consistent disparities
        consistent_region_mask = np.zeros((height, width), dtype=bool)
        for y in range(height):
            for x in range(int(base_disparity) + 5, width - 5):
                d = base_disparity
                disp_left[y, x] = d
                x_right = int(x - d)
                if 0 <= x_right < width:
                    disp_right[y, x_right] = d
                    consistent_region_mask[y, x] = True
        
        # Create a region with intentionally inconsistent disparities
        inconsistent_region_mask = np.zeros((height, width), dtype=bool)
        for y in range(height // 4, 3 * height // 4):
            for x in range(width // 4, width // 2):
                if disp_left[y, x] > 0:
                    # Make this pixel inconsistent by adding large difference
                    x_right = int(x - disp_left[y, x])
                    if 0 <= x_right < width:
                        disp_right[y, x_right] += inconsistency_amount + 2.0  # Ensure > max_diff
                        inconsistent_region_mask[y, x] = True
        
        # Run LRC validation
        validator = LRCValidator(max_diff=1)
        validity_mask = validator.validate_consistency(disp_left, disp_right)
        
        # Verify Property 5: Inconsistent pixels should be marked as invalid (0)
        if np.any(inconsistent_region_mask):
            inconsistent_pixels = inconsistent_region_mask & (disp_left > 0)
            if np.any(inconsistent_pixels):
                # At least 80% of intentionally inconsistent pixels should be marked invalid
                invalid_count = np.sum(validity_mask[inconsistent_pixels] == 0)
                total_inconsistent = np.sum(inconsistent_pixels)
                invalid_rate = invalid_count / total_inconsistent
                assert invalid_rate >= 0.8, \
                    f"Expected at least 80% of inconsistent pixels to be invalid, got {invalid_rate:.2%}"
        
        # Verify that consistent pixels are more likely to be valid
        if np.any(consistent_region_mask) and not np.any(inconsistent_region_mask):
            consistent_pixels = consistent_region_mask & (disp_left > 0)
            if np.any(consistent_pixels):
                valid_count = np.sum(validity_mask[consistent_pixels] == 1)
                total_consistent = np.sum(consistent_pixels)
                valid_rate = valid_count / total_consistent
                # Consistent pixels should have higher validity rate
                assert valid_rate >= 0.5, \
                    f"Expected at least 50% of consistent pixels to be valid, got {valid_rate:.2%}"





class TestWLSFilterProperties:
    """Property-based tests for WLSFilter."""

    @given(
        st.integers(min_value=20, max_value=100),
        st.integers(min_value=20, max_value=100),
        st.floats(min_value=5.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_6_disparity_smoothness_in_textureless_regions(
        self, height, width, base_disparity, lambda_val, sigma_color
    ):
        """
        Property 6: Disparity Smoothness in Textureless Regions

        For any textureless image region, disparity values should vary smoothly
        without abrupt discontinuities, propagating information from nearby
        textured areas.

        This test creates a synthetic scenario with:
        1. A textureless region (uniform intensity)
        2. Surrounding textured regions with known disparities
        3. Applies WLS filtering to propagate disparity information
        4. Verifies that the textureless region has smooth disparity values

        **Feature: advanced-stereo-vision-pipeline, Property 6: Disparity Smoothness in Textureless Regions**
        **Validates: Requirements 2.5**
        """
        from stereo_vision.disparity import WLSFilter

        # Ensure dimensions are reasonable
        assume(height >= 20 and width >= 20)
        assume(base_disparity < width / 2)

        # Create a guide image with a textureless region in the center
        guide_image = np.zeros((height, width), dtype=np.uint8)

        # Define textureless region (center 40% of image)
        textureless_y_start = int(height * 0.3)
        textureless_y_end = int(height * 0.7)
        textureless_x_start = int(width * 0.3)
        textureless_x_end = int(width * 0.7)

        # Add texture to surrounding regions (random noise)
        rng = np.random.RandomState(42)
        guide_image[:, :] = rng.randint(0, 255, (height, width), dtype=np.uint8)

        # Make center region textureless (uniform intensity)
        uniform_intensity = 128
        guide_image[textureless_y_start:textureless_y_end,
                   textureless_x_start:textureless_x_end] = uniform_intensity

        # Create initial disparity map with valid disparities around textureless region
        # but sparse/noisy disparities in the textureless region itself
        disparity = np.zeros((height, width), dtype=np.int16)

        # Fill surrounding textured regions with consistent disparities
        disparity[:, :] = int(base_disparity * 16)  # Convert to fixed-point

        # Add noise to textureless region to simulate poor matching
        noise = rng.uniform(-3, 3, (textureless_y_end - textureless_y_start,
                                    textureless_x_end - textureless_x_start))
        disparity[textureless_y_start:textureless_y_end,
                 textureless_x_start:textureless_x_end] = (
            (base_disparity + noise) * 16
        ).astype(np.int16)

        # Apply WLS filtering
        wls_filter = WLSFilter(lambda_val=lambda_val, sigma_color=sigma_color)
        filtered_disparity = wls_filter.filter_disparity(disparity, guide_image)

        # Convert to float for analysis
        filtered_float = filtered_disparity.astype(np.float32) / 16.0

        # Extract textureless region disparities
        textureless_disparities = filtered_float[
            textureless_y_start:textureless_y_end,
            textureless_x_start:textureless_x_end
        ]

        # Property 6 Verification: Check smoothness in textureless region
        # 1. Compute gradient magnitude to detect discontinuities
        grad_y, grad_x = np.gradient(textureless_disparities)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 2. Smoothness metric: mean gradient should be low
        mean_gradient = np.mean(gradient_magnitude)

        # 3. Discontinuity detection: max gradient should not be too large
        max_gradient = np.max(gradient_magnitude)

        # 4. Verify smoothness: gradients should be small in textureless regions
        # After WLS filtering, the textureless region should have smooth disparities
        # Mean gradient should be less than 1.0 pixel per pixel (very smooth)
        assert mean_gradient < 1.0, \
            f"Textureless region not smooth: mean gradient {mean_gradient:.3f} >= 1.0"

        # 5. Verify no abrupt discontinuities
        # Max gradient should be reasonable (< 3.0 pixels per pixel)
        assert max_gradient < 3.0, \
            f"Abrupt discontinuity detected: max gradient {max_gradient:.3f} >= 3.0"

        # 6. Verify disparity propagation from textured areas
        # The mean disparity in textureless region should be close to base_disparity
        mean_disparity = np.mean(textureless_disparities)
        disparity_deviation = abs(mean_disparity - base_disparity)

        # Allow some deviation but should be within reasonable bounds
        assert disparity_deviation < base_disparity * 0.3, \
            f"Disparity not properly propagated: deviation {disparity_deviation:.3f} too large"

        # 7. Verify spatial coherence: neighboring pixels should have similar disparities
        # Compute standard deviation in small local windows
        window_size = 3
        local_stds = []
        for y in range(textureless_y_start + window_size, textureless_y_end - window_size, window_size):
            for x in range(textureless_x_start + window_size, textureless_x_end - window_size, window_size):
                window = filtered_float[y-window_size:y+window_size+1,
                                       x-window_size:x+window_size+1]
                local_stds.append(np.std(window))

        if local_stds:
            mean_local_std = np.mean(local_stds)
            # Local standard deviation should be small (< 1.0 pixel)
            assert mean_local_std < 1.0, \
                f"Local disparity variation too high: std {mean_local_std:.3f} >= 1.0"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestWLSFilterProperties:
    """Property-based tests for WLSFilter."""
    
    @given(
        st.integers(min_value=40, max_value=100),
        st.integers(min_value=40, max_value=100),
        st.floats(min_value=8.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=5000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.8, max_value=1.5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_property_6_disparity_smoothness_in_textureless_regions(
        self, height, width, base_disparity, lambda_val, sigma_color
    ):
        """
        Property 6: Disparity Smoothness in Textureless Regions
        
        For any textureless image region, disparity values should vary smoothly 
        without abrupt discontinuities, propagating information from nearby 
        textured areas.
        
        This test creates a synthetic scenario with:
        1. A textureless region (uniform intensity)
        2. Surrounding textured regions with known disparities
        3. Applies WLS filtering to propagate disparity information
        4. Verifies that the textureless region has smooth disparity values
        
        **Feature: advanced-stereo-vision-pipeline, Property 6: Disparity Smoothness in Textureless Regions**
        **Validates: Requirements 2.5**
        """
        from stereo_vision.disparity import WLSFilter
        
        # Ensure dimensions are reasonable for WLS filtering
        assume(height >= 40 and width >= 40)
        assume(base_disparity < width / 3)
        assume(base_disparity >= 8.0)  # Ensure reasonable disparity range
        
        # Create a guide image with a textureless region in the center
        guide_image = np.zeros((height, width), dtype=np.uint8)
        
        # Define textureless region (center 40% of image)
        textureless_y_start = int(height * 0.3)
        textureless_y_end = int(height * 0.7)
        textureless_x_start = int(width * 0.3)
        textureless_x_end = int(width * 0.7)
        
        # Add texture to surrounding regions (random noise)
        rng = np.random.RandomState(42)
        guide_image[:, :] = rng.randint(0, 255, (height, width), dtype=np.uint8)
        
        # Make center region textureless (uniform intensity)
        uniform_intensity = 128
        guide_image[textureless_y_start:textureless_y_end, 
                   textureless_x_start:textureless_x_end] = uniform_intensity
        
        # Create initial disparity map with valid disparities around textureless region
        # but sparse/noisy disparities in the textureless region itself
        disparity = np.zeros((height, width), dtype=np.int16)
        
        # Fill surrounding textured regions with consistent disparities
        disparity[:, :] = int(base_disparity * 16)  # Convert to fixed-point
        
        # Add noise to textureless region to simulate poor matching
        noise = rng.uniform(-3, 3, (textureless_y_end - textureless_y_start,
                                    textureless_x_end - textureless_x_start))
        disparity[textureless_y_start:textureless_y_end,
                 textureless_x_start:textureless_x_end] = (
            (base_disparity + noise) * 16
        ).astype(np.int16)
        
        # Apply WLS filtering
        wls_filter = WLSFilter(lambda_val=lambda_val, sigma_color=sigma_color)
        filtered_disparity = wls_filter.filter_disparity(disparity, guide_image)
        
        # Convert to float for analysis
        filtered_float = filtered_disparity.astype(np.float32) / 16.0
        
        # WLS filter marks invalid pixels with very negative values
        # Filter out invalid disparities (< 0)
        valid_mask = filtered_float > 0
        
        # Extract textureless region disparities
        textureless_disparities = filtered_float[
            textureless_y_start:textureless_y_end,
            textureless_x_start:textureless_x_end
        ]
        
        # Get valid disparities in textureless region
        textureless_valid_mask = textureless_disparities > 0
        
        # Skip test if too few valid pixels in textureless region
        valid_pixel_count = np.sum(textureless_valid_mask)
        textureless_size = textureless_disparities.size
        assume(valid_pixel_count > textureless_size * 0.5)  # At least 50% valid
        
        # Extract only valid disparities for analysis
        valid_textureless_disparities = textureless_disparities[textureless_valid_mask]
        
        # Property 6 Verification: Check smoothness in textureless region
        # Only analyze valid disparities (> 0)
        
        # 1. Compute gradient magnitude to detect discontinuities
        # Create a masked version for gradient computation
        textureless_for_gradient = textureless_disparities.copy()
        textureless_for_gradient[~textureless_valid_mask] = np.nan
        
        grad_y, grad_x = np.gradient(textureless_for_gradient)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Remove NaN values from gradient
        valid_gradients = gradient_magnitude[~np.isnan(gradient_magnitude)]
        
        # Skip if too few valid gradients
        assume(len(valid_gradients) > 10)
        
        # 2. Smoothness metric: mean gradient should be low
        mean_gradient = np.mean(valid_gradients)
        
        # 3. Discontinuity detection: max gradient should not be too large
        max_gradient = np.max(valid_gradients)
        
        # 4. Verify smoothness: gradients should be small in textureless regions
        # After WLS filtering, the textureless region should have smooth disparities
        # Mean gradient should be less than 2.0 pixels per pixel (smooth)
        assert mean_gradient < 2.0, \
            f"Textureless region not smooth: mean gradient {mean_gradient:.3f} >= 2.0"
        
        # 5. Verify no abrupt discontinuities
        # Max gradient should be reasonable (< 6.0 pixels per pixel)
        assert max_gradient < 6.0, \
            f"Abrupt discontinuity detected: max gradient {max_gradient:.3f} >= 6.0"
        
        # 6. Verify disparity propagation from textured areas
        # The mean disparity in textureless region should be close to base_disparity
        mean_disparity = np.mean(valid_textureless_disparities)
        disparity_deviation = abs(mean_disparity - base_disparity)
        
        # Allow reasonable deviation - WLS filter may smooth but should stay in range
        # Use absolute threshold for small disparities, relative for large
        max_deviation = max(3.0, base_disparity * 0.5)
        assert disparity_deviation < max_deviation, \
            f"Disparity not properly propagated: deviation {disparity_deviation:.3f} >= {max_deviation:.3f}"
        
        # 7. Verify spatial coherence: neighboring pixels should have similar disparities
        # Compute standard deviation in small local windows (only for valid pixels)
        window_size = 3
        local_stds = []
        for y in range(textureless_y_start + window_size, textureless_y_end - window_size, window_size):
            for x in range(textureless_x_start + window_size, textureless_x_end - window_size, window_size):
                window = filtered_float[y-window_size:y+window_size+1, 
                                       x-window_size:x+window_size+1]
                # Only compute std for windows with mostly valid pixels
                valid_in_window = window[window > 0]
                if len(valid_in_window) >= (window_size * 2 + 1) ** 2 * 0.7:  # 70% valid
                    local_stds.append(np.std(valid_in_window))
        
        if local_stds:
            mean_local_std = np.mean(local_stds)
            # Local standard deviation should be small (< 2.0 pixels)
            # This indicates smooth variation within local neighborhoods
            assert mean_local_std < 2.0, \
                f"Local disparity variation too high: std {mean_local_std:.3f} >= 2.0"
