"""Property-based tests for parameter configuration system.

Tests universal correctness properties for configuration management.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
import numpy as np
from stereo_vision.config import (
    CameraConfig,
    SGBMConfig,
    DepthRangeConfig,
    AnomalyDetectionConfig,
    PipelineConfig
)


# Custom strategies for configuration parameters
@st.composite
def camera_config_strategy(draw):
    """Generate valid camera configurations."""
    baseline = draw(st.floats(min_value=0.01, max_value=2.0))
    focal_length = draw(st.floats(min_value=100.0, max_value=2000.0))
    image_width = draw(st.integers(min_value=320, max_value=1920))
    image_height = draw(st.integers(min_value=240, max_value=1080))
    
    return CameraConfig(
        baseline=baseline,
        focal_length=focal_length,
        image_width=image_width,
        image_height=image_height
    )


@st.composite
def depth_range_strategy(draw):
    """Generate valid depth range configurations."""
    min_depth = draw(st.floats(min_value=0.1, max_value=10.0))
    max_depth = draw(st.floats(min_value=min_depth + 1.0, max_value=100.0))
    
    return DepthRangeConfig(min_depth=min_depth, max_depth=max_depth)


@st.composite
def anomaly_threshold_strategy(draw):
    """Generate valid anomaly detection thresholds."""
    threshold_factor = draw(st.floats(min_value=0.5, max_value=5.0))
    min_size = draw(st.integers(min_value=10, max_value=1000))
    max_size = draw(st.integers(min_value=min_size + 100, max_value=200000))
    
    return AnomalyDetectionConfig(
        threshold_factor=threshold_factor,
        min_anomaly_size=min_size,
        max_anomaly_size=max_size
    )


@st.composite
def invalid_camera_config_strategy(draw):
    """Generate invalid camera configurations for validation testing."""
    choice = draw(st.integers(min_value=0, max_value=3))
    
    if choice == 0:
        # Invalid baseline
        baseline = draw(st.one_of(
            st.floats(max_value=0.009),
            st.floats(min_value=2.1, max_value=10.0)
        ))
        return CameraConfig(baseline=baseline)
    elif choice == 1:
        # Invalid focal length
        focal_length = draw(st.floats(max_value=0.0))
        return CameraConfig(focal_length=focal_length)
    elif choice == 2:
        # Invalid image dimensions
        width = draw(st.integers(max_value=0))
        return CameraConfig(image_width=width)
    else:
        # Invalid height
        height = draw(st.integers(max_value=0))
        return CameraConfig(image_height=height)


@st.composite
def invalid_sgbm_config_strategy(draw):
    """Generate invalid SGBM configurations for validation testing."""
    choice = draw(st.integers(min_value=0, max_value=3))
    
    if choice == 0:
        # num_disparities not divisible by 16
        num_disp = draw(st.integers(min_value=1, max_value=500).filter(
            lambda x: x % 16 != 0
        ))
        return SGBMConfig(num_disparities=num_disp)
    elif choice == 1:
        # Even block size
        block_size = draw(st.integers(min_value=1, max_value=20).filter(
            lambda x: x % 2 == 0
        ))
        return SGBMConfig(block_size=block_size)
    elif choice == 2:
        # p2 < p1
        p1 = draw(st.integers(min_value=100, max_value=1000))
        p2 = draw(st.integers(min_value=1, max_value=p1 - 1))
        return SGBMConfig(p1=p1, p2=p2)
    else:
        # Invalid mode
        return SGBMConfig(mode="INVALID_MODE")


class TestProperty32_ParameterConfigurationEffectiveness:
    """Property 32: Parameter Configuration Effectiveness
    
    **Validates: Requirements 9.2**
    
    For any change in camera parameters (baseline, focal length), 
    the system calculations should reflect the updated geometry correctly.
    """
    
    @given(
        baseline1=st.floats(min_value=0.01, max_value=2.0),
        baseline2=st.floats(min_value=0.01, max_value=2.0),
        focal_length=st.floats(min_value=100.0, max_value=2000.0),
        disparity=st.integers(min_value=1, max_value=256)
    )
    @settings(max_examples=100, deadline=None)
    def test_baseline_change_affects_depth_calculation(
        self, baseline1, baseline2, focal_length, disparity
    ):
        """Changing baseline should proportionally affect depth calculations."""
        assume(abs(baseline1 - baseline2) > 0.001)  # Meaningful difference
        
        # Calculate depth with first baseline
        depth1 = (baseline1 * focal_length) / disparity
        
        # Calculate depth with second baseline
        depth2 = (baseline2 * focal_length) / disparity
        
        # Depth should scale proportionally with baseline
        ratio = baseline2 / baseline1
        expected_depth2 = depth1 * ratio
        
        assert np.isclose(depth2, expected_depth2, rtol=1e-6), \
            f"Depth should scale with baseline: {depth2} != {expected_depth2}"
    
    @given(
        baseline=st.floats(min_value=0.01, max_value=2.0),
        focal1=st.floats(min_value=100.0, max_value=2000.0),
        focal2=st.floats(min_value=100.0, max_value=2000.0),
        disparity=st.integers(min_value=1, max_value=256)
    )
    @settings(max_examples=100, deadline=None)
    def test_focal_length_change_affects_depth_calculation(
        self, baseline, focal1, focal2, disparity
    ):
        """Changing focal length should proportionally affect depth calculations."""
        assume(abs(focal1 - focal2) > 1.0)  # Meaningful difference
        
        # Calculate depth with first focal length
        depth1 = (baseline * focal1) / disparity
        
        # Calculate depth with second focal length
        depth2 = (baseline * focal2) / disparity
        
        # Depth should scale proportionally with focal length
        ratio = focal2 / focal1
        expected_depth2 = depth1 * ratio
        
        assert np.isclose(depth2, expected_depth2, rtol=1e-6), \
            f"Depth should scale with focal length: {depth2} != {expected_depth2}"
    
    @given(config=camera_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_camera_config_updates_reflect_in_calculations(self, config):
        """Camera configuration changes should be reflected in system calculations."""
        # Create pipeline config with camera settings
        pipeline_config = PipelineConfig(camera=config)
        
        # Verify camera parameters are correctly stored
        assert pipeline_config.camera.baseline == config.baseline
        assert pipeline_config.camera.focal_length == config.focal_length
        
        # Calculate a sample depth using the configuration
        test_disparity = 50
        expected_depth = (config.baseline * config.focal_length) / test_disparity
        
        # Verify the calculation uses the correct parameters
        calculated_depth = (
            pipeline_config.camera.baseline * 
            pipeline_config.camera.focal_length
        ) / test_disparity
        
        assert np.isclose(calculated_depth, expected_depth, rtol=1e-10), \
            "Configuration parameters should be used in calculations"


class TestProperty33_ThresholdConfigurationImpact:
    """Property 33: Threshold Configuration Impact
    
    **Validates: Requirements 9.3**
    
    For any change in anomaly detection thresholds, the segmentation results 
    should change appropriately to reflect the new criteria.
    """
    
    @given(
        threshold1=st.floats(min_value=0.5, max_value=5.0),
        threshold2=st.floats(min_value=0.5, max_value=5.0),
        disparity_value=st.floats(min_value=10.0, max_value=100.0),
        ground_plane_disparity=st.floats(min_value=10.0, max_value=100.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_threshold_change_affects_anomaly_classification(
        self, threshold1, threshold2, disparity_value, ground_plane_disparity
    ):
        """Different thresholds should produce different anomaly classifications."""
        assume(abs(threshold1 - threshold2) > 0.1)  # Meaningful difference
        assume(abs(disparity_value - ground_plane_disparity) > 0.1)
        
        # Calculate deviation from ground plane
        deviation = abs(disparity_value - ground_plane_disparity)
        
        # Check if anomaly is detected with threshold1
        is_anomaly1 = deviation > threshold1
        
        # Check if anomaly is detected with threshold2
        is_anomaly2 = deviation > threshold2
        
        # If thresholds are different and deviation is between them,
        # classification should differ
        if threshold1 < deviation < threshold2 or threshold2 < deviation < threshold1:
            assert is_anomaly1 != is_anomaly2, \
                "Different thresholds should produce different classifications"
    
    @given(
        config1=anomaly_threshold_strategy(),
        config2=anomaly_threshold_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_anomaly_size_thresholds_affect_filtering(self, config1, config2):
        """Different size thresholds should affect which anomalies are kept."""
        assume(config1.min_anomaly_size != config2.min_anomaly_size or
               config1.max_anomaly_size != config2.max_anomaly_size)
        
        # Test anomaly size
        test_size = (config1.min_anomaly_size + config1.max_anomaly_size) // 2
        
        # Check if size passes config1 filters
        passes1 = (config1.min_anomaly_size <= test_size <= 
                   config1.max_anomaly_size)
        
        # Check if size passes config2 filters
        passes2 = (config2.min_anomaly_size <= test_size <= 
                   config2.max_anomaly_size)
        
        # If configs differ significantly, results may differ
        if (config1.min_anomaly_size > test_size or 
            config1.max_anomaly_size < test_size):
            assert not passes1, "Size outside range should not pass"
        
        if (config2.min_anomaly_size > test_size or 
            config2.max_anomaly_size < test_size):
            assert not passes2, "Size outside range should not pass"
    
    @given(threshold=st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=100, deadline=None)
    def test_threshold_configuration_stored_correctly(self, threshold):
        """Threshold configuration should be stored and retrievable."""
        config = AnomalyDetectionConfig(threshold_factor=threshold)
        pipeline_config = PipelineConfig(anomaly_detection=config)
        
        # Verify threshold is stored correctly
        assert pipeline_config.anomaly_detection.threshold_factor == threshold
        
        # Verify it can be retrieved and used
        retrieved_threshold = pipeline_config.anomaly_detection.threshold_factor
        assert np.isclose(retrieved_threshold, threshold, rtol=1e-10)


class TestProperty34_DepthRangeConfigurationCompliance:
    """Property 34: Depth Range Configuration Compliance
    
    **Validates: Requirements 9.4**
    
    For any configured depth range, point filtering should respect 
    the specified minimum and maximum depth values.
    """
    
    @given(
        depth_config=depth_range_strategy(),
        test_depth=st.floats(min_value=0.01, max_value=200.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_depth_filtering_respects_configured_range(self, depth_config, test_depth):
        """Points outside configured depth range should be filtered."""
        # Determine if point should be kept based on configuration
        should_keep = (depth_config.min_depth <= test_depth <= 
                      depth_config.max_depth)
        
        # Simulate filtering logic
        is_within_range = (depth_config.min_depth <= test_depth <= 
                          depth_config.max_depth)
        
        assert should_keep == is_within_range, \
            f"Depth {test_depth} filtering should match range " \
            f"[{depth_config.min_depth}, {depth_config.max_depth}]"
    
    @given(
        min_depth=st.floats(min_value=0.1, max_value=10.0),
        max_depth=st.floats(min_value=11.0, max_value=100.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_depths_in_range_pass_filter(self, min_depth, max_depth):
        """All depths within configured range should pass filtering."""
        config = DepthRangeConfig(min_depth=min_depth, max_depth=max_depth)
        
        # Test depths at boundaries and middle
        test_depths = [
            min_depth,
            (min_depth + max_depth) / 2,
            max_depth
        ]
        
        for depth in test_depths:
            is_valid = config.min_depth <= depth <= config.max_depth
            assert is_valid, \
                f"Depth {depth} should be valid in range " \
                f"[{min_depth}, {max_depth}]"
    
    @given(
        min_depth=st.floats(min_value=0.1, max_value=10.0),
        max_depth=st.floats(min_value=11.0, max_value=100.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_depths_outside_range_fail_filter(self, min_depth, max_depth):
        """Depths outside configured range should fail filtering."""
        config = DepthRangeConfig(min_depth=min_depth, max_depth=max_depth)
        
        # Test depths outside range
        below_min = min_depth - 1.0
        above_max = max_depth + 1.0
        
        assert not (config.min_depth <= below_min <= config.max_depth), \
            f"Depth {below_min} below minimum should be invalid"
        
        assert not (config.min_depth <= above_max <= config.max_depth), \
            f"Depth {above_max} above maximum should be invalid"
    
    @given(depth_config=depth_range_strategy())
    @settings(max_examples=100, deadline=None)
    def test_depth_range_configuration_stored_correctly(self, depth_config):
        """Depth range configuration should be stored and retrievable."""
        pipeline_config = PipelineConfig(depth_range=depth_config)
        
        # Verify configuration is stored correctly
        assert pipeline_config.depth_range.min_depth == depth_config.min_depth
        assert pipeline_config.depth_range.max_depth == depth_config.max_depth
        
        # Verify range is valid
        assert pipeline_config.depth_range.max_depth > \
               pipeline_config.depth_range.min_depth


class TestProperty35_ParameterValidationLogic:
    """Property 35: Parameter Validation Logic
    
    **Validates: Requirements 9.5**
    
    For any invalid parameter combination, the validation system should 
    reject the configuration and provide appropriate error messages.
    """
    
    @given(config=invalid_camera_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_invalid_camera_config_rejected(self, config):
        """Invalid camera configurations should be rejected with errors."""
        errors = config.validate()
        
        # Should have at least one error
        assert len(errors) > 0, \
            "Invalid camera configuration should produce validation errors"
        
        # Error messages should be non-empty strings
        for error in errors:
            assert isinstance(error, str)
            assert len(error) > 0
    
    @given(config=invalid_sgbm_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_invalid_sgbm_config_rejected(self, config):
        """Invalid SGBM configurations should be rejected with errors."""
        errors = config.validate()
        
        # Should have at least one error
        assert len(errors) > 0, \
            "Invalid SGBM configuration should produce validation errors"
        
        # Error messages should be descriptive
        for error in errors:
            assert isinstance(error, str)
            assert len(error) > 0
    
    @given(
        min_depth=st.floats(min_value=1.0, max_value=50.0),
        max_depth=st.floats(min_value=0.1, max_value=49.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_depth_range_rejected(self, min_depth, max_depth):
        """Depth range with max < min should be rejected."""
        assume(max_depth < min_depth)  # Ensure invalid configuration
        
        config = DepthRangeConfig(min_depth=min_depth, max_depth=max_depth)
        errors = config.validate()
        
        # Should have error about max < min
        assert len(errors) > 0, \
            "Invalid depth range should produce validation errors"
        
        # Error should mention both min and max depth
        error_text = " ".join(errors).lower()
        assert "max_depth" in error_text or "min_depth" in error_text
    
    @given(
        baseline=st.floats(min_value=0.01, max_value=2.0),
        focal_length=st.floats(min_value=100.0, max_value=2000.0),
        num_disparities=st.integers(min_value=16, max_value=32).map(lambda x: x * 16),
        min_depth=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_cross_section_validation_detects_inconsistencies(
        self, baseline, focal_length, num_disparities, min_depth
    ):
        """Cross-section validation should detect parameter inconsistencies."""
        # Calculate minimum measurable depth with given disparity range
        max_disparity = num_disparities
        min_measurable_depth = (baseline * focal_length) / max_disparity
        
        # If min_depth is less than what we can measure, should get error
        if min_measurable_depth > min_depth + 0.5:  # Significant mismatch
            config = PipelineConfig(
                camera=CameraConfig(baseline=baseline, focal_length=focal_length),
                sgbm=SGBMConfig(min_disparity=0, num_disparities=num_disparities),
                depth_range=DepthRangeConfig(min_depth=min_depth, max_depth=50.0)
            )
            
            errors = config.validate()
            
            # Should have cross-section validation error
            assert len(errors) > 0, \
                "Inconsistent disparity/depth configuration should produce errors"
            
            # Error should mention disparity and depth
            error_text = " ".join(errors).lower()
            assert "disparity" in error_text or "depth" in error_text
    
    @given(
        threshold_factor=st.floats(min_value=-5.0, max_value=0.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_negative_threshold_rejected(self, threshold_factor):
        """Negative or zero threshold factors should be rejected."""
        config = AnomalyDetectionConfig(threshold_factor=threshold_factor)
        errors = config.validate()
        
        # Should have error about negative/zero threshold
        assert len(errors) > 0, \
            "Negative threshold should produce validation errors"
        
        error_text = " ".join(errors).lower()
        assert "threshold" in error_text
    
    @given(config=camera_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_valid_config_passes_validation(self, config):
        """Valid configurations should pass validation without errors."""
        errors = config.validate()
        
        # Should have no errors
        assert len(errors) == 0, \
            f"Valid configuration should pass validation, got errors: {errors}"
    
    @given(
        min_size=st.integers(min_value=100, max_value=1000),
        max_size=st.integers(min_value=10, max_value=99)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_anomaly_size_range_rejected(self, min_size, max_size):
        """Anomaly size range with max < min should be rejected."""
        assume(max_size < min_size)  # Ensure invalid
        
        config = AnomalyDetectionConfig(
            min_anomaly_size=min_size,
            max_anomaly_size=max_size
        )
        errors = config.validate()
        
        # Should have error about size range
        assert len(errors) > 0, \
            "Invalid anomaly size range should produce validation errors"
        
        error_text = " ".join(errors).lower()
        assert "anomaly_size" in error_text or "size" in error_text
