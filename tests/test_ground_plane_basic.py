"""Basic unit tests for V-Disparity ground plane detection module."""

import numpy as np
import cv2
import pytest
from stereo_vision.ground_plane import VDisparityGenerator, HoughLineDetector, GroundPlaneModel


class TestVDisparityGenerator:
    """Unit tests for VDisparityGenerator class."""
    
    def test_initialization(self):
        """Test VDisparityGenerator initialization."""
        # Test with default max_disparity
        generator = VDisparityGenerator()
        assert generator.max_disparity is None
        
        # Test with specified max_disparity
        generator = VDisparityGenerator(max_disparity=128)
        assert generator.max_disparity == 128
    
    def test_generate_v_disparity_basic(self):
        """Test basic V-Disparity generation from a simple disparity map."""
        # Create a simple disparity map with known pattern
        height, width = 100, 200
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        # Create a diagonal pattern (simulating a planar surface)
        # Disparity increases linearly with row number
        for row in range(height):
            disparity_value = 10 + row * 0.5  # Linear increase
            disparity_map[row, :] = disparity_value
        
        # Generate V-Disparity
        generator = VDisparityGenerator(max_disparity=100)
        v_disp = generator.generate_v_disparity(disparity_map)
        
        # Check output shape
        assert v_disp.shape == (height, 100)
        assert v_disp.dtype == np.uint8
        
        # Check that V-Disparity has non-zero values
        assert np.sum(v_disp) > 0
    
    def test_generate_v_disparity_fixed_point(self):
        """Test V-Disparity generation from 16-bit fixed-point disparity."""
        height, width = 50, 100
        
        # Create fixed-point disparity (multiply by 16)
        disparity_float = np.random.uniform(10, 50, (height, width)).astype(np.float32)
        disparity_fixed = (disparity_float * 16).astype(np.int16)
        
        # Generate V-Disparity
        generator = VDisparityGenerator(max_disparity=60)
        v_disp = generator.generate_v_disparity(disparity_fixed)
        
        # Check output
        assert v_disp.shape == (height, 60)
        assert v_disp.dtype == np.uint8
    
    def test_generate_v_disparity_with_invalid_pixels(self):
        """Test V-Disparity generation with invalid disparity pixels."""
        height, width = 80, 150
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        # Set some valid disparities
        disparity_map[20:60, 50:100] = 25.0
        
        # Leave rest as zero (invalid)
        
        # Generate V-Disparity
        generator = VDisparityGenerator(max_disparity=50)
        v_disp = generator.generate_v_disparity(disparity_map)
        
        # Check that only rows 20-60 have non-zero values
        assert np.sum(v_disp[0:20, :]) == 0
        assert np.sum(v_disp[20:60, :]) > 0
        assert np.sum(v_disp[60:, :]) == 0
    
    def test_generate_v_disparity_auto_max_disparity(self):
        """Test automatic max_disparity determination."""
        height, width = 60, 120
        disparity_map = np.random.uniform(5, 40, (height, width)).astype(np.float32)
        
        # Generate V-Disparity without specifying max_disparity
        generator = VDisparityGenerator()
        v_disp = generator.generate_v_disparity(disparity_map)
        
        # Check that width is based on actual max disparity
        assert v_disp.shape[0] == height
        assert v_disp.shape[1] >= 40  # Should be at least max disparity
    
    def test_generate_v_disparity_empty_map(self):
        """Test V-Disparity generation with empty disparity map."""
        height, width = 50, 100
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        # Generate V-Disparity
        generator = VDisparityGenerator()
        v_disp = generator.generate_v_disparity(disparity_map)
        
        # Should return a minimal V-Disparity image
        assert v_disp.shape[0] == height
        assert np.sum(v_disp) == 0  # All zeros
    
    def test_visualize_v_disparity(self):
        """Test V-Disparity visualization."""
        # Create a simple V-Disparity image
        v_disp = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        
        generator = VDisparityGenerator()
        v_disp_colored = generator.visualize_v_disparity(v_disp)
        
        # Check output
        assert v_disp_colored.shape == (100, 80, 3)  # BGR image
        assert v_disp_colored.dtype == np.uint8
    
    def test_visualize_v_disparity_different_colormaps(self):
        """Test V-Disparity visualization with different colormaps."""
        v_disp = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        generator = VDisparityGenerator()
        
        # Test different colormaps
        colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS]
        
        for colormap in colormaps:
            v_disp_colored = generator.visualize_v_disparity(v_disp, colormap=colormap)
            assert v_disp_colored.shape == (100, 80, 3)
            assert v_disp_colored.dtype == np.uint8
    
    def test_visualize_with_annotations(self):
        """Test annotated visualization with ground plane line."""
        v_disp = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        generator = VDisparityGenerator()
        
        # Test without ground line
        v_disp_colored = generator.visualize_with_annotations(v_disp)
        assert v_disp_colored.shape == (100, 80, 3)
        
        # Test with ground line
        ground_line = (0.5, 10.0)  # slope, intercept
        v_disp_annotated = generator.visualize_with_annotations(v_disp, ground_line=ground_line)
        assert v_disp_annotated.shape == (100, 80, 3)
        
        # The annotated version should be different from the non-annotated
        # (due to the drawn line)
        assert not np.array_equal(v_disp_colored, v_disp_annotated)
    
    def test_get_histogram_statistics(self):
        """Test histogram statistics computation."""
        # Create a V-Disparity image with known properties
        v_disp = np.zeros((100, 80), dtype=np.uint8)
        v_disp[20:60, 30:50] = 200  # High intensity region
        v_disp[10:30, 10:20] = 50   # Low intensity region
        
        generator = VDisparityGenerator()
        stats = generator.get_histogram_statistics(v_disp)
        
        # Check that all expected keys are present
        assert 'mean_intensity' in stats
        assert 'max_intensity' in stats
        assert 'non_zero_ratio' in stats
        assert 'dominant_disparity_range' in stats
        
        # Check values
        assert stats['max_intensity'] == 200
        assert 0 < stats['mean_intensity'] < 200
        assert 0 < stats['non_zero_ratio'] < 1
        assert isinstance(stats['dominant_disparity_range'], tuple)
        assert len(stats['dominant_disparity_range']) == 2
    
    def test_get_histogram_statistics_empty(self):
        """Test histogram statistics with empty V-Disparity."""
        v_disp = np.zeros((100, 80), dtype=np.uint8)
        
        generator = VDisparityGenerator()
        stats = generator.get_histogram_statistics(v_disp)
        
        # Check values for empty histogram
        assert stats['mean_intensity'] == 0.0
        assert stats['max_intensity'] == 0
        assert stats['non_zero_ratio'] == 0.0
        assert stats['dominant_disparity_range'] == (0, 0)
    
    def test_road_surface_pattern(self):
        """Test V-Disparity generation for a simulated road surface."""
        # Simulate a road surface where disparity increases with row
        # (closer to camera = higher row = higher disparity)
        height, width = 200, 400
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        # Road surface: linear disparity increase with row
        # Typical for a flat road viewed from above
        for row in range(height):
            # Disparity increases from 20 to 80 as we go down the image
            disparity_value = 20 + (row / height) * 60
            # Add some noise to simulate real disparity
            noise = np.random.normal(0, 1, width)
            disparity_map[row, :] = disparity_value + noise
        
        # Generate V-Disparity
        generator = VDisparityGenerator(max_disparity=100)
        v_disp = generator.generate_v_disparity(disparity_map)
        
        # For a planar road surface, we expect a diagonal line pattern
        # Check that each row has a dominant disparity value
        for row in range(0, height, 20):  # Sample every 20 rows
            row_histogram = v_disp[row, :]
            # Should have a peak (dominant disparity)
            assert np.max(row_histogram) > 0
            
            # The peak should be roughly at the expected disparity
            expected_disp = int(20 + (row / height) * 60)
            peak_disp = np.argmax(row_histogram)
            # Allow some tolerance due to noise
            assert abs(peak_disp - expected_disp) < 10



class TestHoughLineDetector:
    """Unit tests for HoughLineDetector class."""
    
    def test_initialization(self):
        """Test HoughLineDetector initialization."""
        # Test with default parameters
        detector = HoughLineDetector()
        assert detector.rho_resolution == 1.0
        assert detector.theta_resolution == np.pi / 180
        assert detector.threshold is None
        
        # Test with custom parameters
        detector = HoughLineDetector(
            rho_resolution=0.5,
            theta_resolution=np.pi / 360,
            threshold=50
        )
        assert detector.rho_resolution == 0.5
        assert detector.theta_resolution == np.pi / 360
        assert detector.threshold == 50
    
    def test_detect_dominant_line_diagonal_pattern(self):
        """Test Hough line detection on a clear diagonal pattern."""
        # Create a V-Disparity image with a clear diagonal line
        height, width = 200, 100
        v_disp = np.zeros((height, width), dtype=np.uint8)
        
        # Draw a diagonal line: disparity = 0.5 * row + 10
        slope_true = 0.5
        intercept_true = 10.0
        
        for row in range(height):
            disp = int(slope_true * row + intercept_true)
            if 0 <= disp < width:
                # Draw a thicker line with higher intensity for better detection
                for offset in range(-3, 4):
                    if 0 <= disp + offset < width:
                        v_disp[row, disp + offset] = 255
        
        # Detect line (use lower threshold for synthetic data)
        detector = HoughLineDetector(threshold=15)
        line_params = detector.detect_dominant_line(v_disp)
        
        # Check that line was detected
        assert line_params is not None
        slope, intercept = line_params
        
        # Check that detected parameters are close to true values
        # Note: Hough Transform may detect edge of thick line, so allow larger tolerance
        assert abs(slope - slope_true) < 0.3
        assert abs(intercept - intercept_true) < 15.0  # Increased tolerance for thick lines
    
    def test_detect_dominant_line_no_line(self):
        """Test Hough line detection on image with no clear line."""
        # Create a V-Disparity image with random noise
        height, width = 100, 80
        v_disp = np.random.randint(0, 50, (height, width), dtype=np.uint8)
        
        # Detect line
        detector = HoughLineDetector(threshold=50)
        line_params = detector.detect_dominant_line(v_disp)
        
        # May or may not detect a line in noise, but should not crash
        # If detected, should return valid parameters
        if line_params is not None:
            slope, intercept = line_params
            assert isinstance(slope, (int, float, np.number))
            assert isinstance(intercept, (int, float, np.number))
    
    def test_detect_dominant_line_realistic_road(self):
        """Test Hough line detection on realistic road V-Disparity."""
        # Create a more realistic V-Disparity with a diagonal band
        height, width = 150, 120
        v_disp = np.zeros((height, width), dtype=np.uint8)
        
        # Simulate road surface with some width and noise
        slope_true = 0.4
        intercept_true = 15.0
        
        for row in range(height):
            center_disp = slope_true * row + intercept_true
            
            # Add a band of pixels around the center line
            for disp_offset in range(-5, 6):
                disp = int(center_disp + disp_offset)
                if 0 <= disp < width:
                    # Intensity decreases away from center
                    intensity = max(0, 200 - abs(disp_offset) * 30)
                    v_disp[row, disp] = intensity
        
        # Detect line
        detector = HoughLineDetector()
        line_params = detector.detect_dominant_line(v_disp)
        
        # Check detection
        assert line_params is not None
        slope, intercept = line_params
        
        # Should be reasonably close to true values
        assert abs(slope - slope_true) < 0.3
        assert abs(intercept - intercept_true) < 10.0
    
    def test_visualize_detected_line(self):
        """Test line visualization on V-Disparity."""
        # Create a simple V-Disparity
        height, width = 100, 80
        v_disp = np.random.randint(0, 100, (height, width), dtype=np.uint8)
        
        # Create line parameters
        line_params = (0.5, 10.0)
        
        # Visualize
        detector = HoughLineDetector()
        v_disp_with_line = detector.visualize_detected_line(v_disp, line_params)
        
        # Check output
        assert v_disp_with_line.shape == (height, width, 3)
        assert v_disp_with_line.dtype == np.uint8
        
        # Should be different from input (line was drawn)
        assert not np.array_equal(v_disp_with_line[:, :, 0], v_disp)


class TestGroundPlaneModel:
    """Unit tests for GroundPlaneModel class."""
    
    def test_initialization(self):
        """Test GroundPlaneModel initialization."""
        # Test with default threshold
        model = GroundPlaneModel()
        assert model.threshold_factor == 1.5
        assert model.is_fitted is False
        assert model.slope is None
        assert model.intercept is None
        
        # Test with custom threshold
        model = GroundPlaneModel(threshold_factor=2.0)
        assert model.threshold_factor == 2.0
    
    def test_fit_from_line_params(self):
        """Test direct fitting from line parameters."""
        model = GroundPlaneModel()
        
        # Fit with known parameters
        slope = 0.5
        intercept = 10.0
        model.fit_from_line_params(slope, intercept)
        
        # Check fitted state
        assert model.is_fitted is True
        assert model.slope == slope
        assert model.intercept == intercept
    
    def test_get_expected_disparity(self):
        """Test expected disparity calculation."""
        model = GroundPlaneModel()
        model.fit_from_line_params(slope=0.5, intercept=10.0)
        
        # Test at various rows
        assert model.get_expected_disparity(0) == 10.0
        assert model.get_expected_disparity(10) == 15.0
        assert model.get_expected_disparity(20) == 20.0
        assert model.get_expected_disparity(100) == 60.0
    
    def test_get_expected_disparity_not_fitted(self):
        """Test that get_expected_disparity raises error when not fitted."""
        model = GroundPlaneModel()
        
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_expected_disparity(50)
    
    def test_fit_from_v_disparity(self):
        """Test fitting from V-Disparity image."""
        # Create a V-Disparity with clear diagonal line
        height, width = 150, 100
        v_disp = np.zeros((height, width), dtype=np.uint8)
        
        slope_true = 0.4
        intercept_true = 15.0
        
        for row in range(height):
            disp = int(slope_true * row + intercept_true)
            if 0 <= disp < width:
                for offset in range(-3, 4):
                    if 0 <= disp + offset < width:
                        v_disp[row, disp + offset] = 200
        
        # Fit model
        model = GroundPlaneModel()
        success = model.fit_from_v_disparity(v_disp)
        
        # Check fitting
        assert success is True
        assert model.is_fitted is True
        assert model.slope is not None
        assert model.intercept is not None
        
        # Parameters should be reasonably close
        assert abs(model.slope - slope_true) < 0.3
        assert abs(model.intercept - intercept_true) < 10.0
    
    def test_segment_anomalies_basic(self):
        """Test basic anomaly segmentation."""
        # Create a ground plane model
        model = GroundPlaneModel(threshold_factor=1.0)
        model.fit_from_line_params(slope=0.5, intercept=10.0)
        
        # Create a disparity map with known anomalies
        height, width = 100, 150
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        # Fill with ground plane disparities
        for row in range(height):
            expected_disp = model.get_expected_disparity(row)
            disparity_map[row, :] = expected_disp
        
        # Add a pothole (lower disparity = farther)
        disparity_map[40:50, 60:80] -= 10.0
        
        # Add a hump (higher disparity = closer)
        disparity_map[60:70, 90:110] += 10.0
        
        # Segment anomalies
        pothole_mask, hump_mask = model.segment_anomalies(disparity_map)
        
        # Check masks
        assert pothole_mask.shape == (height, width)
        assert hump_mask.shape == (height, width)
        assert pothole_mask.dtype == np.uint8
        assert hump_mask.dtype == np.uint8
        
        # Check that anomalies were detected
        assert np.sum(pothole_mask) > 0
        assert np.sum(hump_mask) > 0
        
        # Check that pothole region has high values in pothole mask
        pothole_region = pothole_mask[40:50, 60:80]
        assert np.mean(pothole_region) > 100
        
        # Check that hump region has high values in hump mask
        hump_region = hump_mask[60:70, 90:110]
        assert np.mean(hump_region) > 100
    
    def test_segment_anomalies_not_fitted(self):
        """Test that segment_anomalies raises error when not fitted."""
        model = GroundPlaneModel()
        disparity_map = np.zeros((100, 150), dtype=np.float32)
        
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.segment_anomalies(disparity_map)
    
    def test_segment_anomalies_fixed_point(self):
        """Test anomaly segmentation with fixed-point disparity."""
        model = GroundPlaneModel()
        model.fit_from_line_params(slope=0.5, intercept=10.0)
        
        # Create fixed-point disparity map
        height, width = 80, 120
        disparity_float = np.zeros((height, width), dtype=np.float32)
        
        for row in range(height):
            disparity_float[row, :] = model.get_expected_disparity(row)
        
        # Convert to fixed-point
        disparity_fixed = (disparity_float * 16).astype(np.int16)
        
        # Segment (should handle fixed-point correctly)
        pothole_mask, hump_mask = model.segment_anomalies(disparity_fixed)
        
        # Should produce valid masks
        assert pothole_mask.shape == (height, width)
        assert hump_mask.shape == (height, width)
    
    def test_get_plane_parameters(self):
        """Test getting plane parameters."""
        model = GroundPlaneModel()
        
        # Before fitting
        assert model.get_plane_parameters() is None
        
        # After fitting
        model.fit_from_line_params(slope=0.6, intercept=12.0)
        params = model.get_plane_parameters()
        
        assert params is not None
        assert params == (0.6, 12.0)
    
    def test_visualize_ground_plane(self):
        """Test ground plane visualization."""
        model = GroundPlaneModel()
        model.fit_from_line_params(slope=0.5, intercept=10.0)
        
        # Create a simple disparity map
        height, width = 100, 150
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        for row in range(height):
            disparity_map[row, :] = model.get_expected_disparity(row) + np.random.normal(0, 1, width)
        
        # Visualize
        vis = model.visualize_ground_plane(disparity_map)
        
        # Check output
        assert vis.shape == (height, width, 3)
        assert vis.dtype == np.uint8
    
    def test_visualize_ground_plane_not_fitted(self):
        """Test that visualization raises error when not fitted."""
        model = GroundPlaneModel()
        disparity_map = np.zeros((100, 150), dtype=np.float32)
        
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.visualize_ground_plane(disparity_map)
