"""
Basic unit tests for quality metrics module.

These tests validate specific examples and edge cases for quality metrics
calculation and diagnostic visualizations.
"""

import pytest
import numpy as np
import cv2
from stereo_vision.quality_metrics import (
    LRCErrorCalculator,
    PlanarityCalculator,
    TemporalStabilityCalculator,
    CalibrationQualityReporter,
    DiagnosticVisualizer
)
from stereo_vision.calibration import CameraParameters, StereoParameters
from stereo_vision.ground_plane import GroundPlaneModel


class TestLRCErrorCalculator:
    """Tests for LRC error rate calculator."""
    
    def test_perfect_consistency(self):
        """Test with perfectly consistent disparity maps."""
        calculator = LRCErrorCalculator(max_diff=1)
        
        # Create consistent disparity pair
        disp_left = np.zeros((100, 150), dtype=np.float32)
        disp_right = np.zeros((100, 150), dtype=np.float32)
        
        # Add some consistent disparities
        for y in range(50, 70):
            for x in range(50, 100):
                d = 20.0
                disp_left[y, x] = d
                x_right = int(x - d)
                if 0 <= x_right < 150:
                    disp_right[y, x_right] = d
        
        error_rate = calculator.calculate_error_rate(disp_left, disp_right)
        
        # Should have very low error rate
        assert error_rate < 5.0
    
    def test_complete_inconsistency(self):
        """Test with completely inconsistent disparity maps."""
        calculator = LRCErrorCalculator(max_diff=1)
        
        # Create inconsistent disparity pair
        disp_left = np.random.rand(100, 150).astype(np.float32) * 50 + 1
        disp_right = np.random.rand(100, 150).astype(np.float32) * 50 + 1
        
        error_rate = calculator.calculate_error_rate(disp_left, disp_right)
        
        # Should have high error rate
        assert error_rate > 50.0
    
    def test_fixed_point_disparity(self):
        """Test with fixed-point disparity format."""
        calculator = LRCErrorCalculator(max_diff=1)
        
        # Create disparity in fixed-point format (16-bit)
        disp_left = np.zeros((100, 150), dtype=np.int16)
        disp_right = np.zeros((100, 150), dtype=np.int16)
        
        # Add disparities (multiply by 16 for fixed-point)
        for y in range(50, 70):
            for x in range(50, 100):
                d = 20 * 16  # 20 pixels in fixed-point
                disp_left[y, x] = d
                x_right = int(x - 20)
                if 0 <= x_right < 150:
                    disp_right[y, x_right] = d
        
        error_rate = calculator.calculate_error_rate(disp_left, disp_right)
        
        assert 0.0 <= error_rate <= 100.0


class TestPlanarityCalculator:
    """Tests for planarity residual calculator."""
    
    def test_perfect_plane(self):
        """Test with disparity perfectly matching ground plane."""
        calculator = PlanarityCalculator()
        
        # Create ground plane
        slope = 0.5
        intercept = 10.0
        ground_plane = GroundPlaneModel()
        ground_plane.fit_from_line_params(slope, intercept)
        
        # Create disparity matching the plane exactly
        height, width = 100, 150
        disparity = np.zeros((height, width), dtype=np.float32)
        
        for row in range(height):
            expected_disp = slope * row + intercept
            disparity[row, :] = expected_disp
        
        rmse = calculator.calculate_planarity_rmse(disparity, ground_plane)
        
        # Should have very low RMSE
        assert rmse < 0.1
    
    def test_noisy_plane(self):
        """Test with noisy disparity around ground plane."""
        calculator = PlanarityCalculator()
        
        # Create ground plane
        slope = 0.5
        intercept = 10.0
        ground_plane = GroundPlaneModel()
        ground_plane.fit_from_line_params(slope, intercept)
        
        # Create disparity with noise
        height, width = 100, 150
        disparity = np.zeros((height, width), dtype=np.float32)
        
        np.random.seed(42)
        for row in range(height):
            expected_disp = slope * row + intercept
            noise = np.random.randn(width) * 0.5
            disparity[row, :] = expected_disp + noise
        
        rmse = calculator.calculate_planarity_rmse(disparity, ground_plane)
        
        # Should have RMSE around the noise level
        assert 0.3 < rmse < 0.7


class TestTemporalStabilityCalculator:
    """Tests for temporal stability calculator."""
    
    def test_stable_sequence(self):
        """Test with very stable volume sequence."""
        calculator = TemporalStabilityCalculator()
        
        # Create stable sequence with small variations
        volumes = [5.0, 5.1, 4.9, 5.0, 5.05, 4.95]
        
        cv = calculator.calculate_temporal_stability(volumes)
        
        # Should have low CV (< 5%)
        assert cv < 5.0
    
    def test_unstable_sequence(self):
        """Test with unstable volume sequence."""
        calculator = TemporalStabilityCalculator()
        
        # Create unstable sequence with large variations
        volumes = [1.0, 5.0, 2.0, 8.0, 3.0, 6.0]
        
        cv = calculator.calculate_temporal_stability(volumes)
        
        # Should have high CV (> 30%)
        assert cv > 30.0
    
    def test_statistics(self):
        """Test comprehensive statistics calculation."""
        calculator = TemporalStabilityCalculator()
        
        volumes = [4.0, 5.0, 6.0, 5.0, 4.5]
        
        stats = calculator.calculate_stability_statistics(volumes)
        
        assert stats['num_samples'] == 5
        assert stats['min'] == 4.0
        assert stats['max'] == 6.0
        assert stats['range'] == 2.0
        assert abs(stats['mean'] - 4.9) < 0.1


class TestCalibrationQualityReporter:
    """Tests for calibration quality reporter."""
    
    def test_camera_quality_report(self):
        """Test single camera quality reporting."""
        reporter = CalibrationQualityReporter()
        
        camera_params = CameraParameters(
            camera_matrix=np.eye(3),
            distortion_coeffs=np.zeros(5),
            reprojection_error=0.25,
            image_size=(640, 480)
        )
        
        quality = reporter.report_camera_quality(camera_params)
        
        assert quality['reprojection_error'] == 0.25
        assert quality['image_width'] == 640
        assert quality['image_height'] == 480
    
    def test_stereo_quality_report(self):
        """Test stereo quality reporting."""
        reporter = CalibrationQualityReporter()
        
        left_camera = CameraParameters(
            camera_matrix=np.eye(3),
            distortion_coeffs=np.zeros(5),
            reprojection_error=0.2,
            image_size=(640, 480)
        )
        
        right_camera = CameraParameters(
            camera_matrix=np.eye(3),
            distortion_coeffs=np.zeros(5),
            reprojection_error=0.3,
            image_size=(640, 480)
        )
        
        stereo_params = StereoParameters(
            left_camera=left_camera,
            right_camera=right_camera,
            rotation_matrix=np.eye(3),
            translation_vector=np.array([0.12, 0, 0]),
            baseline=0.12,
            Q_matrix=np.eye(4),
            rectification_maps_left=(np.zeros((480, 640)), np.zeros((480, 640))),
            rectification_maps_right=(np.zeros((480, 640)), np.zeros((480, 640)))
        )
        
        quality = reporter.report_stereo_quality(stereo_params)
        
        assert quality['left_reprojection_error'] == 0.2
        assert quality['right_reprojection_error'] == 0.3
        assert quality['baseline'] == 0.12
        assert quality['mean_reprojection_error'] == 0.25


class TestDiagnosticVisualizer:
    """Tests for diagnostic visualizer."""
    
    def test_v_disparity_visualization(self):
        """Test V-Disparity visualization."""
        visualizer = DiagnosticVisualizer()
        
        # Create mock V-Disparity
        v_disp = np.random.randint(0, 255, (100, 64), dtype=np.uint8)
        
        vis = visualizer.visualize_v_disparity(v_disp)
        
        assert vis.shape == (100, 64, 3)
        assert vis.dtype == np.uint8
    
    def test_v_disparity_with_ground_line(self):
        """Test V-Disparity visualization with ground plane line."""
        visualizer = DiagnosticVisualizer()
        
        v_disp = np.random.randint(0, 255, (100, 64), dtype=np.uint8)
        ground_line = (0.5, 10.0)  # slope, intercept
        
        vis = visualizer.visualize_v_disparity(v_disp, ground_line)
        
        assert vis.shape == (100, 64, 3)
        # Check that green pixels were added for the line
        assert np.any(vis[:, :, 1] == 255)
    
    def test_anomaly_overlay(self):
        """Test anomaly overlay visualization."""
        visualizer = DiagnosticVisualizer()
        
        # Create mock image and masks
        image = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        pothole_mask = np.zeros((100, 150), dtype=np.uint8)
        hump_mask = np.zeros((100, 150), dtype=np.uint8)
        
        # Add some anomalies
        pothole_mask[30:40, 50:70] = 255
        hump_mask[60:70, 80:100] = 255
        
        overlay = visualizer.visualize_anomaly_overlay(image, pothole_mask, hump_mask)
        
        assert overlay.shape == (100, 150, 3)
        assert overlay.dtype == np.uint8
        
        # Check that anomalies are colored
        # Potholes should have blue component
        assert np.any(overlay[30:40, 50:70, 0] > 0)
        # Humps should have red component
        assert np.any(overlay[60:70, 80:100, 2] > 0)
    
    def test_ground_plane_fit_visualization(self):
        """Test ground plane fit visualization."""
        visualizer = DiagnosticVisualizer()
        
        # Create mock disparity and ground plane
        disparity = np.random.rand(100, 150).astype(np.float32) * 50 + 1
        
        ground_plane = GroundPlaneModel()
        ground_plane.fit_from_line_params(0.5, 10.0)
        
        vis = visualizer.visualize_ground_plane_fit(disparity, ground_plane)
        
        assert vis.shape == (100, 150, 3)
        assert vis.dtype == np.uint8
    
    def test_diagnostic_panel(self):
        """Test comprehensive diagnostic panel creation."""
        visualizer = DiagnosticVisualizer()
        
        # Create mock data
        original = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        disparity = np.random.rand(240, 320).astype(np.float32) * 50 + 1
        v_disp = np.random.randint(0, 255, (240, 64), dtype=np.uint8)
        
        ground_plane = GroundPlaneModel()
        ground_plane.fit_from_line_params(0.5, 10.0)
        
        pothole_mask = np.zeros((240, 320), dtype=np.uint8)
        hump_mask = np.zeros((240, 320), dtype=np.uint8)
        pothole_mask[100:120, 150:180] = 255
        hump_mask[150:170, 200:230] = 255
        
        panel = visualizer.create_diagnostic_panel(
            original, disparity, v_disp, ground_plane, pothole_mask, hump_mask
        )
        
        # Panel should be 2x3 grid
        assert panel.shape[0] == 480 * 2  # 2 rows
        assert panel.shape[2] == 3  # BGR
        assert panel.dtype == np.uint8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
