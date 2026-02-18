"""
Quality Metrics and Diagnostic Visualization Module

This module implements performance evaluation metrics and diagnostic visualizations
for the stereo vision pipeline, including LRC error rates, planarity residuals,
temporal stability measurements, and calibration quality reporting.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 10.2, 10.4
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from stereo_vision.calibration import CameraParameters, StereoParameters
from stereo_vision.ground_plane import GroundPlaneModel


@dataclass
class QualityMetrics:
    """Container for quality metrics from stereo processing."""
    lrc_error_rate: float  # Percentage of pixels failing LRC check
    planarity_rmse: Optional[float]  # RMSE of ground plane fit
    temporal_stability: Optional[float]  # Coefficient of variation for temporal sequence
    calibration_reprojection_error: Optional[float]  # Calibration quality metric


class LRCErrorCalculator:
    """
    Calculator for Left-Right Consistency error rates.
    
    This class computes the percentage of pixels that fail the LRC validation,
    which is a key indicator of disparity map quality and occlusion handling.
    
    Requirements: 7.1
    """
    
    def __init__(self, max_diff: int = 1):
        """
        Initialize LRC error calculator.
        
        Args:
            max_diff: Maximum allowed disparity difference for consistency (pixels)
        """
        self.max_diff = max_diff
    
    def calculate_error_rate(
        self,
        disp_left: np.ndarray,
        disp_right: np.ndarray
    ) -> float:
        """
        Calculate the Left-Right Consistency error rate.
        
        The error rate is the percentage of pixels with valid disparity that
        fail the LRC check, indicating occlusions or matching errors.
        
        Args:
            disp_left: Left disparity map (float32 or int16 fixed-point)
            disp_right: Right disparity map (float32 or int16 fixed-point)
            
        Returns:
            LRC error rate as percentage (0-100)
            
        Requirements: 7.1
        """
        # Convert from fixed-point to float if needed
        if disp_left.dtype == np.int16:
            disp_left_float = disp_left.astype(np.float32) / 16.0
        else:
            disp_left_float = disp_left.astype(np.float32)
            
        if disp_right.dtype == np.int16:
            disp_right_float = disp_right.astype(np.float32) / 16.0
        else:
            disp_right_float = disp_right.astype(np.float32)
        
        height, width = disp_left.shape
        
        # Count valid disparities in left image
        valid_left = disp_left_float > 0
        total_valid = np.sum(valid_left)
        
        if total_valid == 0:
            return 0.0
        
        # Check consistency for each valid pixel
        failed_count = 0
        
        for y in range(height):
            for x in range(width):
                if not valid_left[y, x]:
                    continue
                
                d_left = disp_left_float[y, x]
                x_right = int(x - d_left)
                
                # Check if corresponding pixel is within bounds
                if x_right < 0 or x_right >= width:
                    failed_count += 1
                    continue
                
                d_right = disp_right_float[y, x_right]
                
                # Check consistency
                if abs(d_left - d_right) > self.max_diff:
                    failed_count += 1
        
        # Calculate error rate as percentage
        error_rate = (failed_count / total_valid) * 100.0
        
        return error_rate


class PlanarityCalculator:
    """
    Calculator for ground plane planarity residuals.
    
    This class computes the RMSE of inlier points from the fitted ground plane,
    which indicates the quality of the ground plane model and the flatness of
    the road surface.
    
    Requirements: 7.2
    """
    
    def __init__(self):
        """Initialize planarity calculator."""
        pass
    
    def calculate_planarity_rmse(
        self,
        disparity_map: np.ndarray,
        ground_plane: GroundPlaneModel,
        inlier_threshold: float = 2.0
    ) -> float:
        """
        Calculate RMSE of ground plane fit for inlier points.
        
        This measures how well the fitted ground plane represents the actual
        road surface by computing the root mean square error of disparity
        deviations for inlier points.
        
        Args:
            disparity_map: Input disparity map (float32 or int16 fixed-point)
            ground_plane: Fitted ground plane model
            inlier_threshold: Threshold for classifying inliers (disparity units)
            
        Returns:
            RMSE of planarity residuals in disparity units
            
        Requirements: 7.2
        """
        if not ground_plane.is_fitted:
            raise ValueError("Ground plane model has not been fitted")
        
        # Convert from fixed-point to float if needed
        if disparity_map.dtype == np.int16:
            disp_float = disparity_map.astype(np.float32) / 16.0
        else:
            disp_float = disparity_map.astype(np.float32)
        
        height, width = disp_float.shape
        
        # Calculate expected disparity for each row
        residuals = []
        
        for row in range(height):
            expected_disp = ground_plane.get_expected_disparity(row)
            
            for col in range(width):
                actual_disp = disp_float[row, col]
                
                # Skip invalid disparities
                if actual_disp <= 0:
                    continue
                
                # Calculate residual
                residual = actual_disp - expected_disp
                
                # Only include inliers (points close to the plane)
                if abs(residual) <= inlier_threshold:
                    residuals.append(residual)
        
        if len(residuals) == 0:
            return 0.0
        
        # Calculate RMSE
        residuals_array = np.array(residuals)
        rmse = np.sqrt(np.mean(residuals_array ** 2))
        
        return float(rmse)


class TemporalStabilityCalculator:
    """
    Calculator for temporal stability of volume measurements.
    
    This class measures the consistency of volume estimates across a temporal
    sequence, using the coefficient of variation (CV) as the stability metric.
    
    Requirements: 7.3
    """
    
    def __init__(self):
        """Initialize temporal stability calculator."""
        pass
    
    def calculate_temporal_stability(
        self,
        volume_sequence: List[float]
    ) -> float:
        """
        Calculate temporal stability metric for volume measurements.
        
        Temporal stability is measured using the coefficient of variation (CV),
        which is the ratio of standard deviation to mean. Lower CV indicates
        more stable measurements.
        
        CV = (std / mean) * 100%
        
        Args:
            volume_sequence: List of volume measurements over time (same units)
            
        Returns:
            Coefficient of variation as percentage (lower is more stable)
            
        Requirements: 7.3
        """
        if len(volume_sequence) < 2:
            return 0.0
        
        volumes = np.array(volume_sequence)
        
        # Remove any invalid values (zeros or negatives)
        valid_volumes = volumes[volumes > 0]
        
        if len(valid_volumes) < 2:
            return 0.0
        
        mean_volume = np.mean(valid_volumes)
        std_volume = np.std(valid_volumes)
        
        if mean_volume == 0:
            return 0.0
        
        # Coefficient of variation as percentage
        cv = (std_volume / mean_volume) * 100.0
        
        return float(cv)
    
    def calculate_stability_statistics(
        self,
        volume_sequence: List[float]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive temporal stability statistics.
        
        Args:
            volume_sequence: List of volume measurements over time
            
        Returns:
            Dictionary containing:
            - cv: Coefficient of variation (%)
            - mean: Mean volume
            - std: Standard deviation
            - min: Minimum volume
            - max: Maximum volume
            - range: Range (max - min)
            - num_samples: Number of valid samples
        """
        if len(volume_sequence) < 2:
            return {
                'cv': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'num_samples': len(volume_sequence)
            }
        
        volumes = np.array(volume_sequence)
        valid_volumes = volumes[volumes > 0]
        
        if len(valid_volumes) < 2:
            return {
                'cv': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'num_samples': len(valid_volumes)
            }
        
        mean_vol = np.mean(valid_volumes)
        std_vol = np.std(valid_volumes)
        cv = (std_vol / mean_vol * 100.0) if mean_vol > 0 else 0.0
        
        return {
            'cv': float(cv),
            'mean': float(mean_vol),
            'std': float(std_vol),
            'min': float(np.min(valid_volumes)),
            'max': float(np.max(valid_volumes)),
            'range': float(np.max(valid_volumes) - np.min(valid_volumes)),
            'num_samples': int(len(valid_volumes))
        }


class CalibrationQualityReporter:
    """
    Reporter for calibration quality metrics.
    
    This class extracts and reports calibration quality metrics including
    reprojection errors for both individual cameras and stereo calibration.
    
    Requirements: 7.4
    """
    
    def __init__(self):
        """Initialize calibration quality reporter."""
        pass
    
    def report_camera_quality(
        self,
        camera_params: CameraParameters
    ) -> Dict[str, float]:
        """
        Report quality metrics for a single camera calibration.
        
        Args:
            camera_params: Calibrated camera parameters
            
        Returns:
            Dictionary containing:
            - reprojection_error: RMS reprojection error in pixels
            - image_width: Image width
            - image_height: Image height
            
        Requirements: 7.4
        """
        return {
            'reprojection_error': float(camera_params.reprojection_error),
            'image_width': int(camera_params.image_size[0]),
            'image_height': int(camera_params.image_size[1])
        }
    
    def report_stereo_quality(
        self,
        stereo_params: StereoParameters
    ) -> Dict[str, float]:
        """
        Report quality metrics for stereo calibration.
        
        Args:
            stereo_params: Calibrated stereo parameters
            
        Returns:
            Dictionary containing:
            - left_reprojection_error: Left camera RMS error
            - right_reprojection_error: Right camera RMS error
            - baseline: Stereo baseline in meters
            - mean_reprojection_error: Average of left and right errors
            
        Requirements: 7.4
        """
        left_error = stereo_params.left_camera.reprojection_error
        right_error = stereo_params.right_camera.reprojection_error
        
        return {
            'left_reprojection_error': float(left_error),
            'right_reprojection_error': float(right_error),
            'baseline': float(stereo_params.baseline),
            'mean_reprojection_error': float((left_error + right_error) / 2.0)
        }
    
    def validate_calibration_quality(
        self,
        stereo_params: StereoParameters,
        max_reprojection_error: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Validate calibration quality against thresholds.
        
        Args:
            stereo_params: Calibrated stereo parameters
            max_reprojection_error: Maximum acceptable reprojection error (pixels)
            
        Returns:
            Tuple of (is_valid, message)
        """
        left_error = stereo_params.left_camera.reprojection_error
        right_error = stereo_params.right_camera.reprojection_error
        
        if left_error > max_reprojection_error:
            return False, f"Left camera reprojection error {left_error:.3f} exceeds threshold {max_reprojection_error}"
        
        if right_error > max_reprojection_error:
            return False, f"Right camera reprojection error {right_error:.3f} exceeds threshold {max_reprojection_error}"
        
        return True, "Calibration quality is acceptable"


class DiagnosticVisualizer:
    """
    Generator for diagnostic visualizations.
    
    This class creates visualization images for system validation including
    V-Disparity histograms, ground plane fits, and anomaly overlays.
    
    Requirements: 7.5, 10.2, 10.4
    """
    
    def __init__(self):
        """Initialize diagnostic visualizer."""
        pass
    
    def visualize_v_disparity(
        self,
        v_disparity: np.ndarray,
        ground_line: Optional[Tuple[float, float]] = None,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create V-Disparity visualization with optional ground plane line.
        
        Args:
            v_disparity: V-Disparity histogram (uint8)
            ground_line: Optional (slope, intercept) for ground plane line
            colormap: OpenCV colormap to apply
            
        Returns:
            Color visualization (BGR format)
            
        Requirements: 7.5, 10.4
        """
        # Apply colormap
        v_disp_colored = cv2.applyColorMap(v_disparity, colormap)
        
        # Draw ground plane line if provided
        if ground_line is not None:
            slope, intercept = ground_line
            height, width = v_disparity.shape
            
            for row in range(height):
                disp = int(slope * row + intercept)
                if 0 <= disp < width:
                    cv2.circle(v_disp_colored, (disp, row), 2, (0, 255, 0), -1)
        
        return v_disp_colored
    
    def visualize_ground_plane_fit(
        self,
        disparity_map: np.ndarray,
        ground_plane: GroundPlaneModel,
        show_residuals: bool = True
    ) -> np.ndarray:
        """
        Create visualization showing ground plane fit and residuals.
        
        Args:
            disparity_map: Input disparity map
            ground_plane: Fitted ground plane model
            show_residuals: If True, color-code residuals
            
        Returns:
            Visualization image (BGR format)
            
        Requirements: 7.5, 10.4
        """
        if not ground_plane.is_fitted:
            raise ValueError("Ground plane model has not been fitted")
        
        # Convert disparity to float
        if disparity_map.dtype == np.int16:
            disp_float = disparity_map.astype(np.float32) / 16.0
        else:
            disp_float = disparity_map.astype(np.float32)
        
        height, width = disp_float.shape
        
        if show_residuals:
            # Create residual map
            residual_map = np.zeros((height, width), dtype=np.float32)
            
            for row in range(height):
                expected_disp = ground_plane.get_expected_disparity(row)
                residual_map[row, :] = disp_float[row, :] - expected_disp
            
            # Normalize residuals for visualization
            valid_mask = disp_float > 0
            if np.sum(valid_mask) > 0:
                valid_residuals = residual_map[valid_mask]
                
                # Clip to reasonable range for visualization
                percentile_5 = np.percentile(valid_residuals, 5)
                percentile_95 = np.percentile(valid_residuals, 95)
                
                residual_map_clipped = np.clip(residual_map, percentile_5, percentile_95)
                
                # Normalize to 0-255
                if percentile_95 > percentile_5:
                    residual_normalized = ((residual_map_clipped - percentile_5) / 
                                          (percentile_95 - percentile_5) * 255).astype(np.uint8)
                else:
                    residual_normalized = np.zeros_like(residual_map, dtype=np.uint8)
                
                # Apply diverging colormap (blue = negative, red = positive)
                vis = cv2.applyColorMap(residual_normalized, cv2.COLORMAP_TWILIGHT)
                
                # Mask invalid pixels
                vis[~valid_mask] = [0, 0, 0]
            else:
                vis = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Just show disparity with ground plane overlay
            valid_mask = disp_float > 0
            if np.sum(valid_mask) > 0:
                disp_normalized = disp_float.copy()
                min_disp = np.min(disp_float[valid_mask])
                max_disp = np.max(disp_float[valid_mask])
                
                if max_disp > min_disp:
                    disp_normalized = ((disp_float - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)
                else:
                    disp_normalized = np.zeros_like(disp_float, dtype=np.uint8)
                
                vis = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
                vis[~valid_mask] = [0, 0, 0]
            else:
                vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        return vis
    
    def visualize_anomaly_overlay(
        self,
        original_image: np.ndarray,
        pothole_mask: np.ndarray,
        hump_mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay visualization showing detected anomalies on original image.
        
        Args:
            original_image: Original left image (grayscale or color)
            pothole_mask: Binary mask of pothole pixels
            hump_mask: Binary mask of hump pixels
            alpha: Transparency factor for overlay (0-1)
            
        Returns:
            Overlay visualization (BGR format)
            
        Requirements: 10.2
        """
        # Convert to color if grayscale
        if len(original_image.shape) == 2:
            base_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            base_image = original_image.copy()
        
        # Create overlay
        overlay = base_image.copy()
        
        # Overlay potholes in blue
        overlay[pothole_mask > 0] = [255, 0, 0]  # Blue
        
        # Overlay humps in red
        overlay[hump_mask > 0] = [0, 0, 255]  # Red
        
        # Blend with original
        result = cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def create_diagnostic_panel(
        self,
        original_image: np.ndarray,
        disparity_map: np.ndarray,
        v_disparity: np.ndarray,
        ground_plane: GroundPlaneModel,
        pothole_mask: np.ndarray,
        hump_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create comprehensive diagnostic panel with multiple visualizations.
        
        Args:
            original_image: Original left image
            disparity_map: Computed disparity map
            v_disparity: V-Disparity histogram
            ground_plane: Fitted ground plane model
            pothole_mask: Binary mask of potholes
            hump_mask: Binary mask of humps
            
        Returns:
            Diagnostic panel image (BGR format) with 2x3 grid layout
            
        Requirements: 7.5, 10.2, 10.4
        """
        # Create individual visualizations
        anomaly_overlay = self.visualize_anomaly_overlay(
            original_image, pothole_mask, hump_mask
        )
        
        # Disparity visualization
        if disparity_map.dtype == np.int16:
            disp_float = disparity_map.astype(np.float32) / 16.0
        else:
            disp_float = disparity_map.astype(np.float32)
        
        valid_mask = disp_float > 0
        if np.sum(valid_mask) > 0:
            disp_vis = disp_float.copy()
            min_d = np.min(disp_float[valid_mask])
            max_d = np.max(disp_float[valid_mask])
            if max_d > min_d:
                disp_vis = ((disp_float - min_d) / (max_d - min_d) * 255).astype(np.uint8)
            else:
                disp_vis = np.zeros_like(disp_float, dtype=np.uint8)
            disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            disp_colored[~valid_mask] = [0, 0, 0]
        else:
            disp_colored = np.zeros((disparity_map.shape[0], disparity_map.shape[1], 3), dtype=np.uint8)
        
        # V-Disparity visualization
        ground_line = ground_plane.get_plane_parameters() if ground_plane.is_fitted else None
        v_disp_vis = self.visualize_v_disparity(v_disparity, ground_line)
        
        # Ground plane fit visualization
        ground_fit_vis = self.visualize_ground_plane_fit(disparity_map, ground_plane)
        
        # Resize all to same height for panel
        target_height = 480
        
        def resize_to_height(img, h):
            aspect = img.shape[1] / img.shape[0]
            return cv2.resize(img, (int(h * aspect), h))
        
        # Convert grayscale original to color if needed
        if len(original_image.shape) == 2:
            original_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            original_colored = original_image.copy()
        
        # Resize all images
        img1 = resize_to_height(original_colored, target_height)
        img2 = resize_to_height(anomaly_overlay, target_height)
        img3 = resize_to_height(disp_colored, target_height)
        img4 = resize_to_height(ground_fit_vis, target_height)
        img5 = resize_to_height(v_disp_vis, target_height)
        
        # Create pothole/hump visualization
        combined_mask = np.zeros_like(original_image, dtype=np.uint8)
        if len(combined_mask.shape) == 2:
            combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        combined_mask[pothole_mask > 0] = [255, 0, 0]  # Blue for potholes
        combined_mask[hump_mask > 0] = [0, 0, 255]  # Red for humps
        img6 = resize_to_height(combined_mask, target_height)
        
        # Ensure all images have the same width for proper stacking
        max_width = max(img1.shape[1], img2.shape[1], img3.shape[1], 
                       img4.shape[1], img5.shape[1], img6.shape[1])
        
        def pad_to_width(img, target_width):
            if img.shape[1] < target_width:
                pad_width = target_width - img.shape[1]
                return np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            return img
        
        img1 = pad_to_width(img1, max_width)
        img2 = pad_to_width(img2, max_width)
        img3 = pad_to_width(img3, max_width)
        img4 = pad_to_width(img4, max_width)
        img5 = pad_to_width(img5, max_width)
        img6 = pad_to_width(img6, max_width)
        
        # Create 2x3 grid
        row1 = np.hstack([img1, img2, img3])
        row2 = np.hstack([img4, img5, img6])
        panel = np.vstack([row1, row2])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Anomaly Overlay", (img1.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Disparity", (img1.shape[1] + img2.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Ground Plane Fit", (10, target_height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "V-Disparity", (img1.shape[1] + 10, target_height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Anomaly Masks", (img1.shape[1] + img2.shape[1] + 10, target_height + 30), font, 1, (255, 255, 255), 2)
        
        return panel
