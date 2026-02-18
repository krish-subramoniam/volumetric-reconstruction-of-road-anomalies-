"""V-Disparity ground plane detection module for stereo vision pipeline."""

from typing import Optional, Tuple
import numpy as np
import cv2

from stereo_vision.logging_config import get_logger, PerformanceTimer
from stereo_vision.errors import (
    VDisparityGenerationError, HoughLineDetectionError, GroundPlaneFittingError
)

# Initialize logger
logger = get_logger(__name__)


class VDisparityGenerator:
    """
    V-Disparity histogram generator for ground plane detection.
    
    V-Disparity is a 2D histogram representation that transforms a disparity map
    into a compact representation where:
    - X-axis represents disparity values
    - Y-axis represents image row (v coordinate)
    - Pixel intensity represents the count of pixels at that (row, disparity) pair
    
    In V-Disparity space, planar surfaces like roads appear as diagonal lines,
    making it easier to detect and model the ground plane using techniques like
    Hough Transform.
    
    Key properties:
    - Road surfaces appear as diagonal lines (constant slope)
    - Vertical objects appear as vertical lines
    - The dominant diagonal line represents the ground plane
    """
    
    def __init__(self, max_disparity: Optional[int] = None):
        """
        Initialize V-Disparity generator.
        
        Args:
            max_disparity: Maximum disparity value to consider. If None, will be
                          determined automatically from the disparity map.
        """
        self.max_disparity = max_disparity
    
    def generate_v_disparity(self, disparity_map: np.ndarray) -> np.ndarray:
        """
        Generate V-Disparity histogram from a disparity map.
        
        The V-Disparity image is a 2D histogram where:
        - Each row corresponds to a row in the original image
        - Each column corresponds to a disparity value
        - The pixel value is the count of pixels with that (row, disparity) pair
        
        Args:
            disparity_map: Input disparity map. Can be:
                          - 16-bit fixed point (from SGBM, divide by 16)
                          - float32 (actual disparity values)
                          
        Returns:
            V-Disparity image as uint8 array with shape (height, max_disparity)
            where intensity represents pixel count (normalized to 0-255)
        """
        # Convert from fixed-point to float if needed
        if disparity_map.dtype == np.int16:
            disp_float = disparity_map.astype(np.float32) / 16.0
        else:
            disp_float = disparity_map.astype(np.float32)
        
        height, width = disp_float.shape
        
        # Determine max disparity if not set
        if self.max_disparity is None:
            valid_disparities = disp_float[disp_float > 0]
            if len(valid_disparities) == 0:
                # No valid disparities, return empty V-Disparity
                return np.zeros((height, 1), dtype=np.uint8)
            max_disp = int(np.max(valid_disparities)) + 1
        else:
            max_disp = self.max_disparity
        
        # Initialize V-Disparity histogram
        v_disparity = np.zeros((height, max_disp), dtype=np.uint32)
        
        # Build histogram: for each row, count pixels at each disparity value
        for row in range(height):
            row_disparities = disp_float[row, :]
            
            # Filter valid disparities (> 0 and < max_disp)
            valid_mask = (row_disparities > 0) & (row_disparities < max_disp)
            valid_disparities = row_disparities[valid_mask]
            
            # Round to integer disparity bins
            disparity_bins = valid_disparities.astype(np.int32)
            
            # Count occurrences of each disparity value
            for d in disparity_bins:
                v_disparity[row, d] += 1
        
        # Normalize to 0-255 for visualization
        # Use log scale to enhance visibility of less frequent disparities
        v_disparity_log = np.log1p(v_disparity.astype(np.float32))
        
        # Normalize to 0-255
        if v_disparity_log.max() > 0:
            v_disparity_normalized = (v_disparity_log / v_disparity_log.max() * 255).astype(np.uint8)
        else:
            v_disparity_normalized = np.zeros_like(v_disparity, dtype=np.uint8)
        
        return v_disparity_normalized
    
    def visualize_v_disparity(
        self, 
        v_disp: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create a visualization of the V-Disparity image for debugging.
        
        This method applies a colormap to the V-Disparity histogram to make
        patterns more visible. The dominant diagonal line (ground plane) should
        be clearly visible in the visualization.
        
        Args:
            v_disp: V-Disparity image (uint8, from generate_v_disparity)
            colormap: OpenCV colormap to apply (default: COLORMAP_JET)
                     Options: COLORMAP_JET, COLORMAP_HOT, COLORMAP_VIRIDIS, etc.
                     
        Returns:
            Color visualization of V-Disparity image (BGR format)
        """
        # Apply colormap for better visualization
        v_disp_colored = cv2.applyColorMap(v_disp, colormap)
        
        return v_disp_colored
    
    def visualize_with_annotations(
        self,
        v_disp: np.ndarray,
        ground_line: Optional[Tuple[float, float]] = None,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create annotated visualization showing detected ground plane line.
        
        Args:
            v_disp: V-Disparity image (uint8)
            ground_line: Optional tuple of (slope, intercept) for ground plane line
                        in V-Disparity space. If provided, will be drawn on the image.
            colormap: OpenCV colormap to apply
            
        Returns:
            Annotated color visualization (BGR format)
        """
        # Create base visualization
        v_disp_colored = self.visualize_v_disparity(v_disp, colormap)
        
        # Draw ground plane line if provided
        if ground_line is not None:
            slope, intercept = ground_line
            height, width = v_disp.shape
            
            # Draw line across the V-Disparity image
            # Line equation: disparity = slope * row + intercept
            for row in range(height):
                disp = int(slope * row + intercept)
                if 0 <= disp < width:
                    # Draw a thick line for visibility
                    cv2.circle(v_disp_colored, (disp, row), 2, (0, 255, 0), -1)
        
        return v_disp_colored
    
    def get_histogram_statistics(self, v_disp: np.ndarray) -> dict:
        """
        Compute statistics about the V-Disparity histogram.
        
        These statistics can be useful for:
        - Validating V-Disparity quality
        - Tuning ground plane detection parameters
        - Debugging disparity map issues
        
        Args:
            v_disp: V-Disparity image (uint8)
            
        Returns:
            Dictionary containing:
            - 'mean_intensity': Average pixel intensity
            - 'max_intensity': Maximum pixel intensity
            - 'non_zero_ratio': Ratio of non-zero pixels
            - 'dominant_disparity_range': Range of disparities with highest counts
        """
        stats = {}
        
        # Basic intensity statistics
        stats['mean_intensity'] = float(np.mean(v_disp))
        stats['max_intensity'] = int(np.max(v_disp))
        
        # Non-zero ratio (indicates how much of V-Disparity space is populated)
        non_zero_count = np.count_nonzero(v_disp)
        total_pixels = v_disp.shape[0] * v_disp.shape[1]
        stats['non_zero_ratio'] = non_zero_count / total_pixels
        
        # Find dominant disparity range (where most pixels are concentrated)
        # Sum across rows to get disparity distribution
        disparity_distribution = np.sum(v_disp, axis=0)
        
        if np.max(disparity_distribution) > 0:
            # Find range containing 80% of the mass
            cumsum = np.cumsum(disparity_distribution)
            total_mass = cumsum[-1]
            
            # Find 10th and 90th percentiles
            lower_idx = np.searchsorted(cumsum, total_mass * 0.1)
            upper_idx = np.searchsorted(cumsum, total_mass * 0.9)
            
            stats['dominant_disparity_range'] = (int(lower_idx), int(upper_idx))
        else:
            stats['dominant_disparity_range'] = (0, 0)
        
        return stats



class HoughLineDetector:
    """
    Hough Transform-based line detector for V-Disparity ground plane extraction.
    
    This class detects the dominant diagonal line in V-Disparity space, which
    corresponds to the ground plane in the 3D scene. The Hough Transform is
    particularly effective for this task because:
    - It's robust to noise and outliers
    - It can detect lines even with gaps
    - It provides parametric line representation (rho, theta)
    
    The detected line parameters are then converted to a ground plane model
    that can be used for anomaly segmentation.
    """
    
    def __init__(
        self,
        rho_resolution: float = 1.0,
        theta_resolution: float = np.pi / 180,
        threshold: Optional[int] = None,
        min_line_length: Optional[int] = None,
        max_line_gap: Optional[int] = None
    ):
        """
        Initialize Hough line detector with parameters.
        
        Args:
            rho_resolution: Distance resolution in pixels (default: 1.0)
            theta_resolution: Angle resolution in radians (default: 1 degree)
            threshold: Accumulator threshold for line detection. If None, will be
                      determined automatically based on image size.
            min_line_length: Minimum line length for probabilistic Hough (optional)
            max_line_gap: Maximum gap between line segments (optional)
        """
        self.rho_resolution = rho_resolution
        self.theta_resolution = theta_resolution
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def detect_dominant_line(
        self,
        v_disp: np.ndarray,
        use_probabilistic: bool = False
    ) -> Optional[Tuple[float, float]]:
        """
        Detect the dominant line in V-Disparity image using Hough Transform.
        
        The dominant line represents the ground plane. For road scenes, this
        appears as a diagonal line with positive slope (disparity increases
        with row number as we get closer to the camera).
        
        Args:
            v_disp: V-Disparity image (uint8, from VDisparityGenerator)
            use_probabilistic: If True, use probabilistic Hough transform
                              (faster but may be less accurate)
                              
        Returns:
            Tuple of (slope, intercept) for the detected line in V-Disparity space,
            or None if no line is detected.
            
            Line equation: disparity = slope * row + intercept
            
            For a typical road scene:
            - slope > 0 (disparity increases with row)
            - intercept > 0 (minimum disparity at top of image)
        """
        # Apply edge detection to enhance line features
        # Use Canny edge detector to find strong edges in V-Disparity
        edges = cv2.Canny(v_disp, 50, 150, apertureSize=3)
        
        # Determine threshold if not set
        if self.threshold is None:
            # Adaptive threshold based on image size
            # Larger images need higher thresholds
            height, width = v_disp.shape
            threshold = max(30, int(min(height, width) * 0.15))
        else:
            threshold = self.threshold
        
        if use_probabilistic and self.min_line_length is not None:
            # Probabilistic Hough Transform (faster, returns line segments)
            lines = cv2.HoughLinesP(
                edges,
                self.rho_resolution,
                self.theta_resolution,
                threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap or 10
            )
            
            if lines is None or len(lines) == 0:
                return None
            
            # Find the longest line (most likely to be the ground plane)
            longest_line = None
            max_length = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > max_length:
                    max_length = length
                    longest_line = (x1, y1, x2, y2)
            
            if longest_line is None:
                return None
            
            # Convert line segment to slope-intercept form
            x1, y1, x2, y2 = longest_line
            
            # In V-Disparity: x = disparity, y = row
            # We want: disparity = slope * row + intercept
            # So we need to swap x and y
            if y2 != y1:
                slope = (x2 - x1) / (y2 - y1)
                intercept = x1 - slope * y1
            else:
                # Horizontal line (unlikely for road)
                return None
            
        else:
            # Standard Hough Transform (more robust, returns rho-theta)
            lines = cv2.HoughLines(
                edges,
                self.rho_resolution,
                self.theta_resolution,
                threshold
            )
            
            if lines is None or len(lines) == 0:
                return None
            
            # Filter lines to find the most likely ground plane line
            # Ground plane typically has:
            # - Positive slope (theta between 0 and 90 degrees)
            # - Significant length across the image
            
            height, width = v_disp.shape
            best_line = None
            best_score = 0
            
            for line in lines:
                rho, theta = line[0]
                
                # Convert from rho-theta to slope-intercept form
                # Line equation: x*cos(theta) + y*sin(theta) = rho
                # We want: x = slope * y + intercept (where x=disparity, y=row)
                
                # Avoid vertical lines (theta ≈ 0)
                if abs(np.sin(theta)) < 0.1:
                    continue
                
                # Calculate slope and intercept
                # x = (rho - y*sin(theta)) / cos(theta)
                # x = -(sin(theta)/cos(theta)) * y + rho/cos(theta)
                slope = -np.sin(theta) / np.cos(theta)
                intercept = rho / np.cos(theta)
                
                # Score based on:
                # 1. Positive slope (ground plane characteristic)
                # 2. Line spans a good portion of the image
                # 3. Reasonable intercept (not too negative)
                
                if slope <= 0:
                    continue  # Ground plane should have positive slope
                
                if intercept < -width * 0.2:
                    continue  # Intercept too negative
                
                # Calculate how much of the image the line spans
                disp_at_top = intercept
                disp_at_bottom = slope * (height - 1) + intercept
                
                # Allow lines that partially go out of bounds
                # (common for real road scenes)
                if disp_at_top > width or disp_at_bottom < 0:
                    continue  # Line completely out of bounds
                
                # Score: prefer lines with moderate slope and good coverage
                # Calculate visible portion of the line
                visible_start = max(disp_at_top, 0)
                visible_end = min(disp_at_bottom, width)
                coverage = visible_end - visible_start
                
                if coverage < width * 0.3:
                    continue  # Line doesn't span enough of the image
                
                score = coverage * (1.0 / (1.0 + abs(slope - 0.5)))  # Prefer slope around 0.5
                
                if score > best_score:
                    best_score = score
                    best_line = (slope, intercept)
            
            if best_line is None:
                return None
            
            slope, intercept = best_line
        
        return (slope, intercept)
    
    def visualize_detected_line(
        self,
        v_disp: np.ndarray,
        line_params: Tuple[float, float],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Visualize the detected line on the V-Disparity image.
        
        Args:
            v_disp: V-Disparity image (uint8)
            line_params: Tuple of (slope, intercept)
            color: BGR color for the line (default: green)
            
        Returns:
            V-Disparity image with line overlay (BGR format)
        """
        # Convert to color if grayscale
        if len(v_disp.shape) == 2:
            v_disp_colored = cv2.cvtColor(v_disp, cv2.COLOR_GRAY2BGR)
        else:
            v_disp_colored = v_disp.copy()
        
        slope, intercept = line_params
        height, width = v_disp.shape[:2]
        
        # Draw line across the image
        for row in range(height):
            disp = int(slope * row + intercept)
            if 0 <= disp < width:
                cv2.circle(v_disp_colored, (disp, row), 1, color, -1)
        
        return v_disp_colored


class GroundPlaneModel:
    """
    Parametric ground plane model derived from V-Disparity analysis.
    
    This class represents the ground plane in disparity space and provides
    methods for:
    - Fitting the ground plane from V-Disparity line parameters
    - Predicting expected disparity for any image row
    - Segmenting anomalies (potholes and humps) based on disparity deviation
    
    The ground plane model assumes a planar road surface where disparity
    varies linearly with image row: d(v) = slope * v + intercept
    
    Anomaly detection:
    - Pothole: disparity < expected (farther than ground plane)
    - Hump: disparity > expected (closer than ground plane)
    """
    
    def __init__(self, threshold_factor: float = 1.5):
        """
        Initialize ground plane model.
        
        Args:
            threshold_factor: Multiplier for anomaly detection threshold.
                            Larger values = less sensitive (fewer false positives)
                            Smaller values = more sensitive (may detect noise)
                            Default: 1.5 (reasonable for typical road scenes)
        """
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None
        self.threshold_factor = threshold_factor
        self.is_fitted = False
    
    def fit_from_v_disparity(
        self,
        v_disp: np.ndarray,
        hough_detector: Optional[HoughLineDetector] = None
    ) -> bool:
        """
        Fit ground plane model from V-Disparity image.
        
        This method:
        1. Detects the dominant line in V-Disparity using Hough Transform
        2. Extracts line parameters (slope, intercept)
        3. Stores them as the ground plane model
        
        Args:
            v_disp: V-Disparity image (uint8)
            hough_detector: Optional HoughLineDetector instance. If None,
                          a default detector will be created.
                          
        Returns:
            True if ground plane was successfully fitted, False otherwise
        """
        if hough_detector is None:
            hough_detector = HoughLineDetector()
        
        # Detect dominant line
        line_params = hough_detector.detect_dominant_line(v_disp)
        
        if line_params is None:
            self.is_fitted = False
            return False
        
        self.slope, self.intercept = line_params
        self.is_fitted = True
        return True
    
    def fit_from_line_params(self, slope: float, intercept: float) -> None:
        """
        Directly set ground plane parameters from known line parameters.
        
        Args:
            slope: Slope of the ground plane line in V-Disparity space
            intercept: Intercept of the ground plane line
        """
        self.slope = slope
        self.intercept = intercept
        self.is_fitted = True
    
    def get_expected_disparity(self, row: int) -> float:
        """
        Get expected disparity value for a given image row.
        
        For a fitted ground plane, this returns the disparity value that
        a point on the ground plane would have at the specified row.
        
        Args:
            row: Image row (v coordinate)
            
        Returns:
            Expected disparity value for the ground plane at this row
            
        Raises:
            RuntimeError: If ground plane has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Ground plane model has not been fitted")
        
        return self.slope * row + self.intercept
    
    def segment_anomalies(
        self,
        disparity_map: np.ndarray,
        return_masks: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment potholes and humps based on ground plane model.
        
        This method classifies each pixel as:
        - Pothole: disparity significantly below ground plane (farther away)
        - Hump: disparity significantly above ground plane (closer)
        - Normal: disparity close to ground plane
        
        Args:
            disparity_map: Input disparity map (float32 or int16 fixed-point)
            return_masks: If True, return binary masks. If False, return
                         labeled images with anomaly IDs.
                         
        Returns:
            Tuple of (pothole_mask, hump_mask) where:
            - pothole_mask: Binary mask (uint8) of pothole pixels
            - hump_mask: Binary mask (uint8) of hump pixels
            
        Raises:
            RuntimeError: If ground plane has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Ground plane model has not been fitted")
        
        # Convert from fixed-point to float if needed
        if disparity_map.dtype == np.int16:
            disp_float = disparity_map.astype(np.float32) / 16.0
        else:
            disp_float = disparity_map.astype(np.float32)
        
        height, width = disp_float.shape
        
        # Create expected disparity map for the entire image
        expected_disparity = np.zeros((height, width), dtype=np.float32)
        for row in range(height):
            expected_disparity[row, :] = self.get_expected_disparity(row)
        
        # Calculate deviation from ground plane
        deviation = disp_float - expected_disparity
        
        # Estimate threshold based on disparity variation
        # Use median absolute deviation (MAD) for robust threshold estimation
        valid_mask = disp_float > 0
        if np.sum(valid_mask) > 0:
            valid_deviations = deviation[valid_mask]
            mad = np.median(np.abs(valid_deviations - np.median(valid_deviations)))
            threshold = self.threshold_factor * mad
            
            # Ensure minimum threshold
            threshold = max(threshold, 2.0)
        else:
            threshold = 2.0
        
        # Segment anomalies
        # Pothole: disparity < expected (negative deviation)
        pothole_mask = (valid_mask & (deviation < -threshold)).astype(np.uint8) * 255
        
        # Hump: disparity > expected (positive deviation)
        hump_mask = (valid_mask & (deviation > threshold)).astype(np.uint8) * 255
        
        return pothole_mask, hump_mask
    
    def get_plane_parameters(self) -> Optional[Tuple[float, float]]:
        """
        Get the ground plane parameters.
        
        Returns:
            Tuple of (slope, intercept) if fitted, None otherwise
        """
        if not self.is_fitted:
            return None
        return (self.slope, self.intercept)
    
    def visualize_ground_plane(
        self,
        disparity_map: np.ndarray,
        show_expected: bool = True,
        show_anomalies: bool = True
    ) -> np.ndarray:
        """
        Create visualization showing ground plane and detected anomalies.
        
        Args:
            disparity_map: Input disparity map
            show_expected: If True, overlay expected ground plane disparity
            show_anomalies: If True, highlight detected anomalies
            
        Returns:
            Visualization image (BGR format)
        """
        if not self.is_fitted:
            raise RuntimeError("Ground plane model has not been fitted")
        
        # Convert disparity to visualization
        if disparity_map.dtype == np.int16:
            disp_float = disparity_map.astype(np.float32) / 16.0
        else:
            disp_float = disparity_map.astype(np.float32)
        
        # Normalize for visualization
        valid_mask = disp_float > 0
        if np.sum(valid_mask) > 0:
            disp_vis = disp_float.copy()
            disp_vis[~valid_mask] = 0
            
            # Normalize to 0-255
            min_disp = np.min(disp_vis[valid_mask])
            max_disp = np.max(disp_vis[valid_mask])
            if max_disp > min_disp:
                disp_vis = ((disp_vis - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)
            else:
                disp_vis = np.zeros_like(disp_vis, dtype=np.uint8)
        else:
            disp_vis = np.zeros_like(disp_float, dtype=np.uint8)
        
        # Convert to color
        disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        # Overlay anomalies if requested
        if show_anomalies:
            pothole_mask, hump_mask = self.segment_anomalies(disparity_map)
            
            # Overlay potholes in blue
            disp_colored[pothole_mask > 0] = [255, 0, 0]  # Blue
            
            # Overlay humps in red
            disp_colored[hump_mask > 0] = [0, 0, 255]  # Red
        
        return disp_colored
