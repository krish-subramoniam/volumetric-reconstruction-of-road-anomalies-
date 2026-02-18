"""Advanced disparity estimation module for stereo vision pipeline."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

from stereo_vision.logging_config import get_logger, PerformanceTimer
from stereo_vision.errors import (
    DisparityError, InvalidDisparityMapError, LRCValidationError, WLSFilterError
)

# Initialize logger
logger = get_logger(__name__)


@dataclass
class DisparityResult:
    """Result of disparity computation with metadata."""
    disparity_map: np.ndarray
    validity_mask: np.ndarray
    lrc_error_rate: float
    processing_time: float


class SGBMEstimator:
    """
    Semi-Global Block Matching estimator optimized for road scenes.
    
    This class implements SGBM with parameters specifically tuned for
    road surface characteristics including textureless surfaces and
    smooth asphalt.
    """
    
    def __init__(self, baseline: float, focal_length: float):
        """
        Initialize SGBM estimator with camera parameters.
        
        Args:
            baseline: Distance between camera centers in meters
            focal_length: Camera focal length in pixels
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self.sgbm = None
        self._configure_for_roads()
    
    def _configure_for_roads(self) -> None:
        """
        Configure SGBM parameters optimized for road scenes.
        
        Road scenes have specific characteristics:
        - Large textureless regions (smooth asphalt)
        - Mostly horizontal surfaces
        - Depth range typically 1-50 meters
        
        Parameters are tuned based on baseline and focal length to
        ensure appropriate disparity range coverage.
        """
        # Calculate disparity range based on depth range
        # For road scenes: min_depth = 1m, max_depth = 50m
        min_depth = 1.0  # meters
        max_depth = 50.0  # meters
        
        # Disparity = (baseline * focal_length) / depth
        max_disparity = int((self.baseline * self.focal_length) / min_depth)
        min_disparity = int((self.baseline * self.focal_length) / max_depth)
        
        # numDisparities must be divisible by 16
        num_disparities = ((max_disparity - min_disparity) // 16 + 1) * 16
        
        # Block size: larger for textureless regions, but not too large
        # to preserve detail. 5-11 is typical, we use 7 for road scenes.
        block_size = 7
        
        # P1, P2: Smoothness constraints
        # P1: penalty for disparity changes of +/- 1
        # P2: penalty for larger disparity changes
        # For road scenes, we want strong smoothness to handle textureless areas
        P1 = 8 * 3 * block_size ** 2
        P2 = 32 * 3 * block_size ** 2
        
        # Disparity smoothness: higher values for smoother results
        # Important for textureless road surfaces
        disp12MaxDiff = 1  # Max allowed difference in left-right consistency check
        
        # Pre-filter cap: limits texture strength
        # Lower values for low-texture scenes
        preFilterCap = 31
        
        # Uniqueness ratio: minimum margin by which best computed cost
        # should win over second best. Higher for more reliable matches.
        uniquenessRatio = 10
        
        # Speckle filtering: removes small regions of inconsistent disparity
        # Important for road scenes with noise
        speckleWindowSize = 100  # Max size of smooth disparity regions to consider noise
        speckleRange = 32  # Max disparity variation within speckle region
        
        # Mode: SGBM_MODE_HH (full-scale two-pass) for best quality
        # or SGBM_MODE_SGBM_3WAY for better handling of textureless regions
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        
        # Create SGBM object
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            preFilterCap=preFilterCap,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            mode=mode
        )
    
    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from rectified stereo pair.
        
        Args:
            left: Left rectified image (grayscale or color)
            right: Right rectified image (grayscale or color)
            
        Returns:
            Disparity map in pixels (16-bit fixed point, divide by 16 for actual disparity)
            
        Raises:
            InvalidDisparityMapError: If disparity computation fails
        """
        try:
            if self.sgbm is None:
                raise InvalidDisparityMapError(
                    "SGBM not configured",
                    details={"reason": "Call _configure_for_roads first"}
                )
            
            if left is None or right is None:
                raise InvalidDisparityMapError(
                    "Invalid input images",
                    details={"left_is_none": left is None, "right_is_none": right is None}
                )
            
            if left.shape != right.shape:
                raise InvalidDisparityMapError(
                    "Image dimensions mismatch",
                    details={"left_shape": left.shape, "right_shape": right.shape}
                )
            
            logger.debug("Computing disparity", image_shape=left.shape)
            
            with PerformanceTimer(logger, "SGBM disparity computation"):
                # Convert to grayscale if needed
                if len(left.shape) == 3:
                    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                else:
                    left_gray = left
                    
                if len(right.shape) == 3:
                    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                else:
                    right_gray = right
                
                # Compute disparity
                disparity = self.sgbm.compute(left_gray, right_gray)
                
                if disparity is None:
                    raise InvalidDisparityMapError(
                        "SGBM returned None",
                        details={"reason": "Computation failed"}
                    )
                
                # Check for valid disparities
                valid_count = np.sum(disparity > 0)
                total_pixels = disparity.size
                valid_ratio = valid_count / total_pixels
                
                logger.debug(
                    "Disparity computed",
                    valid_pixels=valid_count,
                    total_pixels=total_pixels,
                    valid_ratio=f"{valid_ratio:.2%}"
                )
                
                if valid_ratio < 0.1:
                    logger.warning(
                        "Low valid disparity ratio",
                        valid_ratio=f"{valid_ratio:.2%}",
                        threshold="10%"
                    )
                
                return disparity
                
        except InvalidDisparityMapError:
            raise
        except cv2.error as e:
            raise InvalidDisparityMapError(
                "OpenCV error during disparity computation",
                details={"opencv_error": str(e)}
            )
        except Exception as e:
            logger.error(f"Unexpected error in disparity computation: {str(e)}", exc_info=True)
            raise InvalidDisparityMapError(
                "Disparity computation failed",
                details={"error": str(e)}
            )
    
    def configure_for_roads(self, baseline: float, focal_length: float) -> None:
        """
        Reconfigure SGBM parameters for different camera setup.
        
        Args:
            baseline: Distance between camera centers in meters
            focal_length: Camera focal length in pixels
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self._configure_for_roads()


class LRCValidator:
    """
    Left-Right Consistency validator for occlusion detection.
    
    This class implements the Left-Right Consistency (LRC) check to identify
    and remove occluded pixels from disparity maps. The LRC check validates
    that the disparity computed from the left image matches the disparity
    computed from the right image when projected back.
    
    For a pixel (x, y) in the left image with disparity d_left, the corresponding
    pixel in the right image is at (x - d_left, y). The LRC check verifies that
    the disparity at this right image location (d_right) is consistent with d_left.
    
    Pixels that fail this consistency check are typically occluded in one of the
    images and should be marked as invalid.
    """
    
    def __init__(self, max_diff: int = 1):
        """
        Initialize LRC validator.
        
        Args:
            max_diff: Maximum allowed disparity difference for consistency (in pixels)
        """
        self.max_diff = max_diff
    
    def validate_consistency(
        self, disp_left: np.ndarray, disp_right: np.ndarray
    ) -> np.ndarray:
        """
        Perform Left-Right Consistency check to remove occluded pixels.
        
        The LRC check validates that for each pixel in the left disparity map,
        the corresponding pixel in the right disparity map (found by subtracting
        the disparity from the x-coordinate) has a consistent disparity value.
        
        Args:
            disp_left: Left disparity map (can be int16 fixed-point or float32)
            disp_right: Right disparity map (can be int16 fixed-point or float32)
            
        Returns:
            Validity mask (1 for valid pixels, 0 for invalid/occluded pixels)
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
        validity_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Vectorized approach for better performance
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Calculate corresponding x coordinates in right image
        x_right = x_coords - disp_left_float
        
        # Create mask for valid disparities and valid right coordinates
        valid_disp = disp_left_float > 0
        valid_x_right = (x_right >= 0) & (x_right < width)
        valid_mask = valid_disp & valid_x_right
        
        # For valid pixels, check consistency
        for y in range(height):
            for x in range(width):
                if not valid_mask[y, x]:
                    continue
                
                d_left = disp_left_float[y, x]
                x_r = int(x_right[y, x])
                d_right = disp_right_float[y, x_r]
                
                # Check consistency: |d_left - d_right| <= max_diff
                if abs(d_left - d_right) <= self.max_diff:
                    validity_mask[y, x] = 1
        
        return validity_mask
    
    def compute_error_rate(
        self, disp_left: np.ndarray, disp_right: np.ndarray
    ) -> float:
        """
        Compute the Left-Right Consistency error rate.
        
        The error rate is the percentage of pixels that fail the LRC check
        among all pixels with valid disparity values.
        
        Args:
            disp_left: Left disparity map
            disp_right: Right disparity map
            
        Returns:
            LRC error rate as a percentage (0-100)
        """
        # Convert from fixed-point to float if needed
        if disp_left.dtype == np.int16:
            disp_left_float = disp_left.astype(np.float32) / 16.0
        else:
            disp_left_float = disp_left.astype(np.float32)
        
        # Get validity mask
        validity_mask = self.validate_consistency(disp_left, disp_right)
        
        # Count valid disparities (pixels with d > 0)
        valid_disparities = np.sum(disp_left_float > 0)
        
        if valid_disparities == 0:
            return 0.0
        
        # Count pixels that passed LRC check
        passed_lrc = np.sum(validity_mask)
        
        # Error rate = (failed / total_valid) * 100
        failed_lrc = valid_disparities - passed_lrc
        error_rate = (failed_lrc / valid_disparities) * 100.0
        
        return error_rate


class WLSFilter:
    """
    Weighted Least Squares filter for sub-pixel disparity refinement.
    
    This class implements edge-preserving smoothing using WLS filtering,
    which refines disparity maps to sub-pixel accuracy while preserving
    edges. The filter uses a guide image (typically the left image) to
    detect edges and applies smoothing that respects these boundaries.
    
    WLS filtering is particularly effective for:
    - Sub-pixel disparity refinement
    - Smoothing textureless regions while preserving edges
    - Reducing noise in disparity maps
    - Improving disparity accuracy after SGBM and LRC validation
    """
    
    def __init__(self, lambda_val: float = 8000.0, sigma_color: float = 1.5):
        """
        Initialize WLS filter with smoothing parameters.
        
        Args:
            lambda_val: Regularization parameter controlling smoothness.
                       Higher values produce smoother results.
                       Typical range: 1000-10000 for road scenes.
            sigma_color: Color sensitivity parameter for edge detection.
                        Controls how much color differences affect smoothing.
                        Typical range: 0.5-2.0
        """
        self.lambda_val = lambda_val
        self.sigma_color = sigma_color
    
    def filter_disparity(
        self, 
        disparity: np.ndarray, 
        guide_image: np.ndarray,
        disparity_right: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply Weighted Least Squares filtering for sub-pixel refinement.
        
        This method applies edge-preserving smoothing to the disparity map
        using the guide image to detect edges. The filter smooths textureless
        regions while preserving sharp transitions at object boundaries.
        
        Args:
            disparity: Input disparity map (16-bit fixed point from SGBM)
            guide_image: Guide image for edge-aware filtering (typically left image).
                        Can be grayscale or color.
            disparity_right: Optional right disparity map for improved filtering.
                           If not provided, a dummy right disparity will be created.
            
        Returns:
            Filtered disparity map (16-bit fixed point format, same as input)
        """
        # Ensure guide image is in correct format
        if len(guide_image.shape) == 2:
            # Convert grayscale to BGR for WLS filter
            guide_bgr = cv2.cvtColor(guide_image, cv2.COLOR_GRAY2BGR)
        else:
            guide_bgr = guide_image
        
        # Create a dummy matcher for WLS filter initialization
        # The matcher is only used to get parameters, not for actual matching
        # We use a minimal SGBM matcher just for initialization
        dummy_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16,
            blockSize=3
        )
        
        # Create WLS filter
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=dummy_matcher)
        wls_filter.setLambda(self.lambda_val)
        wls_filter.setSigmaColor(self.sigma_color)
        
        # If right disparity is not provided, create a dummy one
        if disparity_right is None:
            # Convert disparity to float32 for WLS filter
            disparity_float = disparity.astype(np.float32) / 16.0
            disparity_right = disparity_float.copy()
        else:
            # Ensure right disparity is in float32 format
            if disparity_right.dtype == np.int16:
                disparity_right = disparity_right.astype(np.float32) / 16.0
            else:
                disparity_right = disparity_right.astype(np.float32)
        
        # Apply filter
        filtered_disparity = wls_filter.filter(
            disparity, 
            guide_bgr, 
            disparity_map_right=disparity_right
        )
        
        return filtered_disparity
    
    def filter_with_confidence(
        self,
        disparity: np.ndarray,
        guide_image: np.ndarray,
        disparity_right: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply WLS filtering and return both filtered disparity and confidence map.
        
        The confidence map indicates the reliability of each disparity value,
        with higher values indicating more confident estimates.
        
        Args:
            disparity: Input disparity map (16-bit fixed point)
            guide_image: Guide image for edge-aware filtering
            disparity_right: Optional right disparity map
            
        Returns:
            Tuple of (filtered_disparity, confidence_map)
        """
        # Ensure guide image is in correct format
        if len(guide_image.shape) == 2:
            guide_bgr = cv2.cvtColor(guide_image, cv2.COLOR_GRAY2BGR)
        else:
            guide_bgr = guide_image
        
        # Create dummy matcher and WLS filter
        dummy_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16,
            blockSize=3
        )
        
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=dummy_matcher)
        wls_filter.setLambda(self.lambda_val)
        wls_filter.setSigmaColor(self.sigma_color)
        
        # If right disparity is not provided, create a dummy one
        if disparity_right is None:
            disparity_float = disparity.astype(np.float32) / 16.0
            disparity_right = disparity_float.copy()
        else:
            # Ensure right disparity is in float32 format
            if disparity_right.dtype == np.int16:
                disparity_right = disparity_right.astype(np.float32) / 16.0
            else:
                disparity_right = disparity_right.astype(np.float32)
        
        # Apply filter
        filtered_disparity = wls_filter.filter(
            disparity,
            guide_bgr,
            disparity_map_right=disparity_right
        )
        
        # Get confidence map
        confidence_map = wls_filter.getConfidenceMap()
        
        return filtered_disparity, confidence_map
