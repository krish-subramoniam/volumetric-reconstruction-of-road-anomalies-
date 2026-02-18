"""CharuCo-based calibration system for stereo vision pipeline."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from stereo_vision.logging_config import get_logger, PerformanceTimer
from stereo_vision.errors import (
    CharuCoDetectionError, InsufficientCalibrationDataError,
    CalibrationQualityError, StereoCalibrationError
)

# Initialize logger
logger = get_logger(__name__)


@dataclass
class CameraParameters:
    """Camera intrinsic parameters and calibration quality metrics."""
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    distortion_coeffs: np.ndarray  # 5x1 distortion coefficients
    reprojection_error: float  # RMS reprojection error
    image_size: Tuple[int, int]  # (width, height)


@dataclass
class StereoParameters:
    """Stereo camera system parameters."""
    left_camera: CameraParameters
    right_camera: CameraParameters
    rotation_matrix: np.ndarray  # 3x3 rotation between cameras
    translation_vector: np.ndarray  # 3x1 translation vector
    baseline: float  # Distance between camera centers
    Q_matrix: np.ndarray  # 4x4 reprojection matrix
    rectification_maps_left: Tuple[np.ndarray, np.ndarray]
    rectification_maps_right: Tuple[np.ndarray, np.ndarray]


class CharuCoCalibrator:
    """Handles CharuCo board detection and camera calibration."""
    
    def __init__(self, squares_x: int = 7, squares_y: int = 5, 
                 square_length: float = 0.04, marker_length: float = 0.03,
                 dictionary: int = cv2.aruco.DICT_6X6_250):
        """
        Initialize CharuCo calibrator.
        
        Args:
            squares_x: Number of squares in X direction
            squares_y: Number of squares in Y direction
            square_length: Length of square side in meters
            marker_length: Length of ArUco marker side in meters
            dictionary: ArUco dictionary to use
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        
        # Create CharuCo board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.aruco_dict
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
    
    def detect_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect CharuCo corners in an image.
        
        Args:
            image: Input grayscale or color image
            
        Returns:
            Tuple of (corners, ids) or (None, None) if detection fails
            
        Raises:
            CharuCoDetectionError: If image is invalid or detection fails critically
        """
        try:
            if image is None or image.size == 0:
                raise CharuCoDetectionError(
                    "Invalid input image",
                    details={"reason": "Image is None or empty"}
                )
            
            logger.debug("Detecting CharuCo corners", image_shape=image.shape)
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect CharuCo corners
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                logger.debug(
                    "CharuCo corners detected successfully",
                    num_corners=len(charuco_corners)
                )
                return charuco_corners, charuco_ids
            
            logger.warning("Insufficient CharuCo corners detected", num_corners=0 if charuco_corners is None else len(charuco_corners))
            return None, None
            
        except cv2.error as e:
            raise CharuCoDetectionError(
                "OpenCV error during CharuCo detection",
                details={"opencv_error": str(e)}
            )
        except Exception as e:
            logger.error(f"Unexpected error in CharuCo detection: {str(e)}", exc_info=True)
            raise
    
    def refine_corners(self, corners: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Refine corner positions to sub-pixel accuracy.
        
        Args:
            corners: Detected corner positions
            image: Input grayscale image
            
        Returns:
            Refined corner positions
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sub-pixel corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1), criteria
        )
        return refined_corners
    
    def calibrate_intrinsics(self, images: List[np.ndarray]) -> Optional[CameraParameters]:
        """
        Calibrate camera intrinsic parameters from multiple images.
        
        Args:
            images: List of calibration images
            
        Returns:
            CameraParameters object or None if calibration fails
            
        Raises:
            InsufficientCalibrationDataError: If not enough valid images
            CalibrationQualityError: If calibration quality is poor
        """
        try:
            logger.info("Starting intrinsic calibration", num_images=len(images))
            
            with PerformanceTimer(logger, "Intrinsic calibration"):
                all_corners = []
                all_ids = []
                image_size = None
                
                # Detect corners in all images
                for idx, img in enumerate(images):
                    try:
                        corners, ids = self.detect_corners(img)
                        if corners is not None:
                            all_corners.append(corners)
                            all_ids.append(ids)
                            if image_size is None:
                                image_size = (img.shape[1], img.shape[0])
                            logger.debug(f"Image {idx+1}/{len(images)}: {len(corners)} corners detected")
                        else:
                            logger.warning(f"Image {idx+1}/{len(images)}: No corners detected")
                    except Exception as e:
                        logger.warning(f"Image {idx+1}/{len(images)}: Detection failed - {str(e)}")
                        continue
                
                if len(all_corners) < 10:
                    raise InsufficientCalibrationDataError(
                        "Insufficient calibration images with valid corner detections",
                        details={
                            "required": 10,
                            "found": len(all_corners),
                            "total_images": len(images)
                        }
                    )
                
                logger.info(f"Valid corner detections: {len(all_corners)}/{len(images)}")
                
                # Calibrate camera
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                    all_corners, all_ids, self.board, image_size, None, None
                )
                
                if not ret:
                    raise CalibrationQualityError(
                        "Camera calibration failed to converge",
                        details={"return_value": ret}
                    )
                
                # Calculate reprojection error
                total_error = 0
                total_points = 0
                for i in range(len(all_corners)):
                    obj_points = self.board.getChessboardCorners()[all_ids[i].flatten()]
                    img_points, _ = cv2.projectPoints(
                        obj_points, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(all_corners[i], img_points, cv2.NORM_L2)
                    total_error += error ** 2
                    total_points += len(all_corners[i])
                
                reprojection_error = np.sqrt(total_error / total_points)
                
                logger.info(
                    "Intrinsic calibration completed",
                    reprojection_error=f"{reprojection_error:.4f}",
                    num_valid_images=len(all_corners)
                )
                
                # Warn if reprojection error is high
                if reprojection_error > 0.5:
                    logger.warning(
                        "High reprojection error detected",
                        reprojection_error=f"{reprojection_error:.4f}",
                        threshold=0.5
                    )
                
                return CameraParameters(
                    camera_matrix=camera_matrix,
                    distortion_coeffs=dist_coeffs,
                    reprojection_error=reprojection_error,
                    image_size=image_size
                )
                
        except (InsufficientCalibrationDataError, CalibrationQualityError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during intrinsic calibration: {str(e)}", exc_info=True)
            raise CalibrationQualityError(
                "Intrinsic calibration failed",
                details={"error": str(e)}
            )


class StereoCalibrator:
    """Manages two-stage stereo calibration process."""
    
    def __init__(self, charuco_calibrator: CharuCoCalibrator):
        """
        Initialize stereo calibrator.
        
        Args:
            charuco_calibrator: CharuCo calibrator instance
        """
        self.charuco_calibrator = charuco_calibrator
    
    def calibrate_stereo(
        self,
        left_params: CameraParameters,
        right_params: CameraParameters,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray]
    ) -> Optional[StereoParameters]:
        """
        Calibrate stereo camera system with fixed intrinsic parameters.
        
        Args:
            left_params: Pre-calibrated left camera parameters
            right_params: Pre-calibrated right camera parameters
            left_images: List of left camera calibration images
            right_images: List of right camera calibration images
            
        Returns:
            StereoParameters object or None if calibration fails
            
        Raises:
            StereoCalibrationError: If stereo calibration fails
        """
        try:
            logger.info(
                "Starting stereo calibration",
                num_left_images=len(left_images),
                num_right_images=len(right_images)
            )
            
            with PerformanceTimer(logger, "Stereo calibration"):
                if len(left_images) != len(right_images):
                    raise StereoCalibrationError(
                        "Mismatched number of left and right images",
                        details={
                            "left_images": len(left_images),
                            "right_images": len(right_images)
                        }
                    )
                
                # Collect corresponding corner points
                object_points = []
                left_corners_list = []
                right_corners_list = []
                
                for idx, (left_img, right_img) in enumerate(zip(left_images, right_images)):
                    try:
                        left_corners, left_ids = self.charuco_calibrator.detect_corners(left_img)
                        right_corners, right_ids = self.charuco_calibrator.detect_corners(right_img)
                        
                        if left_corners is None or right_corners is None:
                            logger.debug(f"Pair {idx+1}: Skipping - corners not detected")
                            continue
                        
                        # Find common corner IDs
                        common_ids = np.intersect1d(left_ids, right_ids)
                        if len(common_ids) < 4:
                            logger.debug(f"Pair {idx+1}: Skipping - insufficient common corners ({len(common_ids)})")
                            continue
                        
                        # Extract common corners
                        left_mask = np.isin(left_ids, common_ids)
                        right_mask = np.isin(right_ids, common_ids)
                        
                        left_common = left_corners[left_mask]
                        right_common = right_corners[right_mask]
                        
                        # Get 3D object points for common corners
                        obj_pts = self.charuco_calibrator.board.getChessboardCorners()[common_ids]
                        
                        object_points.append(obj_pts)
                        left_corners_list.append(left_common)
                        right_corners_list.append(right_common)
                        
                        logger.debug(f"Pair {idx+1}: {len(common_ids)} common corners")
                        
                    except Exception as e:
                        logger.warning(f"Pair {idx+1}: Processing failed - {str(e)}")
                        continue
                
                if len(object_points) < 10:
                    raise StereoCalibrationError(
                        "Insufficient valid stereo pairs for calibration",
                        details={
                            "required": 10,
                            "found": len(object_points),
                            "total_pairs": len(left_images)
                        }
                    )
                
                logger.info(f"Valid stereo pairs: {len(object_points)}/{len(left_images)}")
                
                # Stereo calibration with fixed intrinsics (CALIB_FIX_INTRINSIC flag)
                flags = cv2.CALIB_FIX_INTRINSIC
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
                
                ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                    object_points,
                    left_corners_list,
                    right_corners_list,
                    left_params.camera_matrix,
                    left_params.distortion_coeffs,
                    right_params.camera_matrix,
                    right_params.distortion_coeffs,
                    left_params.image_size,
                    criteria=criteria,
                    flags=flags
                )
                
                if not ret:
                    raise StereoCalibrationError(
                        "Stereo calibration failed to converge",
                        details={"return_value": ret}
                    )
                
                # Calculate baseline
                baseline = np.linalg.norm(T)
                
                logger.info(f"Stereo calibration successful", baseline=f"{baseline:.4f}m")
                
                # Compute rectification transforms
                R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
                    left_params.camera_matrix,
                    left_params.distortion_coeffs,
                    right_params.camera_matrix,
                    right_params.distortion_coeffs,
                    left_params.image_size,
                    R, T,
                    alpha=0
                )
                
                # Generate rectification maps
                map_left = cv2.initUndistortRectifyMap(
                    left_params.camera_matrix,
                    left_params.distortion_coeffs,
                    R1, P1,
                    left_params.image_size,
                    cv2.CV_32FC1
                )
                
                map_right = cv2.initUndistortRectifyMap(
                    right_params.camera_matrix,
                    right_params.distortion_coeffs,
                    R2, P2,
                    right_params.image_size,
                    cv2.CV_32FC1
                )
                
                logger.info("Rectification maps generated successfully")
                
                return StereoParameters(
                    left_camera=left_params,
                    right_camera=right_params,
                    rotation_matrix=R,
                    translation_vector=T,
                    baseline=baseline,
                    Q_matrix=Q,
                    rectification_maps_left=map_left,
                    rectification_maps_right=map_right
                )
                
        except StereoCalibrationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during stereo calibration: {str(e)}", exc_info=True)
            raise StereoCalibrationError(
                "Stereo calibration failed",
                details={"error": str(e)}
            )
    
    def compute_rectification_maps(
        self, stereo_params: StereoParameters
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Get rectification maps from stereo parameters.
        
        Args:
            stereo_params: Calibrated stereo parameters
            
        Returns:
            Tuple of (left_maps, right_maps)
        """
        return (
            stereo_params.rectification_maps_left,
            stereo_params.rectification_maps_right
        )
