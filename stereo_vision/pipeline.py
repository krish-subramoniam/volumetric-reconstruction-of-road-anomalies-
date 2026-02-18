"""
Integrated Pipeline Controller for Advanced Stereo Vision System

This module provides the main pipeline controller that wires all modules together
into a cohesive processing pipeline for volumetric road anomaly detection.

The pipeline integrates:
- CharuCo-based calibration
- Advanced disparity estimation with SGBM, LRC, and WLS
- V-Disparity ground plane detection
- 3D point cloud reconstruction
- Alpha Shape mesh generation
- Watertight volume calculation
- Quality metrics and diagnostics

Requirements: All requirements integration
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
import cv2
from pathlib import Path
import json
import time

# Import all pipeline modules
from stereo_vision.config import PipelineConfig
from stereo_vision.calibration import (
    CharuCoCalibrator, StereoCalibrator, CameraParameters, StereoParameters
)
from stereo_vision.preprocessing import ImagePreprocessor
from stereo_vision.disparity import SGBMEstimator, LRCValidator, WLSFilter, DisparityResult
from stereo_vision.ground_plane import (
    VDisparityGenerator, HoughLineDetector, GroundPlaneModel
)
from stereo_vision.reconstruction import PointCloudGenerator, OutlierRemover
from stereo_vision.volumetric import (
    AlphaShapeGenerator, MeshCapper, VolumeCalculator, Mesh
)
from stereo_vision.quality_metrics import (
    LRCErrorCalculator, PlanarityCalculator, TemporalStabilityCalculator,
    CalibrationQualityReporter, DiagnosticVisualizer, QualityMetrics
)
from stereo_vision.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class AnomalyResult:
    """Result for a single detected anomaly."""
    anomaly_type: str  # "pothole" or "hump"
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    point_cloud: np.ndarray  # Nx3 or Nx6 array
    mesh: Optional[Mesh]  # Generated mesh (if successful)
    volume_cubic_meters: float
    volume_liters: float
    volume_cubic_cm: float
    uncertainty_cubic_meters: float
    is_valid: bool
    validation_message: str
    area_square_meters: float
    depth_statistics: Dict[str, float]


@dataclass
class PipelineResult:
    """Complete result from pipeline processing."""
    anomalies: List[AnomalyResult]
    disparity_map: np.ndarray
    v_disparity: np.ndarray
    ground_plane_params: Optional[Tuple[float, float]]
    quality_metrics: QualityMetrics
    processing_time: float
    diagnostic_panel: Optional[np.ndarray]


class StereoVisionPipeline:
    """
    Main integrated pipeline controller for stereo vision processing.
    
    This class orchestrates all processing stages from calibration through
    volume calculation, providing a unified interface for the complete
    stereo vision system.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the stereo vision pipeline.
        
        Args:
            config: Pipeline configuration. If None, uses default configuration.
        """
        self.config = config if config is not None else PipelineConfig()
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration:\n" + "\n".join(errors))
        
        # Initialize all pipeline components
        self._initialize_components()
        
        # Calibration parameters (set via calibrate() or load_calibration())
        self.stereo_params: Optional[StereoParameters] = None
        self.is_calibrated = False
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline processing components."""
        # Preprocessing
        self.preprocessor = ImagePreprocessor()
        
        # Disparity estimation
        self.sgbm_estimator = SGBMEstimator(
            baseline=self.config.camera.baseline,
            focal_length=self.config.camera.focal_length
        )
        self.lrc_validator = LRCValidator(max_diff=self.config.lrc.max_diff)
        self.wls_filter = WLSFilter(
            lambda_val=self.config.wls.lambda_val,
            sigma_color=self.config.wls.sigma_color
        )
        
        # Ground plane detection
        self.v_disparity_generator = VDisparityGenerator(
            max_disparity=self.config.v_disparity.max_disparity
        )
        self.hough_detector = HoughLineDetector(
            threshold=self.config.v_disparity.hough_threshold,
            min_line_length=self.config.v_disparity.hough_min_line_length,
            max_line_gap=self.config.v_disparity.hough_max_line_gap
        )
        self.ground_plane = GroundPlaneModel(
            threshold_factor=self.config.anomaly_detection.threshold_factor
        )
        
        # 3D reconstruction
        # Q matrix will be set after calibration
        self.point_cloud_generator: Optional[PointCloudGenerator] = None
        self.outlier_remover = OutlierRemover(
            k_neighbors=self.config.outlier_removal.k_neighbors,
            std_ratio=self.config.outlier_removal.std_ratio
        )
        
        # Volumetric analysis
        self.alpha_shape_generator = AlphaShapeGenerator(alpha=1.0)
        self.mesh_capper = MeshCapper()
        self.volume_calculator = VolumeCalculator()
        
        # Quality metrics
        self.lrc_error_calculator = LRCErrorCalculator(
            max_diff=self.config.lrc.max_diff
        )
        self.planarity_calculator = PlanarityCalculator()
        self.temporal_stability_calculator = TemporalStabilityCalculator()
        self.calibration_quality_reporter = CalibrationQualityReporter()
        self.diagnostic_visualizer = DiagnosticVisualizer()
    
    def calibrate(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        charuco_params: Optional[Dict] = None
    ) -> StereoParameters:
        """
        Perform stereo calibration using CharuCo boards.
        
        Args:
            left_images: List of left camera calibration images
            right_images: List of right camera calibration images
            charuco_params: Optional CharuCo board parameters
            
        Returns:
            Calibrated stereo parameters
            
        Raises:
            ValueError: If calibration fails
        """
        # Create CharuCo calibrator
        if charuco_params is None:
            charuco_calibrator = CharuCoCalibrator()
        else:
            charuco_calibrator = CharuCoCalibrator(**charuco_params)
        
        # Calibrate individual cameras
        print("Calibrating left camera...")
        left_params = charuco_calibrator.calibrate_intrinsics(left_images)
        if left_params is None:
            raise ValueError("Left camera calibration failed")
        
        print(f"Left camera calibration: RMS error = {left_params.reprojection_error:.4f} pixels")
        
        print("Calibrating right camera...")
        right_params = charuco_calibrator.calibrate_intrinsics(right_images)
        if right_params is None:
            raise ValueError("Right camera calibration failed")
        
        print(f"Right camera calibration: RMS error = {right_params.reprojection_error:.4f} pixels")
        
        # Stereo calibration
        print("Performing stereo calibration...")
        stereo_calibrator = StereoCalibrator(charuco_calibrator)
        stereo_params = stereo_calibrator.calibrate_stereo(
            left_params, right_params, left_images, right_images
        )
        
        if stereo_params is None:
            raise ValueError("Stereo calibration failed")
        
        print(f"Stereo calibration complete: baseline = {stereo_params.baseline:.4f} m")
        
        # Store calibration
        self.stereo_params = stereo_params
        self.is_calibrated = True
        
        # Initialize point cloud generator with Q matrix
        self.point_cloud_generator = PointCloudGenerator(
            Q_matrix=stereo_params.Q_matrix,
            min_depth=self.config.depth_range.min_depth,
            max_depth=self.config.depth_range.max_depth
        )
        
        return stereo_params
    
    def load_calibration(self, calibration_file: str) -> None:
        """
        Load calibration parameters from file.
        
        Args:
            calibration_file: Path to calibration file (JSON or NPZ format)
        """
        path = Path(calibration_file)
        
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
        
        # Load calibration data
        if path.suffix == '.npz':
            data = np.load(calibration_file)
            
            # Reconstruct stereo parameters
            left_camera = CameraParameters(
                camera_matrix=data['left_camera_matrix'],
                distortion_coeffs=data['left_distortion'],
                reprojection_error=float(data['left_reprojection_error']),
                image_size=tuple(data['left_image_size'])
            )
            
            right_camera = CameraParameters(
                camera_matrix=data['right_camera_matrix'],
                distortion_coeffs=data['right_distortion'],
                reprojection_error=float(data['right_reprojection_error']),
                image_size=tuple(data['right_image_size'])
            )
            
            self.stereo_params = StereoParameters(
                left_camera=left_camera,
                right_camera=right_camera,
                rotation_matrix=data['rotation_matrix'],
                translation_vector=data['translation_vector'],
                baseline=float(data['baseline']),
                Q_matrix=data['Q_matrix'],
                rectification_maps_left=(data['map_left_x'], data['map_left_y']),
                rectification_maps_right=(data['map_right_x'], data['map_right_y'])
            )
        else:
            raise ValueError(f"Unsupported calibration file format: {path.suffix}")
        
        self.is_calibrated = True
        
        # Initialize point cloud generator
        self.point_cloud_generator = PointCloudGenerator(
            Q_matrix=self.stereo_params.Q_matrix,
            min_depth=self.config.depth_range.min_depth,
            max_depth=self.config.depth_range.max_depth
        )
        
        print(f"Calibration loaded from {calibration_file}")
    
    def save_calibration(self, calibration_file: str) -> None:
        """
        Save calibration parameters to file.
        
        Args:
            calibration_file: Path to save calibration file (NPZ format)
        """
        if not self.is_calibrated or self.stereo_params is None:
            raise RuntimeError("No calibration to save. Run calibrate() first.")
        
        path = Path(calibration_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as NPZ
        np.savez(
            calibration_file,
            left_camera_matrix=self.stereo_params.left_camera.camera_matrix,
            left_distortion=self.stereo_params.left_camera.distortion_coeffs,
            left_reprojection_error=self.stereo_params.left_camera.reprojection_error,
            left_image_size=self.stereo_params.left_camera.image_size,
            right_camera_matrix=self.stereo_params.right_camera.camera_matrix,
            right_distortion=self.stereo_params.right_camera.distortion_coeffs,
            right_reprojection_error=self.stereo_params.right_camera.reprojection_error,
            right_image_size=self.stereo_params.right_camera.image_size,
            rotation_matrix=self.stereo_params.rotation_matrix,
            translation_vector=self.stereo_params.translation_vector,
            baseline=self.stereo_params.baseline,
            Q_matrix=self.stereo_params.Q_matrix,
            map_left_x=self.stereo_params.rectification_maps_left[0],
            map_left_y=self.stereo_params.rectification_maps_left[1],
            map_right_x=self.stereo_params.rectification_maps_right[0],
            map_right_y=self.stereo_params.rectification_maps_right[1]
        )
        
        print(f"Calibration saved to {calibration_file}")
    
    def process_stereo_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        generate_diagnostics: bool = True
    ) -> PipelineResult:
        """
        Process a stereo image pair through the complete pipeline.
        
        This is the main entry point for processing. It executes all stages:
        1. Preprocessing
        2. Rectification
        3. Disparity estimation
        4. Ground plane detection
        5. 3D reconstruction
        6. Anomaly segmentation
        7. Volume calculation
        8. Quality metrics
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            generate_diagnostics: Whether to generate diagnostic visualizations
            
        Returns:
            PipelineResult containing all processing outputs
            
        Raises:
            RuntimeError: If pipeline is not calibrated
        """
        if not self.is_calibrated or self.stereo_params is None:
            raise RuntimeError("Pipeline not calibrated. Run calibrate() or load_calibration() first.")
        
        start_time = time.time()
        
        # Stage 1: Preprocessing
        left_processed, right_processed = self.preprocessor.preprocess_stereo_pair(
            left_image, right_image
        )
        
        # Stage 2: Rectification
        left_rectified = cv2.remap(
            left_processed,
            self.stereo_params.rectification_maps_left[0],
            self.stereo_params.rectification_maps_left[1],
            cv2.INTER_LINEAR
        )
        
        right_rectified = cv2.remap(
            right_processed,
            self.stereo_params.rectification_maps_right[0],
            self.stereo_params.rectification_maps_right[1],
            cv2.INTER_LINEAR
        )
        
        # Stage 3: Disparity estimation
        disparity_result = self._compute_disparity(left_rectified, right_rectified)
        
        # Stage 4: Ground plane detection
        v_disparity = self.v_disparity_generator.generate_v_disparity(
            disparity_result.disparity_map
        )
        
        ground_plane_fitted = self.ground_plane.fit_from_v_disparity(
            v_disparity, self.hough_detector
        )
        
        if not ground_plane_fitted:
            raise RuntimeError("Ground plane detection failed")
        
        # Stage 5: Anomaly segmentation
        pothole_mask, hump_mask = self.ground_plane.segment_anomalies(
            disparity_result.disparity_map
        )
        
        # Stage 6: 3D reconstruction and volume calculation
        anomalies = self._process_anomalies(
            disparity_result.disparity_map,
            pothole_mask,
            hump_mask,
            left_rectified
        )
        
        # Stage 7: Quality metrics
        quality_metrics = self._calculate_quality_metrics(
            disparity_result,
            self.ground_plane
        )
        
        # Stage 8: Diagnostic visualization
        diagnostic_panel = None
        if generate_diagnostics:
            diagnostic_panel = self.diagnostic_visualizer.create_diagnostic_panel(
                left_rectified,
                disparity_result.disparity_map,
                v_disparity,
                self.ground_plane,
                pothole_mask,
                hump_mask
            )
        
        processing_time = time.time() - start_time
        
        return PipelineResult(
            anomalies=anomalies,
            disparity_map=disparity_result.disparity_map,
            v_disparity=v_disparity,
            ground_plane_params=self.ground_plane.get_plane_parameters(),
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            diagnostic_panel=diagnostic_panel
        )
    
    def _compute_disparity(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray
    ) -> DisparityResult:
        """Compute disparity with SGBM, LRC, and WLS filtering."""
        start_time = time.time()
        
        # SGBM computation
        disp_left = self.sgbm_estimator.compute_disparity(left_rectified, right_rectified)
        
        # Compute right disparity for LRC check
        disp_right = self.sgbm_estimator.compute_disparity(right_rectified, left_rectified)
        
        # LRC validation
        if self.config.lrc.enabled:
            validity_mask = self.lrc_validator.validate_consistency(disp_left, disp_right)
            lrc_error_rate = self.lrc_validator.compute_error_rate(disp_left, disp_right)
        else:
            validity_mask = np.ones(disp_left.shape, dtype=np.uint8)
            lrc_error_rate = 0.0
        
        # WLS filtering
        if self.config.wls.enabled:
            disp_filtered = self.wls_filter.filter_disparity(
                disp_left, left_rectified, disp_right
            )
        else:
            disp_filtered = disp_left.astype(np.float32) / 16.0
        
        # Apply validity mask
        disp_filtered[validity_mask == 0] = 0
        
        processing_time = time.time() - start_time
        
        return DisparityResult(
            disparity_map=disp_filtered,
            validity_mask=validity_mask,
            lrc_error_rate=lrc_error_rate,
            processing_time=processing_time
        )
    
    def _process_anomalies(
        self,
        disparity_map: np.ndarray,
        pothole_mask: np.ndarray,
        hump_mask: np.ndarray,
        left_image: np.ndarray
    ) -> List[AnomalyResult]:
        """Process detected anomalies to calculate volumes."""
        anomalies = []
        
        # Process potholes
        anomalies.extend(self._process_anomaly_type(
            disparity_map, pothole_mask, left_image, "pothole"
        ))
        
        # Process humps
        anomalies.extend(self._process_anomaly_type(
            disparity_map, hump_mask, left_image, "hump"
        ))
        
        return anomalies
    
    def _process_anomaly_type(
        self,
        disparity_map: np.ndarray,
        mask: np.ndarray,
        left_image: np.ndarray,
        anomaly_type: str
    ) -> List[AnomalyResult]:
        """Process all anomalies of a specific type."""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        anomalies = []
        
        for label in range(1, num_labels):  # Skip background (label 0)
            # Get bounding box
            x, y, w, h, area = stats[label]
            
            # Filter by size
            if area < self.config.anomaly_detection.min_anomaly_size:
                continue
            if area > self.config.anomaly_detection.max_anomaly_size:
                continue
            
            # Extract anomaly region mask
            anomaly_mask = (labels == label).astype(np.uint8)
            mask_bool = anomaly_mask > 0
            
            # Extract disparity values at mask locations BEFORE any modification
            disparity_at_mask = disparity_map[mask_bool]
            
            # Generate 3D point cloud for the full disparity map
            disp_float = disparity_map.astype(np.float32)
            points_3d = cv2.reprojectImageTo3D(disp_float, self.point_cloud_generator.Q_matrix, handleMissingValues=True)
            
            # Extract points only from the anomaly mask
            anomaly_points_3d = points_3d[mask_bool]
            
            logger.info(
                f"Anomaly {label}: Extracted {anomaly_points_3d.shape[0]} points from mask",
                mask_pixels=np.sum(mask_bool),
                bbox=f"{x},{y},{w}x{h}",
                disparity_range=f"{disparity_at_mask[disparity_at_mask>0].min():.2f}-{disparity_at_mask.max():.2f}" if np.any(disparity_at_mask > 0) else "all_zero"
            )
            
            # Filter valid points (finite Z, positive disparity, within depth range)
            valid_mask = np.isfinite(anomaly_points_3d[:, 2])
            valid_mask &= (disparity_at_mask > 0)  # Use the extracted disparity values
            valid_mask &= (anomaly_points_3d[:, 2] >= self.config.depth_range.min_depth)
            valid_mask &= (anomaly_points_3d[:, 2] <= self.config.depth_range.max_depth)
            
            anomaly_points = anomaly_points_3d[valid_mask]
            
            logger.info(
                f"Anomaly {label}: {anomaly_points.shape[0]} valid points after filtering",
                depth_range=f"{anomaly_points[:, 2].min():.3f}-{anomaly_points[:, 2].max():.3f}m" if anomaly_points.shape[0] > 0 else "N/A"
            )
            
            if anomaly_points.shape[0] < 4:
                continue  # Not enough points for meshing
            
            # Remove outliers
            if self.config.outlier_removal.enabled and anomaly_points.shape[0] > 10:
                try:
                    anomaly_points = self.outlier_remover.remove_statistical_outliers(
                        anomaly_points
                    )
                except:
                    pass  # Continue with original points if outlier removal fails
            
            if anomaly_points.shape[0] < 4:
                continue
            
            # Calculate depth statistics
            depth_stats = {
                'mean_depth': float(np.mean(anomaly_points[:, 2])),
                'std_depth': float(np.std(anomaly_points[:, 2])),
                'min_depth': float(np.min(anomaly_points[:, 2])),
                'max_depth': float(np.max(anomaly_points[:, 2]))
            }
            
            # Generate mesh and calculate volume
            volume_result = self._calculate_anomaly_volume(anomaly_points[:, :3])
            
            # Calculate area
            area_m2 = self._calculate_anomaly_area(anomaly_points[:, :3])
            
            anomalies.append(AnomalyResult(
                anomaly_type=anomaly_type,
                bounding_box=(x, y, w, h),
                point_cloud=anomaly_points,
                mesh=volume_result.get('mesh'),
                volume_cubic_meters=volume_result['volume_cubic_meters'],
                volume_liters=volume_result['volume_liters'],
                volume_cubic_cm=volume_result['volume_cubic_cm'],
                uncertainty_cubic_meters=volume_result.get('uncertainty_cubic_meters', 0.0),
                is_valid=volume_result['is_valid'],
                validation_message=volume_result['validation_message'],
                area_square_meters=area_m2,
                depth_statistics=depth_stats
            ))
        
        return anomalies
    
    def _calculate_anomaly_volume(self, points: np.ndarray) -> Dict:
        """Calculate volume for an anomaly point cloud."""
        try:
            # Log point cloud info for debugging
            logger.info(
                f"Calculating volume for {points.shape[0]} points",
                num_points=points.shape[0],
                depth_range=f"{points[:, 2].min():.3f}-{points[:, 2].max():.3f}m"
            )
            
            # Check if we have enough points
            if points.shape[0] < 4:
                return {
                    'mesh': None,
                    'volume_cubic_meters': 0.0,
                    'volume_liters': 0.0,
                    'volume_cubic_cm': 0.0,
                    'is_valid': False,
                    'validation_message': f'Insufficient points: {points.shape[0]} (need at least 4)'
                }
            
            # Try multiple alpha values if first one fails
            alpha_values = [0.5, 1.0, 2.0, 5.0, 10.0]
            mesh = None
            alpha_used = None
            
            for alpha in alpha_values:
                try:
                    self.alpha_shape_generator.update_alpha(alpha)
                    mesh = self.alpha_shape_generator.generate_alpha_shape(points)
                    
                    if mesh.faces.shape[0] > 0:
                        alpha_used = alpha
                        logger.info(f"Alpha shape successful with alpha={alpha}, faces={mesh.faces.shape[0]}")
                        break
                    else:
                        logger.warning(f"Alpha shape with alpha={alpha} produced no faces")
                except Exception as e:
                    logger.warning(f"Alpha shape failed with alpha={alpha}: {str(e)}")
                    continue
            
            if mesh is None or mesh.faces.shape[0] == 0:
                # Fallback: Use convex hull for volume estimation
                logger.warning("Alpha shape failed, using convex hull fallback")
                return self._calculate_volume_convex_hull(points)
            
            # Extract boundary edges
            boundary_edges = self.alpha_shape_generator.extract_boundary_edges(mesh)
            logger.info(f"Extracted {len(boundary_edges)} boundary edges")
            
            if len(boundary_edges) == 0:
                # Mesh might already be closed
                logger.info("No boundary edges - checking if mesh is already watertight")
                is_watertight = self.mesh_capper.validate_watertightness(mesh)
                
                if is_watertight:
                    # Calculate volume directly
                    volume_result = self.volume_calculator.calculate_volume_with_units(mesh)
                    volume_result['mesh'] = mesh
                    logger.info(f"Volume calculated: {volume_result['volume_cubic_meters']:.6f} m³")
                    return volume_result
                else:
                    logger.warning("Mesh has no boundary edges but is not watertight")
                    return self._calculate_volume_convex_hull(points)
            
            # Generate caps
            try:
                cap_mesh = self.mesh_capper.triangulate_boundary(
                    boundary_edges, mesh.vertices
                )
                logger.info(f"Generated {cap_mesh.faces.shape[0]} cap faces")
            except Exception as e:
                logger.error(f"Mesh capping failed: {str(e)}")
                return self._calculate_volume_convex_hull(points)
            
            # Create watertight mesh
            watertight_mesh = self.mesh_capper.create_watertight_mesh(mesh, cap_mesh)
            
            # Validate watertightness
            is_watertight = self.mesh_capper.validate_watertightness(watertight_mesh)
            logger.info(f"Watertight validation: {is_watertight}")
            
            if not is_watertight:
                logger.warning("Mesh is not watertight after capping, using convex hull")
                return self._calculate_volume_convex_hull(points)
            
            # Calculate volume
            volume_result = self.volume_calculator.calculate_volume_with_units(
                watertight_mesh
            )
            volume_result['mesh'] = watertight_mesh
            
            logger.info(
                f"Volume calculated successfully: {volume_result['volume_cubic_meters']:.6f} m³",
                alpha=alpha_used,
                is_valid=volume_result['is_valid']
            )
            
            return volume_result
            
        except Exception as e:
            logger.error(f"Volume calculation failed: {str(e)}", exc_info=True)
            # Try convex hull as last resort
            try:
                return self._calculate_volume_convex_hull(points)
            except:
                return {
                    'mesh': None,
                    'volume_cubic_meters': 0.0,
                    'volume_liters': 0.0,
                    'volume_cubic_cm': 0.0,
                    'is_valid': False,
                    'validation_message': f'Volume calculation failed: {str(e)}'
                }
    
    def _calculate_volume_convex_hull(self, points: np.ndarray) -> Dict:
        """Calculate volume using convex hull as fallback method."""
        try:
            from scipy.spatial import ConvexHull
            
            logger.info(f"Using convex hull fallback for {points.shape[0]} points")
            
            # Calculate convex hull
            hull = ConvexHull(points)
            volume_m3 = hull.volume
            
            logger.info(f"Convex hull volume: {volume_m3:.6f} m³")
            
            # Convert units
            volume_units = self.volume_calculator.convert_volume_units(volume_m3)
            
            # Validate constraints
            is_valid, validation_message = self.volume_calculator.validate_volume_constraints(volume_m3)
            
            # Create a simple mesh from convex hull
            mesh = Mesh(
                vertices=points[hull.vertices],
                faces=hull.simplices,
                is_watertight=True
            )
            
            return {
                'mesh': mesh,
                'volume_cubic_meters': volume_units['cubic_meters'],
                'volume_liters': volume_units['liters'],
                'volume_cubic_cm': volume_units['cubic_centimeters'],
                'uncertainty_cubic_meters': 0.0,
                'is_valid': is_valid,
                'validation_message': f'Convex hull approximation: {validation_message}'
            }
            
        except Exception as e:
            logger.error(f"Convex hull fallback failed: {str(e)}")
            return {
                'mesh': None,
                'volume_cubic_meters': 0.0,
                'volume_liters': 0.0,
                'volume_cubic_cm': 0.0,
                'is_valid': False,
                'validation_message': f'Convex hull failed: {str(e)}'
            }
    
    def _calculate_anomaly_area(self, points: np.ndarray) -> float:
        """Calculate approximate surface area of anomaly."""
        if points.shape[0] < 3:
            return 0.0
        
        # Simple approximation: project to XY plane and calculate 2D area
        xy_points = points[:, :2]
        
        # Use convex hull for area estimation
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(xy_points)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0
    
    def _calculate_quality_metrics(
        self,
        disparity_result: DisparityResult,
        ground_plane: GroundPlaneModel
    ) -> QualityMetrics:
        """Calculate quality metrics for the processing result."""
        # Planarity RMSE
        try:
            planarity_rmse = self.planarity_calculator.calculate_planarity_rmse(
                disparity_result.disparity_map,
                ground_plane
            )
        except:
            planarity_rmse = None
        
        # Calibration quality
        calibration_error = None
        if self.stereo_params is not None:
            quality_report = self.calibration_quality_reporter.report_stereo_quality(
                self.stereo_params
            )
            calibration_error = quality_report['mean_reprojection_error']
        
        return QualityMetrics(
            lrc_error_rate=disparity_result.lrc_error_rate,
            planarity_rmse=planarity_rmse,
            temporal_stability=None,  # Requires sequence processing
            calibration_reprojection_error=calibration_error
        )
    
    def process_batch(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        output_dir: Optional[str] = None
    ) -> List[PipelineResult]:
        """
        Process a batch of stereo image pairs.
        
        Args:
            image_pairs: List of (left_image, right_image) tuples
            output_dir: Optional directory to save results
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        for idx, (left_img, right_img) in enumerate(image_pairs):
            print(f"Processing pair {idx + 1}/{len(image_pairs)}...")
            
            result = self.process_stereo_pair(left_img, right_img)
            results.append(result)
            
            # Save results if output directory specified
            if output_dir is not None:
                self._save_result(result, output_dir, idx)
        
        # Calculate batch statistics
        self._print_batch_statistics(results)
        
        return results
    
    def _save_result(
        self,
        result: PipelineResult,
        output_dir: str,
        index: int
    ) -> None:
        """Save processing result to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save disparity map
        np.save(output_path / f"disparity_{index:04d}.npy", result.disparity_map)
        
        # Save diagnostic panel
        if result.diagnostic_panel is not None:
            cv2.imwrite(
                str(output_path / f"diagnostics_{index:04d}.png"),
                result.diagnostic_panel
            )
        
        # Save anomaly results as JSON
        anomaly_data = []
        for anomaly in result.anomalies:
            anomaly_data.append({
                'type': anomaly.anomaly_type,
                'bounding_box': anomaly.bounding_box,
                'volume_cubic_meters': anomaly.volume_cubic_meters,
                'volume_liters': anomaly.volume_liters,
                'volume_cubic_cm': anomaly.volume_cubic_cm,
                'uncertainty_cubic_meters': anomaly.uncertainty_cubic_meters,
                'is_valid': anomaly.is_valid,
                'validation_message': anomaly.validation_message,
                'area_square_meters': anomaly.area_square_meters,
                'depth_statistics': anomaly.depth_statistics
            })
        
        with open(output_path / f"anomalies_{index:04d}.json", 'w') as f:
            json.dump(anomaly_data, f, indent=2)
    
    def _print_batch_statistics(self, results: List[PipelineResult]) -> None:
        """Print summary statistics for batch processing."""
        total_anomalies = sum(len(r.anomalies) for r in results)
        total_potholes = sum(
            len([a for a in r.anomalies if a.anomaly_type == "pothole"])
            for r in results
        )
        total_humps = sum(
            len([a for a in r.anomalies if a.anomaly_type == "hump"])
            for r in results
        )
        
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total image pairs processed: {len(results)}")
        print(f"Total anomalies detected: {total_anomalies}")
        print(f"  - Potholes: {total_potholes}")
        print(f"  - Humps: {total_humps}")
        print(f"Average processing time: {avg_processing_time:.2f} seconds")
        print("="*60 + "\n")


def create_pipeline(config_file: Optional[str] = None) -> StereoVisionPipeline:
    """
    Factory function to create a configured pipeline.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configured StereoVisionPipeline instance
    """
    if config_file is not None:
        config = PipelineConfig.load_from_file(config_file)
    else:
        config = PipelineConfig()
    
    return StereoVisionPipeline(config)
