"""3D reconstruction engine for stereo vision pipeline."""

from typing import Optional, Tuple
import numpy as np
import cv2

from stereo_vision.logging_config import get_logger, PerformanceTimer
from stereo_vision.errors import (
    PointCloudGenerationError, InsufficientPointsError,
    OutlierRemovalError, DepthFilterError
)

# Initialize logger
logger = get_logger(__name__)


class PointCloudGenerator:
    """
    3D point cloud generator for disparity-to-3D reprojection.
    
    This class converts 2D disparity maps into 3D point clouds using the Q matrix
    from stereo rectification. The Q matrix encodes the geometric relationship
    between disparity and 3D coordinates, allowing us to reproject each pixel
    to its corresponding 3D position in world coordinates.
    
    The reprojection formula is:
    [X, Y, Z, W]^T = Q * [x, y, disparity, 1]^T
    
    Where the final 3D coordinates are: (X/W, Y/W, Z/W)
    
    Key features:
    - Metric-accurate 3D reconstruction
    - Depth range filtering for realistic road distances
    - Handles invalid disparities gracefully
    - Preserves spatial relationships between neighboring points
    """
    
    def __init__(self, Q_matrix: np.ndarray, min_depth: float = 1.0, max_depth: float = 50.0):
        """
        Initialize point cloud generator with calibration parameters.
        
        Args:
            Q_matrix: 4x4 reprojection matrix from stereo rectification
            min_depth: Minimum valid depth in meters (default: 1.0m)
            max_depth: Maximum valid depth in meters (default: 50.0m for road scenes)
        """
        if Q_matrix.shape != (4, 4):
            raise ValueError(f"Q_matrix must be 4x4, got shape {Q_matrix.shape}")
        
        self.Q_matrix = Q_matrix
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def reproject_to_3d(
        self,
        disparity: np.ndarray,
        colors: Optional[np.ndarray] = None,
        apply_depth_filter: bool = True
    ) -> np.ndarray:
        """
        Reproject disparity map to 3D point cloud using Q matrix.
        
        This method performs the core 3D reconstruction by:
        1. Converting each pixel (x, y, disparity) to 3D coordinates
        2. Filtering points based on depth range (if enabled)
        3. Optionally associating colors with each point
        
        Args:
            disparity: Disparity map (float32 or int16 fixed-point)
            colors: Optional color image (BGR or RGB) to associate with points.
                   Must have same dimensions as disparity map.
            apply_depth_filter: If True, filter points outside depth range
            
        Returns:
            Point cloud as Nx3 or Nx6 array:
            - Nx3: [X, Y, Z] coordinates in meters
            - Nx6: [X, Y, Z, R, G, B] if colors are provided
            
            Points are in the left camera coordinate system where:
            - X: right (positive to the right)
            - Y: down (positive downward)
            - Z: forward (positive away from camera)
        """
        # Convert from fixed-point to float if needed
        if disparity.dtype == np.int16:
            disp_float = disparity.astype(np.float32) / 16.0
        else:
            disp_float = disparity.astype(np.float32)
        
        # Use OpenCV's reprojectImageTo3D for efficient reprojection
        # This applies the Q matrix to convert disparity to 3D
        points_3d = cv2.reprojectImageTo3D(disp_float, self.Q_matrix, handleMissingValues=True)
        
        # Filter valid points
        # Invalid points have infinite Z values
        valid_mask = np.isfinite(points_3d[:, :, 2])
        
        # Also filter points with zero or negative disparity
        valid_mask &= (disp_float > 0)
        
        # Apply depth range filtering if requested
        if apply_depth_filter:
            depth = points_3d[:, :, 2]
            valid_mask &= (depth >= self.min_depth) & (depth <= self.max_depth)
        
        # Extract valid points
        valid_points = points_3d[valid_mask]
        
        # Add colors if provided
        if colors is not None:
            if colors.shape[:2] != disparity.shape:
                raise ValueError(
                    f"Color image shape {colors.shape[:2]} must match "
                    f"disparity shape {disparity.shape}"
                )
            
            # Extract colors for valid points
            if len(colors.shape) == 3:
                # Color image (BGR or RGB)
                valid_colors = colors[valid_mask]
                # Combine points and colors
                point_cloud = np.hstack([valid_points, valid_colors])
            else:
                # Grayscale image - replicate to RGB
                valid_gray = colors[valid_mask]
                valid_colors = np.stack([valid_gray, valid_gray, valid_gray], axis=1)
                point_cloud = np.hstack([valid_points, valid_colors])
        else:
            point_cloud = valid_points
        
        return point_cloud
    
    def filter_depth_range(
        self,
        points: np.ndarray,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter point cloud to realistic depth range for road scenes.
        
        This method removes points that are too close or too far to be
        part of the road surface or relevant anomalies. This is important for:
        - Removing noise from invalid disparities
        - Focusing on the region of interest (road surface)
        - Improving computational efficiency for downstream processing
        
        Args:
            points: Point cloud as Nx3 or Nx6 array (X, Y, Z, [R, G, B])
            min_depth: Minimum depth in meters (if None, use instance default)
            max_depth: Maximum depth in meters (if None, use instance default)
            
        Returns:
            Filtered point cloud with same format as input
        """
        if points.shape[0] == 0:
            return points
        
        if points.shape[1] < 3:
            raise ValueError(f"Points must have at least 3 columns (X, Y, Z), got {points.shape[1]}")
        
        # Use provided depths or instance defaults
        min_d = min_depth if min_depth is not None else self.min_depth
        max_d = max_depth if max_depth is not None else self.max_depth
        
        # Extract Z coordinates (depth)
        depth = points[:, 2]
        
        # Create filter mask
        valid_mask = (depth >= min_d) & (depth <= max_d)
        
        # Apply filter
        filtered_points = points[valid_mask]
        
        return filtered_points
    
    def get_depth_statistics(self, points: np.ndarray) -> dict:
        """
        Compute depth statistics for a point cloud.
        
        These statistics are useful for:
        - Validating reconstruction quality
        - Tuning depth range parameters
        - Debugging disparity issues
        
        Args:
            points: Point cloud as Nx3 or Nx6 array
            
        Returns:
            Dictionary containing:
            - 'min_depth': Minimum depth value
            - 'max_depth': Maximum depth value
            - 'mean_depth': Average depth
            - 'median_depth': Median depth
            - 'std_depth': Standard deviation of depth
            - 'num_points': Total number of points
        """
        if points.shape[0] == 0:
            return {
                'min_depth': 0.0,
                'max_depth': 0.0,
                'mean_depth': 0.0,
                'median_depth': 0.0,
                'std_depth': 0.0,
                'num_points': 0
            }
        
        depth = points[:, 2]
        
        return {
            'min_depth': float(np.min(depth)),
            'max_depth': float(np.max(depth)),
            'mean_depth': float(np.mean(depth)),
            'median_depth': float(np.median(depth)),
            'std_depth': float(np.std(depth)),
            'num_points': int(points.shape[0])
        }
    
    def update_depth_range(self, min_depth: float, max_depth: float) -> None:
        """
        Update the depth range for filtering.
        
        Args:
            min_depth: New minimum depth in meters
            max_depth: New maximum depth in meters
        """
        if min_depth >= max_depth:
            raise ValueError(f"min_depth ({min_depth}) must be less than max_depth ({max_depth})")
        
        if min_depth <= 0:
            raise ValueError(f"min_depth must be positive, got {min_depth}")
        
        self.min_depth = min_depth
        self.max_depth = max_depth


class OutlierRemover:
    """
    Statistical outlier removal for 3D point clouds.

    This class implements k-nearest neighbor (k-NN) based statistical outlier removal
    to eliminate noise points while preserving the spatial structure of the point cloud.
    The algorithm works by:

    1. For each point, finding its k nearest neighbors
    2. Computing the mean distance to these neighbors
    3. Computing global statistics (mean and std) of all mean distances
    4. Removing points whose mean distance exceeds: global_mean + std_ratio * global_std

    This approach effectively removes isolated noise points that are far from the main
    surface structure, while preserving points that are part of dense regions.

    Key features:
    - Preserves spatial coherence of the main surface
    - Removes isolated noise points
    - Configurable sensitivity via k_neighbors and std_ratio
    - Efficient implementation using vectorized operations
    """

    def __init__(self, k_neighbors: int = 20, std_ratio: float = 2.0):
        """
        Initialize outlier remover with filtering parameters.

        Args:
            k_neighbors: Number of nearest neighbors to consider (default: 20)
                        Higher values make filtering more conservative
            std_ratio: Standard deviation multiplier for outlier threshold (default: 2.0)
                      Higher values make filtering more permissive
        """
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be at least 1, got {k_neighbors}")

        if std_ratio <= 0:
            raise ValueError(f"std_ratio must be positive, got {std_ratio}")

        self.k_neighbors = k_neighbors
        self.std_ratio = std_ratio

    def remove_statistical_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        Remove statistical outliers from point cloud using k-NN filtering.

        This method implements the core outlier removal algorithm:
        1. Compute k-nearest neighbors for each point
        2. Calculate mean distance to neighbors for each point
        3. Compute global mean and standard deviation of these distances
        4. Remove points that exceed the threshold: mean + std_ratio * std

        Args:
            points: Point cloud as Nx3 or Nx6 array (X, Y, Z, [R, G, B])

        Returns:
            Filtered point cloud with outliers removed, same format as input

        Note:
            - If the point cloud has fewer points than k_neighbors, all points are kept
            - Color information (if present) is preserved for inlier points
            - The method preserves spatial coherence by keeping dense regions
        """
        if points.shape[0] == 0:
            return points

        if points.shape[1] < 3:
            raise ValueError(f"Points must have at least 3 columns (X, Y, Z), got {points.shape[1]}")

        # If we have fewer points than k_neighbors, keep all points
        if points.shape[0] <= self.k_neighbors:
            return points

        # Extract XYZ coordinates (first 3 columns)
        xyz = points[:, :3]

        # Compute k-nearest neighbors for each point
        # We use a simple distance-based approach for efficiency
        mean_distances = self._compute_mean_neighbor_distances(xyz)

        # Compute global statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)

        # Compute outlier threshold
        threshold = global_mean + self.std_ratio * global_std

        # Create inlier mask
        inlier_mask = mean_distances <= threshold

        # Filter points
        filtered_points = points[inlier_mask]

        return filtered_points

    def _compute_mean_neighbor_distances(self, xyz: np.ndarray) -> np.ndarray:
        """
        Compute mean distance to k nearest neighbors for each point.

        This is a helper method that efficiently computes the k-NN distances
        using vectorized operations where possible.

        Args:
            xyz: Nx3 array of 3D coordinates

        Returns:
            N-length array of mean distances to k nearest neighbors
        """
        n_points = xyz.shape[0]
        mean_distances = np.zeros(n_points, dtype=np.float32)

        # For each point, compute distances to all other points
        # and find k nearest neighbors
        for i in range(n_points):
            # Compute distances to all other points
            point = xyz[i]
            distances = np.linalg.norm(xyz - point, axis=1)

            # Sort distances and get k+1 nearest (including self at distance 0)
            sorted_distances = np.sort(distances)

            # Take k nearest neighbors (excluding self)
            k_nearest = sorted_distances[1:self.k_neighbors + 1]

            # Compute mean distance
            mean_distances[i] = np.mean(k_nearest)

        return mean_distances

    def update_parameters(self, k_neighbors: int, std_ratio: float) -> None:
        """
        Update filtering parameters.

        Args:
            k_neighbors: New number of nearest neighbors
            std_ratio: New standard deviation multiplier
        """
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be at least 1, got {k_neighbors}")

        if std_ratio <= 0:
            raise ValueError(f"std_ratio must be positive, got {std_ratio}")

        self.k_neighbors = k_neighbors
        self.std_ratio = std_ratio

    def get_outlier_statistics(self, points: np.ndarray) -> dict:
        """
        Compute outlier statistics without removing points.

        This method is useful for:
        - Analyzing point cloud quality
        - Tuning filtering parameters
        - Debugging outlier detection

        Args:
            points: Point cloud as Nx3 or Nx6 array

        Returns:
            Dictionary containing:
            - 'num_points': Total number of points
            - 'num_inliers': Number of inlier points
            - 'num_outliers': Number of outlier points
            - 'outlier_ratio': Fraction of points classified as outliers
            - 'mean_distance': Global mean of neighbor distances
            - 'std_distance': Global std of neighbor distances
            - 'threshold': Outlier threshold used
        """
        if points.shape[0] == 0:
            return {
                'num_points': 0,
                'num_inliers': 0,
                'num_outliers': 0,
                'outlier_ratio': 0.0,
                'mean_distance': 0.0,
                'std_distance': 0.0,
                'threshold': 0.0
            }

        if points.shape[0] <= self.k_neighbors:
            return {
                'num_points': points.shape[0],
                'num_inliers': points.shape[0],
                'num_outliers': 0,
                'outlier_ratio': 0.0,
                'mean_distance': 0.0,
                'std_distance': 0.0,
                'threshold': 0.0
            }

        # Extract XYZ coordinates
        xyz = points[:, :3]

        # Compute mean neighbor distances
        mean_distances = self._compute_mean_neighbor_distances(xyz)

        # Compute global statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + self.std_ratio * global_std

        # Count inliers and outliers
        num_inliers = np.sum(mean_distances <= threshold)
        num_outliers = np.sum(mean_distances > threshold)

        return {
            'num_points': int(points.shape[0]),
            'num_inliers': int(num_inliers),
            'num_outliers': int(num_outliers),
            'outlier_ratio': float(num_outliers / points.shape[0]),
            'mean_distance': float(global_mean),
            'std_distance': float(global_std),
            'threshold': float(threshold)
        }

