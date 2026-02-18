"""Basic unit tests for 3D reconstruction module."""

import pytest
import numpy as np
import cv2
from stereo_vision.reconstruction import PointCloudGenerator


class TestPointCloudGenerator:
    """Test suite for PointCloudGenerator class."""
    
    @pytest.fixture
    def sample_Q_matrix(self):
        """Create a sample Q matrix for testing."""
        # Typical Q matrix for a stereo system
        # baseline = 0.1m, focal_length = 500 pixels
        baseline = 0.1
        focal_length = 500.0
        cx = 320.0  # Principal point x
        cy = 240.0  # Principal point y
        
        Q = np.array([
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, focal_length],
            [0.0, 0.0, 1.0/baseline, 0.0]
        ], dtype=np.float32)
        
        return Q
    
    @pytest.fixture
    def sample_disparity(self):
        """Create a sample disparity map."""
        # Create a simple disparity map with known values
        disparity = np.zeros((480, 640), dtype=np.float32)
        
        # Add a plane at constant disparity (simulating a flat surface)
        disparity[100:400, 100:500] = 50.0  # 50 pixels disparity
        
        return disparity
    
    def test_initialization_valid_Q_matrix(self, sample_Q_matrix):
        """Test initialization with valid Q matrix."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        assert generator.Q_matrix.shape == (4, 4)
        assert generator.min_depth == 1.0
        assert generator.max_depth == 50.0
        assert np.array_equal(generator.Q_matrix, sample_Q_matrix)
    
    def test_initialization_invalid_Q_matrix(self):
        """Test initialization with invalid Q matrix raises error."""
        invalid_Q = np.eye(3)  # 3x3 instead of 4x4
        
        with pytest.raises(ValueError, match="Q_matrix must be 4x4"):
            PointCloudGenerator(invalid_Q)
    
    def test_initialization_custom_depth_range(self, sample_Q_matrix):
        """Test initialization with custom depth range."""
        generator = PointCloudGenerator(sample_Q_matrix, min_depth=2.0, max_depth=30.0)
        
        assert generator.min_depth == 2.0
        assert generator.max_depth == 30.0
    
    def test_reproject_to_3d_basic(self, sample_Q_matrix, sample_disparity):
        """Test basic 3D reprojection."""
        generator = PointCloudGenerator(sample_Q_matrix, min_depth=0.5, max_depth=100.0)
        
        points = generator.reproject_to_3d(sample_disparity, apply_depth_filter=False)
        
        # Should have Nx3 shape (X, Y, Z)
        assert points.shape[1] == 3
        
        # Should have points (non-zero disparity region)
        assert points.shape[0] > 0
        
        # All coordinates should be finite
        assert np.all(np.isfinite(points))
    
    def test_reproject_to_3d_with_colors(self, sample_Q_matrix, sample_disparity):
        """Test 3D reprojection with color information."""
        generator = PointCloudGenerator(sample_Q_matrix, min_depth=0.5, max_depth=100.0)
        
        # Create a color image
        colors = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        points = generator.reproject_to_3d(sample_disparity, colors=colors, apply_depth_filter=False)
        
        # Should have Nx6 shape (X, Y, Z, R, G, B)
        assert points.shape[1] == 6
        
        # Should have points
        assert points.shape[0] > 0
        
        # XYZ coordinates should be finite
        assert np.all(np.isfinite(points[:, :3]))
        
        # Colors should be in valid range
        assert np.all(points[:, 3:6] >= 0)
        assert np.all(points[:, 3:6] <= 255)
    
    def test_reproject_to_3d_with_grayscale(self, sample_Q_matrix, sample_disparity):
        """Test 3D reprojection with grayscale image."""
        generator = PointCloudGenerator(sample_Q_matrix, min_depth=0.5, max_depth=100.0)
        
        # Create a grayscale image
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        points = generator.reproject_to_3d(sample_disparity, colors=gray, apply_depth_filter=False)
        
        # Should have Nx6 shape (grayscale replicated to RGB)
        assert points.shape[1] == 6
        
        # Should have points
        assert points.shape[0] > 0
    
    def test_reproject_to_3d_fixed_point_disparity(self, sample_Q_matrix):
        """Test reprojection with fixed-point disparity (int16)."""
        generator = PointCloudGenerator(sample_Q_matrix, min_depth=0.5, max_depth=100.0)
        
        # Create fixed-point disparity (SGBM output format)
        disparity_fixed = np.zeros((480, 640), dtype=np.int16)
        disparity_fixed[100:400, 100:500] = 50 * 16  # 50 pixels * 16 (fixed-point)
        
        points = generator.reproject_to_3d(disparity_fixed, apply_depth_filter=False)
        
        # Should successfully convert and reproject
        assert points.shape[1] == 3
        assert points.shape[0] > 0
        assert np.all(np.isfinite(points))
    
    def test_reproject_to_3d_with_depth_filter(self, sample_Q_matrix, sample_disparity):
        """Test that depth filtering removes out-of-range points."""
        # Use a restrictive depth range
        generator = PointCloudGenerator(sample_Q_matrix, min_depth=5.0, max_depth=10.0)
        
        points_filtered = generator.reproject_to_3d(sample_disparity, apply_depth_filter=True)
        points_unfiltered = generator.reproject_to_3d(sample_disparity, apply_depth_filter=False)
        
        # Filtered should have fewer or equal points
        assert points_filtered.shape[0] <= points_unfiltered.shape[0]
        
        # All filtered points should be within depth range
        if points_filtered.shape[0] > 0:
            depths = points_filtered[:, 2]
            assert np.all(depths >= 5.0)
            assert np.all(depths <= 10.0)
    
    def test_filter_depth_range(self, sample_Q_matrix):
        """Test depth range filtering on existing point cloud."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        # Create a point cloud with known depths
        points = np.array([
            [0, 0, 0.5],   # Too close
            [0, 0, 5.0],   # Valid
            [0, 0, 25.0],  # Valid
            [0, 0, 60.0],  # Too far
        ], dtype=np.float32)
        
        filtered = generator.filter_depth_range(points, min_depth=1.0, max_depth=50.0)
        
        # Should keep only the two valid points
        assert filtered.shape[0] == 2
        assert np.all(filtered[:, 2] >= 1.0)
        assert np.all(filtered[:, 2] <= 50.0)
    
    def test_filter_depth_range_with_colors(self, sample_Q_matrix):
        """Test depth filtering preserves color information."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        # Create a point cloud with colors
        points = np.array([
            [0, 0, 0.5, 255, 0, 0],   # Too close, red
            [0, 0, 5.0, 0, 255, 0],   # Valid, green
            [0, 0, 25.0, 0, 0, 255],  # Valid, blue
            [0, 0, 60.0, 255, 255, 0], # Too far, yellow
        ], dtype=np.float32)
        
        filtered = generator.filter_depth_range(points, min_depth=1.0, max_depth=50.0)
        
        # Should keep only the two valid points with their colors
        assert filtered.shape == (2, 6)
        assert np.array_equal(filtered[0, 3:6], [0, 255, 0])  # Green
        assert np.array_equal(filtered[1, 3:6], [0, 0, 255])  # Blue
    
    def test_filter_depth_range_empty_input(self, sample_Q_matrix):
        """Test depth filtering with empty point cloud."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        empty_points = np.array([]).reshape(0, 3)
        filtered = generator.filter_depth_range(empty_points)
        
        assert filtered.shape[0] == 0
    
    def test_filter_depth_range_invalid_input(self, sample_Q_matrix):
        """Test depth filtering with invalid input raises error."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        # Points with only 2 columns (missing Z)
        invalid_points = np.array([[0, 0], [1, 1]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="must have at least 3 columns"):
            generator.filter_depth_range(invalid_points)
    
    def test_get_depth_statistics(self, sample_Q_matrix):
        """Test depth statistics calculation."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        # Create a point cloud with known depths
        points = np.array([
            [0, 0, 5.0],
            [0, 0, 10.0],
            [0, 0, 15.0],
            [0, 0, 20.0],
        ], dtype=np.float32)
        
        stats = generator.get_depth_statistics(points)
        
        assert stats['min_depth'] == 5.0
        assert stats['max_depth'] == 20.0
        assert stats['mean_depth'] == 12.5
        assert stats['median_depth'] == 12.5
        assert stats['num_points'] == 4
        assert stats['std_depth'] > 0
    
    def test_get_depth_statistics_empty(self, sample_Q_matrix):
        """Test depth statistics with empty point cloud."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        empty_points = np.array([]).reshape(0, 3)
        stats = generator.get_depth_statistics(empty_points)
        
        assert stats['min_depth'] == 0.0
        assert stats['max_depth'] == 0.0
        assert stats['mean_depth'] == 0.0
        assert stats['median_depth'] == 0.0
        assert stats['std_depth'] == 0.0
        assert stats['num_points'] == 0
    
    def test_update_depth_range(self, sample_Q_matrix):
        """Test updating depth range."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        generator.update_depth_range(2.0, 30.0)
        
        assert generator.min_depth == 2.0
        assert generator.max_depth == 30.0
    
    def test_update_depth_range_invalid(self, sample_Q_matrix):
        """Test updating depth range with invalid values."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        # min >= max
        with pytest.raises(ValueError, match="must be less than"):
            generator.update_depth_range(30.0, 20.0)
        
        # min <= 0
        with pytest.raises(ValueError, match="must be positive"):
            generator.update_depth_range(-1.0, 20.0)
    
    def test_reproject_handles_invalid_disparities(self, sample_Q_matrix):
        """Test that reprojection handles invalid disparities correctly."""
        generator = PointCloudGenerator(sample_Q_matrix, min_depth=0.5, max_depth=100.0)
        
        # Create disparity with invalid values
        disparity = np.zeros((100, 100), dtype=np.float32)
        disparity[20:40, 20:40] = 50.0  # Valid region
        disparity[50:60, 50:60] = 0.0   # Invalid (zero disparity)
        disparity[70:80, 70:80] = -1.0  # Invalid (negative disparity)
        
        points = generator.reproject_to_3d(disparity, apply_depth_filter=False)
        
        # Should only have points from valid region
        assert points.shape[0] > 0
        
        # All points should be finite
        assert np.all(np.isfinite(points))
    
    def test_color_image_size_mismatch(self, sample_Q_matrix, sample_disparity):
        """Test that mismatched color image size raises error."""
        generator = PointCloudGenerator(sample_Q_matrix)
        
        # Create color image with wrong size
        wrong_size_colors = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="must match"):
            generator.reproject_to_3d(sample_disparity, colors=wrong_size_colors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestOutlierRemover:
    """Test suite for OutlierRemover class."""
    
    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        from stereo_vision.reconstruction import OutlierRemover
        
        remover = OutlierRemover()
        
        assert remover.k_neighbors == 20
        assert remover.std_ratio == 2.0
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        from stereo_vision.reconstruction import OutlierRemover
        
        remover = OutlierRemover(k_neighbors=10, std_ratio=1.5)
        
        assert remover.k_neighbors == 10
        assert remover.std_ratio == 1.5
    
    def test_initialization_invalid_k_neighbors(self):
        """Test initialization with invalid k_neighbors raises error."""
        from stereo_vision.reconstruction import OutlierRemover
        
        with pytest.raises(ValueError, match="k_neighbors must be at least 1"):
            OutlierRemover(k_neighbors=0)
        
        with pytest.raises(ValueError, match="k_neighbors must be at least 1"):
            OutlierRemover(k_neighbors=-5)
    
    def test_initialization_invalid_std_ratio(self):
        """Test initialization with invalid std_ratio raises error."""
        from stereo_vision.reconstruction import OutlierRemover
        
        with pytest.raises(ValueError, match="std_ratio must be positive"):
            OutlierRemover(std_ratio=0.0)
        
        with pytest.raises(ValueError, match="std_ratio must be positive"):
            OutlierRemover(std_ratio=-1.0)
    
    def test_remove_outliers_basic(self):
        """Test basic outlier removal with a simple point cloud."""
        from stereo_vision.reconstruction import OutlierRemover
        
        # Create a point cloud with a dense cluster and an outlier
        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0.1, 0.1, 0],
            [0, 0, 0.1],
            [0.1, 0, 0.1],
            [0, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            # Outlier far from the cluster
            [10, 10, 10],
        ], dtype=np.float32)
        
        remover = OutlierRemover(k_neighbors=5, std_ratio=2.0)
        filtered = remover.remove_statistical_outliers(points)
        
        # Should remove the outlier
        assert filtered.shape[0] < points.shape[0]
        assert filtered.shape[0] >= 8  # Should keep the cluster
        
        # The outlier should be removed
        # Check that no point is far from origin
        distances_from_origin = np.linalg.norm(filtered, axis=1)
        assert np.all(distances_from_origin < 5.0)
    
    def test_remove_outliers_preserves_colors(self):
        """Test that outlier removal preserves color information."""
        from stereo_vision.reconstruction import OutlierRemover
        
        # Create a point cloud with colors
        points = np.array([
            [0, 0, 0, 255, 0, 0],
            [0.1, 0, 0, 0, 255, 0],
            [0, 0.1, 0, 0, 0, 255],
            [0.1, 0.1, 0, 255, 255, 0],
            [0, 0, 0.1, 255, 0, 255],
            [0.1, 0, 0.1, 0, 255, 255],
            [0, 0.1, 0.1, 128, 128, 128],
            [0.1, 0.1, 0.1, 64, 64, 64],
            # Outlier
            [10, 10, 10, 255, 255, 255],
        ], dtype=np.float32)
        
        remover = OutlierRemover(k_neighbors=5, std_ratio=2.0)
        filtered = remover.remove_statistical_outliers(points)
        
        # Should have 6 columns (XYZ + RGB)
        assert filtered.shape[1] == 6
        
        # Colors should be preserved
        assert np.all(filtered[:, 3:6] >= 0)
        assert np.all(filtered[:, 3:6] <= 255)
    
    def test_remove_outliers_empty_input(self):
        """Test outlier removal with empty point cloud."""
        from stereo_vision.reconstruction import OutlierRemover
        
        empty_points = np.array([]).reshape(0, 3)
        
        remover = OutlierRemover()
        filtered = remover.remove_statistical_outliers(empty_points)
        
        assert filtered.shape[0] == 0
    
    def test_remove_outliers_few_points(self):
        """Test outlier removal when point cloud has fewer points than k_neighbors."""
        from stereo_vision.reconstruction import OutlierRemover
        
        # Create a small point cloud
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        remover = OutlierRemover(k_neighbors=10)
        filtered = remover.remove_statistical_outliers(points)
        
        # Should keep all points (not enough for k-NN)
        assert filtered.shape[0] == points.shape[0]
        assert np.array_equal(filtered, points)
    
    def test_remove_outliers_invalid_input(self):
        """Test outlier removal with invalid input raises error."""
        from stereo_vision.reconstruction import OutlierRemover
        
        # Points with only 2 columns
        invalid_points = np.array([[0, 0], [1, 1]], dtype=np.float32)
        
        remover = OutlierRemover()
        
        with pytest.raises(ValueError, match="must have at least 3 columns"):
            remover.remove_statistical_outliers(invalid_points)
    
    def test_remove_outliers_no_outliers(self):
        """Test outlier removal when all points are inliers."""
        from stereo_vision.reconstruction import OutlierRemover
        
        # Create a uniform grid (no outliers)
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
        
        remover = OutlierRemover(k_neighbors=10, std_ratio=2.0)
        filtered = remover.remove_statistical_outliers(points)
        
        # Should keep most or all points
        assert filtered.shape[0] >= points.shape[0] * 0.9
    
    def test_update_parameters(self):
        """Test updating filtering parameters."""
        from stereo_vision.reconstruction import OutlierRemover
        
        remover = OutlierRemover(k_neighbors=10, std_ratio=1.5)
        
        remover.update_parameters(k_neighbors=15, std_ratio=2.5)
        
        assert remover.k_neighbors == 15
        assert remover.std_ratio == 2.5
    
    def test_update_parameters_invalid(self):
        """Test updating parameters with invalid values."""
        from stereo_vision.reconstruction import OutlierRemover
        
        remover = OutlierRemover()
        
        with pytest.raises(ValueError, match="k_neighbors must be at least 1"):
            remover.update_parameters(k_neighbors=0, std_ratio=2.0)
        
        with pytest.raises(ValueError, match="std_ratio must be positive"):
            remover.update_parameters(k_neighbors=10, std_ratio=-1.0)
    
    def test_get_outlier_statistics(self):
        """Test outlier statistics calculation."""
        from stereo_vision.reconstruction import OutlierRemover
        
        # Create a point cloud with known structure
        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0.1, 0.1, 0],
            [0, 0, 0.1],
            [0.1, 0, 0.1],
            [0, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            # Outlier
            [10, 10, 10],
        ], dtype=np.float32)
        
        remover = OutlierRemover(k_neighbors=5, std_ratio=2.0)
        stats = remover.get_outlier_statistics(points)
        
        assert stats['num_points'] == 9
        assert stats['num_inliers'] > 0
        assert stats['num_outliers'] >= 0
        assert stats['num_inliers'] + stats['num_outliers'] == 9
        assert 0.0 <= stats['outlier_ratio'] <= 1.0
        assert stats['mean_distance'] > 0
        assert stats['std_distance'] >= 0
        assert stats['threshold'] > 0
    
    def test_get_outlier_statistics_empty(self):
        """Test outlier statistics with empty point cloud."""
        from stereo_vision.reconstruction import OutlierRemover
        
        empty_points = np.array([]).reshape(0, 3)
        
        remover = OutlierRemover()
        stats = remover.get_outlier_statistics(empty_points)
        
        assert stats['num_points'] == 0
        assert stats['num_inliers'] == 0
        assert stats['num_outliers'] == 0
        assert stats['outlier_ratio'] == 0.0
        assert stats['mean_distance'] == 0.0
        assert stats['std_distance'] == 0.0
        assert stats['threshold'] == 0.0
    
    def test_get_outlier_statistics_few_points(self):
        """Test outlier statistics when point cloud has fewer points than k_neighbors."""
        from stereo_vision.reconstruction import OutlierRemover
        
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        remover = OutlierRemover(k_neighbors=10)
        stats = remover.get_outlier_statistics(points)
        
        assert stats['num_points'] == 3
        assert stats['num_inliers'] == 3
        assert stats['num_outliers'] == 0
        assert stats['outlier_ratio'] == 0.0
    
    def test_std_ratio_sensitivity(self):
        """Test that std_ratio affects filtering sensitivity."""
        from stereo_vision.reconstruction import OutlierRemover
        
        # Create a point cloud with a moderate outlier
        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0.1, 0.1, 0],
            [0, 0, 0.1],
            [0.1, 0, 0.1],
            [0, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            # Moderate outlier
            [2, 2, 2],
        ], dtype=np.float32)
        
        # Strict filtering (low std_ratio)
        remover_strict = OutlierRemover(k_neighbors=5, std_ratio=1.0)
        filtered_strict = remover_strict.remove_statistical_outliers(points)
        
        # Permissive filtering (high std_ratio)
        remover_permissive = OutlierRemover(k_neighbors=5, std_ratio=3.0)
        filtered_permissive = remover_permissive.remove_statistical_outliers(points)
        
        # Strict filtering should remove more points
        assert filtered_strict.shape[0] <= filtered_permissive.shape[0]
