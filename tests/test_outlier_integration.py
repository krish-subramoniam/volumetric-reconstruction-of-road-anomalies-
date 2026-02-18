"""Integration test for OutlierRemover with PointCloudGenerator."""

import pytest
import numpy as np
from stereo_vision.reconstruction import PointCloudGenerator, OutlierRemover


def test_outlier_removal_integration():
    """Test that OutlierRemover works with PointCloudGenerator output."""
    # Create a Q matrix
    baseline = 0.1
    focal_length = 500.0
    cx = 320.0
    cy = 240.0
    
    Q = np.array([
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, focal_length],
        [0.0, 0.0, 1.0/baseline, 0.0]
    ], dtype=np.float32)
    
    # Create a disparity map with some noise
    disparity = np.zeros((100, 100), dtype=np.float32)
    
    # Main surface
    disparity[20:80, 20:80] = 50.0
    
    # Add some noise pixels (will create outliers in 3D)
    rng = np.random.RandomState(42)
    noise_mask = rng.random((100, 100)) < 0.02  # 2% noise
    disparity[noise_mask] = rng.uniform(10, 100, size=np.sum(noise_mask))
    
    # Generate point cloud
    generator = PointCloudGenerator(Q, min_depth=0.5, max_depth=100.0)
    points = generator.reproject_to_3d(disparity, apply_depth_filter=False)
    
    print(f"Original point cloud: {points.shape[0]} points")
    
    # Remove outliers
    remover = OutlierRemover(k_neighbors=10, std_ratio=2.0)
    
    # Get statistics before filtering
    stats_before = remover.get_outlier_statistics(points)
    print(f"Before filtering: {stats_before['num_outliers']} outliers detected")
    
    # Apply filtering
    filtered_points = remover.remove_statistical_outliers(points)
    
    print(f"Filtered point cloud: {filtered_points.shape[0]} points")
    print(f"Removed: {points.shape[0] - filtered_points.shape[0]} points")
    
    # Verify filtering worked
    assert filtered_points.shape[0] < points.shape[0], "Should remove some outliers"
    assert filtered_points.shape[0] > 0, "Should keep some inliers"
    assert filtered_points.shape[1] == 3, "Should preserve XYZ format"
    
    # Verify all coordinates are finite
    assert np.all(np.isfinite(filtered_points))
    
    # Verify depth statistics improved (lower std deviation)
    stats_original = generator.get_depth_statistics(points)
    stats_filtered = generator.get_depth_statistics(filtered_points)
    
    print(f"Original depth std: {stats_original['std_depth']:.4f}")
    print(f"Filtered depth std: {stats_filtered['std_depth']:.4f}")
    
    # Filtered should have lower or similar std (more consistent)
    # This verifies spatial coherence is preserved
    assert stats_filtered['std_depth'] <= stats_original['std_depth'] * 1.1


def test_outlier_removal_with_colors():
    """Test that OutlierRemover preserves colors in the pipeline."""
    # Create a Q matrix
    baseline = 0.1
    focal_length = 500.0
    cx = 320.0
    cy = 240.0
    
    Q = np.array([
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, focal_length],
        [0.0, 0.0, 1.0/baseline, 0.0]
    ], dtype=np.float32)
    
    # Create a disparity map
    disparity = np.zeros((100, 100), dtype=np.float32)
    disparity[20:80, 20:80] = 50.0
    
    # Create a color image
    colors = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Generate colored point cloud
    generator = PointCloudGenerator(Q, min_depth=0.5, max_depth=100.0)
    points = generator.reproject_to_3d(disparity, colors=colors, apply_depth_filter=False)
    
    assert points.shape[1] == 6, "Should have XYZ + RGB"
    
    # Remove outliers
    remover = OutlierRemover(k_neighbors=10, std_ratio=2.0)
    filtered_points = remover.remove_statistical_outliers(points)
    
    # Verify colors are preserved
    assert filtered_points.shape[1] == 6, "Should preserve XYZ + RGB format"
    assert np.all(filtered_points[:, 3:6] >= 0), "Colors should be non-negative"
    assert np.all(filtered_points[:, 3:6] <= 255), "Colors should be in valid range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
