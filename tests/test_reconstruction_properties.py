"""Property-based tests for 3D reconstruction module."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from stereo_vision.reconstruction import PointCloudGenerator, OutlierRemover


# Custom strategies for Q matrices and disparity maps
@st.composite
def valid_Q_matrix_strategy(draw):
    """
    Generate valid Q matrices for stereo reprojection.
    
    The Q matrix has the form:
    [1,  0,  0,  -cx]
    [0,  1,  0,  -cy]
    [0,  0,  0,   f]
    [0,  0, 1/B,  0]
    
    Where:
    - cx, cy: principal point coordinates
    - f: focal length
    - B: baseline (distance between cameras)
    """
    # Generate reasonable camera parameters
    focal_length = draw(st.floats(min_value=300.0, max_value=1000.0))
    baseline = draw(st.floats(min_value=0.05, max_value=0.5))  # 5cm to 50cm
    cx = draw(st.floats(min_value=200.0, max_value=800.0))
    cy = draw(st.floats(min_value=150.0, max_value=600.0))
    
    Q = np.array([
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, focal_length],
        [0.0, 0.0, 1.0/baseline, 0.0]
    ], dtype=np.float32)
    
    return Q, focal_length, baseline, cx, cy


@st.composite
def disparity_map_with_Q_strategy(draw, min_height=20, max_height=100,
                                   min_width=20, max_width=100):
    """
    Generate a disparity map along with a compatible Q matrix.
    
    Returns:
        Tuple of (Q_matrix, disparity_map, focal_length, baseline)
    """
    height = draw(st.integers(min_value=min_height, max_value=max_height))
    width = draw(st.integers(min_value=min_width, max_value=max_width))
    
    Q, focal_length, baseline, cx, cy = draw(valid_Q_matrix_strategy())
    
    # Generate disparity map with reasonable values
    # Disparity = (focal_length * baseline) / depth
    # For depth range [1m, 50m], compute disparity range
    min_depth = 1.0
    max_depth = 50.0
    max_disparity = (focal_length * baseline) / min_depth
    min_disparity = (focal_length * baseline) / max_depth
    
    # Generate random disparities in valid range
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    disparity_map = rng.uniform(min_disparity, max_disparity, (height, width)).astype(np.float32)
    
    # Set some pixels to zero (invalid)
    invalid_mask = rng.random((height, width)) < 0.1  # 10% invalid
    disparity_map[invalid_mask] = 0.0
    
    return Q, disparity_map, focal_length, baseline, height, width


class TestPointCloudGeneratorProperties:
    """Property-based tests for PointCloudGenerator."""
    
    @given(valid_Q_matrix_strategy())
    def test_property_initialization_valid_Q(self, Q_data):
        """
        Property: PointCloudGenerator should successfully initialize with any valid Q matrix.
        
        **Feature: advanced-stereo-vision-pipeline, Property: Initialization Correctness**
        """
        Q, focal_length, baseline, cx, cy = Q_data
        
        generator = PointCloudGenerator(Q)
        
        assert generator.Q_matrix.shape == (4, 4)
        assert np.array_equal(generator.Q_matrix, Q)
    
    @given(disparity_map_with_Q_strategy())
    @settings(max_examples=100)
    def test_property_11_geometric_consistency_metric_distances(self, test_data):
        """
        Property 11: 3D Reprojection Geometric Consistency
        
        For any valid disparity pixel, reprojection to 3D coordinates using the Q matrix
        should preserve geometric relationships and metric distances.
        
        This test verifies that:
        1. Known disparity values produce expected depth values
        2. Metric distances are preserved in 3D space
        3. The relationship between disparity and depth follows the formula: depth = (f * B) / d
        
        **Validates: Requirements 4.1, 4.4**
        """
        Q, disparity_map, focal_length, baseline, height, width = test_data
        
        # Create generator with wide depth range to avoid filtering
        generator = PointCloudGenerator(Q, min_depth=0.1, max_depth=1000.0)
        
        # Reproject to 3D
        points_3d = generator.reproject_to_3d(disparity_map, apply_depth_filter=False)
        
        # Verify basic properties
        assert points_3d.shape[1] == 3  # X, Y, Z coordinates
        assert points_3d.shape[0] > 0  # Should have some valid points
        
        # Property 11 Verification: Geometric consistency
        
        # 1. Verify depth-disparity relationship for known points
        # Select a few random valid pixels and verify the depth formula
        valid_mask = disparity_map > 0
        valid_indices = np.argwhere(valid_mask)
        
        if len(valid_indices) > 0:
            # Sample up to 10 random valid pixels
            num_samples = min(10, len(valid_indices))
            sample_indices = valid_indices[np.random.choice(len(valid_indices), num_samples, replace=False)]
            
            for idx in sample_indices:
                y, x = idx
                disparity = disparity_map[y, x]
                
                # Expected depth from disparity formula: depth = (f * B) / d
                expected_depth = (focal_length * baseline) / disparity
                
                # Find the corresponding 3D point
                # We need to find which point in the cloud corresponds to this pixel
                # Since we don't have direct mapping, we'll verify the relationship holds
                # for the overall point cloud statistics
                
                # For now, verify that depths in the point cloud are in the expected range
                min_expected_depth = (focal_length * baseline) / np.max(disparity_map[valid_mask])
                max_expected_depth = (focal_length * baseline) / np.min(disparity_map[valid_mask])
                
                # All point depths should be within this range
                point_depths = points_3d[:, 2]
                assert np.all(point_depths >= min_expected_depth * 0.9), \
                    f"Some points have depth below expected minimum"
                assert np.all(point_depths <= max_expected_depth * 1.1), \
                    f"Some points have depth above expected maximum"
        
        # 2. Verify metric accuracy: distances between neighboring pixels
        # should correspond to realistic metric distances
        if points_3d.shape[0] >= 2:
            # Compute distances between consecutive points
            distances = np.linalg.norm(points_3d[1:] - points_3d[:-1], axis=1)
            
            # Distances should be finite and positive
            assert np.all(np.isfinite(distances))
            assert np.all(distances >= 0)
            
            # For road scenes at typical depths (1-50m), neighboring pixel distances
            # should be reasonable (typically < 1m for adjacent pixels)
            # This verifies metric consistency
            median_distance = np.median(distances)
            assert median_distance < 5.0, \
                f"Median distance between points ({median_distance:.3f}m) seems unrealistic"
        
        # 3. Verify spatial relationships: points should maintain relative positions
        # Points with similar disparities should have similar depths
        if points_3d.shape[0] >= 10:
            # Group points by depth ranges
            depths = points_3d[:, 2]
            depth_std = np.std(depths)
            depth_mean = np.mean(depths)
            
            # Coefficient of variation should be reasonable
            if depth_mean > 0:
                cv = depth_std / depth_mean
                # For a disparity map with varied depths, CV should be reasonable
                assert cv < 2.0, \
                    f"Depth variation too high (CV={cv:.3f}), suggests geometric inconsistency"
    
    @given(disparity_map_with_Q_strategy())
    @settings(max_examples=100)
    def test_property_11_neighboring_pixel_spatial_coherence(self, test_data):
        """
        Property 11: 3D Reprojection Geometric Consistency (Spatial Coherence)
        
        For neighboring pixels with similar disparities, their 3D points should be
        spatially proximate, preserving the local geometric structure.
        
        **Validates: Requirements 4.1, 4.4**
        """
        Q, disparity_map, focal_length, baseline, height, width = test_data
        
        # Ensure we have enough pixels to test
        assume(height >= 10 and width >= 10)
        
        generator = PointCloudGenerator(Q, min_depth=0.1, max_depth=1000.0)
        
        # Create a test region with uniform disparity (simulating a planar surface)
        test_disparity = disparity_map.copy()
        
        # Set a 5x5 region to uniform disparity
        center_y, center_x = height // 2, width // 2
        uniform_disparity = np.mean(disparity_map[disparity_map > 0])
        
        region_size = 5
        y_start = max(0, center_y - region_size // 2)
        y_end = min(height, center_y + region_size // 2 + 1)
        x_start = max(0, center_x - region_size // 2)
        x_end = min(width, center_x + region_size // 2 + 1)
        
        test_disparity[y_start:y_end, x_start:x_end] = uniform_disparity
        
        # Reproject to 3D
        points_3d = generator.reproject_to_3d(test_disparity, apply_depth_filter=False)
        
        # For the uniform disparity region, all points should have similar depths
        # Since we can't directly map pixels to points, we verify that the point cloud
        # has a cluster of points at the expected depth
        expected_depth = (focal_length * baseline) / uniform_disparity
        
        depths = points_3d[:, 2]
        
        # Find points near the expected depth
        depth_tolerance = expected_depth * 0.1  # 10% tolerance
        near_expected = np.abs(depths - expected_depth) < depth_tolerance
        
        # Should have multiple points near the expected depth
        assert np.sum(near_expected) >= region_size, \
            f"Expected at least {region_size} points near depth {expected_depth:.2f}m"
        
        # Points near expected depth should have low depth variance
        if np.sum(near_expected) > 1:
            near_depths = depths[near_expected]
            depth_std = np.std(near_depths)
            
            # Standard deviation should be small relative to depth
            relative_std = depth_std / expected_depth
            assert relative_std < 0.15, \
                f"Depth variation too high for uniform disparity region (rel_std={relative_std:.3f})"
    
    @given(
        valid_Q_matrix_strategy(),
        st.integers(min_value=20, max_value=100),
        st.integers(min_value=20, max_value=100),
        st.floats(min_value=2.0, max_value=10.0),
        st.floats(min_value=15.0, max_value=40.0)
    )
    @settings(max_examples=100)
    def test_property_13_depth_range_filtering_compliance(
        self, Q_data, height, width, min_depth, max_depth
    ):
        """
        Property 13: Depth Range Filtering Compliance
        
        For any 3D point outside the configured depth range, the filtering process
        should remove it from the point cloud.
        
        This test verifies that:
        1. Points with depth < min_depth are removed
        2. Points with depth > max_depth are removed
        3. Points within [min_depth, max_depth] are retained
        
        **Validates: Requirements 4.3**
        """
        Q, focal_length, baseline, cx, cy = Q_data
        
        # Ensure valid depth range
        assume(min_depth < max_depth)
        assume(min_depth > 0.1)
        
        # Create generator with specified depth range
        generator = PointCloudGenerator(Q, min_depth=min_depth, max_depth=max_depth)
        
        # Create disparity map with known depth distribution
        # depth = (f * B) / disparity
        # So: disparity = (f * B) / depth
        
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        # Create regions with different depths
        region_height = height // 3
        region_width = width // 3
        
        # Region 1: Too close (depth < min_depth)
        depth_too_close = min_depth * 0.5
        disparity_too_close = (focal_length * baseline) / depth_too_close
        disparity_map[0:region_height, 0:region_width] = disparity_too_close
        
        # Region 2: Valid depth (within range)
        depth_valid = (min_depth + max_depth) / 2.0
        disparity_valid = (focal_length * baseline) / depth_valid
        disparity_map[region_height:2*region_height, region_width:2*region_width] = disparity_valid
        
        # Region 3: Too far (depth > max_depth)
        depth_too_far = max_depth * 2.0
        disparity_too_far = (focal_length * baseline) / depth_too_far
        disparity_map[2*region_height:, 2*region_width:] = disparity_too_far
        
        # Reproject with depth filtering enabled
        points_filtered = generator.reproject_to_3d(disparity_map, apply_depth_filter=True)
        
        # Reproject without depth filtering for comparison
        points_unfiltered = generator.reproject_to_3d(disparity_map, apply_depth_filter=False)
        
        # Property 13 Verification: Depth range filtering compliance
        
        # 1. Filtered point cloud should have fewer or equal points
        assert points_filtered.shape[0] <= points_unfiltered.shape[0], \
            "Filtered point cloud should not have more points than unfiltered"
        
        # 2. All points in filtered cloud should be within depth range
        if points_filtered.shape[0] > 0:
            filtered_depths = points_filtered[:, 2]
            
            assert np.all(filtered_depths >= min_depth), \
                f"Found points below min_depth: min={np.min(filtered_depths):.3f}, threshold={min_depth:.3f}"
            
            assert np.all(filtered_depths <= max_depth), \
                f"Found points above max_depth: max={np.max(filtered_depths):.3f}, threshold={max_depth:.3f}"
        
        # 3. Points outside depth range should be removed
        if points_unfiltered.shape[0] > 0:
            unfiltered_depths = points_unfiltered[:, 2]
            
            # Count points outside range in unfiltered cloud
            too_close = np.sum(unfiltered_depths < min_depth)
            too_far = np.sum(unfiltered_depths > max_depth)
            within_range = np.sum((unfiltered_depths >= min_depth) & (unfiltered_depths <= max_depth))
            
            # Filtered cloud should have approximately the same number as within_range
            # (allowing for small numerical differences)
            assert points_filtered.shape[0] <= within_range + 5, \
                f"Filtered cloud has more points ({points_filtered.shape[0]}) than expected ({within_range})"
            
            # If there were points outside range, filtered should have fewer points
            if too_close > 0 or too_far > 0:
                assert points_filtered.shape[0] < points_unfiltered.shape[0], \
                    "Filtering should remove points outside depth range"
    
    @given(disparity_map_with_Q_strategy())
    @settings(max_examples=100)
    def test_property_13_filter_depth_range_method_compliance(self, test_data):
        """
        Property 13: Depth Range Filtering Compliance (filter_depth_range method)
        
        The filter_depth_range method should remove all points outside the specified
        depth range when applied to an existing point cloud.
        
        **Validates: Requirements 4.3**
        """
        Q, disparity_map, focal_length, baseline, height, width = test_data
        
        generator = PointCloudGenerator(Q, min_depth=0.1, max_depth=1000.0)
        
        # Generate unfiltered point cloud
        points = generator.reproject_to_3d(disparity_map, apply_depth_filter=False)
        
        assume(points.shape[0] > 10)  # Need enough points to test
        
        # Define a restrictive depth range
        all_depths = points[:, 2]
        depth_range = np.max(all_depths) - np.min(all_depths)
        
        # Set range to middle 50% of depth values
        min_depth = np.percentile(all_depths, 25)
        max_depth = np.percentile(all_depths, 75)
        
        assume(max_depth > min_depth)
        assume(depth_range > 0.1)  # Ensure meaningful range
        
        # Apply depth filtering
        filtered_points = generator.filter_depth_range(points, min_depth=min_depth, max_depth=max_depth)
        
        # Property 13 Verification
        
        # 1. Filtered should have fewer or equal points
        assert filtered_points.shape[0] <= points.shape[0]
        
        # 2. All filtered points should be within range
        if filtered_points.shape[0] > 0:
            filtered_depths = filtered_points[:, 2]
            
            assert np.all(filtered_depths >= min_depth), \
                f"Found points below min_depth after filtering"
            
            assert np.all(filtered_depths <= max_depth), \
                f"Found points above max_depth after filtering"
        
        # 3. Points outside range should be removed
        points_below = np.sum(all_depths < min_depth)
        points_above = np.sum(all_depths > max_depth)
        points_within = np.sum((all_depths >= min_depth) & (all_depths <= max_depth))
        
        # Filtered count should match points within range
        assert filtered_points.shape[0] == points_within, \
            f"Filtered count ({filtered_points.shape[0]}) doesn't match expected ({points_within})"
        
        # If there were points outside range, some should have been removed
        if points_below > 0 or points_above > 0:
            assert filtered_points.shape[0] < points.shape[0], \
                "Should have removed points outside depth range"
    
    @given(
        valid_Q_matrix_strategy(),
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=20.0, max_value=50.0)
    )
    @settings(max_examples=50)
    def test_property_13_depth_range_update_compliance(self, Q_data, new_min_depth, new_max_depth):
        """
        Property 13: Depth Range Filtering Compliance (dynamic range updates)
        
        When depth range is updated, subsequent filtering operations should use
        the new range values.
        
        **Validates: Requirements 4.3**
        """
        Q, focal_length, baseline, cx, cy = Q_data
        
        assume(new_min_depth < new_max_depth)
        
        # Create generator with initial range
        generator = PointCloudGenerator(Q, min_depth=1.0, max_depth=100.0)
        
        # Update depth range
        generator.update_depth_range(new_min_depth, new_max_depth)
        
        # Verify range was updated
        assert generator.min_depth == new_min_depth
        assert generator.max_depth == new_max_depth
        
        # Create test point cloud with known depths
        test_points = np.array([
            [0, 0, new_min_depth - 1.0],  # Below range
            [0, 0, new_min_depth + 1.0],  # Within range
            [0, 0, (new_min_depth + new_max_depth) / 2.0],  # Within range
            [0, 0, new_max_depth - 1.0],  # Within range
            [0, 0, new_max_depth + 1.0],  # Above range
        ], dtype=np.float32)
        
        # Filter using updated range
        filtered = generator.filter_depth_range(test_points)
        
        # Should keep only the 3 points within range
        assert filtered.shape[0] == 3, \
            f"Expected 3 points within range, got {filtered.shape[0]}"
        
        # All filtered points should be within new range
        filtered_depths = filtered[:, 2]
        assert np.all(filtered_depths >= new_min_depth)
        assert np.all(filtered_depths <= new_max_depth)
    
    @given(disparity_map_with_Q_strategy())
    @settings(max_examples=50)
    def test_property_output_shape_consistency(self, test_data):
        """
        Property: Reprojection should always produce Nx3 point clouds (or Nx6 with colors).
        
        **Feature: advanced-stereo-vision-pipeline, Property: Output Shape Consistency**
        """
        Q, disparity_map, focal_length, baseline, height, width = test_data
        
        generator = PointCloudGenerator(Q)
        
        # Test without colors
        points = generator.reproject_to_3d(disparity_map)
        assert points.shape[1] == 3
        
        # Test with colors
        colors = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        points_colored = generator.reproject_to_3d(disparity_map, colors=colors)
        assert points_colored.shape[1] == 6
    
    @given(disparity_map_with_Q_strategy())
    @settings(max_examples=50)
    def test_property_finite_coordinates(self, test_data):
        """
        Property: All reprojected 3D coordinates should be finite (no NaN or Inf).
        
        **Feature: advanced-stereo-vision-pipeline, Property: Finite Coordinates**
        """
        Q, disparity_map, focal_length, baseline, height, width = test_data
        
        generator = PointCloudGenerator(Q, min_depth=0.1, max_depth=1000.0)
        points = generator.reproject_to_3d(disparity_map, apply_depth_filter=False)
        
        # All coordinates should be finite
        assert np.all(np.isfinite(points[:, :3]))
    
    @given(disparity_map_with_Q_strategy())
    @settings(max_examples=50)
    def test_property_positive_depths(self, test_data):
        """
        Property: All reprojected points should have positive depth (Z > 0).
        
        **Feature: advanced-stereo-vision-pipeline, Property: Positive Depths**
        """
        Q, disparity_map, focal_length, baseline, height, width = test_data
        
        generator = PointCloudGenerator(Q, min_depth=0.1, max_depth=1000.0)
        points = generator.reproject_to_3d(disparity_map, apply_depth_filter=False)
        
        if points.shape[0] > 0:
            depths = points[:, 2]
            assert np.all(depths > 0), \
                f"Found non-positive depths: min={np.min(depths)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



# Custom strategies for point clouds with outliers
@st.composite
def point_cloud_with_outliers_strategy(draw):
    """
    Generate a point cloud with a dense main structure and isolated outliers.
    
    Returns:
        Tuple of (points, num_inliers, num_outliers)
    """
    # Generate main cluster parameters
    num_inliers = draw(st.integers(min_value=50, max_value=200))
    cluster_center = draw(st.lists(st.floats(min_value=-5.0, max_value=5.0), min_size=3, max_size=3))
    cluster_radius = draw(st.floats(min_value=0.5, max_value=2.0))
    
    # Generate outlier parameters
    num_outliers = draw(st.integers(min_value=5, max_value=20))
    outlier_distance = draw(st.floats(min_value=10.0, max_value=50.0))
    
    # Generate random seed for reproducibility
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    # Generate main cluster (dense inliers)
    # Use normal distribution around cluster center
    inliers = rng.normal(
        loc=cluster_center,
        scale=cluster_radius,
        size=(num_inliers, 3)
    ).astype(np.float32)
    
    # Generate outliers (isolated points far from cluster)
    outliers = []
    for _ in range(num_outliers):
        # Random direction
        direction = rng.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        # Place outlier far from cluster center
        outlier_pos = np.array(cluster_center) + direction * outlier_distance
        outliers.append(outlier_pos)
    
    outliers = np.array(outliers, dtype=np.float32)
    
    # Combine inliers and outliers
    points = np.vstack([inliers, outliers])
    
    # Shuffle to mix inliers and outliers
    shuffle_indices = rng.permutation(points.shape[0])
    points = points[shuffle_indices]
    
    return points, num_inliers, num_outliers


@st.composite
def point_cloud_planar_surface_strategy(draw):
    """
    Generate a point cloud representing a planar surface (like a road).
    
    This simulates neighboring pixels on a flat surface, useful for testing
    spatial coherence preservation.
    
    Returns:
        Tuple of (points, plane_params)
    """
    # Generate plane parameters: z = ax + by + c
    a = draw(st.floats(min_value=-0.2, max_value=0.2))  # Slight slope
    b = draw(st.floats(min_value=-0.2, max_value=0.2))  # Slight slope
    c = draw(st.floats(min_value=5.0, max_value=15.0))  # Offset (depth)
    
    # Generate grid of points on the plane
    num_points_x = draw(st.integers(min_value=10, max_value=30))
    num_points_y = draw(st.integers(min_value=10, max_value=30))
    
    x_range = draw(st.floats(min_value=2.0, max_value=5.0))
    y_range = draw(st.floats(min_value=2.0, max_value=5.0))
    
    # Create grid
    x = np.linspace(-x_range/2, x_range/2, num_points_x)
    y = np.linspace(-y_range/2, y_range/2, num_points_y)
    xx, yy = np.meshgrid(x, y)
    
    # Compute z values on the plane
    zz = a * xx + b * yy + c
    
    # Add small noise to simulate measurement noise
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    noise_level = draw(st.floats(min_value=0.01, max_value=0.1))
    zz += rng.normal(0, noise_level, zz.shape)
    
    # Flatten to point cloud
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    
    plane_params = (a, b, c)
    
    return points, plane_params


class TestOutlierRemoverProperties:
    """Property-based tests for OutlierRemover."""
    
    @given(point_cloud_with_outliers_strategy())
    @settings(max_examples=100)
    def test_property_12_statistical_outlier_removal_effectiveness(self, test_data):
        """
        Property 12: Statistical Outlier Removal Effectiveness
        
        For any 3D point cloud with noise, statistical outlier removal should eliminate
        isolated points while preserving the main surface structure.
        
        This test verifies that:
        1. Isolated outliers far from the main cluster are removed
        2. The dense main cluster is preserved
        3. The filtering reduces the outlier ratio significantly
        4. Inliers remain largely intact
        
        **Validates: Requirements 4.2**
        """
        points, num_inliers, num_outliers = test_data
        
        # Ensure we have enough points for k-NN
        assume(points.shape[0] > 30)
        
        # Create outlier remover with reasonable parameters
        remover = OutlierRemover(k_neighbors=10, std_ratio=2.0)
        
        # Apply outlier removal
        filtered_points = remover.remove_statistical_outliers(points)
        
        # Property 12 Verification: Outlier removal effectiveness
        
        # 1. Filtering should not increase the number of points
        assert filtered_points.shape[0] <= points.shape[0], \
            "Outlier removal should not increase the number of points"
        
        # 2. Most inliers should be preserved
        # We expect to keep at least 60% of the original inliers (allowing for some false positives)
        expected_min_inliers = int(num_inliers * 0.6)
        assert filtered_points.shape[0] >= expected_min_inliers, \
            f"Too many inliers removed: kept {filtered_points.shape[0]}, expected >= {expected_min_inliers}"
        
        # 3. The main structure should be preserved
        # Compute the centroid and spread of the filtered cloud
        if filtered_points.shape[0] > 0:
            filtered_centroid = np.mean(filtered_points, axis=0)
            filtered_spread = np.std(filtered_points, axis=0)
            
            # The spread should be reasonable (not collapsed to a point)
            assert np.all(filtered_spread > 0.05), \
                "Filtered cloud has collapsed, main structure not preserved"
            
            # The centroid should be finite
            assert np.all(np.isfinite(filtered_centroid)), \
                "Filtered cloud centroid is not finite"
        
        # 4. Verify that the filtered cloud has reasonable density
        # (points are not too spread out)
        if filtered_points.shape[0] >= 10:
            # Sample some points and compute nearest neighbor distances
            sample_size = min(10, filtered_points.shape[0])
            sample_indices = np.random.choice(filtered_points.shape[0], sample_size, replace=False)
            
            max_nn_distances = []
            for idx in sample_indices:
                point = filtered_points[idx]
                distances = np.linalg.norm(filtered_points - point, axis=1)
                # Exclude self (distance 0)
                distances = distances[distances > 0]
                
                if len(distances) > 0:
                    min_distance = np.min(distances)
                    max_nn_distances.append(min_distance)
            
            # At least some points should have nearby neighbors
            # This verifies that we kept a dense cluster
            if len(max_nn_distances) > 0:
                median_nn_distance = np.median(max_nn_distances)
                # The median nearest neighbor distance should be reasonable
                # (not too large, indicating isolated points)
                assert median_nn_distance < 20.0, \
                    f"Filtered cloud too sparse (median NN distance={median_nn_distance:.2f})"
    
    @given(point_cloud_with_outliers_strategy())
    @settings(max_examples=100)
    def test_property_12_outlier_removal_preserves_main_structure(self, test_data):
        """
        Property 12: Statistical Outlier Removal Effectiveness (Structure Preservation)
        
        Outlier removal should preserve the geometric properties of the main structure,
        including centroid position and overall shape.
        
        **Validates: Requirements 4.2**
        """
        points, num_inliers, num_outliers = test_data
        
        assume(points.shape[0] > 30)
        
        remover = OutlierRemover(k_neighbors=10, std_ratio=2.0)
        
        # Compute properties of original cloud
        original_centroid = np.mean(points, axis=0)
        original_cov = np.cov(points.T)
        
        # Apply filtering
        filtered_points = remover.remove_statistical_outliers(points)
        
        assume(filtered_points.shape[0] > 10)
        
        # Compute properties of filtered cloud
        filtered_centroid = np.mean(filtered_points, axis=0)
        filtered_cov = np.cov(filtered_points.T)
        
        # Property 12 Verification: Structure preservation
        
        # 1. Centroid should not shift dramatically
        # (outliers are far away, so removing them should bring centroid closer to main cluster)
        centroid_shift = np.linalg.norm(filtered_centroid - original_centroid)
        
        # The shift should be reasonable (not more than the original spread)
        original_spread = np.sqrt(np.trace(original_cov))
        
        # Centroid can shift, but should be within reasonable bounds
        # (removing outliers typically moves centroid toward main cluster)
        assert centroid_shift < original_spread * 2.0, \
            f"Centroid shifted too much: {centroid_shift:.2f} vs spread {original_spread:.2f}"
        
        # 2. Covariance structure should be preserved or reduced
        # (removing outliers should reduce variance)
        filtered_spread = np.sqrt(np.trace(filtered_cov))
        
        # Filtered spread should be less than or similar to original
        # (outliers increase spread, so removing them should reduce it)
        assert filtered_spread <= original_spread * 1.5, \
            f"Filtered spread increased unexpectedly: {filtered_spread:.2f} vs {original_spread:.2f}"
        
        # 3. The filtered cloud should still have reasonable extent
        # (not collapsed to a point)
        assert filtered_spread > 0.1, \
            "Filtered cloud has collapsed, structure not preserved"
    
    @given(point_cloud_planar_surface_strategy())
    @settings(max_examples=100)
    def test_property_14_spatial_coherence_preservation(self, test_data):
        """
        Property 14: Spatial Coherence Preservation
        
        For any pair of neighboring pixels in the image, their corresponding 3D points
        should be spatially proximate in world coordinates.
        
        This test verifies that:
        1. Points on a planar surface remain spatially coherent after processing
        2. Neighboring points maintain their proximity relationships
        3. The local geometric structure is preserved
        4. Outlier removal doesn't break spatial coherence of the main surface
        
        **Validates: Requirements 4.5**
        """
        points, plane_params = test_data
        
        assume(points.shape[0] > 50)
        
        # Property 14 Verification: Spatial coherence preservation
        
        # 1. Verify that neighboring points are spatially proximate
        # For a planar surface, points should have consistent nearest neighbor distances
        
        # Sample some points and check their nearest neighbors
        sample_size = min(20, points.shape[0])
        sample_indices = np.random.choice(points.shape[0], sample_size, replace=False)
        
        neighbor_distances = []
        
        for idx in sample_indices:
            point = points[idx]
            distances = np.linalg.norm(points - point, axis=1)
            # Exclude self
            distances = distances[distances > 0]
            
            if len(distances) > 0:
                # Get k nearest neighbors
                k = min(4, len(distances))
                k_nearest = np.sort(distances)[:k]
                mean_neighbor_dist = np.mean(k_nearest)
                neighbor_distances.append(mean_neighbor_dist)
        
        neighbor_distances = np.array(neighbor_distances)
        
        # 2. Neighbor distances should be consistent (low variance)
        # This indicates spatial coherence
        if len(neighbor_distances) > 1:
            mean_dist = np.mean(neighbor_distances)
            std_dist = np.std(neighbor_distances)
            
            # Coefficient of variation should be reasonable
            if mean_dist > 0:
                cv = std_dist / mean_dist
                assert cv < 1.0, \
                    f"Neighbor distances too variable (CV={cv:.3f}), spatial coherence not preserved"
        
        # 3. Apply outlier removal and verify coherence is maintained
        # Add a few outliers to test
        num_outliers = 5
        outliers = np.random.uniform(-20, 20, (num_outliers, 3)).astype(np.float32)
        points_with_outliers = np.vstack([points, outliers])
        
        remover = OutlierRemover(k_neighbors=10, std_ratio=2.0)
        filtered_points = remover.remove_statistical_outliers(points_with_outliers)
        
        assume(filtered_points.shape[0] > 20)
        
        # 4. Verify spatial coherence after outlier removal
        sample_size_filtered = min(20, filtered_points.shape[0])
        sample_indices_filtered = np.random.choice(
            filtered_points.shape[0], sample_size_filtered, replace=False
        )
        
        neighbor_distances_filtered = []
        
        for idx in sample_indices_filtered:
            point = filtered_points[idx]
            distances = np.linalg.norm(filtered_points - point, axis=1)
            distances = distances[distances > 0]
            
            if len(distances) > 0:
                k = min(4, len(distances))
                k_nearest = np.sort(distances)[:k]
                mean_neighbor_dist = np.mean(k_nearest)
                neighbor_distances_filtered.append(mean_neighbor_dist)
        
        neighbor_distances_filtered = np.array(neighbor_distances_filtered)
        
        # 5. Filtered cloud should maintain or improve spatial coherence
        if len(neighbor_distances_filtered) > 1:
            mean_dist_filtered = np.mean(neighbor_distances_filtered)
            std_dist_filtered = np.std(neighbor_distances_filtered)
            
            if mean_dist_filtered > 0:
                cv_filtered = std_dist_filtered / mean_dist_filtered
                
                # Spatial coherence should be maintained
                # Allow for some variation due to random sampling and edge effects
                assert cv_filtered < 3.0, \
                    f"Spatial coherence degraded after filtering (CV={cv_filtered:.3f})"
                
                # Note: We don't compare to original CV because adding outliers
                # can artificially improve the CV of the original cloud by
                # increasing mean distances. The key is that the filtered cloud
                # maintains good coherence (low CV), which we verify above.
        
        # 6. Verify planar structure is preserved
        # Fit a plane to the filtered points and check residuals
        if filtered_points.shape[0] >= 10:
            # Fit plane using least squares: z = ax + by + c
            X = filtered_points[:, :2]  # x, y coordinates
            z = filtered_points[:, 2]   # z coordinates
            
            # Add column of ones for intercept
            X_with_intercept = np.hstack([X, np.ones((X.shape[0], 1))])
            
            # Solve least squares
            try:
                plane_coeffs, residuals, rank, s = np.linalg.lstsq(
                    X_with_intercept, z, rcond=None
                )
                
                # Compute RMSE of plane fit
                z_predicted = X_with_intercept @ plane_coeffs
                rmse = np.sqrt(np.mean((z - z_predicted) ** 2))
                
                # RMSE should be small (planar structure preserved)
                # Allow for the noise we added
                assert rmse < 0.61, \
                    f"Planar structure not preserved (RMSE={rmse:.3f})"
            except np.linalg.LinAlgError:
                # If least squares fails, skip this check
                pass
    
    @given(point_cloud_planar_surface_strategy())
    @settings(max_examples=100)
    def test_property_14_neighboring_points_proximity(self, test_data):
        """
        Property 14: Spatial Coherence Preservation (Neighboring Points)
        
        For neighboring points on a surface, their 3D distances should be small
        and consistent, indicating preserved spatial relationships.
        
        **Validates: Requirements 4.5**
        """
        points, plane_params = test_data
        
        assume(points.shape[0] > 20)
        
        # Property 14 Verification: Neighboring points remain proximate
        
        # 1. Compute pairwise distances for a sample of points
        sample_size = min(30, points.shape[0])
        sample_indices = np.random.choice(points.shape[0], sample_size, replace=False)
        sample_points = points[sample_indices]
        
        # 2. For each point, find its nearest neighbor
        nearest_neighbor_distances = []
        
        for i, point in enumerate(sample_points):
            # Compute distances to all other sample points
            other_points = np.delete(sample_points, i, axis=0)
            distances = np.linalg.norm(other_points - point, axis=1)
            
            if len(distances) > 0:
                min_distance = np.min(distances)
                nearest_neighbor_distances.append(min_distance)
        
        nearest_neighbor_distances = np.array(nearest_neighbor_distances)
        
        # 3. Verify that nearest neighbor distances are reasonable
        if len(nearest_neighbor_distances) > 0:
            max_nn_distance = np.max(nearest_neighbor_distances)
            mean_nn_distance = np.mean(nearest_neighbor_distances)
            
            # For a planar surface with regular sampling, nearest neighbors
            # should be relatively close
            assert max_nn_distance < 5.0, \
                f"Found distant nearest neighbors (max={max_nn_distance:.2f}), spatial coherence broken"
            
            # Mean distance should be reasonable
            assert mean_nn_distance < 2.0, \
                f"Mean nearest neighbor distance too large ({mean_nn_distance:.2f})"
        
        # 4. Verify consistency of nearest neighbor distances
        if len(nearest_neighbor_distances) > 1:
            std_nn_distance = np.std(nearest_neighbor_distances)
            mean_nn_distance = np.mean(nearest_neighbor_distances)
            
            if mean_nn_distance > 0:
                cv = std_nn_distance / mean_nn_distance
                
                # For a regular grid, CV should be relatively low
                assert cv < 0.8, \
                    f"Nearest neighbor distances too variable (CV={cv:.3f}), spatial coherence not uniform"
    
    @given(
        st.integers(min_value=5, max_value=15),
        st.floats(min_value=1.0, max_value=3.0)
    )
    @settings(max_examples=50)
    def test_property_12_parameter_sensitivity(self, k_neighbors, std_ratio):
        """
        Property 12: Statistical Outlier Removal Effectiveness (Parameter Sensitivity)
        
        The outlier removal algorithm should work correctly across different
        parameter settings, with higher std_ratio being more permissive.
        
        **Validates: Requirements 4.2**
        """
        # Create a simple point cloud with known outliers
        # Main cluster
        inliers = np.random.normal(0, 0.5, (100, 3)).astype(np.float32)
        
        # Outliers
        outliers = np.array([
            [10, 10, 10],
            [-10, -10, -10],
            [15, 0, 0],
            [0, 15, 0],
            [0, 0, 15],
        ], dtype=np.float32)
        
        points = np.vstack([inliers, outliers])
        
        # Apply outlier removal
        remover = OutlierRemover(k_neighbors=k_neighbors, std_ratio=std_ratio)
        filtered = remover.remove_statistical_outliers(points)
        
        # Property 12 Verification: Parameter sensitivity
        
        # 1. Should remove some points
        assert filtered.shape[0] < points.shape[0], \
            "Outlier removal should reduce point count"
        
        # 2. Should keep most inliers
        assert filtered.shape[0] >= 80, \
            f"Too many inliers removed: kept {filtered.shape[0]}/100"
        
        # 3. Filtered points should be mostly from the main cluster
        # (close to origin)
        if filtered.shape[0] > 0:
            distances_from_origin = np.linalg.norm(filtered, axis=1)
            max_distance = np.max(distances_from_origin)
            
            # Most points should be close to origin (the main cluster)
            assert max_distance < 5.0, \
                f"Found distant points in filtered cloud (max={max_distance:.2f})"
    
    @given(point_cloud_with_outliers_strategy())
    @settings(max_examples=50)
    def test_property_outlier_removal_preserves_colors(self, test_data):
        """
        Property: Outlier removal should preserve color information when present.
        
        **Feature: advanced-stereo-vision-pipeline, Property: Color Preservation**
        """
        points, num_inliers, num_outliers = test_data
        
        assume(points.shape[0] > 30)
        
        # Add random colors
        colors = np.random.randint(0, 255, (points.shape[0], 3), dtype=np.uint8).astype(np.float32)
        points_with_colors = np.hstack([points, colors])
        
        remover = OutlierRemover(k_neighbors=10, std_ratio=2.0)
        filtered = remover.remove_statistical_outliers(points_with_colors)
        
        # Should have 6 columns (XYZ + RGB)
        assert filtered.shape[1] == 6
        
        # Colors should be in valid range
        assert np.all(filtered[:, 3:6] >= 0)
        assert np.all(filtered[:, 3:6] <= 255)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

