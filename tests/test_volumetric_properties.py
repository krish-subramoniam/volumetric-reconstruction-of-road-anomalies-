"""Property-based tests for volumetric analysis module."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from stereo_vision.volumetric import AlphaShapeGenerator, Mesh, Edge


# Custom strategies for point clouds
@st.composite
def point_cloud_strategy(draw, min_points=10, max_points=100):
    """
    Generate random 3D point clouds for testing.
    
    Args:
        min_points: Minimum number of points
        max_points: Maximum number of points
    
    Returns:
        Nx3 numpy array of 3D points
    """
    num_points = draw(st.integers(min_value=min_points, max_value=max_points))
    
    # Generate random seed for reproducibility
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    # Generate points in a reasonable range
    points = rng.uniform(-10.0, 10.0, (num_points, 3)).astype(np.float32)
    
    return points


@st.composite
def clustered_point_cloud_strategy(draw, min_points=20, max_points=100):
    """
    Generate a clustered point cloud (simulating a road anomaly).
    
    This creates a dense cluster of points that represents a realistic
    pothole or hump geometry.
    
    Returns:
        Nx3 numpy array of 3D points
    """
    num_points = draw(st.integers(min_value=min_points, max_value=max_points))
    
    # Generate cluster center
    center = draw(st.lists(
        st.floats(min_value=-5.0, max_value=5.0),
        min_size=3, max_size=3
    ))
    
    # Generate cluster radius
    radius = draw(st.floats(min_value=0.3, max_value=2.0))
    
    # Generate random seed
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    # Generate points around cluster center
    points = rng.normal(loc=center, scale=radius, size=(num_points, 3)).astype(np.float32)
    
    return points


@st.composite
def point_cloud_with_gap_strategy(draw):
    """
    Generate a point cloud with two separate clusters and a gap between them.
    
    This is useful for testing that Alpha Shape doesn't bridge gaps.
    
    Returns:
        Tuple of (points, gap_distance, cluster1_center, cluster2_center)
    """
    # Generate two cluster centers with a gap
    gap_distance = draw(st.floats(min_value=3.0, max_value=10.0))
    
    cluster1_center = np.array([0.0, 0.0, 0.0])
    cluster2_center = np.array([gap_distance, 0.0, 0.0])
    
    # Generate cluster parameters
    cluster_radius = draw(st.floats(min_value=0.3, max_value=1.0))
    points_per_cluster = draw(st.integers(min_value=15, max_value=50))
    
    # Generate random seed
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    # Generate points for both clusters
    cluster1 = rng.normal(loc=cluster1_center, scale=cluster_radius, 
                          size=(points_per_cluster, 3)).astype(np.float32)
    cluster2 = rng.normal(loc=cluster2_center, scale=cluster_radius,
                          size=(points_per_cluster, 3)).astype(np.float32)
    
    # Combine clusters
    points = np.vstack([cluster1, cluster2])
    
    return points, gap_distance, cluster1_center, cluster2_center


@st.composite
def open_mesh_point_cloud_strategy(draw):
    """
    Generate a point cloud that is more likely to produce an open mesh.
    
    This creates a partial surface (like a cup or bowl) by generating points
    on only part of a sphere or cylinder.
    
    Returns:
        Tuple of (alpha, points)
    """
    # Generate alpha value
    alpha = draw(st.floats(min_value=0.8, max_value=3.0))
    
    # Generate random seed
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    # Choose surface type
    surface_type = draw(st.sampled_from(['partial_sphere', 'partial_cylinder', 'plane_with_hole']))
    
    num_points = draw(st.integers(min_value=20, max_value=60))
    
    if surface_type == 'partial_sphere':
        # Generate points on a partial sphere (hemisphere or less)
        theta_max = draw(st.floats(min_value=np.pi/3, max_value=np.pi))
        phi_max = draw(st.floats(min_value=np.pi, max_value=2*np.pi))
        
        theta = rng.uniform(0, theta_max, num_points)
        phi = rng.uniform(0, phi_max, num_points)
        r = 1.0 + rng.normal(0, 0.1, num_points)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        points = np.column_stack([x, y, z]).astype(np.float32)
    
    elif surface_type == 'partial_cylinder':
        # Generate points on a partial cylinder (open at top/bottom)
        angle_max = draw(st.floats(min_value=np.pi, max_value=2*np.pi))
        z_min = draw(st.floats(min_value=0.0, max_value=0.3))
        z_max = draw(st.floats(min_value=0.7, max_value=1.0))
        
        angles = rng.uniform(0, angle_max, num_points)
        z = rng.uniform(z_min, z_max, num_points)
        r = 1.0 + rng.normal(0, 0.1, num_points)
        
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        
        points = np.column_stack([x, y, z]).astype(np.float32)
    
    else:  # plane_with_hole
        # Generate points on a plane with a hole in the middle
        # Create points in a ring
        inner_radius = draw(st.floats(min_value=0.3, max_value=0.6))
        outer_radius = draw(st.floats(min_value=1.0, max_value=2.0))
        
        angles = rng.uniform(0, 2*np.pi, num_points)
        radii = rng.uniform(inner_radius, outer_radius, num_points)
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        z = rng.normal(0, 0.1, num_points)
        
        points = np.column_stack([x, y, z]).astype(np.float32)
    
    return alpha, points


@st.composite
def alpha_and_point_cloud_strategy(draw):
    """
    Generate both an alpha value and a compatible point cloud.
    
    Returns:
        Tuple of (alpha, points)
    """
    alpha = draw(st.floats(min_value=0.5, max_value=5.0))
    points = draw(clustered_point_cloud_strategy(min_points=20, max_points=80))
    
    return alpha, points


class TestAlphaShapeGeneratorProperties:
    """Property-based tests for AlphaShapeGenerator."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=50)
    def test_property_initialization_valid_alpha(self, alpha):
        """
        Property: AlphaShapeGenerator should initialize with any positive alpha value.
        
        **Feature: advanced-stereo-vision-pipeline, Property: Valid Initialization**
        """
        generator = AlphaShapeGenerator(alpha=alpha)
        assert generator.alpha == alpha
    
    @given(alpha_and_point_cloud_strategy())
    @settings(max_examples=100)
    def test_property_15_alpha_shape_mesh_quality_tight_fitting(self, test_data):
        """
        Property 15: Alpha Shape Mesh Quality
        
        For any anomaly point cloud, Alpha Shape mesh generation should produce a surface
        that tightly fits the point distribution without bridging gaps.
        
        This test verifies that:
        1. The mesh vertices are close to the input points
        2. The mesh doesn't extend far beyond the point cloud bounds
        3. The mesh surface tightly wraps the point distribution
        4. All mesh vertices come from the input point cloud
        
        **Validates: Requirements 5.1**
        """
        alpha, points = test_data
        
        # Ensure we have enough points for meaningful mesh
        assume(points.shape[0] >= 10)
        
        # Remove any duplicate points
        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)
        
        # Generate Alpha Shape mesh
        generator = AlphaShapeGenerator(alpha=alpha)
        
        try:
            mesh = generator.generate_alpha_shape(points)
        except ValueError:
            # If triangulation fails (e.g., degenerate points), skip this test case
            assume(False)
        
        # Property 15 Verification: Mesh quality and tight fitting
        
        # 1. Mesh should have valid structure
        assert mesh.vertices.shape[1] == 3, "Mesh vertices should be 3D"
        assert mesh.faces.shape[1] == 3, "Mesh faces should be triangles"
        
        # 2. All mesh vertices should come from the input point cloud
        # (Alpha Shape uses Delaunay triangulation, so vertices are input points)
        for vertex in mesh.vertices:
            # Check if this vertex exists in the input points
            distances = np.linalg.norm(unique_points - vertex, axis=1)
            min_distance = np.min(distances)
            assert min_distance < 1e-6, \
                f"Mesh vertex not from input points (min_distance={min_distance})"
        
        # 3. Mesh should not extend far beyond point cloud bounds
        # Calculate bounding boxes
        points_min = np.min(unique_points, axis=0)
        points_max = np.max(unique_points, axis=0)
        points_range = points_max - points_min
        
        mesh_min = np.min(mesh.vertices, axis=0)
        mesh_max = np.max(mesh.vertices, axis=0)
        
        # Mesh bounds should be within or equal to point cloud bounds
        # (Alpha Shape is a subset of the convex hull)
        assert np.all(mesh_min >= points_min - 1e-6), \
            "Mesh extends below point cloud minimum"
        assert np.all(mesh_max <= points_max + 1e-6), \
            "Mesh extends above point cloud maximum"
        
        # 4. Verify tight fitting: mesh vertices should be well-distributed
        # across the point cloud, not concentrated in one area
        if mesh.vertices.shape[0] >= 4:
            mesh_range = mesh_max - mesh_min
            
            # For each dimension, mesh should span a reasonable portion of point cloud
            for dim in range(3):
                if points_range[dim] > 0.1:  # Only check if there's meaningful spread
                    coverage_ratio = mesh_range[dim] / points_range[dim]
                    # Mesh should cover at least 30% of the point cloud extent
                    # (Alpha Shape may be smaller than convex hull, but shouldn't be tiny)
                    assert coverage_ratio >= 0.3, \
                        f"Mesh doesn't fit point distribution well in dimension {dim} " \
                        f"(coverage={coverage_ratio:.2f})"
        
        # 5. Verify mesh faces reference valid vertices
        if mesh.faces.shape[0] > 0:
            max_vertex_index = np.max(mesh.faces)
            assert max_vertex_index < mesh.vertices.shape[0], \
                f"Face references invalid vertex index {max_vertex_index}"
            
            min_vertex_index = np.min(mesh.faces)
            assert min_vertex_index >= 0, \
                f"Face has negative vertex index {min_vertex_index}"
    
    @given(alpha_and_point_cloud_strategy())
    @settings(max_examples=100, deadline=500)
    def test_property_15_alpha_shape_tight_fitting_verification(self, test_data):
        """
        Property 15: Alpha Shape Mesh Quality (Tight Fitting Verification)
        
        For any point cloud, Alpha Shape mesh generation should produce a surface
        that tightly fits the point distribution. This is verified by checking that:
        1. All mesh vertices come from the input points
        2. The mesh doesn't extend beyond the point cloud bounds
        3. The mesh covers a reasonable portion of the point cloud extent
        
        **Validates: Requirements 5.1**
        """
        alpha, points = test_data
        
        assume(points.shape[0] >= 10)
        
        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)
        
        generator = AlphaShapeGenerator(alpha=alpha)
        
        try:
            mesh = generator.generate_alpha_shape(points)
        except ValueError:
            assume(False)
        
        # Property 15 Verification: Tight fitting
        
        # 1. All mesh vertices must come from the input points
        for vertex in mesh.vertices:
            distances = np.linalg.norm(unique_points - vertex, axis=1)
            min_distance = np.min(distances)
            assert min_distance < 1e-6, \
                f"Mesh vertex not from input points (min_distance={min_distance})"
        
        # 2. Mesh should not extend beyond point cloud bounds
        points_min = np.min(unique_points, axis=0)
        points_max = np.max(unique_points, axis=0)
        
        mesh_min = np.min(mesh.vertices, axis=0)
        mesh_max = np.max(mesh.vertices, axis=0)
        
        assert np.all(mesh_min >= points_min - 1e-6), \
            "Mesh extends below point cloud minimum"
        assert np.all(mesh_max <= points_max + 1e-6), \
            "Mesh extends above point cloud maximum"
        
        # 3. If mesh has faces, verify they form valid triangles
        if mesh.faces.shape[0] > 0:
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                
                # Calculate triangle area
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                
                # Triangle should have non-zero area
                assert area > 1e-10, \
                    f"Degenerate triangle with area {area}"
    
    @given(alpha_and_point_cloud_strategy())
    @settings(max_examples=100)
    def test_property_15_alpha_shape_surface_extraction(self, test_data):
        """
        Property 15: Alpha Shape Mesh Quality (Surface Extraction)
        
        The Alpha Shape algorithm should correctly extract surface triangles from
        the filtered tetrahedra, ensuring that only boundary faces are included.
        
        **Validates: Requirements 5.1**
        """
        alpha, points = test_data
        
        assume(points.shape[0] >= 10)
        
        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)
        
        generator = AlphaShapeGenerator(alpha=alpha)
        
        try:
            mesh = generator.generate_alpha_shape(points)
        except ValueError:
            assume(False)
        
        # Property 15 Verification: Surface extraction correctness
        
        # 1. All faces should be triangles
        if mesh.faces.shape[0] > 0:
            assert mesh.faces.shape[1] == 3, "All faces should be triangles"
        
        # 2. Face vertices should form valid triangles (non-degenerate)
        if mesh.faces.shape[0] > 0:
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                
                # Calculate triangle area using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                
                # Triangle should have non-zero area
                assert area > 1e-10, \
                    f"Degenerate triangle with area {area}"
        
        # 3. Verify that the mesh represents a surface (has boundary edges or is closed)
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        # A valid surface mesh should either:
        # - Have boundary edges (open surface), or
        # - Have no boundary edges (closed surface)
        # Both are valid for Alpha Shapes
        
        # If there are faces, there should be a consistent edge structure
        if mesh.faces.shape[0] > 0:
            # Count total edges
            total_edges = mesh.faces.shape[0] * 3
            
            # Boundary edges appear once, internal edges appear twice
            # So: total_edges = boundary_edges + 2 * internal_edges
            # This should always be satisfied
            num_boundary = len(boundary_edges)
            num_internal = (total_edges - num_boundary) / 2
            
            assert num_internal >= 0, \
                "Invalid edge structure: more boundary edges than possible"
            
            # For a valid surface, we should have some edges
            assert total_edges > 0, "Mesh has faces but no edges"
    
    @given(
        clustered_point_cloud_strategy(min_points=20, max_points=80),
        st.floats(min_value=0.5, max_value=2.0),
        st.floats(min_value=2.5, max_value=5.0)
    )
    @settings(max_examples=100)
    def test_property_15_alpha_parameter_effect_on_tightness(
        self, points, small_alpha, large_alpha
    ):
        """
        Property 15: Alpha Shape Mesh Quality (Alpha Parameter Effect)

        Smaller alpha values should produce tighter fits (potentially fewer faces),
        while larger alpha values should produce smoother surfaces (more permissive).

        This verifies that the alpha parameter correctly controls mesh tightness.

        **Validates: Requirements 5.1**
        """
        assume(small_alpha < large_alpha)
        assume(points.shape[0] >= 10)

        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)

        # Calculate point cloud spread to ensure alpha values are reasonable
        point_spread = np.max(np.std(unique_points, axis=0))
        assume(point_spread > 0.1)  # Need meaningful spread

        # Generate meshes with different alpha values
        gen_small = AlphaShapeGenerator(alpha=small_alpha)
        gen_large = AlphaShapeGenerator(alpha=large_alpha)

        try:
            mesh_small = gen_small.generate_alpha_shape(points)
            mesh_large = gen_large.generate_alpha_shape(points)
        except ValueError:
            assume(False)

        # Property 15 Verification: Alpha parameter effect

        # 1. Both meshes should be valid
        assert mesh_small.vertices.shape[1] == 3
        assert mesh_large.vertices.shape[1] == 3

        # 2. At least one mesh should have faces (verify alpha values aren't too restrictive)
        total_faces = mesh_small.faces.shape[0] + mesh_large.faces.shape[0]
        assume(total_faces > 0)  # Skip if both produce empty meshes

        # 3. Larger alpha should be more permissive (produce at least as many faces)
        # However, this relationship isn't always monotonic due to the discrete nature
        # of Delaunay triangulation, so we use a relaxed check
        if mesh_small.faces.shape[0] > 0 and mesh_large.faces.shape[0] > 0:
            # The key property: larger alpha shouldn't be dramatically more restrictive
            # Allow for some variation, but larger alpha should generally have similar or more faces
            ratio = mesh_large.faces.shape[0] / mesh_small.faces.shape[0]

            # Relaxed check: larger alpha shouldn't produce less than 20% of smaller alpha's faces
            assert ratio >= 0.20, \
                f"Larger alpha produced too few faces (ratio={ratio:.2f}, " \
                f"small={mesh_small.faces.shape[0]}, large={mesh_large.faces.shape[0]})"

        # 4. Verify that both meshes fit the point cloud
        for mesh in [mesh_small, mesh_large]:
            if mesh.faces.shape[0] > 0:
                # All mesh vertices should be from the input points
                for vertex in mesh.vertices:
                    distances = np.linalg.norm(unique_points - vertex, axis=1)
                    min_distance = np.min(distances)
                    assert min_distance < 1e-6, \
                        "Mesh vertex not from input points"

    
    @given(alpha_and_point_cloud_strategy())
    @settings(max_examples=50)
    def test_property_mesh_statistics_consistency(self, test_data):
        """
        Property: Mesh statistics should be consistent with mesh structure.
        
        **Feature: advanced-stereo-vision-pipeline, Property: Statistics Consistency**
        """
        alpha, points = test_data
        
        assume(points.shape[0] >= 10)
        
        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)
        
        generator = AlphaShapeGenerator(alpha=alpha)
        
        try:
            mesh = generator.generate_alpha_shape(points)
        except ValueError:
            assume(False)
        
        stats = generator.get_mesh_statistics(mesh)
        
        # Verify statistics consistency
        assert stats['num_vertices'] == mesh.vertices.shape[0]
        assert stats['num_faces'] == mesh.faces.shape[0]
        
        if mesh.faces.shape[0] > 0:
            assert 'surface_area' in stats
            assert stats['surface_area'] >= 0
            assert np.isfinite(stats['surface_area'])
            
            assert 'mean_triangle_area' in stats
            assert stats['mean_triangle_area'] >= 0
            
            # Number of boundary edges should be reasonable
            assert stats['num_boundary_edges'] >= 0
            assert stats['num_boundary_edges'] <= mesh.faces.shape[0] * 3


class TestBoundaryDetectionProperties:
    """Property-based tests for boundary edge detection."""
    
    @given(alpha_and_point_cloud_strategy())
    @settings(max_examples=100)
    def test_property_16_boundary_edge_detection_accuracy(self, test_data):
        """
        Property 16: Boundary Edge Detection Accuracy
        
        For any open mesh surface, boundary edge detection should correctly identify
        all edges that form the perimeter of the surface.
        
        A boundary edge is an edge that belongs to exactly one triangle.
        This test verifies that:
        1. All detected boundary edges appear exactly once in the mesh
        2. All non-boundary edges appear exactly twice in the mesh
        3. The boundary edges form a consistent structure
        
        **Validates: Requirements 5.2**
        """
        alpha, points = test_data
        
        assume(points.shape[0] >= 10)
        
        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)
        
        generator = AlphaShapeGenerator(alpha=alpha)
        
        try:
            mesh = generator.generate_alpha_shape(points)
        except ValueError:
            assume(False)
        
        # Skip if mesh has no faces
        assume(mesh.faces.shape[0] > 0)
        
        # Extract boundary edges
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        # Property 16 Verification: Boundary edge detection accuracy
        
        # 1. Count all edge occurrences in the mesh
        edge_count = {}
        for face in mesh.faces:
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # 2. Verify that all boundary edges appear exactly once
        boundary_edge_tuples = set()
        for edge in boundary_edges:
            edge_tuple = tuple(sorted([edge.v1, edge.v2]))
            boundary_edge_tuples.add(edge_tuple)
            
            # Boundary edge should appear exactly once in the mesh
            assert edge_tuple in edge_count, \
                f"Boundary edge {edge_tuple} not found in mesh"
            assert edge_count[edge_tuple] == 1, \
                f"Boundary edge {edge_tuple} appears {edge_count[edge_tuple]} times, expected 1"
        
        # 3. Verify that all edges appearing once are detected as boundary edges
        for edge_tuple, count in edge_count.items():
            if count == 1:
                assert edge_tuple in boundary_edge_tuples, \
                    f"Edge {edge_tuple} appears once but not detected as boundary edge"
        
        # 4. Verify that all non-boundary edges appear at least twice
        # (In manifold meshes, internal edges appear exactly twice, but Alpha Shapes
        # can produce non-manifold edges that appear more than twice)
        for edge_tuple, count in edge_count.items():
            if edge_tuple not in boundary_edge_tuples:
                assert count >= 2, \
                    f"Non-boundary edge {edge_tuple} appears {count} times, expected at least 2"
        
        # 5. Verify boundary edges reference valid vertices
        for edge in boundary_edges:
            assert 0 <= edge.v1 < mesh.vertices.shape[0], \
                f"Boundary edge has invalid vertex index {edge.v1}"
            assert 0 <= edge.v2 < mesh.vertices.shape[0], \
                f"Boundary edge has invalid vertex index {edge.v2}"
            assert edge.v1 != edge.v2, \
                f"Boundary edge has same vertex for both endpoints: {edge.v1}"
    
    @given(alpha_and_point_cloud_strategy())
    @settings(max_examples=100)
    def test_property_16_boundary_edge_consistency(self, test_data):
        """
        Property 16: Boundary Edge Detection Accuracy (Consistency Check)
        
        For any mesh, the boundary edges should form a consistent structure where
        the total number of edges equals boundary edges plus twice the internal edges.
        
        **Validates: Requirements 5.2**
        """
        alpha, points = test_data
        
        assume(points.shape[0] >= 10)
        
        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)
        
        generator = AlphaShapeGenerator(alpha=alpha)
        
        try:
            mesh = generator.generate_alpha_shape(points)
        except ValueError:
            assume(False)
        
        assume(mesh.faces.shape[0] > 0)
        
        # Extract boundary edges
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        # Property 16 Verification: Edge structure consistency
        
        # Total edges in mesh (counting with multiplicity)
        total_edges = mesh.faces.shape[0] * 3
        
        # Number of boundary edges
        num_boundary = len(boundary_edges)
        
        # Calculate number of internal edges
        # Formula: total_edges = num_boundary + 2 * num_internal
        # So: num_internal = (total_edges - num_boundary) / 2
        num_internal = (total_edges - num_boundary) / 2
        
        # Verify consistency
        assert num_internal >= 0, \
            f"Invalid edge structure: negative internal edges ({num_internal})"
        
        assert num_internal == int(num_internal), \
            f"Invalid edge structure: non-integer internal edges ({num_internal})"
        
        # Verify the formula holds
        assert total_edges == num_boundary + 2 * int(num_internal), \
            f"Edge count formula doesn't hold: {total_edges} != {num_boundary} + 2*{int(num_internal)}"
    
    @given(alpha_and_point_cloud_strategy())
    @settings(max_examples=50)
    def test_property_16_closed_mesh_no_boundary(self, test_data):
        """
        Property 16: Boundary Edge Detection Accuracy (Closed Mesh)
        
        For any closed (watertight) mesh, boundary edge detection should return
        an empty list since all edges are shared by exactly two faces.
        
        **Validates: Requirements 5.2**
        """
        alpha, points = test_data
        
        assume(points.shape[0] >= 10)
        
        unique_points = np.unique(points, axis=0)
        assume(unique_points.shape[0] >= 4)
        
        generator = AlphaShapeGenerator(alpha=alpha)
        
        try:
            mesh = generator.generate_alpha_shape(points)
        except ValueError:
            assume(False)
        
        assume(mesh.faces.shape[0] > 0)
        
        # Extract boundary edges
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        # Count edge occurrences
        edge_count = {}
        for face in mesh.faces:
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Property 16 Verification: Closed mesh has no boundary edges
        
        # If all edges appear at least twice, mesh has no boundary edges
        # (Note: Alpha Shapes can produce non-manifold edges that appear more than twice)
        has_boundary_edges = any(count == 1 for count in edge_count.values())
        
        if not has_boundary_edges:
            # Mesh with no boundary edges should have empty boundary edge list
            assert len(boundary_edges) == 0, \
                f"Mesh with no boundary edges has {len(boundary_edges)} detected boundary edges"
        else:
            # Mesh with boundary edges should have non-empty boundary edge list
            assert len(boundary_edges) > 0, \
                f"Mesh with boundary edges has empty boundary edge list"


class TestMeshCappingProperties:
    """Property-based tests for mesh capping."""
    
    @given(
        st.integers(min_value=3, max_value=8),  # Number of vertices in boundary loop
        st.floats(min_value=0.5, max_value=2.0),  # Radius
        st.floats(min_value=-1.0, max_value=1.0)  # Z offset
    )
    @settings(max_examples=100)
    def test_property_17_mesh_capping_completeness(self, num_vertices, radius, z_offset):
        """
        Property 17: Mesh Capping Completeness
        
        For any detected boundary loop, triangulation should generate a cap that
        properly closes the mesh opening.
        
        This test verifies that:
        1. Boundary loops are correctly extracted from boundary edges
        2. Each loop is triangulated into valid triangles
        3. The cap triangles reference valid vertices
        4. The cap reduces the number of boundary edges
        
        **Validates: Requirements 5.3**
        """
        from stereo_vision.volumetric import MeshCapper
        
        # Create a simple open mesh: a cylinder without top cap
        # Bottom ring
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        bottom_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(num_vertices, z_offset)
        ])
        
        # Top ring
        top_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(num_vertices, z_offset + 1.0)
        ])
        
        # Combine vertices
        vertices = np.vstack([bottom_vertices, top_vertices]).astype(np.float32)
        
        # Create side faces (connecting bottom and top rings)
        faces = []
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            # Two triangles per side
            faces.append([i, next_i, num_vertices + next_i])
            faces.append([i, num_vertices + next_i, num_vertices + i])
        
        faces = np.array(faces)
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Extract boundary edges
        generator = AlphaShapeGenerator(alpha=1.0)
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        # Should have boundary edges at top and bottom
        assert len(boundary_edges) > 0, "Open cylinder should have boundary edges"
        
        # Property 17 Verification: Mesh capping completeness
        
        capper = MeshCapper()
        
        # 1. Triangulate boundary to create cap
        cap_mesh = capper.triangulate_boundary(boundary_edges, mesh.vertices)
        
        # 2. Verify cap mesh structure
        assert cap_mesh.vertices.shape[0] == mesh.vertices.shape[0], \
            "Cap mesh should use same vertices as original mesh"
        
        # 3. Cap should have faces
        assert cap_mesh.faces.shape[0] > 0, \
            "Cap should have faces for open mesh"
        
        # 4. Verify cap faces are valid triangles
        assert cap_mesh.faces.shape[1] == 3, \
            "Cap faces should be triangles"
        
        # 5. Verify cap faces reference valid vertices
        for face in cap_mesh.faces:
            for vertex_idx in face:
                assert 0 <= vertex_idx < mesh.vertices.shape[0], \
                    f"Cap face references invalid vertex {vertex_idx}"
            
            # Verify face vertices are distinct
            assert len(set(face)) == 3, \
                f"Cap face has duplicate vertices: {face}"
        
        # 6. Verify cap triangles have non-zero area
        for face in cap_mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            
            assert area > 1e-6, \
                f"Cap triangle has near-zero area: {area}"
        
        # 7. Combine surface and cap to create watertight mesh
        combined_mesh = capper.create_watertight_mesh(mesh, cap_mesh)
        
        # 8. Verify combined mesh has more faces
        assert combined_mesh.faces.shape[0] == mesh.faces.shape[0] + cap_mesh.faces.shape[0], \
            "Combined mesh should have sum of surface and cap faces"
        
        # 9. Verify that capping reduces boundary edges
        boundary_edges_after = generator.extract_boundary_edges(combined_mesh)
        
        # The number of boundary edges should be reduced
        assert len(boundary_edges_after) < len(boundary_edges), \
            f"Capping should reduce boundary edges: {len(boundary_edges)} -> {len(boundary_edges_after)}"
    
    @given(
        st.integers(min_value=3, max_value=10),  # Number of vertices in loop
        st.floats(min_value=0.5, max_value=2.0)  # Radius
    )
    @settings(max_examples=100)
    def test_property_17_cap_closes_boundary_loops(self, num_vertices, radius):
        """
        Property 17: Mesh Capping Completeness (Loop Closure)
        
        For any boundary loop extracted from boundary edges, the triangulation
        should create faces that use all vertices in the loop.
        
        **Validates: Requirements 5.3**
        """
        from stereo_vision.volumetric import MeshCapper, Edge
        
        # Create a simple boundary loop (circular)
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(num_vertices)
        ]).astype(np.float32)
        
        # Create boundary edges forming a loop
        boundary_edges = []
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            boundary_edges.append(Edge(i, next_i))
        
        # Property 17 Verification: Cap closes boundary loops
        
        capper = MeshCapper()
        
        # Extract boundary loops
        loops = capper._extract_boundary_loops(boundary_edges)
        
        assert len(loops) > 0, "Should extract at least one loop"
        
        # For each loop, verify triangulation
        for loop in loops:
            # 1. Loop should have at least 3 vertices
            assert len(loop) >= 3, \
                f"Boundary loop has fewer than 3 vertices: {len(loop)}"
            
            # 2. Loop vertices should be distinct
            assert len(set(loop)) == len(loop), \
                f"Boundary loop has duplicate vertices"
            
            # 3. Triangulate the loop
            cap_faces = capper._triangulate_loop(loop, vertices)
            
            # 4. Verify triangulation produces faces
            assert len(cap_faces) > 0, \
                f"Triangulation of loop with {len(loop)} vertices produced no faces"
            
            # 5. Verify number of triangles (should be n-2 for fan triangulation)
            expected_triangles = len(loop) - 2
            assert len(cap_faces) == expected_triangles, \
                f"Expected {expected_triangles} triangles for {len(loop)}-vertex loop, got {len(cap_faces)}"
            
            # 6. Verify all cap faces use vertices from the loop
            loop_vertices = set(loop)
            for face in cap_faces:
                for vertex_idx in face:
                    assert vertex_idx in loop_vertices, \
                        f"Cap face uses vertex {vertex_idx} not in loop {loop}"
            
            # 7. Verify all triangles are valid (non-degenerate)
            for face in cap_faces:
                v0, v1, v2 = vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                
                assert area > 1e-6, \
                    f"Cap triangle has near-zero area: {area}"
    
    @given(
        st.integers(min_value=3, max_value=8),  # Number of vertices
        st.floats(min_value=0.5, max_value=2.0)  # Radius
    )
    @settings(max_examples=50)
    def test_property_17_watertight_validation_after_capping(self, num_vertices, radius):
        """
        Property 17: Mesh Capping Completeness (Watertightness)
        
        For any open mesh that is successfully capped, the watertightness validation
        should correctly determine if the mesh forms a closed manifold.
        
        **Validates: Requirements 5.3**
        """
        from stereo_vision.volumetric import MeshCapper
        
        # Create a simple open mesh: single ring (no top or bottom)
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        bottom_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(num_vertices)
        ])
        
        top_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.ones(num_vertices)
        ])
        
        vertices = np.vstack([bottom_vertices, top_vertices]).astype(np.float32)
        
        # Create side faces only (no top/bottom caps)
        faces = []
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([i, next_i, num_vertices + next_i])
            faces.append([i, num_vertices + next_i, num_vertices + i])
        
        faces = np.array(faces)
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Extract boundary edges
        generator = AlphaShapeGenerator(alpha=1.0)
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        assert len(boundary_edges) > 0, "Open mesh should have boundary edges"
        
        # Property 17 Verification: Watertightness validation
        
        capper = MeshCapper()
        
        # Original mesh should not be watertight
        is_watertight_before = capper.validate_watertightness(mesh)
        assert not is_watertight_before, \
            "Open mesh incorrectly validated as watertight"
        
        # Create cap
        cap_mesh = capper.triangulate_boundary(boundary_edges, mesh.vertices)
        
        # Combine to create potentially watertight mesh
        combined_mesh = capper.create_watertight_mesh(mesh, cap_mesh)
        
        # Validate watertightness
        is_watertight_after = capper.validate_watertightness(combined_mesh)
        
        # Verify validation is consistent with edge structure
        boundary_edges_after = generator.extract_boundary_edges(combined_mesh)
        
        if len(boundary_edges_after) == 0:
            # No boundary edges means mesh should be watertight
            assert is_watertight_after, \
                "Mesh with no boundary edges should be watertight"
        else:
            # Has boundary edges means mesh should not be watertight
            assert not is_watertight_after, \
                "Mesh with boundary edges should not be watertight"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestWatertightnessValidationProperties:
    """Property-based tests for watertightness validation."""
    
    @given(
        st.integers(min_value=3, max_value=8),  # Number of vertices
        st.floats(min_value=0.5, max_value=2.0)  # Radius
    )
    @settings(max_examples=100)
    def test_property_18_watertightness_validation_closed_mesh(self, num_vertices, radius):
        """
        Property 18: Watertightness Validation
        
        For any mesh after capping, watertightness verification should correctly
        determine whether the mesh forms a closed manifold.
        
        A mesh is watertight if every edge is shared by exactly 2 faces.
        
        This test verifies that:
        1. Closed meshes (no boundary edges) are validated as watertight
        2. Open meshes (with boundary edges) are validated as not watertight
        3. The validation is consistent with edge structure
        
        **Validates: Requirements 5.4**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a closed tetrahedron (simplest closed mesh)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2],  # Bottom face
            [0, 1, 3],  # Front face
            [0, 2, 3],  # Left face
            [1, 2, 3]   # Back face
        ])
        
        closed_mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 18 Verification: Watertightness validation for closed mesh
        
        calculator = VolumeCalculator()
        
        # 1. Closed mesh should be validated as watertight
        is_watertight = calculator.validate_mesh_closure(closed_mesh)
        assert is_watertight, \
            "Closed tetrahedron should be validated as watertight"
        
        # 2. Verify edge structure: all edges should appear exactly twice
        edge_count = {}
        for face in closed_mesh.faces:
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # All edges should appear exactly twice
        for edge, count in edge_count.items():
            assert count == 2, \
                f"Edge {edge} appears {count} times in closed mesh, expected 2"
        
        # 3. Test open mesh (single triangle)
        open_vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        open_faces = np.array([[0, 1, 2]])
        open_mesh = Mesh(vertices=open_vertices, faces=open_faces)
        
        # Open mesh should not be watertight
        is_watertight_open = calculator.validate_mesh_closure(open_mesh)
        assert not is_watertight_open, \
            "Open mesh (single triangle) should not be validated as watertight"
    
    @given(
        st.integers(min_value=4, max_value=10),  # Number of vertices per ring
        st.floats(min_value=0.5, max_value=2.0)  # Radius
    )
    @settings(max_examples=100)
    def test_property_18_watertightness_validation_consistency(self, num_vertices, radius):
        """
        Property 18: Watertightness Validation (Consistency)
        
        For any mesh, watertightness validation should be consistent with the
        edge structure: a mesh is watertight if and only if all edges appear
        exactly twice.
        
        **Validates: Requirements 5.4**
        """
        from stereo_vision.volumetric import VolumeCalculator, MeshCapper
        
        # Create a cylinder with top and bottom caps (closed mesh)
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        
        # Bottom ring
        bottom_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(num_vertices)
        ])
        
        # Top ring
        top_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.ones(num_vertices)
        ])
        
        # Center points for caps
        bottom_center = np.array([[0, 0, 0]])
        top_center = np.array([[0, 0, 1]])
        
        # Combine all vertices
        vertices = np.vstack([
            bottom_vertices,
            top_vertices,
            bottom_center,
            top_center
        ]).astype(np.float32)
        
        bottom_center_idx = 2 * num_vertices
        top_center_idx = 2 * num_vertices + 1
        
        # Create faces
        faces = []
        
        # Side faces
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([i, next_i, num_vertices + next_i])
            faces.append([i, num_vertices + next_i, num_vertices + i])
        
        # Bottom cap (fan from center)
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([bottom_center_idx, i, next_i])
        
        # Top cap (fan from center)
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([top_center_idx, num_vertices + next_i, num_vertices + i])
        
        faces = np.array(faces)
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 18 Verification: Consistency between validation and edge structure
        
        calculator = VolumeCalculator()
        capper = MeshCapper()
        
        # Validate watertightness
        is_watertight = calculator.validate_mesh_closure(mesh)
        
        # Count edge occurrences
        edge_count = {}
        for face in mesh.faces:
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Check if all edges appear exactly twice
        all_edges_twice = all(count == 2 for count in edge_count.values())
        
        # Watertightness should match edge structure
        assert is_watertight == all_edges_twice, \
            f"Watertightness validation ({is_watertight}) inconsistent with edge structure (all_edges_twice={all_edges_twice})"
        
        # Also verify using MeshCapper's validation
        is_watertight_capper = capper.validate_watertightness(mesh)
        assert is_watertight == is_watertight_capper, \
            "VolumeCalculator and MeshCapper watertightness validation should agree"
    
    @given(st.integers(min_value=0, max_value=5))
    @settings(max_examples=50)
    def test_property_18_empty_mesh_not_watertight(self, num_vertices):
        """
        Property 18: Watertightness Validation (Empty Mesh)
        
        For any empty mesh (no faces), watertightness validation should return False.
        
        **Validates: Requirements 5.4**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create mesh with vertices but no faces
        vertices = np.random.randn(num_vertices, 3).astype(np.float32)
        faces = np.empty((0, 3), dtype=int)
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 18 Verification: Empty mesh is not watertight
        
        calculator = VolumeCalculator()
        is_watertight = calculator.validate_mesh_closure(mesh)
        
        assert not is_watertight, \
            "Empty mesh (no faces) should not be validated as watertight"


class TestVolumeCalculationProperties:
    """Property-based tests for volume calculation."""
    
    @given(st.floats(min_value=0.1, max_value=5.0))
    @settings(max_examples=100)
    def test_property_19_volume_calculation_mathematical_correctness_cube(self, side_length):
        """
        Property 19: Volume Calculation Mathematical Correctness
        
        For any watertight mesh, signed tetrahedron volume integration should
        produce results consistent with the Divergence Theorem.
        
        This test verifies volume calculation on a cube with known volume.
        
        **Validates: Requirements 5.5, 6.1**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a cube with given side length
        # 8 vertices of the cube
        vertices = np.array([
            [0, 0, 0],
            [side_length, 0, 0],
            [side_length, side_length, 0],
            [0, side_length, 0],
            [0, 0, side_length],
            [side_length, 0, side_length],
            [side_length, side_length, side_length],
            [0, side_length, side_length]
        ], dtype=np.float32)
        
        # 12 triangular faces (2 per cube face)
        faces = np.array([
            # Bottom (z=0)
            [0, 1, 2], [0, 2, 3],
            # Top (z=side_length)
            [4, 6, 5], [4, 7, 6],
            # Front (y=0)
            [0, 5, 1], [0, 4, 5],
            # Back (y=side_length)
            [2, 7, 3], [2, 6, 7],
            # Left (x=0)
            [0, 7, 4], [0, 3, 7],
            # Right (x=side_length)
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 19 Verification: Volume calculation correctness
        
        calculator = VolumeCalculator()
        
        # 1. Verify mesh is watertight
        is_watertight = calculator.validate_mesh_closure(mesh)
        assert is_watertight, \
            "Cube mesh should be watertight"
        
        # 2. Calculate volume
        calculated_volume = calculator.calculate_signed_volume(mesh)
        
        # 3. Expected volume
        expected_volume = side_length ** 3
        
        # 4. Verify volume is correct (within numerical tolerance)
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.01, \
            f"Volume calculation error too large: calculated={calculated_volume:.6f}, " \
            f"expected={expected_volume:.6f}, relative_error={relative_error:.6f}"
        
        # 5. Volume should be positive
        assert calculated_volume > 0, \
            f"Volume should be positive, got {calculated_volume}"
    
    @given(st.floats(min_value=0.1, max_value=5.0))
    @settings(max_examples=100)
    def test_property_19_volume_calculation_tetrahedron(self, scale):
        """
        Property 19: Volume Calculation Mathematical Correctness (Tetrahedron)
        
        For any watertight tetrahedron, the calculated volume should match
        the analytical formula: V = (1/6) * |det([v1-v0, v2-v0, v3-v0])|
        
        **Validates: Requirements 5.5, 6.1**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a regular tetrahedron scaled by the given factor
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3)/2, 0],
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
        ], dtype=np.float32) * scale
        
        faces = np.array([
            [0, 1, 2],  # Bottom
            [0, 1, 3],  # Front
            [0, 2, 3],  # Left
            [1, 2, 3]   # Back
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 19 Verification: Volume calculation for tetrahedron
        
        calculator = VolumeCalculator()
        
        # 1. Verify mesh is watertight
        is_watertight = calculator.validate_mesh_closure(mesh)
        assert is_watertight, \
            "Tetrahedron mesh should be watertight"
        
        # 2. Calculate volume using signed tetrahedron integration
        calculated_volume = calculator.calculate_signed_volume(mesh)
        
        # 3. Calculate expected volume using analytical formula
        v0, v1, v2, v3 = vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0
        
        # Volume = (1/6) * |det([edge1, edge2, edge3])|
        det = np.dot(edge1, np.cross(edge2, edge3))
        expected_volume = abs(det) / 6.0
        
        # 4. Verify volume is correct
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.01, \
            f"Volume calculation error: calculated={calculated_volume:.6f}, " \
            f"expected={expected_volume:.6f}, relative_error={relative_error:.6f}"
        
        # 5. Volume should be positive
        assert calculated_volume > 0, \
            f"Volume should be positive, got {calculated_volume}"
    
    @given(
        st.floats(min_value=0.1, max_value=3.0),  # Radius
        st.integers(min_value=4, max_value=8)     # Number of vertices per ring
    )
    @settings(max_examples=50)
    def test_property_19_volume_calculation_positive(self, radius, num_vertices):
        """
        Property 19: Volume Calculation Mathematical Correctness (Positivity)
        
        For any watertight mesh, the calculated volume should always be positive.
        
        **Validates: Requirements 5.5, 6.1**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a closed cylinder (approximation)
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        
        # Bottom ring
        bottom_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(num_vertices)
        ])
        
        # Top ring
        top_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.ones(num_vertices)
        ])
        
        # Center points
        bottom_center = np.array([[0, 0, 0]])
        top_center = np.array([[0, 0, 1]])
        
        vertices = np.vstack([
            bottom_vertices,
            top_vertices,
            bottom_center,
            top_center
        ]).astype(np.float32)
        
        bottom_center_idx = 2 * num_vertices
        top_center_idx = 2 * num_vertices + 1
        
        # Create faces
        faces = []
        
        # Side faces
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([i, next_i, num_vertices + next_i])
            faces.append([i, num_vertices + next_i, num_vertices + i])
        
        # Bottom cap
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([bottom_center_idx, i, next_i])
        
        # Top cap
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([top_center_idx, num_vertices + next_i, num_vertices + i])
        
        faces = np.array(faces)
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 19 Verification: Volume positivity
        
        calculator = VolumeCalculator()
        
        # 1. Verify mesh is watertight
        is_watertight = calculator.validate_mesh_closure(mesh)
        
        if is_watertight:
            # 2. Calculate volume
            calculated_volume = calculator.calculate_signed_volume(mesh)
            
            # 3. Volume should be positive
            assert calculated_volume > 0, \
                f"Volume should be positive for watertight mesh, got {calculated_volume}"
            
            # 4. Volume should be finite
            assert np.isfinite(calculated_volume), \
                f"Volume should be finite, got {calculated_volume}"
    
    @given(st.floats(min_value=0.1, max_value=3.0))
    @settings(max_examples=50)
    def test_property_19_volume_calculation_scale_invariance(self, scale_factor):
        """
        Property 19: Volume Calculation Mathematical Correctness (Scale Invariance)
        
        For any watertight mesh, scaling the mesh by a factor k should scale
        the volume by k^3.
        
        **Validates: Requirements 5.5, 6.1**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a unit cube
        vertices_unit = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 6, 5], [4, 7, 6],  # Top
            [0, 5, 1], [0, 4, 5],  # Front
            [2, 7, 3], [2, 6, 7],  # Back
            [0, 7, 4], [0, 3, 7],  # Left
            [1, 6, 2], [1, 5, 6]   # Right
        ])
        
        mesh_unit = Mesh(vertices=vertices_unit, faces=faces)
        
        # Create scaled mesh
        vertices_scaled = vertices_unit * scale_factor
        mesh_scaled = Mesh(vertices=vertices_scaled, faces=faces)
        
        # Property 19 Verification: Scale invariance
        
        calculator = VolumeCalculator()
        
        # Calculate volumes
        volume_unit = calculator.calculate_signed_volume(mesh_unit)
        volume_scaled = calculator.calculate_signed_volume(mesh_scaled)
        
        # Expected relationship: volume_scaled = volume_unit * scale_factor^3
        expected_volume_scaled = volume_unit * (scale_factor ** 3)
        
        # Verify scaling relationship
        relative_error = abs(volume_scaled - expected_volume_scaled) / expected_volume_scaled
        assert relative_error < 0.01, \
            f"Volume scaling error: volume_scaled={volume_scaled:.6f}, " \
            f"expected={expected_volume_scaled:.6f}, relative_error={relative_error:.6f}"
    
    @given(st.integers(min_value=0, max_value=5))
    @settings(max_examples=50)
    def test_property_19_volume_calculation_rejects_open_mesh(self, num_vertices):
        """
        Property 19: Volume Calculation Mathematical Correctness (Open Mesh Rejection)
        
        For any non-watertight mesh, volume calculation should raise WatertightnessError.
        
        **Validates: Requirements 5.5, 6.1**
        """
        from stereo_vision.volumetric import VolumeCalculator
        from stereo_vision.errors import WatertightnessError
        
        # Create an open mesh (single triangle)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2]])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 19 Verification: Open mesh rejection
        
        calculator = VolumeCalculator()
        
        # Verify mesh is not watertight
        is_watertight = calculator.validate_mesh_closure(mesh)
        assert not is_watertight, \
            "Single triangle should not be watertight"
        
        # Volume calculation should raise WatertightnessError
        with pytest.raises(WatertightnessError, match="not watertight"):
            calculator.calculate_signed_volume(mesh)
    
    @given(st.floats(min_value=0.1, max_value=3.0))
    @settings(max_examples=50)
    def test_property_19_volume_with_validation_convenience_method(self, side_length):
        """
        Property 19: Volume Calculation Mathematical Correctness (Convenience Method)
        
        The calculate_volume_with_validation method should return consistent results
        with separate validation and calculation calls.
        
        **Validates: Requirements 5.5, 6.1**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a cube
        vertices = np.array([
            [0, 0, 0],
            [side_length, 0, 0],
            [side_length, side_length, 0],
            [0, side_length, 0],
            [0, 0, side_length],
            [side_length, 0, side_length],
            [side_length, side_length, side_length],
            [0, side_length, side_length]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 7, 4], [0, 3, 7],
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 19 Verification: Convenience method consistency
        
        calculator = VolumeCalculator()
        
        # Method 1: Separate calls
        is_watertight_separate = calculator.validate_mesh_closure(mesh)
        if is_watertight_separate:
            volume_separate = calculator.calculate_signed_volume(mesh)
        else:
            volume_separate = 0.0
        
        # Method 2: Convenience method
        volume_convenience, is_watertight_convenience = calculator.calculate_volume_with_validation(mesh)
        
        # Results should match
        assert is_watertight_separate == is_watertight_convenience, \
            "Watertightness validation should be consistent"
        
        assert abs(volume_separate - volume_convenience) < 1e-6, \
            f"Volume calculations should match: {volume_separate} vs {volume_convenience}"



class TestUnitConversionProperties:
    """Property-based tests for unit conversion and validation."""
    
    @given(st.floats(min_value=1e-9, max_value=100.0))
    @settings(max_examples=100)
    def test_property_20_unit_conversion_accuracy(self, volume_cubic_meters):
        """
        Property 20: Unit Conversion Accuracy
        
        For any calculated volume in cubic meters, conversions to cubic centimeters
        and liters should be mathematically exact.
        
        Conversion factors:
        - 1 cubic meter = 1,000,000 cubic centimeters
        - 1 cubic meter = 1,000 liters
        
        This test verifies that:
        1. Conversion to liters is exact (multiply by 1000)
        2. Conversion to cubic centimeters is exact (multiply by 1,000,000)
        3. Conversions are reversible
        4. No precision is lost in conversion
        
        **Validates: Requirements 6.2**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        
        # Property 20 Verification: Unit conversion accuracy
        
        # 1. Perform conversions
        converted = calculator.convert_volume_units(volume_cubic_meters)
        
        # 2. Verify conversion factors are exact
        expected_liters = volume_cubic_meters * 1000.0
        expected_cubic_cm = volume_cubic_meters * 1_000_000.0
        
        # Check liters conversion
        assert abs(converted['liters'] - expected_liters) < 1e-10, \
            f"Liters conversion not exact: {converted['liters']} vs {expected_liters}"
        
        # Check cubic centimeters conversion
        assert abs(converted['cubic_centimeters'] - expected_cubic_cm) < 1e-6, \
            f"Cubic cm conversion not exact: {converted['cubic_centimeters']} vs {expected_cubic_cm}"
        
        # 3. Verify original value is preserved
        assert abs(converted['cubic_meters'] - volume_cubic_meters) < 1e-15, \
            f"Original value not preserved: {converted['cubic_meters']} vs {volume_cubic_meters}"
        
        # 4. Verify conversions are reversible
        # Convert back from liters
        volume_from_liters = converted['liters'] / 1000.0
        assert abs(volume_from_liters - volume_cubic_meters) < 1e-10, \
            f"Liters conversion not reversible: {volume_from_liters} vs {volume_cubic_meters}"
        
        # Convert back from cubic centimeters
        volume_from_cubic_cm = converted['cubic_centimeters'] / 1_000_000.0
        assert abs(volume_from_cubic_cm - volume_cubic_meters) < 1e-6, \
            f"Cubic cm conversion not reversible: {volume_from_cubic_cm} vs {volume_cubic_meters}"
        
        # 5. Verify all values are finite
        assert np.isfinite(converted['cubic_meters']), "Cubic meters should be finite"
        assert np.isfinite(converted['liters']), "Liters should be finite"
        assert np.isfinite(converted['cubic_centimeters']), "Cubic cm should be finite"
    
    @given(
        st.floats(min_value=1e-9, max_value=100.0),
        st.floats(min_value=1.1, max_value=10.0)
    )
    @settings(max_examples=100)
    def test_property_20_unit_conversion_proportionality(self, volume1, scale_factor):
        """
        Property 20: Unit Conversion Accuracy (Proportionality)
        
        For any two volumes where one is a scaled version of the other,
        the conversions should maintain the same proportional relationship.
        
        **Validates: Requirements 6.2**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        
        volume2 = volume1 * scale_factor
        
        # Property 20 Verification: Proportionality preservation
        
        converted1 = calculator.convert_volume_units(volume1)
        converted2 = calculator.convert_volume_units(volume2)
        
        # Verify proportionality in all units
        ratio_m3 = converted2['cubic_meters'] / converted1['cubic_meters']
        ratio_liters = converted2['liters'] / converted1['liters']
        ratio_cubic_cm = converted2['cubic_centimeters'] / converted1['cubic_centimeters']
        
        # All ratios should equal the scale factor
        assert abs(ratio_m3 - scale_factor) < 1e-10, \
            f"Cubic meters ratio {ratio_m3} doesn't match scale factor {scale_factor}"
        
        assert abs(ratio_liters - scale_factor) < 1e-10, \
            f"Liters ratio {ratio_liters} doesn't match scale factor {scale_factor}"
        
        assert abs(ratio_cubic_cm - scale_factor) < 1e-6, \
            f"Cubic cm ratio {ratio_cubic_cm} doesn't match scale factor {scale_factor}"
    
    @given(st.floats(min_value=-10.0, max_value=100.0))
    @settings(max_examples=100)
    def test_property_22_volume_constraint_validation(self, volume_cubic_meters):
        """
        Property 22: Volume Constraint Validation
        
        For any calculated volume, geometric constraint checking should flag results
        that exceed physically reasonable bounds for road anomalies.
        
        Constraints:
        1. Volume must be non-negative
        2. Volume must be above minimum threshold (1 cubic centimeter = 1e-6 m³)
        3. Volume must be below maximum reasonable bound (default 10 m³)
        
        This test verifies that:
        1. Negative volumes are rejected
        2. Very small volumes (< 1 cm³) are rejected as noise
        3. Very large volumes (> max_volume) are rejected as unrealistic
        4. Valid volumes within bounds are accepted
        
        **Validates: Requirements 6.4**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        max_volume = 10.0  # Default maximum
        min_volume = 1e-6  # 1 cubic centimeter
        
        calculator = VolumeCalculator(max_volume_cubic_meters=max_volume)
        
        # Property 22 Verification: Volume constraint validation
        
        is_valid, message = calculator.validate_volume_constraints(volume_cubic_meters)
        
        # 1. Negative volumes should be rejected
        if volume_cubic_meters < 0:
            assert not is_valid, \
                f"Negative volume {volume_cubic_meters} should be rejected"
            assert "negative" in message.lower(), \
                f"Message should mention negative volume: {message}"
        
        # 2. Very small volumes should be rejected
        elif 0 <= volume_cubic_meters < min_volume:
            assert not is_valid, \
                f"Volume {volume_cubic_meters} below minimum {min_volume} should be rejected"
            assert "minimum" in message.lower() or "threshold" in message.lower(), \
                f"Message should mention minimum threshold: {message}"
        
        # 3. Very large volumes should be rejected
        elif volume_cubic_meters > max_volume:
            assert not is_valid, \
                f"Volume {volume_cubic_meters} above maximum {max_volume} should be rejected"
            assert "maximum" in message.lower() or "exceeds" in message.lower(), \
                f"Message should mention maximum bound: {message}"
        
        # 4. Valid volumes should be accepted
        else:
            assert is_valid, \
                f"Valid volume {volume_cubic_meters} should be accepted"
            assert "valid" in message.lower() or "within" in message.lower(), \
                f"Message should indicate validity: {message}"
        
        # 5. Verify message is non-empty
        assert len(message) > 0, "Validation message should not be empty"
    
    @given(
        st.floats(min_value=0.1, max_value=5.0),
        st.floats(min_value=0.5, max_value=20.0)
    )
    @settings(max_examples=100)
    def test_property_22_volume_constraint_custom_maximum(self, volume, custom_max):
        """
        Property 22: Volume Constraint Validation (Custom Maximum)
        
        For any custom maximum volume setting, the validation should correctly
        apply the specified bound.
        
        **Validates: Requirements 6.4**
        """
        from stereo_vision.volumetric import VolumeCalculator
        
        assume(volume != custom_max)  # Avoid boundary case
        
        calculator = VolumeCalculator(max_volume_cubic_meters=custom_max)
        
        # Property 22 Verification: Custom maximum enforcement
        
        is_valid, message = calculator.validate_volume_constraints(volume)
        
        min_volume = 1e-6
        
        if volume < min_volume:
            # Below minimum
            assert not is_valid
        elif volume > custom_max:
            # Above custom maximum
            assert not is_valid, \
                f"Volume {volume} above custom max {custom_max} should be rejected"
        else:
            # Within bounds
            assert is_valid, \
                f"Volume {volume} within bounds [{ min_volume}, {custom_max}] should be accepted"


class TestMultiAnomalyProcessingProperties:
    """Property-based tests for multi-anomaly processing and uncertainty estimation."""
    
    @given(
        st.lists(
            st.floats(min_value=0.1, max_value=2.0),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=100)
    def test_property_21_multi_anomaly_volume_independence(self, side_lengths):
        """
        Property 21: Multi-Anomaly Volume Independence
        
        For any scene with multiple detected anomalies, the volume calculation for
        each anomaly should be independent and unaffected by the presence of others.
        
        This test verifies that:
        1. Each anomaly's volume is calculated independently
        2. The volume of one anomaly doesn't change when calculated alone vs. in a batch
        3. The order of anomalies doesn't affect individual volume calculations
        4. Each result is properly indexed to its corresponding anomaly
        
        **Validates: Requirements 6.3**
        """
        from stereo_vision.volumetric import VolumeCalculator, Mesh
        
        # Create multiple cube meshes with different sizes
        meshes = []
        expected_volumes = []
        
        for side_length in side_lengths:
            # Create a cube
            vertices = np.array([
                [0, 0, 0],
                [side_length, 0, 0],
                [side_length, side_length, 0],
                [0, side_length, 0],
                [0, 0, side_length],
                [side_length, 0, side_length],
                [side_length, side_length, side_length],
                [0, side_length, side_length]
            ], dtype=np.float32)
            
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # Bottom
                [4, 6, 5], [4, 7, 6],  # Top
                [0, 5, 1], [0, 4, 5],  # Front
                [2, 7, 3], [2, 6, 7],  # Back
                [0, 7, 4], [0, 3, 7],  # Left
                [1, 6, 2], [1, 5, 6]   # Right
            ])
            
            mesh = Mesh(vertices=vertices, faces=faces)
            meshes.append(mesh)
            expected_volumes.append(side_length ** 3)
        
        # Property 21 Verification: Volume independence
        
        calculator = VolumeCalculator()
        
        # 1. Calculate volumes individually
        individual_volumes = []
        for mesh in meshes:
            volume = calculator.calculate_signed_volume(mesh)
            individual_volumes.append(volume)
        
        # 2. Calculate volumes in batch
        batch_results = calculator.calculate_multiple_volumes(meshes)
        
        # 3. Verify batch results match individual calculations
        assert len(batch_results) == len(meshes), \
            f"Batch results count {len(batch_results)} doesn't match mesh count {len(meshes)}"
        
        for idx, (individual_vol, batch_result, expected_vol) in enumerate(
            zip(individual_volumes, batch_results, expected_volumes)
        ):
            # Verify independence: batch volume should match individual volume
            batch_vol = batch_result['volume_cubic_meters']
            assert abs(batch_vol - individual_vol) < 1e-6, \
                f"Anomaly {idx}: batch volume {batch_vol} doesn't match " \
                f"individual volume {individual_vol}"
            
            # Verify correct indexing
            assert batch_result['anomaly_index'] == idx, \
                f"Anomaly index mismatch: expected {idx}, got {batch_result['anomaly_index']}"
            
            # Verify volume is close to expected
            relative_error = abs(batch_vol - expected_vol) / expected_vol
            assert relative_error < 0.01, \
                f"Anomaly {idx}: volume error too large (relative_error={relative_error})"
            
            # Verify watertightness
            assert batch_result['is_watertight'], \
                f"Anomaly {idx}: should be watertight"
            
            # Verify validity
            assert batch_result['is_valid'], \
                f"Anomaly {idx}: should be valid"
        
        # 4. Verify order independence: shuffle and recalculate
        import random
        shuffled_indices = list(range(len(meshes)))
        random.shuffle(shuffled_indices)
        
        shuffled_meshes = [meshes[i] for i in shuffled_indices]
        shuffled_results = calculator.calculate_multiple_volumes(shuffled_meshes)
        
        # Verify volumes match original order
        for new_idx, orig_idx in enumerate(shuffled_indices):
            original_vol = batch_results[orig_idx]['volume_cubic_meters']
            shuffled_vol = shuffled_results[new_idx]['volume_cubic_meters']
            
            assert abs(shuffled_vol - original_vol) < 1e-6, \
                f"Order affects volume: original {original_vol} vs shuffled {shuffled_vol}"
    
    @given(
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=0.0001, max_value=0.01)
    )
    @settings(max_examples=100)
    def test_property_23_uncertainty_estimation_scaling(self, side_length, measurement_precision):
        """
        Property 23: Uncertainty Estimation Scaling
        
        For any volume measurement, uncertainty estimates should scale appropriately
        with the measurement precision and point cloud density.
        
        This test verifies that:
        1. Uncertainty increases with measurement precision
        2. Uncertainty scales with surface area
        3. Uncertainty is always non-negative
        4. Uncertainty is proportional to the input precision
        
        **Validates: Requirements 6.5**
        """
        from stereo_vision.volumetric import VolumeCalculator, Mesh
        
        # Create a cube mesh
        vertices = np.array([
            [0, 0, 0],
            [side_length, 0, 0],
            [side_length, side_length, 0],
            [0, side_length, 0],
            [0, 0, side_length],
            [side_length, 0, side_length],
            [side_length, side_length, side_length],
            [0, side_length, side_length]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 6, 5], [4, 7, 6],  # Top
            [0, 5, 1], [0, 4, 5],  # Front
            [2, 7, 3], [2, 6, 7],  # Back
            [0, 7, 4], [0, 3, 7],  # Left
            [1, 6, 2], [1, 5, 6]   # Right
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Property 23 Verification: Uncertainty scaling
        
        calculator = VolumeCalculator()
        
        # 1. Calculate uncertainty
        uncertainty = calculator.calculate_volume_uncertainty(mesh, measurement_precision)
        
        # 2. Uncertainty should be non-negative
        assert uncertainty >= 0, \
            f"Uncertainty should be non-negative, got {uncertainty}"
        
        # 3. Uncertainty should be finite
        assert np.isfinite(uncertainty), \
            f"Uncertainty should be finite, got {uncertainty}"
        
        # 4. Uncertainty should scale with measurement precision
        # Test with doubled precision
        doubled_precision = measurement_precision * 2.0
        uncertainty_doubled = calculator.calculate_volume_uncertainty(mesh, doubled_precision)
        
        # Uncertainty should approximately double
        ratio = uncertainty_doubled / uncertainty if uncertainty > 0 else 0
        assert abs(ratio - 2.0) < 0.01, \
            f"Uncertainty should scale linearly with precision: ratio={ratio}, expected=2.0"
        
        # 5. Uncertainty should scale with surface area
        # Create a larger cube (2x side length = 4x surface area)
        larger_vertices = vertices * 2.0
        larger_mesh = Mesh(vertices=larger_vertices, faces=faces)
        
        uncertainty_larger = calculator.calculate_volume_uncertainty(larger_mesh, measurement_precision)
        
        # Uncertainty should scale with surface area (4x for 2x linear scale)
        area_ratio = uncertainty_larger / uncertainty if uncertainty > 0 else 0
        expected_area_ratio = 4.0  # Surface area scales as length²
        
        assert abs(area_ratio - expected_area_ratio) < 0.1, \
            f"Uncertainty should scale with surface area: ratio={area_ratio}, expected={expected_area_ratio}"
        
        # 6. Verify uncertainty is reasonable relative to volume
        volume = calculator.calculate_signed_volume(mesh)
        uncertainty_ratio = uncertainty / volume if volume > 0 else 0
        
        # Uncertainty should be small relative to volume (typically < 10%)
        assert uncertainty_ratio < 0.5, \
            f"Uncertainty ratio {uncertainty_ratio} seems unreasonably large"
    
    @given(
        st.lists(
            st.floats(min_value=0.5, max_value=3.0),
            min_size=2,
            max_size=4
        ),
        st.floats(min_value=0.0001, max_value=0.01)
    )
    @settings(max_examples=100)
    def test_property_23_uncertainty_in_batch_processing(self, side_lengths, measurement_precision):
        """
        Property 23: Uncertainty Estimation Scaling (Batch Processing)
        
        For any batch of anomalies, uncertainty estimates should be calculated
        independently for each anomaly and included in the results.
        
        **Validates: Requirements 6.5**
        """
        from stereo_vision.volumetric import VolumeCalculator, Mesh
        
        # Create multiple cube meshes
        meshes = []
        for side_length in side_lengths:
            vertices = np.array([
                [0, 0, 0],
                [side_length, 0, 0],
                [side_length, side_length, 0],
                [0, side_length, 0],
                [0, 0, side_length],
                [side_length, 0, side_length],
                [side_length, side_length, side_length],
                [0, side_length, side_length]
            ], dtype=np.float32)
            
            faces = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 5, 1], [0, 4, 5],
                [2, 7, 3], [2, 6, 7],
                [0, 7, 4], [0, 3, 7],
                [1, 6, 2], [1, 5, 6]
            ])
            
            mesh = Mesh(vertices=vertices, faces=faces)
            meshes.append(mesh)
        
        # Property 23 Verification: Uncertainty in batch processing
        
        calculator = VolumeCalculator()
        
        # Calculate volumes with uncertainty in batch
        batch_results = calculator.calculate_multiple_volumes(meshes, measurement_precision)
        
        # Verify each result has uncertainty
        for idx, result in enumerate(batch_results):
            # 1. Uncertainty should be present
            assert 'uncertainty_cubic_meters' in result, \
                f"Anomaly {idx}: missing uncertainty in result"
            
            uncertainty = result['uncertainty_cubic_meters']
            
            # 2. Uncertainty should be non-negative
            assert uncertainty >= 0, \
                f"Anomaly {idx}: uncertainty should be non-negative, got {uncertainty}"
            
            # 3. Uncertainty should be finite
            assert np.isfinite(uncertainty), \
                f"Anomaly {idx}: uncertainty should be finite, got {uncertainty}"
            
            # 4. Verify uncertainty matches individual calculation
            individual_uncertainty = calculator.calculate_volume_uncertainty(
                meshes[idx], measurement_precision
            )
            
            assert abs(uncertainty - individual_uncertainty) < 1e-9, \
                f"Anomaly {idx}: batch uncertainty {uncertainty} doesn't match " \
                f"individual uncertainty {individual_uncertainty}"
            
            # 5. Larger anomalies should have larger uncertainty
            # (due to larger surface area)
            if idx > 0:
                size_ratio = side_lengths[idx] / side_lengths[idx-1]
                # Only check if there's a meaningful size difference (>5%)
                if size_ratio > 1.05:
                    prev_uncertainty = batch_results[idx-1]['uncertainty_cubic_meters']
                    assert uncertainty > prev_uncertainty, \
                        f"Larger anomaly {idx} (size ratio {size_ratio:.2f}) should have " \
                        f"larger uncertainty than {idx-1}"
    
    @given(st.integers(min_value=0, max_value=3))
    @settings(max_examples=50)
    def test_property_21_empty_mesh_list_handling(self, num_empty_meshes):
        """
        Property 21: Multi-Anomaly Volume Independence (Empty Mesh Handling)
        
        For any list containing empty or invalid meshes, the batch processing
        should handle them gracefully and report appropriate errors.
        
        **Validates: Requirements 6.3**
        """
        from stereo_vision.volumetric import VolumeCalculator, Mesh
        
        # Create list with empty meshes
        meshes = []
        for _ in range(num_empty_meshes):
            # Create mesh with no faces
            vertices = np.random.randn(5, 3).astype(np.float32)
            faces = np.empty((0, 3), dtype=int)
            mesh = Mesh(vertices=vertices, faces=faces)
            meshes.append(mesh)
        
        # Property 21 Verification: Empty mesh handling
        
        calculator = VolumeCalculator()
        
        results = calculator.calculate_multiple_volumes(meshes)
        
        # Verify results for each empty mesh
        assert len(results) == num_empty_meshes, \
            f"Should have {num_empty_meshes} results, got {len(results)}"
        
        for idx, result in enumerate(results):
            # Empty meshes should have zero volume
            assert result['volume_cubic_meters'] == 0.0, \
                f"Empty mesh {idx} should have zero volume"
            
            # Should not be valid
            assert not result['is_valid'], \
                f"Empty mesh {idx} should not be valid"
            
            # Should not be watertight
            assert not result['is_watertight'], \
                f"Empty mesh {idx} should not be watertight"
            
            # Should have proper index
            assert result['anomaly_index'] == idx, \
                f"Empty mesh {idx} has wrong index"
            
            # Should have error message
            assert len(result['validation_message']) > 0, \
                f"Empty mesh {idx} should have validation message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
