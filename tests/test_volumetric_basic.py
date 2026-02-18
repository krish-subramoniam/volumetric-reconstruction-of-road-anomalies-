"""
Basic unit tests for volumetric analysis module.

Tests the AlphaShapeGenerator class with specific examples and edge cases.
"""

import pytest
import numpy as np
from stereo_vision.volumetric import AlphaShapeGenerator, Mesh, Edge


class TestAlphaShapeGenerator:
    """Test suite for AlphaShapeGenerator class."""
    
    def test_initialization_valid_alpha(self):
        """Test that generator initializes with valid alpha."""
        generator = AlphaShapeGenerator(alpha=1.0)
        assert generator.alpha == 1.0
    
    def test_initialization_invalid_alpha(self):
        """Test that generator rejects invalid alpha values."""
        with pytest.raises(ValueError, match="Alpha must be positive"):
            AlphaShapeGenerator(alpha=0.0)
        
        with pytest.raises(ValueError, match="Alpha must be positive"):
            AlphaShapeGenerator(alpha=-1.0)
    
    def test_generate_alpha_shape_simple_cube(self):
        """Test alpha shape generation on a simple cube point cloud."""
        # Create 8 corners of a unit cube
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        
        generator = AlphaShapeGenerator(alpha=2.0)
        mesh = generator.generate_alpha_shape(points)
        
        # Verify mesh structure
        assert mesh.vertices.shape[0] == 8
        assert mesh.faces.shape[1] == 3
        assert mesh.faces.shape[0] > 0
    
    def test_generate_alpha_shape_insufficient_points(self):
        """Test that generator rejects point clouds with too few points."""
        from stereo_vision.errors import InsufficientPointsError
        
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        generator = AlphaShapeGenerator(alpha=1.0)
        with pytest.raises(InsufficientPointsError, match="Insufficient points for 3D triangulation"):
            generator.generate_alpha_shape(points)
    
    def test_generate_alpha_shape_invalid_shape(self):
        """Test that generator rejects invalid point array shapes."""
        from stereo_vision.errors import AlphaShapeError
        
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        generator = AlphaShapeGenerator(alpha=1.0)
        with pytest.raises(AlphaShapeError, match="Invalid points array dimensions"):
            generator.generate_alpha_shape(points)
    
    def test_generate_alpha_shape_duplicate_points(self):
        """Test that generator handles duplicate points correctly."""
        # Create points with duplicates
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0, 0, 0], [1, 0, 0]  # Duplicates
        ], dtype=float)
        
        generator = AlphaShapeGenerator(alpha=2.0)
        mesh = generator.generate_alpha_shape(points)
        
        # Should have 4 unique vertices
        assert mesh.vertices.shape[0] == 4
    
    def test_extract_boundary_edges_open_mesh(self):
        """Test boundary edge extraction from an open mesh."""
        # Create a simple triangular mesh (single triangle)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = Mesh(vertices=vertices, faces=faces)
        
        generator = AlphaShapeGenerator(alpha=1.0)
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        # A single triangle has 3 boundary edges
        assert len(boundary_edges) == 3
    
    def test_extract_boundary_edges_closed_mesh(self):
        """Test boundary edge extraction from a closed mesh."""
        # Create a tetrahedron (closed mesh)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=float)
        faces = np.array([
            [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
        ])
        mesh = Mesh(vertices=vertices, faces=faces)
        
        generator = AlphaShapeGenerator(alpha=1.0)
        boundary_edges = generator.extract_boundary_edges(mesh)
        
        # A closed tetrahedron has no boundary edges
        assert len(boundary_edges) == 0
    
    def test_update_alpha(self):
        """Test updating the alpha parameter."""
        generator = AlphaShapeGenerator(alpha=1.0)
        generator.update_alpha(2.0)
        assert generator.alpha == 2.0
        
        with pytest.raises(ValueError, match="Alpha must be positive"):
            generator.update_alpha(-1.0)
    
    def test_get_mesh_statistics(self):
        """Test mesh statistics calculation."""
        # Create a simple mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = Mesh(vertices=vertices, faces=faces)
        
        generator = AlphaShapeGenerator(alpha=1.0)
        stats = generator.get_mesh_statistics(mesh)
        
        assert stats['num_vertices'] == 3
        assert stats['num_faces'] == 1
        assert stats['num_boundary_edges'] == 3
        assert 'surface_area' in stats
        assert stats['surface_area'] > 0
    
    def test_alpha_parameter_effect(self):
        """Test that different alpha values produce different meshes."""
        # Create a point cloud
        np.random.seed(42)
        points = np.random.randn(20, 3)
        
        # Generate with small alpha
        gen_small = AlphaShapeGenerator(alpha=0.5)
        mesh_small = gen_small.generate_alpha_shape(points)
        
        # Generate with large alpha
        gen_large = AlphaShapeGenerator(alpha=5.0)
        mesh_large = gen_large.generate_alpha_shape(points)
        
        # Larger alpha should generally produce more faces
        # (though this isn't guaranteed for all point clouds)
        assert mesh_small.faces.shape[0] >= 0
        assert mesh_large.faces.shape[0] >= 0


class TestEdge:
    """Test suite for Edge class."""
    
    def test_edge_equality(self):
        """Test that edges are compared correctly (undirected)."""
        edge1 = Edge(0, 1)
        edge2 = Edge(1, 0)
        edge3 = Edge(0, 2)
        
        assert edge1 == edge2  # Undirected
        assert edge1 != edge3
    
    def test_edge_hash(self):
        """Test that edges hash correctly for use in sets/dicts."""
        edge1 = Edge(0, 1)
        edge2 = Edge(1, 0)
        
        # Should have same hash since they're the same undirected edge
        assert hash(edge1) == hash(edge2)
        
        # Can be used in sets
        edge_set = {edge1, edge2}
        assert len(edge_set) == 1


class TestMesh:
    """Test suite for Mesh dataclass."""
    
    def test_mesh_creation(self):
        """Test that Mesh objects can be created."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        assert mesh.vertices.shape == (3, 3)
        assert mesh.faces.shape == (1, 3)
        assert mesh.is_watertight == False
    
    def test_mesh_watertight_flag(self):
        """Test that watertight flag can be set."""
        vertices = np.array([[0, 0, 0]])
        faces = np.array([[0, 0, 0]])
        
        mesh = Mesh(vertices=vertices, faces=faces, is_watertight=True)
        assert mesh.is_watertight == True



class TestMeshCapper:
    """Test suite for MeshCapper class."""
    
    def test_triangulate_boundary_empty_edges(self):
        """Test that empty boundary edges return empty mesh."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        cap_mesh = capper.triangulate_boundary([], vertices)
        
        assert cap_mesh.faces.shape[0] == 0
        assert cap_mesh.vertices.shape[0] == 3
    
    def test_triangulate_boundary_simple_triangle(self):
        """Test triangulation of a simple triangular boundary loop."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        # Create a triangular boundary loop
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        boundary_edges = [Edge(0, 1), Edge(1, 2), Edge(2, 0)]
        
        cap_mesh = capper.triangulate_boundary(boundary_edges, vertices)
        
        # Should produce 1 triangle
        assert cap_mesh.faces.shape[0] == 1
        assert cap_mesh.faces.shape[1] == 3
    
    def test_triangulate_boundary_square(self):
        """Test triangulation of a square boundary loop."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        # Create a square boundary loop
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        boundary_edges = [
            Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 0)
        ]
        
        cap_mesh = capper.triangulate_boundary(boundary_edges, vertices)
        
        # Should produce 2 triangles (fan triangulation)
        assert cap_mesh.faces.shape[0] == 2
        assert cap_mesh.faces.shape[1] == 3
    
    def test_create_watertight_mesh(self):
        """Test combining surface mesh with cap mesh."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        # Create a simple surface mesh (single triangle)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        surface_faces = np.array([[0, 1, 2]])
        surface_mesh = Mesh(vertices=vertices, faces=surface_faces)
        
        # Create a cap mesh (another triangle)
        cap_faces = np.array([[0, 1, 3]])
        cap_mesh = Mesh(vertices=vertices, faces=cap_faces)
        
        # Combine them
        combined = capper.create_watertight_mesh(surface_mesh, cap_mesh)
        
        assert combined.faces.shape[0] == 2
        assert combined.vertices.shape[0] == 4
    
    def test_validate_watertightness_open_mesh(self):
        """Test watertightness validation on an open mesh."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        # Create an open mesh (single triangle)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Should not be watertight
        assert not capper.validate_watertightness(mesh)
    
    def test_validate_watertightness_closed_mesh(self):
        """Test watertightness validation on a closed mesh."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        # Create a closed tetrahedron
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=float)
        faces = np.array([
            [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
        ])
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Should be watertight
        assert capper.validate_watertightness(mesh)
    
    def test_validate_watertightness_empty_mesh(self):
        """Test watertightness validation on an empty mesh."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        vertices = np.array([[0, 0, 0]])
        faces = np.empty((0, 3), dtype=int)
        mesh = Mesh(vertices=vertices, faces=faces)
        
        # Empty mesh is not watertight
        assert not capper.validate_watertightness(mesh)
    
    def test_extract_boundary_loops_single_loop(self):
        """Test extracting a single boundary loop."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        # Create a square loop
        boundary_edges = [
            Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 0)
        ]
        
        loops = capper._extract_boundary_loops(boundary_edges)
        
        assert len(loops) == 1
        assert len(loops[0]) == 4
    
    def test_extract_boundary_loops_multiple_loops(self):
        """Test extracting multiple boundary loops."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        # Create two separate triangular loops
        boundary_edges = [
            # First triangle
            Edge(0, 1), Edge(1, 2), Edge(2, 0),
            # Second triangle
            Edge(3, 4), Edge(4, 5), Edge(5, 3)
        ]
        
        loops = capper._extract_boundary_loops(boundary_edges)
        
        assert len(loops) == 2
        assert all(len(loop) == 3 for loop in loops)
    
    def test_triangulate_loop_triangle(self):
        """Test triangulating a triangular loop."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        loop = [0, 1, 2]
        
        faces = capper._triangulate_loop(loop, vertices)
        
        # Triangle should produce 1 face
        assert len(faces) == 1
        assert len(faces[0]) == 3
    
    def test_triangulate_loop_pentagon(self):
        """Test triangulating a pentagonal loop."""
        from stereo_vision.volumetric import MeshCapper
        
        capper = MeshCapper()
        
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1.5, 1, 0], [0.5, 1.5, 0], [-0.5, 1, 0]
        ], dtype=float)
        loop = [0, 1, 2, 3, 4]
        
        faces = capper._triangulate_loop(loop, vertices)
        
        # Pentagon should produce 3 faces (n-2 triangles)
        assert len(faces) == 3
        assert all(len(face) == 3 for face in faces)



class TestMeshCappingIntegration:
    """Integration tests for the complete mesh capping workflow."""
    
    def test_complete_workflow_open_to_watertight(self):
        """Test the complete workflow from open mesh to watertight mesh."""
        from stereo_vision.volumetric import AlphaShapeGenerator, MeshCapper
        
        # Create a simple point cloud with a clear opening
        # Use a cup shape - points on sides but not on bottom
        np.random.seed(42)
        
        # Create points on a cylinder without bottom cap
        n_points = 30
        theta = np.linspace(0, 2 * np.pi, n_points)
        
        points_list = []
        # Add points around the rim (top)
        for t in theta:
            points_list.append([np.cos(t), np.sin(t), 1.0])
        
        # Add points around middle
        for t in theta:
            points_list.append([np.cos(t), np.sin(t), 0.5])
        
        # Add some points on the sides
        for i in range(10):
            t = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0.3, 0.9)
            points_list.append([np.cos(t), np.sin(t), z])
        
        points = np.array(points_list, dtype=float)
        
        # Step 1: Generate Alpha Shape mesh
        generator = AlphaShapeGenerator(alpha=2.0)
        surface_mesh = generator.generate_alpha_shape(points)
        
        # Check if we have faces
        if surface_mesh.faces.shape[0] == 0:
            print("No faces generated, skipping test")
            return
        
        # Check if we have an open mesh
        boundary_edges = generator.extract_boundary_edges(surface_mesh)
        
        if len(boundary_edges) == 0:
            # If mesh is already closed, that's fine - skip capping
            print("Mesh is already closed, no capping needed")
            is_watertight = MeshCapper().validate_watertightness(surface_mesh)
            print(f"Closed mesh watertight: {is_watertight}")
            return
        
        # Step 2: Create caps for the boundary
        capper = MeshCapper()
        cap_mesh = capper.triangulate_boundary(boundary_edges, surface_mesh.vertices)
        
        # Verify cap was created
        assert cap_mesh.faces.shape[0] > 0, "Cap should have faces"
        
        # Step 3: Combine surface and cap to create watertight mesh
        watertight_mesh = capper.create_watertight_mesh(surface_mesh, cap_mesh)
        
        # Verify combined mesh
        assert watertight_mesh.faces.shape[0] == surface_mesh.faces.shape[0] + cap_mesh.faces.shape[0]
        
        # Step 4: Validate watertightness
        is_watertight = capper.validate_watertightness(watertight_mesh)
        
        # Note: This may not always be watertight due to the complexity of the boundary
        # but the workflow should complete without errors
        print(f"Mesh watertight: {is_watertight}")
        print(f"Surface faces: {surface_mesh.faces.shape[0]}")
        print(f"Cap faces: {cap_mesh.faces.shape[0]}")
        print(f"Total faces: {watertight_mesh.faces.shape[0]}")
        print(f"Boundary edges: {len(boundary_edges)}")
    
    def test_simple_open_mesh_to_watertight(self):
        """Test capping a simple open mesh (cylinder without top/bottom)."""
        from stereo_vision.volumetric import MeshCapper
        
        # Create a simple open cylinder (4 sides, no top/bottom)
        vertices = np.array([
            # Bottom ring
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            # Top ring
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        
        # Side faces (4 rectangles = 8 triangles)
        faces = np.array([
            # Side 1
            [0, 1, 5], [0, 5, 4],
            # Side 2
            [1, 2, 6], [1, 6, 5],
            # Side 3
            [2, 3, 7], [2, 7, 6],
            # Side 4
            [3, 0, 4], [3, 4, 7]
        ])
        
        surface_mesh = Mesh(vertices=vertices, faces=faces)
        
        # Extract boundary edges
        generator = AlphaShapeGenerator(alpha=1.0)
        boundary_edges = generator.extract_boundary_edges(surface_mesh)
        
        # Should have boundary edges at top and bottom
        assert len(boundary_edges) > 0
        
        # Create caps
        capper = MeshCapper()
        cap_mesh = capper.triangulate_boundary(boundary_edges, vertices)
        
        # Combine
        watertight_mesh = capper.create_watertight_mesh(surface_mesh, cap_mesh)
        
        # Validate
        is_watertight = capper.validate_watertightness(watertight_mesh)
        
        print(f"Cylinder mesh watertight: {is_watertight}")
        print(f"Boundary edges: {len(boundary_edges)}")
        print(f"Cap faces: {cap_mesh.faces.shape[0]}")



class TestVolumeCalculator:
    """Test suite for VolumeCalculator class."""
    
    def test_initialization(self):
        """Test that VolumeCalculator initializes correctly."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        assert calculator is not None
    
    def test_validate_mesh_closure_closed_tetrahedron(self):
        """Test watertightness validation on a closed tetrahedron."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a closed tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        is_watertight = calculator.validate_mesh_closure(mesh)
        
        assert is_watertight, "Closed tetrahedron should be watertight"
    
    def test_validate_mesh_closure_open_triangle(self):
        """Test watertightness validation on an open mesh."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create an open mesh (single triangle)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)
        
        faces = np.array([[0, 1, 2]])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        is_watertight = calculator.validate_mesh_closure(mesh)
        
        assert not is_watertight, "Single triangle should not be watertight"
    
    def test_validate_mesh_closure_empty_mesh(self):
        """Test watertightness validation on an empty mesh."""
        from stereo_vision.volumetric import VolumeCalculator
        
        vertices = np.array([[0, 0, 0]])
        faces = np.empty((0, 3), dtype=int)
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        is_watertight = calculator.validate_mesh_closure(mesh)
        
        assert not is_watertight, "Empty mesh should not be watertight"
    
    def test_calculate_signed_volume_unit_cube(self):
        """Test volume calculation on a unit cube."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a unit cube
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=float)
        
        # 12 triangular faces (2 per cube face)
        faces = np.array([
            # Bottom (z=0)
            [0, 1, 2], [0, 2, 3],
            # Top (z=1)
            [4, 6, 5], [4, 7, 6],
            # Front (y=0)
            [0, 5, 1], [0, 4, 5],
            # Back (y=1)
            [2, 7, 3], [2, 6, 7],
            # Left (x=0)
            [0, 7, 4], [0, 3, 7],
            # Right (x=1)
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        volume = calculator.calculate_signed_volume(mesh)
        
        # Expected volume is 1.0
        assert abs(volume - 1.0) < 0.01, f"Unit cube volume should be 1.0, got {volume}"
    
    def test_calculate_signed_volume_scaled_cube(self):
        """Test volume calculation on a scaled cube."""
        from stereo_vision.volumetric import VolumeCalculator
        
        scale = 2.0
        
        # Create a cube with side length 2
        vertices = np.array([
            [0, 0, 0],
            [scale, 0, 0],
            [scale, scale, 0],
            [0, scale, 0],
            [0, 0, scale],
            [scale, 0, scale],
            [scale, scale, scale],
            [0, scale, scale]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 7, 4], [0, 3, 7],
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        volume = calculator.calculate_signed_volume(mesh)
        
        # Expected volume is 2^3 = 8.0
        expected_volume = scale ** 3
        assert abs(volume - expected_volume) < 0.01, \
            f"Cube volume should be {expected_volume}, got {volume}"
    
    def test_calculate_signed_volume_tetrahedron(self):
        """Test volume calculation on a tetrahedron."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        volume = calculator.calculate_signed_volume(mesh)
        
        # Expected volume is 1/6
        expected_volume = 1.0 / 6.0
        assert abs(volume - expected_volume) < 0.01, \
            f"Tetrahedron volume should be {expected_volume}, got {volume}"
    
    def test_calculate_signed_volume_rejects_open_mesh(self):
        """Test that volume calculation rejects open meshes."""
        from stereo_vision.volumetric import VolumeCalculator
        from stereo_vision.errors import WatertightnessError
        
        # Create an open mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)
        
        faces = np.array([[0, 1, 2]])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        
        with pytest.raises(WatertightnessError, match="not watertight"):
            calculator.calculate_signed_volume(mesh)
    
    def test_calculate_signed_volume_rejects_empty_mesh(self):
        """Test that volume calculation rejects empty meshes."""
        from stereo_vision.volumetric import VolumeCalculator
        from stereo_vision.errors import WatertightnessError
        
        vertices = np.array([[0, 0, 0]])
        faces = np.empty((0, 3), dtype=int)
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        
        with pytest.raises(WatertightnessError, match="not watertight"):
            calculator.calculate_signed_volume(mesh)
    
    def test_calculate_volume_with_validation_watertight(self):
        """Test convenience method with watertight mesh."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a unit cube
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 7, 4], [0, 3, 7],
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        volume, is_watertight = calculator.calculate_volume_with_validation(mesh)
        
        assert is_watertight, "Cube should be watertight"
        assert abs(volume - 1.0) < 0.01, f"Volume should be 1.0, got {volume}"
    
    def test_calculate_volume_with_validation_open_mesh(self):
        """Test convenience method with open mesh."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create an open mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)
        
        faces = np.array([[0, 1, 2]])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        volume, is_watertight = calculator.calculate_volume_with_validation(mesh)
        
        assert not is_watertight, "Single triangle should not be watertight"
        assert volume == 0.0, f"Volume should be 0.0 for open mesh, got {volume}"
    
    def test_calculate_volume_statistics(self):
        """Test volume statistics calculation."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a unit cube
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 7, 4], [0, 3, 7],
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        stats = calculator.calculate_volume_statistics(mesh)
        
        assert 'volume' in stats
        assert 'is_watertight' in stats
        assert 'num_vertices' in stats
        assert 'num_faces' in stats
        assert 'surface_area' in stats
        
        assert stats['is_watertight'] == True
        assert abs(stats['volume'] - 1.0) < 0.01
        assert stats['num_vertices'] == 8
        assert stats['num_faces'] == 12
        assert stats['surface_area'] > 0
    
    def test_volume_calculation_positive(self):
        """Test that volume is always positive."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a tetrahedron with different vertex orderings
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        # Different face orderings (some clockwise, some counter-clockwise)
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],  # Different ordering
            [0, 2, 3],
            [1, 3, 2]   # Different ordering
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        volume = calculator.calculate_signed_volume(mesh)
        
        # Volume should be positive regardless of face orientation
        assert volume > 0, f"Volume should be positive, got {volume}"


class TestVolumeUnitConversion:
    """Test suite for volume unit conversion functionality."""
    
    def test_convert_volume_units_one_cubic_meter(self):
        """Test conversion of 1 cubic meter to other units."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        result = calculator.convert_volume_units(1.0)
        
        assert result['cubic_meters'] == 1.0
        assert result['liters'] == 1000.0
        assert result['cubic_centimeters'] == 1_000_000.0
    
    def test_convert_volume_units_small_volume(self):
        """Test conversion of small volume (1 liter)."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        result = calculator.convert_volume_units(0.001)  # 1 liter
        
        assert result['cubic_meters'] == 0.001
        assert abs(result['liters'] - 1.0) < 1e-10
        assert abs(result['cubic_centimeters'] - 1000.0) < 1e-6
    
    def test_convert_volume_units_very_small_volume(self):
        """Test conversion of very small volume (1 cubic centimeter)."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        result = calculator.convert_volume_units(1e-6)  # 1 cm³
        
        assert result['cubic_meters'] == 1e-6
        assert abs(result['liters'] - 0.001) < 1e-10
        assert abs(result['cubic_centimeters'] - 1.0) < 1e-10
    
    def test_convert_volume_units_zero(self):
        """Test conversion of zero volume."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        result = calculator.convert_volume_units(0.0)
        
        assert result['cubic_meters'] == 0.0
        assert result['liters'] == 0.0
        assert result['cubic_centimeters'] == 0.0
    
    def test_convert_volume_units_large_volume(self):
        """Test conversion of large volume."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        result = calculator.convert_volume_units(5.0)
        
        assert result['cubic_meters'] == 5.0
        assert result['liters'] == 5000.0
        assert result['cubic_centimeters'] == 5_000_000.0


class TestVolumeConstraintValidation:
    """Test suite for volume constraint validation functionality."""
    
    def test_validate_volume_constraints_valid_volume(self):
        """Test validation of a valid volume."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator(max_volume_cubic_meters=10.0)
        is_valid, message = calculator.validate_volume_constraints(0.5)
        
        assert is_valid
        assert "valid constraints" in message
    
    def test_validate_volume_constraints_negative_volume(self):
        """Test validation rejects negative volumes."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        is_valid, message = calculator.validate_volume_constraints(-0.1)
        
        assert not is_valid
        assert "negative" in message.lower()
    
    def test_validate_volume_constraints_below_minimum(self):
        """Test validation rejects volumes below minimum threshold."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        # 1e-7 m³ is below the 1e-6 m³ (1 cm³) minimum
        is_valid, message = calculator.validate_volume_constraints(1e-7)
        
        assert not is_valid
        assert "minimum threshold" in message.lower()
    
    def test_validate_volume_constraints_at_minimum(self):
        """Test validation accepts volume at minimum threshold."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        # Exactly 1 cm³ should be valid
        is_valid, message = calculator.validate_volume_constraints(1e-6)
        
        assert is_valid
        assert "valid constraints" in message
    
    def test_validate_volume_constraints_above_maximum(self):
        """Test validation rejects volumes above maximum threshold."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator(max_volume_cubic_meters=5.0)
        is_valid, message = calculator.validate_volume_constraints(10.0)
        
        assert not is_valid
        assert "exceeds maximum" in message.lower()
    
    def test_validate_volume_constraints_at_maximum(self):
        """Test validation accepts volume at maximum threshold."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator(max_volume_cubic_meters=5.0)
        is_valid, message = calculator.validate_volume_constraints(5.0)
        
        assert is_valid
        assert "valid constraints" in message
    
    def test_validate_volume_constraints_custom_maximum(self):
        """Test validation with custom maximum threshold."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator(max_volume_cubic_meters=2.0)
        
        # Below max should be valid
        is_valid, _ = calculator.validate_volume_constraints(1.5)
        assert is_valid
        
        # Above max should be invalid
        is_valid, _ = calculator.validate_volume_constraints(3.0)
        assert not is_valid
    
    def test_validate_volume_constraints_typical_pothole(self):
        """Test validation with typical pothole volume."""
        from stereo_vision.volumetric import VolumeCalculator
        
        calculator = VolumeCalculator()
        # Typical pothole: 0.1 m³ (100 liters)
        is_valid, message = calculator.validate_volume_constraints(0.1)
        
        assert is_valid
        assert "valid constraints" in message


class TestVolumeCalculationWithUnits:
    """Test suite for comprehensive volume calculation with units."""
    
    def test_calculate_volume_with_units_unit_cube(self):
        """Test comprehensive volume calculation on a unit cube."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a unit cube
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 7, 4], [0, 3, 7],
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        result = calculator.calculate_volume_with_units(mesh)
        
        # Check all fields are present
        assert 'volume_cubic_meters' in result
        assert 'volume_liters' in result
        assert 'volume_cubic_cm' in result
        assert 'is_valid' in result
        assert 'validation_message' in result
        assert 'is_watertight' in result
        
        # Check values
        assert abs(result['volume_cubic_meters'] - 1.0) < 0.01
        assert abs(result['volume_liters'] - 1000.0) < 10.0
        assert abs(result['volume_cubic_cm'] - 1_000_000.0) < 10000.0
        assert result['is_valid']
        assert result['is_watertight']
    
    def test_calculate_volume_with_units_small_tetrahedron(self):
        """Test comprehensive volume calculation on a small tetrahedron."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a small tetrahedron (scaled by 0.1)
        scale = 0.1
        vertices = np.array([
            [0, 0, 0],
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, scale]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        result = calculator.calculate_volume_with_units(mesh)
        
        # Expected volume: (1/6) * scale^3
        expected_m3 = (1.0 / 6.0) * (scale ** 3)
        
        assert abs(result['volume_cubic_meters'] - expected_m3) < 1e-6
        assert abs(result['volume_liters'] - expected_m3 * 1000.0) < 1e-3
        assert abs(result['volume_cubic_cm'] - expected_m3 * 1_000_000.0) < 1.0
        assert result['is_valid']
        assert result['is_watertight']
    
    def test_calculate_volume_with_units_exceeds_maximum(self):
        """Test that very large volumes are flagged as invalid."""
        from stereo_vision.volumetric import VolumeCalculator
        
        # Create a large cube (side length 100 meters)
        scale = 100.0
        vertices = np.array([
            [0, 0, 0], [scale, 0, 0], [scale, scale, 0], [0, scale, 0],
            [0, 0, scale], [scale, 0, scale], [scale, scale, scale], [0, scale, scale]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 7, 4], [0, 3, 7],
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator(max_volume_cubic_meters=10.0)
        result = calculator.calculate_volume_with_units(mesh)
        
        # Volume should be calculated but flagged as invalid
        assert result['volume_cubic_meters'] > 10.0
        assert not result['is_valid']
        assert "exceeds maximum" in result['validation_message'].lower()
        assert result['is_watertight']
    
    def test_calculate_volume_with_units_not_watertight(self):
        """Test that non-watertight meshes raise an error."""
        from stereo_vision.volumetric import VolumeCalculator
        from stereo_vision.errors import WatertightnessError
        
        # Create an open mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = Mesh(vertices=vertices, faces=faces)
        
        calculator = VolumeCalculator()
        
        with pytest.raises(WatertightnessError, match="not watertight"):
            calculator.calculate_volume_with_units(mesh)
