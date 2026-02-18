"""
Volumetric Analysis Module for Advanced Stereo Vision Pipeline

This module implements advanced mesh generation and volume calculation algorithms
for road anomaly quantification. It includes Alpha Shape mesh generation for
tight-fitting concave hulls and watertight mesh processing for accurate volume
computation.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy.spatial import Delaunay
import trimesh

from stereo_vision.logging_config import get_logger, PerformanceTimer
from stereo_vision.errors import (
    AlphaShapeError, MeshCappingError, WatertightnessError,
    VolumeCalculationError, VolumeConstraintError, InsufficientPointsError
)

# Initialize logger
logger = get_logger(__name__)


@dataclass
class Mesh:
    """
    Represents a 3D triangular mesh.
    
    Attributes:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of triangle vertex indices
        is_watertight: Whether the mesh forms a closed manifold
    """
    vertices: np.ndarray
    faces: np.ndarray
    is_watertight: bool = False


@dataclass
class Edge:
    """
    Represents an edge in a mesh.
    
    Attributes:
        v1: Index of first vertex
        v2: Index of second vertex
    """
    v1: int
    v2: int
    
    def __hash__(self):
        # Ensure edges are undirected by sorting indices
        return hash(tuple(sorted([self.v1, self.v2])))
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return set([self.v1, self.v2]) == set([other.v1, other.v2])


class AlphaShapeGenerator:
    """
    Generates Alpha Shape meshes from 3D point clouds.
    
    Alpha shapes are a generalization of convex hulls that can create concave
    surfaces by filtering triangles based on their circumradius. This creates
    tight-fitting meshes around point clouds without bridging large gaps.
    
    The algorithm:
    1. Compute Delaunay triangulation of the point cloud
    2. Calculate circumradius for each tetrahedron
    3. Keep only tetrahedra with circumradius <= alpha
    4. Extract surface triangles from the filtered tetrahedra
    
    Requirements: 5.1
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Alpha Shape generator.
        
        Args:
            alpha: The alpha parameter controlling mesh tightness.
                   Smaller values create tighter fits but may create holes.
                   Larger values create smoother surfaces but may bridge gaps.
                   Typical range: 0.5 to 5.0 for road anomalies.
        
        Raises:
            ValueError: If alpha is not positive
        """
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
        
        self.alpha = alpha
    
    def generate_alpha_shape(self, points: np.ndarray) -> Mesh:
        """
        Generate an Alpha Shape mesh from a 3D point cloud.
        
        Args:
            points: Nx3 array of 3D point coordinates
        
        Returns:
            Mesh object containing vertices and faces
        
        Raises:
            InsufficientPointsError: If points array has insufficient points
            AlphaShapeError: If Alpha Shape generation fails
        """
        try:
            # Validate input
            if points.ndim != 2 or points.shape[1] != 3:
                raise AlphaShapeError(
                    "Invalid points array dimensions",
                    details={"expected": "Nx3", "got": points.shape}
                )
            
            if points.shape[0] < 4:
                raise InsufficientPointsError(
                    "Insufficient points for 3D triangulation",
                    details={"required": 4, "got": points.shape[0]}
                )
            
            logger.debug("Generating Alpha Shape", num_points=points.shape[0], alpha=self.alpha)
            
            with PerformanceTimer(logger, "Alpha Shape generation"):
                # Remove duplicate points
                unique_points = np.unique(points, axis=0)
                if unique_points.shape[0] < 4:
                    raise InsufficientPointsError(
                        "Insufficient unique points after deduplication",
                        details={"required": 4, "got": unique_points.shape[0]}
                    )
                
                # Compute Delaunay triangulation
                try:
                    delaunay = Delaunay(unique_points)
                except Exception as e:
                    raise AlphaShapeError(
                        "Delaunay triangulation failed",
                        details={"error": str(e)}
                    )
                
                # Filter tetrahedra by alpha criterion
                filtered_simplices = self._filter_by_alpha(unique_points, delaunay.simplices)
                
                if filtered_simplices.shape[0] == 0:
                    logger.warning(
                        "No tetrahedra passed alpha filter",
                        alpha=self.alpha,
                        total_simplices=delaunay.simplices.shape[0]
                    )
                
                # Extract surface triangles
                surface_faces = self._extract_surface_triangles(filtered_simplices)
                
                logger.debug(
                    "Alpha Shape generated",
                    num_vertices=unique_points.shape[0],
                    num_faces=surface_faces.shape[0]
                )
                
                return Mesh(vertices=unique_points, faces=surface_faces, is_watertight=False)
                
        except (InsufficientPointsError, AlphaShapeError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Alpha Shape generation: {str(e)}", exc_info=True)
            raise AlphaShapeError(
                "Alpha Shape generation failed",
                details={"error": str(e)}
            )
    
    def _filter_by_alpha(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        """
        Filter tetrahedra by circumradius criterion.
        
        Args:
            points: Nx3 array of point coordinates
            simplices: Mx4 array of tetrahedron vertex indices
        
        Returns:
            Filtered array of tetrahedron indices
        """
        filtered = []
        
        for simplex in simplices:
            # Get the four vertices of the tetrahedron
            tetra_points = points[simplex]
            
            # Calculate circumradius
            circumradius = self._compute_circumradius(tetra_points)
            
            # Keep tetrahedron if circumradius <= alpha
            if circumradius <= self.alpha:
                filtered.append(simplex)
        
        return np.array(filtered) if filtered else np.empty((0, 4), dtype=int)
    
    def _compute_circumradius(self, tetra_points: np.ndarray) -> float:
        """
        Compute the circumradius of a tetrahedron.
        
        The circumradius is the radius of the sphere passing through all four vertices.
        
        Args:
            tetra_points: 4x3 array of tetrahedron vertex coordinates
        
        Returns:
            Circumradius value
        """
        # Use the formula: R = |a-d| * |b-d| * |c-d| / (6 * V)
        # where V is the volume of the tetrahedron
        
        a, b, c, d = tetra_points
        
        # Calculate edge vectors from vertex d
        da = a - d
        db = b - d
        dc = c - d
        
        # Calculate volume using scalar triple product
        volume = abs(np.dot(da, np.cross(db, dc))) / 6.0
        
        # Handle degenerate case
        if volume < 1e-10:
            return float('inf')
        
        # Calculate edge lengths
        len_da = np.linalg.norm(da)
        len_db = np.linalg.norm(db)
        len_dc = np.linalg.norm(dc)
        len_ab = np.linalg.norm(a - b)
        len_bc = np.linalg.norm(b - c)
        len_ca = np.linalg.norm(c - a)
        
        # Use Cayley-Menger determinant for more stable computation
        # R = sqrt(sum of products of opposite edges) / (24 * V)
        numerator = len_da * len_bc + len_db * len_ca + len_dc * len_ab
        circumradius = numerator / (24.0 * volume)
        
        return circumradius
    
    def _extract_surface_triangles(self, simplices: np.ndarray) -> np.ndarray:
        """
        Extract surface triangles from tetrahedra.
        
        A triangle is on the surface if it belongs to only one tetrahedron.
        
        Args:
            simplices: Mx4 array of tetrahedron vertex indices
        
        Returns:
            Nx3 array of surface triangle vertex indices
        """
        if simplices.shape[0] == 0:
            return np.empty((0, 3), dtype=int)
        
        # Dictionary to count face occurrences
        face_count = {}
        
        # Each tetrahedron has 4 triangular faces
        for simplex in simplices:
            faces = [
                tuple(sorted([simplex[0], simplex[1], simplex[2]])),
                tuple(sorted([simplex[0], simplex[1], simplex[3]])),
                tuple(sorted([simplex[0], simplex[2], simplex[3]])),
                tuple(sorted([simplex[1], simplex[2], simplex[3]]))
            ]
            
            for face in faces:
                face_count[face] = face_count.get(face, 0) + 1
        
        # Surface faces appear exactly once
        surface_faces = [face for face, count in face_count.items() if count == 1]
        
        return np.array(surface_faces) if surface_faces else np.empty((0, 3), dtype=int)
    
    def extract_boundary_edges(self, mesh: Mesh) -> List[Edge]:
        """
        Extract boundary edges from a mesh.
        
        Boundary edges are edges that belong to only one triangle.
        
        Args:
            mesh: Input mesh
        
        Returns:
            List of boundary Edge objects
        
        Requirements: 5.2
        """
        if mesh.faces.shape[0] == 0:
            return []
        
        # Dictionary to count edge occurrences
        edge_count = {}
        
        # Each triangle has 3 edges
        for face in mesh.faces:
            edges = [
                Edge(face[0], face[1]),
                Edge(face[1], face[2]),
                Edge(face[2], face[0])
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Boundary edges appear exactly once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        return boundary_edges
    
    def update_alpha(self, alpha: float) -> None:
        """
        Update the alpha parameter.
        
        Args:
            alpha: New alpha value
        
        Raises:
            ValueError: If alpha is not positive
        """
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
        
        self.alpha = alpha
    
    def get_mesh_statistics(self, mesh: Mesh) -> Dict[str, any]:
        """
        Calculate statistics about the generated mesh.
        
        Args:
            mesh: Input mesh
        
        Returns:
            Dictionary containing mesh statistics
        """
        stats = {
            'num_vertices': mesh.vertices.shape[0],
            'num_faces': mesh.faces.shape[0],
            'is_watertight': mesh.is_watertight,
            'num_boundary_edges': len(self.extract_boundary_edges(mesh))
        }
        
        if mesh.faces.shape[0] > 0:
            # Calculate surface area
            areas = []
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                areas.append(area)
            
            stats['surface_area'] = sum(areas)
            stats['mean_triangle_area'] = np.mean(areas)
            stats['std_triangle_area'] = np.std(areas)
        
        return stats


class MeshCapper:
    """
    Generates caps to close open mesh surfaces for watertight closure.
    
    This class detects boundary loops in open meshes and triangulates them
    to create caps that close the mesh openings. This is essential for
    accurate volume calculation using the Divergence Theorem.
    
    The algorithm:
    1. Identify boundary edges from the mesh
    2. Organize edges into closed loops
    3. Triangulate each loop to create a cap
    4. Combine original mesh with caps for watertight closure
    
    Requirements: 5.2, 5.3
    """
    
    def __init__(self):
        """Initialize the MeshCapper."""
        pass
    
    def triangulate_boundary(self, boundary_edges: List[Edge], vertices: np.ndarray) -> Mesh:
        """
        Triangulate boundary edges to create caps.
        
        Args:
            boundary_edges: List of boundary Edge objects
            vertices: Nx3 array of vertex coordinates
        
        Returns:
            Mesh object containing the cap triangles
        
        Raises:
            ValueError: If boundary edges are invalid or cannot form loops
        
        Requirements: 5.3
        """
        if not boundary_edges:
            # No boundary edges, return empty mesh
            return Mesh(vertices=vertices, faces=np.empty((0, 3), dtype=int), is_watertight=False)
        
        # Organize edges into loops
        loops = self._extract_boundary_loops(boundary_edges)
        
        if not loops:
            raise ValueError("Could not extract valid boundary loops from edges")
        
        # Triangulate each loop
        all_cap_faces = []
        for loop in loops:
            cap_faces = self._triangulate_loop(loop, vertices)
            all_cap_faces.extend(cap_faces)
        
        if not all_cap_faces:
            return Mesh(vertices=vertices, faces=np.empty((0, 3), dtype=int), is_watertight=False)
        
        return Mesh(vertices=vertices, faces=np.array(all_cap_faces), is_watertight=False)
    
    def _extract_boundary_loops(self, boundary_edges: List[Edge]) -> List[List[int]]:
        """
        Extract closed loops from boundary edges.
        
        Args:
            boundary_edges: List of boundary Edge objects
        
        Returns:
            List of loops, where each loop is a list of vertex indices
        """
        if not boundary_edges:
            return []
        
        # Build adjacency map
        adjacency = {}
        for edge in boundary_edges:
            if edge.v1 not in adjacency:
                adjacency[edge.v1] = []
            if edge.v2 not in adjacency:
                adjacency[edge.v2] = []
            adjacency[edge.v1].append(edge.v2)
            adjacency[edge.v2].append(edge.v1)
        
        # Extract loops
        loops = []
        visited_edges = set()
        
        for start_vertex in adjacency.keys():
            if start_vertex in [v for loop in loops for v in loop]:
                continue
            
            # Try to build a loop starting from this vertex
            loop = self._build_loop(start_vertex, adjacency, visited_edges)
            if loop and len(loop) >= 3:
                loops.append(loop)
        
        return loops
    
    def _build_loop(self, start_vertex: int, adjacency: Dict[int, List[int]], 
                    visited_edges: set) -> Optional[List[int]]:
        """
        Build a single loop starting from a vertex.
        
        Args:
            start_vertex: Starting vertex index
            adjacency: Adjacency map of vertices
            visited_edges: Set of already visited edges
        
        Returns:
            List of vertex indices forming a loop, or None if no loop found
        """
        loop = [start_vertex]
        current = start_vertex
        
        while True:
            # Find next unvisited neighbor
            next_vertex = None
            for neighbor in adjacency.get(current, []):
                edge_key = tuple(sorted([current, neighbor]))
                if edge_key not in visited_edges:
                    next_vertex = neighbor
                    visited_edges.add(edge_key)
                    break
            
            if next_vertex is None:
                # No unvisited neighbors
                break
            
            if next_vertex == start_vertex:
                # Loop closed
                return loop
            
            if next_vertex in loop:
                # Hit a vertex already in the loop (but not start) - invalid
                break
            
            loop.append(next_vertex)
            current = next_vertex
            
            # Prevent infinite loops
            if len(loop) > len(adjacency) * 2:
                break
        
        # Check if we have a valid loop
        if len(loop) >= 3:
            # Check if last vertex connects back to start
            edge_key = tuple(sorted([loop[-1], start_vertex]))
            if edge_key not in visited_edges:
                for neighbor in adjacency.get(loop[-1], []):
                    if neighbor == start_vertex:
                        visited_edges.add(edge_key)
                        return loop
        
        return None
    
    def _triangulate_loop(self, loop: List[int], vertices: np.ndarray) -> List[np.ndarray]:
        """
        Triangulate a boundary loop using ear clipping or fan triangulation.
        
        Args:
            loop: List of vertex indices forming a closed loop
            vertices: Nx3 array of vertex coordinates
        
        Returns:
            List of triangle faces (each as array of 3 vertex indices)
        """
        if len(loop) < 3:
            return []
        
        if len(loop) == 3:
            # Already a triangle
            return [np.array(loop)]
        
        # Use simple fan triangulation from first vertex
        # This works well for convex or nearly-convex loops
        faces = []
        for i in range(1, len(loop) - 1):
            face = np.array([loop[0], loop[i], loop[i + 1]])
            faces.append(face)
        
        return faces
    
    def create_watertight_mesh(self, surface_mesh: Mesh, cap_mesh: Mesh) -> Mesh:
        """
        Combine surface mesh with cap mesh to create watertight closure.
        
        Args:
            surface_mesh: Original open surface mesh
            cap_mesh: Cap mesh generated from boundary triangulation
        
        Returns:
            Combined watertight mesh
        
        Requirements: 5.3
        """
        # Combine faces from both meshes
        combined_faces = np.vstack([surface_mesh.faces, cap_mesh.faces])
        
        # Use the same vertices (both meshes should reference the same vertex array)
        combined_mesh = Mesh(
            vertices=surface_mesh.vertices,
            faces=combined_faces,
            is_watertight=False  # Will be validated separately
        )
        
        return combined_mesh
    
    def validate_watertightness(self, mesh: Mesh) -> bool:
        """
        Validate whether a mesh is watertight.
        
        A mesh is watertight if:
        1. Every edge is shared by exactly 2 faces
        2. The mesh forms a closed manifold
        
        Args:
            mesh: Mesh to validate
        
        Returns:
            True if mesh is watertight, False otherwise
        
        Requirements: 5.4
        """
        if mesh.faces.shape[0] == 0:
            return False
        
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
        
        # Check if all edges are shared by exactly 2 faces
        for count in edge_count.values():
            if count != 2:
                return False
        
        return True


class VolumeCalculator:
    """
    Calculates volumes of watertight meshes using signed tetrahedron integration.
    
    This class implements the Divergence Theorem for volume calculation by
    decomposing the mesh into tetrahedra formed by each triangle and the origin,
    then summing their signed volumes. This method is mathematically exact for
    closed manifolds.
    
    The algorithm:
    1. Validate that the mesh is watertight
    2. For each triangle, form a tetrahedron with the origin
    3. Calculate the signed volume of each tetrahedron
    4. Sum all signed volumes to get the total mesh volume
    
    Requirements: 5.4, 5.5, 6.1
    """
    
    def __init__(self, max_volume_cubic_meters: float = 10.0):
        """
        Initialize the VolumeCalculator.
        
        Args:
            max_volume_cubic_meters: Maximum physically reasonable volume for road anomalies
                                    in cubic meters. Default is 10.0 m³ (very large pothole).
        """
        self.max_volume_cubic_meters = max_volume_cubic_meters
    
    def validate_mesh_closure(self, mesh: Mesh) -> bool:
        """
        Validate that a mesh is closed (watertight) before volume computation.
        
        A mesh is closed if every edge is shared by exactly two faces, forming
        a manifold without boundaries. This is a prerequisite for accurate
        volume calculation using the Divergence Theorem.
        
        Args:
            mesh: Mesh to validate
        
        Returns:
            True if mesh is watertight, False otherwise
        
        Requirements: 5.4
        """
        if mesh.faces.shape[0] == 0:
            return False
        
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
        
        # Check if all edges are shared by exactly 2 faces
        for count in edge_count.values():
            if count != 2:
                return False
        
        return True
    
    def calculate_signed_volume(self, watertight_mesh: Mesh) -> float:
        """
        Calculate volume of a watertight mesh using signed tetrahedron integration.
        
        This method implements the Divergence Theorem by decomposing the mesh
        into tetrahedra and summing their signed volumes. The formula for each
        tetrahedron formed by triangle (v0, v1, v2) and origin is:
        
        V = (1/6) * |v0 · (v1 × v2)|
        
        The sign is determined by the triangle orientation, and summing all
        signed volumes gives the total enclosed volume.
        
        Args:
            watertight_mesh: A closed mesh with validated watertightness
        
        Returns:
            Volume in cubic units (same units as vertex coordinates)
        
        Raises:
            WatertightnessError: If mesh is not watertight
            VolumeCalculationError: If volume calculation fails
        
        Requirements: 5.5, 6.1
        """
        try:
            # Validate mesh closure
            if not self.validate_mesh_closure(watertight_mesh):
                raise WatertightnessError(
                    "Mesh is not watertight - cannot calculate volume",
                    details={
                        "num_vertices": watertight_mesh.vertices.shape[0],
                        "num_faces": watertight_mesh.faces.shape[0]
                    }
                )
            
            if watertight_mesh.faces.shape[0] == 0:
                raise VolumeCalculationError(
                    "Mesh has no faces",
                    details={"num_faces": 0}
                )
            
            logger.debug(
                "Calculating signed volume",
                num_vertices=watertight_mesh.vertices.shape[0],
                num_faces=watertight_mesh.faces.shape[0]
            )
            
            with PerformanceTimer(logger, "Volume calculation"):
                # Calculate signed volume using tetrahedron decomposition
                total_volume = 0.0
                
                for face in watertight_mesh.faces:
                    # Get the three vertices of the triangle
                    v0 = watertight_mesh.vertices[face[0]]
                    v1 = watertight_mesh.vertices[face[1]]
                    v2 = watertight_mesh.vertices[face[2]]
                    
                    # Calculate signed volume of tetrahedron formed by triangle and origin
                    # V = (1/6) * v0 · (v1 × v2)
                    cross_product = np.cross(v1, v2)
                    signed_volume = np.dot(v0, cross_product) / 6.0
                    
                    total_volume += signed_volume
                
                # Return absolute value (sign depends on face orientation)
                volume = abs(total_volume)
                
                logger.debug("Volume calculated", volume_cubic_meters=f"{volume:.6f}")
                
                return volume
                
        except (WatertightnessError, VolumeCalculationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in volume calculation: {str(e)}", exc_info=True)
            raise VolumeCalculationError(
                "Volume calculation failed",
                details={"error": str(e)}
            )
    
    def calculate_volume_with_validation(self, mesh: Mesh) -> Tuple[float, bool]:
        """
        Calculate volume with explicit watertightness validation.
        
        This is a convenience method that validates mesh closure and calculates
        volume in a single call, returning both the volume and validation status.
        
        Args:
            mesh: Mesh to calculate volume for
        
        Returns:
            Tuple of (volume, is_watertight)
            - volume: Calculated volume (0.0 if not watertight)
            - is_watertight: Whether the mesh passed validation
        
        Requirements: 5.4, 5.5, 6.1
        """
        is_watertight = self.validate_mesh_closure(mesh)
        
        if not is_watertight:
            return 0.0, False
        
        try:
            volume = self.calculate_signed_volume(mesh)
            return volume, True
        except ValueError:
            return 0.0, False
    
    def calculate_volume_statistics(self, mesh: Mesh) -> Dict[str, any]:
        """
        Calculate volume and related statistics for a mesh.
        
        Args:
            mesh: Mesh to analyze
        
        Returns:
            Dictionary containing volume statistics including:
            - volume: Calculated volume
            - is_watertight: Watertightness validation result
            - num_vertices: Number of vertices
            - num_faces: Number of faces
            - surface_area: Total surface area (if calculable)
        """
        volume, is_watertight = self.calculate_volume_with_validation(mesh)
        
        stats = {
            'volume': volume,
            'is_watertight': is_watertight,
            'num_vertices': mesh.vertices.shape[0],
            'num_faces': mesh.faces.shape[0]
        }
        
        # Calculate surface area
        if mesh.faces.shape[0] > 0:
            total_area = 0.0
            for face in mesh.faces:
                v0 = mesh.vertices[face[0]]
                v1 = mesh.vertices[face[1]]
                v2 = mesh.vertices[face[2]]
                
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                total_area += area
            
            stats['surface_area'] = total_area
        
        return stats
    
    def convert_volume_units(self, volume_cubic_meters: float) -> Dict[str, float]:
        """
        Convert volume from cubic meters to multiple standard units.
        
        Conversion factors:
        - 1 cubic meter = 1,000,000 cubic centimeters
        - 1 cubic meter = 1,000 liters
        
        Args:
            volume_cubic_meters: Volume in cubic meters
        
        Returns:
            Dictionary with volume in different units:
            - cubic_meters: Original volume in m³
            - liters: Volume in liters (L)
            - cubic_centimeters: Volume in cubic centimeters (cm³)
        
        Requirements: 6.2
        """
        return {
            'cubic_meters': volume_cubic_meters,
            'liters': volume_cubic_meters * 1000.0,
            'cubic_centimeters': volume_cubic_meters * 1_000_000.0
        }
    
    def validate_volume_constraints(self, volume_cubic_meters: float) -> Tuple[bool, str]:
        """
        Validate that calculated volume is within physically reasonable bounds.
        
        This checks geometric constraints to ensure the volume is:
        1. Non-negative (mathematical requirement)
        2. Greater than a minimum threshold (avoids numerical noise)
        3. Less than maximum expected size for road anomalies
        
        Args:
            volume_cubic_meters: Volume to validate in cubic meters
        
        Returns:
            Tuple of (is_valid, message)
            - is_valid: True if volume passes all constraints
            - message: Description of validation result or failure reason
        
        Requirements: 6.4
        """
        # Minimum threshold: 1 cubic centimeter = 1e-6 cubic meters
        min_volume = 1e-6
        
        if volume_cubic_meters < 0:
            return False, f"Volume is negative: {volume_cubic_meters} m³"
        
        if volume_cubic_meters < min_volume:
            return False, f"Volume {volume_cubic_meters} m³ is below minimum threshold {min_volume} m³ (likely numerical noise)"
        
        if volume_cubic_meters > self.max_volume_cubic_meters:
            return False, f"Volume {volume_cubic_meters} m³ exceeds maximum reasonable bound {self.max_volume_cubic_meters} m³"
        
        return True, "Volume is within valid constraints"
    
    def calculate_volume_with_units(self, watertight_mesh: Mesh) -> Dict[str, any]:
        """
        Calculate volume with unit conversion and constraint validation.
        
        This is a comprehensive method that:
        1. Calculates the volume in cubic meters
        2. Converts to multiple standard units
        3. Validates against geometric constraints
        4. Returns all results in a structured format
        
        Args:
            watertight_mesh: A closed mesh with validated watertightness
        
        Returns:
            Dictionary containing:
            - volume_cubic_meters: Volume in m³
            - volume_liters: Volume in liters
            - volume_cubic_cm: Volume in cm³
            - is_valid: Whether volume passes constraint validation
            - validation_message: Description of validation result
            - is_watertight: Whether mesh is watertight
        
        Raises:
            ValueError: If mesh is not watertight
        
        Requirements: 6.2, 6.4
        """
        # Calculate volume
        volume_m3 = self.calculate_signed_volume(watertight_mesh)
        
        # Convert units
        volume_units = self.convert_volume_units(volume_m3)
        
        # Validate constraints
        is_valid, validation_message = self.validate_volume_constraints(volume_m3)
        
        return {
            'volume_cubic_meters': volume_units['cubic_meters'],
            'volume_liters': volume_units['liters'],
            'volume_cubic_cm': volume_units['cubic_centimeters'],
            'is_valid': is_valid,
            'validation_message': validation_message,
            'is_watertight': True
        }
    
    def calculate_volume_uncertainty(self, watertight_mesh: Mesh, 
                                     measurement_precision: float = 0.001) -> float:
        """
        Estimate volume uncertainty based on measurement precision.
        
        The uncertainty in volume calculation depends on:
        1. Measurement precision of vertex coordinates
        2. Number of vertices (more vertices = more error accumulation)
        3. Surface area (larger surface = more uncertainty)
        
        This uses a simplified error propagation model:
        σ_V ≈ A * σ_p
        
        where:
        - σ_V is the volume uncertainty
        - A is the surface area
        - σ_p is the measurement precision
        
        Args:
            watertight_mesh: A closed mesh with validated watertightness
            measurement_precision: Precision of coordinate measurements in meters
                                  (default: 0.001 m = 1 mm)
        
        Returns:
            Estimated volume uncertainty in cubic meters
        
        Requirements: 6.5
        """
        if not self.validate_mesh_closure(watertight_mesh):
            raise ValueError("Mesh is not watertight - cannot estimate uncertainty")
        
        if measurement_precision <= 0:
            raise ValueError(f"Measurement precision must be positive, got {measurement_precision}")
        
        # Calculate surface area
        total_area = 0.0
        for face in watertight_mesh.faces:
            v0 = watertight_mesh.vertices[face[0]]
            v1 = watertight_mesh.vertices[face[1]]
            v2 = watertight_mesh.vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            total_area += area
        
        # Estimate uncertainty: σ_V ≈ A * σ_p
        # This is a conservative estimate based on surface area
        uncertainty = total_area * measurement_precision
        
        return uncertainty
    
    def calculate_multiple_volumes(self, meshes: List[Mesh], 
                                   measurement_precision: float = 0.001) -> List[Dict[str, any]]:
        """
        Calculate volumes for multiple anomalies independently.
        
        This method processes multiple anomaly meshes and calculates their volumes
        independently, ensuring that the presence of one anomaly doesn't affect
        the volume calculation of another.
        
        Args:
            meshes: List of watertight meshes representing different anomalies
            measurement_precision: Precision of coordinate measurements in meters
        
        Returns:
            List of dictionaries, each containing:
            - volume_cubic_meters: Volume in m³
            - volume_liters: Volume in liters
            - volume_cubic_cm: Volume in cm³
            - uncertainty_cubic_meters: Estimated uncertainty in m³
            - is_valid: Whether volume passes constraint validation
            - validation_message: Description of validation result
            - is_watertight: Whether mesh is watertight
            - anomaly_index: Index of the anomaly in the input list
        
        Requirements: 6.3, 6.5
        """
        results = []
        
        for idx, mesh in enumerate(meshes):
            try:
                # Validate watertightness
                if not self.validate_mesh_closure(mesh):
                    results.append({
                        'volume_cubic_meters': 0.0,
                        'volume_liters': 0.0,
                        'volume_cubic_cm': 0.0,
                        'uncertainty_cubic_meters': 0.0,
                        'is_valid': False,
                        'validation_message': 'Mesh is not watertight',
                        'is_watertight': False,
                        'anomaly_index': idx
                    })
                    continue
                
                # Calculate volume
                volume_m3 = self.calculate_signed_volume(mesh)
                
                # Convert units
                volume_units = self.convert_volume_units(volume_m3)
                
                # Validate constraints
                is_valid, validation_message = self.validate_volume_constraints(volume_m3)
                
                # Calculate uncertainty
                uncertainty = self.calculate_volume_uncertainty(mesh, measurement_precision)
                
                results.append({
                    'volume_cubic_meters': volume_units['cubic_meters'],
                    'volume_liters': volume_units['liters'],
                    'volume_cubic_cm': volume_units['cubic_centimeters'],
                    'uncertainty_cubic_meters': uncertainty,
                    'is_valid': is_valid,
                    'validation_message': validation_message,
                    'is_watertight': True,
                    'anomaly_index': idx
                })
                
            except Exception as e:
                results.append({
                    'volume_cubic_meters': 0.0,
                    'volume_liters': 0.0,
                    'volume_cubic_cm': 0.0,
                    'uncertainty_cubic_meters': 0.0,
                    'is_valid': False,
                    'validation_message': f'Error calculating volume: {str(e)}',
                    'is_watertight': False,
                    'anomaly_index': idx
                })
        
        return results

    def calculate_volume_uncertainty(self, watertight_mesh: Mesh,
                                     measurement_precision: float = 0.001) -> float:
        """
        Estimate volume uncertainty based on measurement precision.

        The uncertainty in volume calculation depends on:
        1. Measurement precision of vertex coordinates
        2. Number of vertices (more vertices = more error accumulation)
        3. Surface area (larger surface = more uncertainty)

        This uses a simplified error propagation model:
        σ_V ≈ A * σ_p

        where:
        - σ_V is the volume uncertainty
        - A is the surface area
        - σ_p is the measurement precision

        Args:
            watertight_mesh: A closed mesh with validated watertightness
            measurement_precision: Precision of coordinate measurements in meters
                                  (default: 0.001 m = 1 mm)

        Returns:
            Estimated volume uncertainty in cubic meters

        Requirements: 6.5
        """
        if not self.validate_mesh_closure(watertight_mesh):
            raise ValueError("Mesh is not watertight - cannot estimate uncertainty")

        if measurement_precision <= 0:
            raise ValueError(f"Measurement precision must be positive, got {measurement_precision}")

        # Calculate surface area
        total_area = 0.0
        for face in watertight_mesh.faces:
            v0 = watertight_mesh.vertices[face[0]]
            v1 = watertight_mesh.vertices[face[1]]
            v2 = watertight_mesh.vertices[face[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            total_area += area

        # Estimate uncertainty: σ_V ≈ A * σ_p
        # This is a conservative estimate based on surface area
        uncertainty = total_area * measurement_precision

        return uncertainty

    def calculate_multiple_volumes(self, meshes: List[Mesh],
                                   measurement_precision: float = 0.001) -> List[Dict[str, any]]:
        """
        Calculate volumes for multiple anomalies independently.

        This method processes multiple anomaly meshes and calculates their volumes
        independently, ensuring that the presence of one anomaly doesn't affect
        the volume calculation of another.

        Args:
            meshes: List of watertight meshes representing different anomalies
            measurement_precision: Precision of coordinate measurements in meters

        Returns:
            List of dictionaries, each containing:
            - volume_cubic_meters: Volume in m³
            - volume_liters: Volume in liters
            - volume_cubic_cm: Volume in cm³
            - uncertainty_cubic_meters: Estimated uncertainty in m³
            - is_valid: Whether volume passes constraint validation
            - validation_message: Description of validation result
            - is_watertight: Whether mesh is watertight
            - anomaly_index: Index of the anomaly in the input list

        Requirements: 6.3, 6.5
        """
        results = []

        for idx, mesh in enumerate(meshes):
            try:
                # Validate watertightness
                if not self.validate_mesh_closure(mesh):
                    results.append({
                        'volume_cubic_meters': 0.0,
                        'volume_liters': 0.0,
                        'volume_cubic_cm': 0.0,
                        'uncertainty_cubic_meters': 0.0,
                        'is_valid': False,
                        'validation_message': 'Mesh is not watertight',
                        'is_watertight': False,
                        'anomaly_index': idx
                    })
                    continue

                # Calculate volume
                volume_m3 = self.calculate_signed_volume(mesh)

                # Convert units
                volume_units = self.convert_volume_units(volume_m3)

                # Validate constraints
                is_valid, validation_message = self.validate_volume_constraints(volume_m3)

                # Calculate uncertainty
                uncertainty = self.calculate_volume_uncertainty(mesh, measurement_precision)

                results.append({
                    'volume_cubic_meters': volume_units['cubic_meters'],
                    'volume_liters': volume_units['liters'],
                    'volume_cubic_cm': volume_units['cubic_centimeters'],
                    'uncertainty_cubic_meters': uncertainty,
                    'is_valid': is_valid,
                    'validation_message': validation_message,
                    'is_watertight': True,
                    'anomaly_index': idx
                })

            except Exception as e:
                results.append({
                    'volume_cubic_meters': 0.0,
                    'volume_liters': 0.0,
                    'volume_cubic_cm': 0.0,
                    'uncertainty_cubic_meters': 0.0,
                    'is_valid': False,
                    'validation_message': f'Error calculating volume: {str(e)}',
                    'is_watertight': False,
                    'anomaly_index': idx
                })

        return results

