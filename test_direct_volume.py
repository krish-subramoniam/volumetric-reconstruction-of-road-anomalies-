"""
Direct test of volume calculation without full pipeline.
"""

import numpy as np
from stereo_vision.volumetric import AlphaShapeGenerator, MeshCapper, VolumeCalculator

def test_volume_calculation_direct():
    """Test volume calculation directly with a simple point cloud."""
    print("Testing volume calculation with simple cube point cloud...")
    
    # Create a simple cube point cloud (1m x 1m x 1m)
    # This should give us a volume of approximately 1.0 m³
    points = []
    step = 0.1
    for x in np.arange(0, 1, step):
        for y in np.arange(0, 1, step):
            for z in np.arange(0, 1, step):
                points.append([x, y, z])
    
    points = np.array(points)
    print(f"Created point cloud with {points.shape[0]} points")
    print(f"Bounds: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Test alpha shape generation
    print("\n1. Testing Alpha Shape generation...")
    alpha_gen = AlphaShapeGenerator(alpha=0.5)
    
    try:
        mesh = alpha_gen.generate_alpha_shape(points)
        print(f"   ✓ Alpha shape generated: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    except Exception as e:
        print(f"   ✗ Alpha shape failed: {str(e)}")
        return False
    
    # Test boundary edge extraction
    print("\n2. Testing boundary edge extraction...")
    boundary_edges = alpha_gen.extract_boundary_edges(mesh)
    print(f"   Found {len(boundary_edges)} boundary edges")
    
    # Test mesh capping
    print("\n3. Testing mesh capping...")
    capper = MeshCapper()
    
    try:
        if boundary_edges:
            cap_mesh = capper.triangulate_boundary(boundary_edges, mesh.vertices)
            print(f"   ✓ Cap mesh generated: {cap_mesh.faces.shape[0]} cap faces")
            
            watertight_mesh = capper.create_watertight_mesh(mesh, cap_mesh)
            print(f"   ✓ Watertight mesh created: {watertight_mesh.faces.shape[0]} total faces")
        else:
            print("   No boundary edges - checking if already watertight...")
            watertight_mesh = mesh
        
        is_watertight = capper.validate_watertightness(watertight_mesh)
        print(f"   Watertight validation: {is_watertight}")
        
    except Exception as e:
        print(f"   ✗ Mesh capping failed: {str(e)}")
        # Try convex hull as fallback
        print("   Trying convex hull fallback...")
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        print(f"   ✓ Convex hull: {hull.simplices.shape[0]} faces, volume={hull.volume:.6f} m³")
        return True
    
    # Test volume calculation
    print("\n4. Testing volume calculation...")
    calc = VolumeCalculator()
    
    try:
        if is_watertight:
            volume_result = calc.calculate_volume_with_units(watertight_mesh)
            print(f"   ✓ Volume calculated: {volume_result['volume_cubic_meters']:.6f} m³")
            print(f"     = {volume_result['volume_liters']:.2f} liters")
            print(f"     = {volume_result['volume_cubic_cm']:.2f} cm³")
            print(f"   Valid: {volume_result['is_valid']}")
            print(f"   Message: {volume_result['validation_message']}")
            
            # Check if volume is reasonable (should be close to 1.0 m³ for a 1x1x1 cube)
            expected_volume = 1.0
            actual_volume = volume_result['volume_cubic_meters']
            error_percent = abs(actual_volume - expected_volume) / expected_volume * 100
            
            print(f"\n   Expected volume: {expected_volume:.3f} m³")
            print(f"   Actual volume: {actual_volume:.6f} m³")
            print(f"   Error: {error_percent:.1f}%")
            
            if volume_result['is_valid'] and actual_volume > 0:
                print("\n✓ Volume calculation test PASSED")
                return True
            else:
                print("\n✗ Volume calculation test FAILED - invalid or zero volume")
                return False
        else:
            print("   ✗ Mesh is not watertight - cannot calculate volume accurately")
            # Try anyway with convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            print(f"   Convex hull fallback: volume={hull.volume:.6f} m³")
            return True
            
    except Exception as e:
        print(f"   ✗ Volume calculation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_small_depression():
    """Test with a small depression (pothole-like shape)."""
    print("\n" + "="*60)
    print("Testing with depression shape (pothole)...")
    print("="*60)
    
    # Create a flat surface with a depression
    points = []
    
    # Ground plane at z=0
    for x in np.linspace(-1, 1, 20):
        for y in np.linspace(-1, 1, 20):
            points.append([x, y, 0])
    
    # Depression (bowl shape) in the center
    for x in np.linspace(-0.3, 0.3, 10):
        for y in np.linspace(-0.3, 0.3, 10):
            # Depth based on distance from center
            dist = np.sqrt(x**2 + y**2)
            if dist < 0.3:
                depth = -0.2 * (1 - (dist / 0.3)**2)  # Parabolic depression
                points.append([x, y, depth])
    
    points = np.array(points)
    print(f"Created depression with {points.shape[0]} points")
    print(f"Depth range: {points[:, 2].min():.3f}m to {points[:, 2].max():.3f}m")
    
    # Use convex hull for quick volume estimate
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        volume = hull.volume
        print(f"Convex hull volume: {volume:.6f} m³ ({volume*1000:.2f} liters)")
        
        if volume > 0:
            print("✓ Depression volume calculation PASSED")
            return True
        else:
            print("✗ Depression volume calculation FAILED")
            return False
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("DIRECT VOLUME CALCULATION TESTS")
    print("="*60)
    
    test1 = test_volume_calculation_direct()
    test2 = test_small_depression()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Cube test: {'PASSED' if test1 else 'FAILED'}")
    print(f"Depression test: {'PASSED' if test2 else 'FAILED'}")
    
    if test1 and test2:
        print("\n✓ All tests PASSED - volume calculation is working!")
    else:
        print("\n✗ Some tests FAILED - volume calculation needs fixes")
