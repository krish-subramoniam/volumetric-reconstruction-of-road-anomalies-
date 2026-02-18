"""Generate synthetic stereo images with known potholes for testing."""

import numpy as np
import cv2

def generate_stereo_pair_with_pothole():
    """Generate synthetic stereo pair with a visible pothole."""
    
    print("Generating synthetic stereo images with pothole...")
    
    # Image dimensions
    height, width = 480, 640
    
    # Create textured ground plane
    np.random.seed(42)
    base_texture = np.random.randint(80, 120, (height, width), dtype=np.uint8)
    
    # Add some structure (grid pattern)
    for i in range(0, height, 20):
        base_texture[i:i+2, :] = 150
    for j in range(0, width, 20):
        base_texture[:, j:j+2] = 150
    
    # Create depth map (disparity)
    # Flat ground at disparity 40
    disparity = np.ones((height, width), dtype=np.float32) * 40.0
    
    # Add 3 potholes with different depths
    potholes = [
        {'center': (240, 200), 'radius': 40, 'depth_disp': 20},  # Deep pothole
        {'center': (240, 400), 'radius': 30, 'depth_disp': 15},  # Medium pothole
        {'center': (350, 320), 'radius': 25, 'depth_disp': 12},  # Shallow pothole
    ]
    
    for pothole in potholes:
        cy, cx = pothole['center']
        radius = pothole['radius']
        depth = pothole['depth_disp']
        
        # Create circular depression
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Smooth depression (parabolic)
        mask = dist < radius
        depression = np.zeros_like(disparity)
        depression[mask] = depth * (1 - (dist[mask] / radius)**2)
        
        # Add to disparity (higher disparity = closer = depression)
        disparity += depression
        
        # Darken pothole in texture
        base_texture[mask] = (base_texture[mask] * 0.7).astype(np.uint8)
    
    # Create left image
    left_image = base_texture.copy()
    
    # Create right image by shifting based on disparity
    right_image = np.zeros_like(left_image)
    
    for y in range(height):
        for x in range(width):
            disp = int(disparity[y, x])
            if x - disp >= 0:
                right_image[y, x] = left_image[y, x - disp]
    
    # Add some noise for realism
    noise_left = np.random.randint(-5, 5, left_image.shape, dtype=np.int16)
    noise_right = np.random.randint(-5, 5, right_image.shape, dtype=np.int16)
    
    left_image = np.clip(left_image.astype(np.int16) + noise_left, 0, 255).astype(np.uint8)
    right_image = np.clip(right_image.astype(np.int16) + noise_right, 0, 255).astype(np.uint8)
    
    # Save images
    cv2.imwrite('synthetic_left.png', left_image)
    cv2.imwrite('synthetic_right.png', right_image)
    
    print("✓ Generated synthetic_left.png")
    print("✓ Generated synthetic_right.png")
    print(f"\nImage details:")
    print(f"  Size: {width}x{height}")
    print(f"  Potholes: {len(potholes)}")
    print(f"  Disparity range: {disparity.min():.1f} - {disparity.max():.1f} pixels")
    print(f"\nPothole details:")
    for i, p in enumerate(potholes, 1):
        print(f"  Pothole {i}: center={p['center']}, radius={p['radius']}px, depth={p['depth_disp']}px")
    
    # Calculate expected volumes (approximate)
    print(f"\nExpected volumes (approximate):")
    baseline = 0.12  # meters
    focal_length = 700.0  # pixels
    
    for i, p in enumerate(potholes, 1):
        radius_m = p['radius'] * baseline / focal_length
        depth_m = p['depth_disp'] * baseline / focal_length
        # Approximate as hemisphere: V = (2/3) * π * r² * h
        volume_m3 = (2/3) * np.pi * (radius_m**2) * depth_m
        print(f"  Pothole {i}: ~{volume_m3:.6f} m³ ({volume_m3*1000:.2f} liters)")
    
    print(f"\n✓ Upload these images to the Gradio app at http://localhost:7860")
    print(f"  Use baseline=0.12m and focal_length=700px")

def generate_simple_test_pair():
    """Generate very simple stereo pair for basic testing."""
    
    print("\nGenerating simple test stereo pair...")
    
    # Create simple textured image
    height, width = 480, 640
    left = np.zeros((height, width), dtype=np.uint8)
    
    # Add checkerboard pattern
    square_size = 40
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                left[i:i+square_size, j:j+square_size] = 200
            else:
                left[i:i+square_size, j:j+square_size] = 100
    
    # Create right image with uniform disparity shift
    disparity_shift = 30
    right = np.zeros_like(left)
    right[:, disparity_shift:] = left[:, :-disparity_shift]
    
    # Add a "pothole" - region with larger disparity
    cy, cx = 240, 320
    radius = 50
    y, x = np.ogrid[:height, :width]
    mask = ((x - cx)**2 + (y - cy)**2) < radius**2
    
    # Shift this region more (simulating depression)
    extra_shift = 20
    pothole_region = left[mask]
    right_pothole = np.zeros_like(right)
    
    # This is simplified - just darken the pothole area
    right[mask] = (right[mask] * 0.6).astype(np.uint8)
    
    cv2.imwrite('simple_left.png', left)
    cv2.imwrite('simple_right.png', right)
    
    print("✓ Generated simple_left.png")
    print("✓ Generated simple_right.png")
    print("  (Checkerboard pattern with central depression)")

if __name__ == "__main__":
    print("="*60)
    print("SYNTHETIC STEREO IMAGE GENERATOR")
    print("="*60)
    
    generate_stereo_pair_with_pothole()
    generate_simple_test_pair()
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("1. Go to http://localhost:7860")
    print("2. Click 'Full Pipeline' tab")
    print("3. Upload synthetic_left.png as Left Image")
    print("4. Upload synthetic_right.png as Right Image")
    print("5. Set Baseline = 0.12")
    print("6. Set Focal Length = 700")
    print("7. Click 'Run Full Pipeline'")
    print("\nYou should see NON-ZERO volumes!")
    print("="*60)
