# -*- coding: utf-8 -*-
"""
Gradio UI for Advanced Stereo Vision Pipeline Testing
"""

import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from stereo_vision.pipeline import StereoVisionPipeline
from stereo_vision.config import PipelineConfig, CameraConfig
from stereo_vision.preprocessing import ImagePreprocessor
from stereo_vision.disparity import SGBMEstimator
from stereo_vision.ground_plane import VDisparityGenerator
from stereo_vision.reconstruction import PointCloudGenerator, OutlierRemover
from stereo_vision.volumetric import AlphaShapeGenerator, MeshCapper, VolumeCalculator
from stereo_vision.quality_metrics import QualityMetrics


# Global state
pipeline = None
current_disparity = None
current_points = None


def run_full_pipeline(left_image, right_image, baseline, focal_length):
    """Run the complete pipeline end-to-end."""
    global pipeline, current_disparity, current_points
    
    if left_image is None or right_image is None:
        return None, None, None, None, "Please upload both images"
    
    try:
        from stereo_vision.calibration import CameraParameters, StereoParameters
        
        camera_config = CameraConfig(baseline=baseline, focal_length=focal_length)
        config = PipelineConfig(camera=camera_config)
        # Disable WLS filtering as it requires opencv-contrib
        config.wls.enabled = False
        pipeline = StereoVisionPipeline(config)
        
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        # Create synthetic calibration for testing
        h, w = left_gray.shape
        
        # Create identity camera matrices
        camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        distortion = np.zeros(5, dtype=np.float32)
        
        left_cam = CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion,
            reprojection_error=0.0,
            image_size=(w, h)
        )
        
        right_cam = CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion,
            reprojection_error=0.0,
            image_size=(w, h)
        )
        
        # Create Q matrix for reprojection
        Q = np.array([
            [1, 0, 0, -w/2],
            [0, 1, 0, -h/2],
            [0, 0, 0, focal_length],
            [0, 0, 1/baseline, 0]
        ], dtype=np.float32)
        
        # Create identity rectification maps (no rectification needed for synthetic)
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                map_x[y, x] = x
                map_y[y, x] = y
        
        # Create stereo parameters
        stereo_params = StereoParameters(
            left_camera=left_cam,
            right_camera=right_cam,
            rotation_matrix=np.eye(3, dtype=np.float32),
            translation_vector=np.array([[baseline], [0], [0]], dtype=np.float32),
            baseline=baseline,
            Q_matrix=Q,
            rectification_maps_left=(map_x, map_y),
            rectification_maps_right=(map_x, map_y)
        )
        
        # Set calibration
        pipeline.stereo_params = stereo_params
        pipeline.is_calibrated = True
        
        # Initialize point cloud generator
        pipeline.point_cloud_generator = PointCloudGenerator(
            Q_matrix=Q,
            min_depth=config.depth_range.min_depth,
            max_depth=config.depth_range.max_depth
        )
        
        result = pipeline.process_stereo_pair(left_gray, right_gray, generate_diagnostics=False)
        current_disparity = result.disparity_map
        
        # Get point cloud from first anomaly if available, otherwise generate from disparity
        if result.anomalies and len(result.anomalies) > 0:
            current_points = result.anomalies[0].point_cloud
        else:
            # Generate point cloud from full disparity map
            current_points = pipeline.point_cloud_generator.reproject_to_3d(
                result.disparity_map, apply_depth_filter=True
            )
        
        disp_norm = cv2.normalize(result.disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        
        v_disp = result.v_disparity
        v_disp_colored = None
        if v_disp is not None:
            v_disp_vis = cv2.normalize(v_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            v_disp_colored = cv2.applyColorMap(v_disp_vis, cv2.COLORMAP_HOT)
        
        points = current_points
    
    except RuntimeError as e:
        if "Ground plane detection failed" in str(e):
            # Ground plane failed, but we can still show disparity and point cloud
            # Generate basic visualizations without anomaly detection
            try:
                disp_norm = cv2.normalize(result.disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                
                # Generate point cloud from full disparity
                full_points = pipeline.point_cloud_generator.reproject_to_3d(
                    result.disparity_map, apply_depth_filter=True
                )
                
                # Create 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                if len(full_points) > 10000:
                    indices = np.random.choice(len(full_points), 10000, replace=False)
                    points_vis = full_points[indices]
                else:
                    points_vis = full_points
                
                if len(points_vis) > 0:
                    ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                              c=points_vis[:, 2], cmap='viridis', s=1)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Z (m)')
                    ax.set_title('3D Point Cloud (No Ground Plane)')
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                img = Image.open(buf)
                pointcloud_img = np.array(img)
                
                stats = (
                    "Ground plane detection failed, but generated basic visualizations.\n\n"
                    f"Disparity range: {result.disparity_map[result.disparity_map>0].min():.2f} - {result.disparity_map.max():.2f}\n"
                    f"Point cloud size: {len(full_points)} points\n"
                    f"Depth range: {full_points[:, 2].min():.2f}m - {full_points[:, 2].max():.2f}m\n\n"
                    "Ground plane detection failed because:\n"
                    "- Images don't contain a clear ground plane\n"
                    "- Images are not proper stereo pairs\n"
                    "- Images have insufficient texture/features\n\n"
                    "Try with real stereo images that show a road surface."
                )
                
                return disp_colored, None, pointcloud_img, None, stats
            except:
                return None, None, None, None, (
                    "Ground plane detection failed.\n\n"
                    "This typically happens when:\n"
                    "- Images don't contain a clear ground plane\n"
                    "- Images are not proper stereo pairs\n"
                    "- Images have insufficient texture/features\n\n"
                    "Please try with real stereo images that show a road or ground surface."
                )
        else:
            return None, None, None, None, f"Error: {str(e)}"
    
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"
    
    # If we got here, ground plane detection succeeded
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points_vis = points[indices]
        else:
            points_vis = points
        
        ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                  c=points_vis[:, 2], cmap='viridis', s=1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Point Cloud')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        img = Image.open(buf)
        pointcloud_img = np.array(img)
        
        # Create mesh visualization
        mesh_img = None
        if result.anomalies and len(result.anomalies) > 0:
            # Try to find an anomaly with a valid mesh
            mesh_found = False
            for anomaly in result.anomalies:
                if anomaly.mesh is not None and anomaly.mesh.vertices.shape[0] > 0:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    vertices = np.array(anomaly.mesh.vertices)
                    if len(vertices) > 5000:
                        indices = np.random.choice(len(vertices), 5000, replace=False)
                        vertices_vis = vertices[indices]
                    else:
                        vertices_vis = vertices
                    
                    ax.scatter(vertices_vis[:, 0], vertices_vis[:, 1], vertices_vis[:, 2], 
                              c=vertices_vis[:, 2], cmap='plasma', s=2)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Z (m)')
                    ax.set_title(f'Volume Mesh ({anomaly.anomaly_type})')
                    
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    plt.close()
                    
                    img = Image.open(buf)
                    mesh_img = np.array(img)
                    mesh_found = True
                    break
            
            # If no mesh found, visualize the point cloud of first anomaly instead
            if not mesh_found and result.anomalies[0].point_cloud.shape[0] > 0:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                anomaly_points = result.anomalies[0].point_cloud
                if len(anomaly_points) > 5000:
                    indices = np.random.choice(len(anomaly_points), 5000, replace=False)
                    points_vis = anomaly_points[indices]
                else:
                    points_vis = anomaly_points
                
                if len(points_vis) > 0:
                    ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                              c=points_vis[:, 2], cmap='plasma', s=2)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Z (m)')
                    ax.set_title(f'Anomaly Point Cloud (No Mesh Generated)')
                else:
                    # Create empty plot with message
                    ax.text(0.5, 0.5, 0.5, 'No valid points for mesh', 
                           ha='center', va='center', fontsize=12)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Z (m)')
                    ax.set_title('Volume Mesh (Insufficient Data)')
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                img = Image.open(buf)
                mesh_img = np.array(img)
        
        stats = f"Pipeline completed successfully\n\n"
        stats += f"Disparity range: {result.disparity_map[result.disparity_map>0].min():.2f} - {result.disparity_map.max():.2f}\n"
        stats += f"Point cloud size: {len(current_points)} points\n"
        
        if len(current_points) > 0:
            stats += f"Depth range: {current_points[:, 2].min():.2f}m - {current_points[:, 2].max():.2f}m\n"
        
        stats += f"Anomalies detected: {len(result.anomalies)}\n"
        
        if result.anomalies:
            total_volume = sum(a.volume_cubic_meters for a in result.anomalies if a.is_valid)
            stats += f"Total volume: {total_volume:.6f} m3 ({total_volume*1000:.2f} liters)\n"
            for i, anomaly in enumerate(result.anomalies):
                stats += f"\nAnomaly {i+1} ({anomaly.anomaly_type}):\n"
                stats += f"  Volume: {anomaly.volume_cubic_meters:.6f} m3 ({anomaly.volume_liters:.2f} L)\n"
                stats += f"  Valid: {anomaly.is_valid}\n"
        
        return disp_colored, v_disp_colored, pointcloud_img, mesh_img, stats
    
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"


def test_preprocessing(left_image, right_image, enhance_contrast, normalize_brightness, filter_noise):
    """Test preprocessing functionality."""
    if left_image is None or right_image is None:
        return None, None, "Please upload both images"
    
    try:
        preprocessor = ImagePreprocessor()
        
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        left_processed = left_gray.copy()
        right_processed = right_gray.copy()
        
        if enhance_contrast:
            left_processed = preprocessor.enhance_contrast(left_processed)
            right_processed = preprocessor.enhance_contrast(right_processed)
        
        if normalize_brightness:
            left_processed, right_processed = preprocessor.normalize_brightness(
                left_processed, right_processed
            )
        
        if filter_noise:
            left_processed = preprocessor.filter_noise(left_processed)
            right_processed = preprocessor.filter_noise(right_processed)
        
        status = f"Preprocessing complete\n"
        status += f"Contrast enhancement: {'ON' if enhance_contrast else 'OFF'}\n"
        status += f"Brightness normalization: {'ON' if normalize_brightness else 'OFF'}\n"
        status += f"Noise filtering: {'ON' if filter_noise else 'OFF'}"
        
        return left_processed, right_processed, status
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def test_disparity(left_image, right_image):
    """Test disparity estimation."""
    global current_disparity
    
    if left_image is None or right_image is None:
        return None, "Please upload both images"
    
    try:
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        sgbm = SGBMEstimator(baseline=0.12, focal_length=700.0)
        disparity = sgbm.compute_disparity(left_gray, right_gray)
        current_disparity = disparity
        
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        valid_disparity = disparity[disparity > 0]
        stats = f"Disparity computed\n"
        stats += f"Min disparity: {valid_disparity.min():.2f}\n"
        stats += f"Max disparity: {valid_disparity.max():.2f}\n"
        stats += f"Mean disparity: {valid_disparity.mean():.2f}\n"
        stats += f"Valid pixels: {len(valid_disparity)} ({100*len(valid_disparity)/disparity.size:.1f}%)"
        
        return disparity_colored, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def test_ground_plane():
    """Test ground plane detection."""
    global current_disparity
    
    if current_disparity is None:
        return None, "Please compute disparity first"
    
    try:
        v_disp_gen = VDisparityGenerator(max_disparity=128)
        v_disparity = v_disp_gen.generate_v_disparity(current_disparity)
        
        v_disp_vis = cv2.normalize(v_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        v_disp_colored = cv2.applyColorMap(v_disp_vis, cv2.COLORMAP_HOT)
        
        stats = f"V-disparity generated\n"
        stats += f"Shape: {v_disparity.shape}\n"
        stats += f"Non-zero pixels: {np.count_nonzero(v_disparity)}"
        
        return v_disp_colored, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def test_reconstruction(min_depth, max_depth, remove_outliers):
    """Test 3D reconstruction."""
    global current_disparity, current_points
    
    if current_disparity is None:
        return None, "Please compute disparity first"
    
    try:
        Q = np.array([
            [1, 0, 0, -320],
            [0, 1, 0, -240],
            [0, 0, 0, 700],
            [0, 0, 1/0.12, 0]
        ], dtype=np.float32)
        
        pcg = PointCloudGenerator(Q_matrix=Q, min_depth=min_depth, max_depth=max_depth)
        points = pcg.reproject_to_3d(current_disparity, apply_depth_filter=True)
        
        if remove_outliers and len(points) > 100:
            outlier_remover = OutlierRemover()
            points = outlier_remover.remove_statistical_outliers(points)
        
        current_points = points
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points_vis = points[indices]
        else:
            points_vis = points
        
        ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                  c=points_vis[:, 2], cmap='viridis', s=1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Point Cloud')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        img = Image.open(buf)
        img_array = np.array(img)
        
        stats = f"Point cloud generated\n"
        stats += f"Total points: {len(points)}\n"
        stats += f"Depth range: {points[:, 2].min():.2f}m to {points[:, 2].max():.2f}m\n"
        stats += f"Outlier removal: {'ON' if remove_outliers else 'OFF'}"
        
        return img_array, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def test_volume(alpha, ground_plane_z):
    """Test volume calculation."""
    global current_points
    
    if current_points is None:
        return None, "Please generate point cloud first"
    
    try:
        alpha_gen = AlphaShapeGenerator(alpha=alpha)
        mesh = alpha_gen.generate_alpha_shape(current_points)
        
        capper = MeshCapper()
        capped_mesh = capper.cap_mesh(mesh, ground_plane_z=ground_plane_z)
        
        calc = VolumeCalculator()
        volume = calc.calculate_volume(capped_mesh)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = np.array(capped_mesh.vertices)
        if len(vertices) > 5000:
            indices = np.random.choice(len(vertices), 5000, replace=False)
            vertices_vis = vertices[indices]
        else:
            vertices_vis = vertices
        
        ax.scatter(vertices_vis[:, 0], vertices_vis[:, 1], vertices_vis[:, 2], 
                  c=vertices_vis[:, 2], cmap='plasma', s=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Alpha Shape Mesh')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        img = Image.open(buf)
        img_array = np.array(img)
        
        stats = f"Volume calculated\n"
        stats += f"Volume: {volume:.6f} m3 ({volume*1000:.2f} liters)\n"
        stats += f"Alpha parameter: {alpha}\n"
        stats += f"Mesh vertices: {len(capped_mesh.vertices)}\n"
        stats += f"Mesh faces: {len(capped_mesh.faces)}"
        
        return img_array, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Advanced Stereo Vision Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Advanced Stereo Vision Pipeline - Interactive Testing Interface
    
    Test all functionalities of the stereo vision pipeline.
    """)
    
    with gr.Tabs():
        with gr.Tab("Full Pipeline"):
            gr.Markdown("### Run the complete pipeline end-to-end")
            
            with gr.Row():
                with gr.Column():
                    full_left = gr.Image(label="Left Image", type="numpy")
                    full_right = gr.Image(label="Right Image", type="numpy")
                    
                    with gr.Row():
                        full_baseline = gr.Slider(0.01, 1.0, value=0.12, label="Baseline (m)")
                        full_focal = gr.Slider(100, 2000, value=700, label="Focal Length (px)")
                    
                    full_run_btn = gr.Button("Run Full Pipeline", variant="primary")
                
                with gr.Column():
                    full_disparity = gr.Image(label="Disparity Map")
                    full_vdisp = gr.Image(label="V-Disparity")
            
            with gr.Row():
                full_pointcloud = gr.Image(label="Point Cloud")
                full_mesh = gr.Image(label="Volume Mesh")
            
            full_status = gr.Textbox(label="Status", lines=10)
            
            full_run_btn.click(
                run_full_pipeline,
                inputs=[full_left, full_right, full_baseline, full_focal],
                outputs=[full_disparity, full_vdisp, full_pointcloud, full_mesh, full_status]
            )
        
        with gr.Tab("Preprocessing"):
            gr.Markdown("### Test image preprocessing capabilities")
            
            with gr.Row():
                with gr.Column():
                    prep_left = gr.Image(label="Left Image", type="numpy")
                    prep_right = gr.Image(label="Right Image", type="numpy")
                    
                    prep_contrast = gr.Checkbox(label="Enhance Contrast (CLAHE)", value=True)
                    prep_brightness = gr.Checkbox(label="Normalize Brightness", value=True)
                    prep_noise = gr.Checkbox(label="Filter Noise (Bilateral)", value=False)
                    
                    prep_btn = gr.Button("Process Images", variant="primary")
                
                with gr.Column():
                    prep_left_out = gr.Image(label="Processed Left")
                    prep_right_out = gr.Image(label="Processed Right")
                    prep_status = gr.Textbox(label="Status", lines=5)
            
            prep_btn.click(
                test_preprocessing,
                inputs=[prep_left, prep_right, prep_contrast, prep_brightness, prep_noise],
                outputs=[prep_left_out, prep_right_out, prep_status]
            )
        
        with gr.Tab("Disparity Estimation"):
            gr.Markdown("### Compute disparity maps using SGBM")
            
            with gr.Row():
                with gr.Column():
                    disp_left = gr.Image(label="Left Image", type="numpy")
                    disp_right = gr.Image(label="Right Image", type="numpy")
                    disp_btn = gr.Button("Compute Disparity", variant="primary")
                
                with gr.Column():
                    disp_out = gr.Image(label="Disparity Map (Colored)")
                    disp_status = gr.Textbox(label="Statistics", lines=8)
            
            disp_btn.click(
                test_disparity,
                inputs=[disp_left, disp_right],
                outputs=[disp_out, disp_status]
            )
        
        with gr.Tab("Ground Plane Detection"):
            gr.Markdown("### Detect ground plane using V-disparity")
            
            with gr.Row():
                with gr.Column():
                    gp_btn = gr.Button("Detect Ground Plane", variant="primary")
                
                with gr.Column():
                    gp_out = gr.Image(label="V-Disparity Map")
                    gp_status = gr.Textbox(label="Results", lines=8)
            
            gp_btn.click(
                test_ground_plane,
                inputs=[],
                outputs=[gp_out, gp_status]
            )
        
        with gr.Tab("3D Reconstruction"):
            gr.Markdown("### Generate 3D point cloud from disparity")
            
            with gr.Row():
                with gr.Column():
                    recon_min_depth = gr.Slider(0.1, 5.0, value=0.5, label="Min Depth (m)")
                    recon_max_depth = gr.Slider(5.0, 50.0, value=20.0, label="Max Depth (m)")
                    recon_outliers = gr.Checkbox(label="Remove Outliers", value=True)
                    recon_btn = gr.Button("Generate Point Cloud", variant="primary")
                
                with gr.Column():
                    recon_out = gr.Image(label="Point Cloud Visualization")
                    recon_status = gr.Textbox(label="Statistics", lines=8)
            
            recon_btn.click(
                test_reconstruction,
                inputs=[recon_min_depth, recon_max_depth, recon_outliers],
                outputs=[recon_out, recon_status]
            )
        
        with gr.Tab("Volume Calculation"):
            gr.Markdown("### Calculate volume using alpha shapes")
            
            with gr.Row():
                with gr.Column():
                    vol_alpha = gr.Slider(0.01, 1.0, value=0.1, label="Alpha Parameter")
                    vol_ground_z = gr.Slider(-5.0, 5.0, value=0.0, label="Ground Plane Z (m)")
                    vol_btn = gr.Button("Calculate Volume", variant="primary")
                
                with gr.Column():
                    vol_out = gr.Image(label="Alpha Shape Mesh")
                    vol_status = gr.Textbox(label="Results", lines=8)
            
            vol_btn.click(
                test_volume,
                inputs=[vol_alpha, vol_ground_z],
                outputs=[vol_out, vol_status]
            )
    
    gr.Markdown("""
    ---
    ### Usage Instructions
    
    1. **Full Pipeline**: Upload stereo images and run the complete pipeline
    2. **Preprocessing**: Test contrast enhancement, brightness normalization, and noise filtering
    3. **Disparity Estimation**: Compute disparity maps
    4. **Ground Plane Detection**: Detect ground plane using V-disparity
    5. **3D Reconstruction**: Generate point clouds
    6. **Volume Calculation**: Calculate volumes using alpha shapes
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
