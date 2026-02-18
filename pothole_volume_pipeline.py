#!/usr/bin/env python3
"""
Advanced Stereo Vision Pipeline for Volumetric Road Anomaly Detection

This is the main entry point for the advanced stereo vision system. It provides
a command-line interface for calibration, processing, and batch operations.

Usage:
    # Calibrate the system
    python pothole_volume_pipeline.py calibrate --left-images calib/left/*.png --right-images calib/right/*.png --output calibration.npz
    
    # Process a single stereo pair
    python pothole_volume_pipeline.py process --left left.png --right right.png --calibration calibration.npz --output results/
    
    # Process a batch of images
    python pothole_volume_pipeline.py batch --left-dir images/left/ --right-dir images/right/ --calibration calibration.npz --output results/

Requirements: All requirements integration
"""

import argparse
import sys
from pathlib import Path
from typing import List
import numpy as np
import cv2
import glob

from stereo_vision.pipeline import StereoVisionPipeline, create_pipeline
from stereo_vision.config import PipelineConfig, create_default_config, create_high_accuracy_config


def load_images_from_pattern(pattern: str) -> List[np.ndarray]:
    """Load images matching a glob pattern."""
    image_paths = sorted(glob.glob(pattern))
    
    if not image_paths:
        raise ValueError(f"No images found matching pattern: {pattern}")
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image {path}")
            continue
        images.append(img)
    
    print(f"Loaded {len(images)} images from {pattern}")
    return images


def load_images_from_directory(directory: str, extension: str = "*.png") -> List[np.ndarray]:
    """Load all images from a directory."""
    pattern = str(Path(directory) / extension)
    return load_images_from_pattern(pattern)


def calibrate_command(args):
    """Execute calibration command."""
    print("="*60)
    print("STEREO CALIBRATION")
    print("="*60)
    
    # Load calibration images
    print("\nLoading calibration images...")
    left_images = load_images_from_pattern(args.left_images)
    right_images = load_images_from_pattern(args.right_images)
    
    if len(left_images) != len(right_images):
        print(f"Error: Number of left images ({len(left_images)}) != right images ({len(right_images)})")
        sys.exit(1)
    
    if len(left_images) < 10:
        print(f"Warning: Only {len(left_images)} image pairs. Recommend at least 10 for good calibration.")
    
    # Create pipeline
    config = create_high_accuracy_config() if args.high_accuracy else create_default_config()
    pipeline = StereoVisionPipeline(config)
    
    # Perform calibration
    print("\nStarting calibration...")
    try:
        stereo_params = pipeline.calibrate(left_images, right_images)
        
        # Save calibration
        pipeline.save_calibration(args.output)
        
        print("\n" + "="*60)
        print("CALIBRATION SUCCESSFUL")
        print("="*60)
        print(f"Baseline: {stereo_params.baseline:.4f} m")
        print(f"Left camera RMS error: {stereo_params.left_camera.reprojection_error:.4f} pixels")
        print(f"Right camera RMS error: {stereo_params.right_camera.reprojection_error:.4f} pixels")
        print(f"Calibration saved to: {args.output}")
        print("="*60)
        
    except Exception as e:
        print(f"\nCalibration failed: {e}")
        sys.exit(1)


def process_command(args):
    """Execute single pair processing command."""
    print("="*60)
    print("PROCESSING STEREO PAIR")
    print("="*60)
    
    # Load images
    print("\nLoading images...")
    left_image = cv2.imread(args.left)
    right_image = cv2.imread(args.right)
    
    if left_image is None:
        print(f"Error: Could not load left image: {args.left}")
        sys.exit(1)
    
    if right_image is None:
        print(f"Error: Could not load right image: {args.right}")
        sys.exit(1)
    
    # Create pipeline
    config_file = args.config if hasattr(args, 'config') and args.config else None
    pipeline = create_pipeline(config_file)
    
    # Load calibration
    print(f"Loading calibration from {args.calibration}...")
    pipeline.load_calibration(args.calibration)
    
    # Process
    print("\nProcessing stereo pair...")
    try:
        result = pipeline.process_stereo_pair(
            left_image,
            right_image,
            generate_diagnostics=True
        )
        
        # Print results
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Anomalies detected: {len(result.anomalies)}")
        
        for idx, anomaly in enumerate(result.anomalies):
            print(f"\nAnomaly {idx + 1}: {anomaly.anomaly_type.upper()}")
            print(f"  Bounding box: {anomaly.bounding_box}")
            print(f"  Volume: {anomaly.volume_liters:.2f} liters ({anomaly.volume_cubic_cm:.0f} cm³)")
            print(f"  Uncertainty: ±{anomaly.uncertainty_cubic_meters * 1000:.2f} liters")
            print(f"  Area: {anomaly.area_square_meters:.3f} m²")
            print(f"  Valid: {anomaly.is_valid}")
            if not anomaly.is_valid:
                print(f"  Message: {anomaly.validation_message}")
        
        print(f"\nQuality Metrics:")
        print(f"  LRC error rate: {result.quality_metrics.lrc_error_rate:.2f}%")
        if result.quality_metrics.planarity_rmse is not None:
            print(f"  Planarity RMSE: {result.quality_metrics.planarity_rmse:.3f}")
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save disparity
            np.save(output_path / "disparity.npy", result.disparity_map)
            
            # Save diagnostic panel
            if result.diagnostic_panel is not None:
                cv2.imwrite(str(output_path / "diagnostics.png"), result.diagnostic_panel)
                print(f"\nDiagnostic panel saved to: {output_path / 'diagnostics.png'}")
            
            print(f"Results saved to: {args.output}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def batch_command(args):
    """Execute batch processing command."""
    print("="*60)
    print("BATCH PROCESSING")
    print("="*60)
    
    # Load images
    print("\nLoading images...")
    left_images = load_images_from_directory(args.left_dir)
    right_images = load_images_from_directory(args.right_dir)
    
    if len(left_images) != len(right_images):
        print(f"Error: Number of left images ({len(left_images)}) != right images ({len(right_images)})")
        sys.exit(1)
    
    # Create image pairs
    image_pairs = list(zip(left_images, right_images))
    
    # Create pipeline
    config_file = args.config if hasattr(args, 'config') and args.config else None
    pipeline = create_pipeline(config_file)
    
    # Load calibration
    print(f"Loading calibration from {args.calibration}...")
    pipeline.load_calibration(args.calibration)
    
    # Process batch
    print(f"\nProcessing {len(image_pairs)} stereo pairs...")
    try:
        results = pipeline.process_batch(image_pairs, output_dir=args.output)
        
        print("\nBatch processing complete!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\nBatch processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Stereo Vision Pipeline for Road Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate the system
  %(prog)s calibrate --left-images "calib/left/*.png" --right-images "calib/right/*.png" --output calibration.npz
  
  # Process a single stereo pair
  %(prog)s process --left left.png --right right.png --calibration calibration.npz --output results/
  
  # Process a batch of images
  %(prog)s batch --left-dir images/left/ --right-dir images/right/ --calibration calibration.npz --output results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate stereo camera system')
    calibrate_parser.add_argument('--left-images', required=True, help='Glob pattern for left calibration images')
    calibrate_parser.add_argument('--right-images', required=True, help='Glob pattern for right calibration images')
    calibrate_parser.add_argument('--output', required=True, help='Output calibration file (.npz)')
    calibrate_parser.add_argument('--high-accuracy', action='store_true', help='Use high-accuracy configuration')
    calibrate_parser.set_defaults(func=calibrate_command)
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single stereo pair')
    process_parser.add_argument('--left', required=True, help='Left image file')
    process_parser.add_argument('--right', required=True, help='Right image file')
    process_parser.add_argument('--calibration', required=True, help='Calibration file (.npz)')
    process_parser.add_argument('--output', help='Output directory for results')
    process_parser.add_argument('--config', help='Configuration file (JSON)')
    process_parser.set_defaults(func=process_command)
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process a batch of stereo pairs')
    batch_parser.add_argument('--left-dir', required=True, help='Directory containing left images')
    batch_parser.add_argument('--right-dir', required=True, help='Directory containing right images')
    batch_parser.add_argument('--calibration', required=True, help='Calibration file (.npz)')
    batch_parser.add_argument('--output', required=True, help='Output directory for results')
    batch_parser.add_argument('--config', help='Configuration file (JSON)')
    batch_parser.set_defaults(func=batch_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
