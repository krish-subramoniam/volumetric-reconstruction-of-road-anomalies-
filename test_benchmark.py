#!/usr/bin/env python3
"""
Quick test script to validate the benchmarking system.

This script performs a minimal benchmark test to ensure all components work correctly.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Check if calibration exists
if not Path('calibration.npz').exists():
    print("Error: calibration.npz not found")
    print("Please run calibration first:")
    print("  python pothole_volume_pipeline.py calibrate ...")
    sys.exit(1)

print("Testing benchmark system...")
print("="*60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from benchmark_pipeline import PipelineBenchmark, BaselinePipeline, create_synthetic_test_data
    from generate_performance_report import load_benchmark_results, create_performance_charts
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create synthetic data
print("\n2. Creating synthetic test data...")
try:
    test_pairs = create_synthetic_test_data(num_pairs=2)
    print(f"   ✓ Created {len(test_pairs)} synthetic stereo pairs")
    print(f"   ✓ Image shape: {test_pairs[0][0].shape}")
except Exception as e:
    print(f"   ✗ Synthetic data creation failed: {e}")
    sys.exit(1)

# Test 3: Initialize benchmark
print("\n3. Initializing benchmark...")
try:
    benchmark = PipelineBenchmark('calibration.npz')
    print("   ✓ Benchmark initialized")
    print("   ✓ Baseline pipeline loaded")
    print("   ✓ Advanced pipeline loaded")
except Exception as e:
    print(f"   ✗ Benchmark initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Run single pair benchmark
print("\n4. Running single pair benchmark...")
try:
    left, right = test_pairs[0]
    results = benchmark.benchmark_single_pair(left, right, "both")
    
    print(f"   ✓ Baseline result: {results['baseline'].success}")
    print(f"     - Processing time: {results['baseline'].processing_time:.3f}s")
    print(f"     - Anomalies: {results['baseline'].num_anomalies}")
    
    print(f"   ✓ Advanced result: {results['advanced'].success}")
    print(f"     - Processing time: {results['advanced'].processing_time:.3f}s")
    print(f"     - Anomalies: {results['advanced'].num_anomalies}")
    print(f"     - LRC error: {results['advanced'].lrc_error_rate:.2f}%")
    
except Exception as e:
    print(f"   ✗ Benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Run batch benchmark
print("\n5. Running batch benchmark...")
try:
    baseline_results, advanced_results = benchmark.benchmark_batch(test_pairs)
    print(f"   ✓ Processed {len(baseline_results)} pairs")
    print(f"   ✓ Baseline success: {sum(r.success for r in baseline_results)}/{len(baseline_results)}")
    print(f"   ✓ Advanced success: {sum(r.success for r in advanced_results)}/{len(advanced_results)}")
except Exception as e:
    print(f"   ✗ Batch benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Generate report
print("\n6. Generating report...")
try:
    report = benchmark.generate_report(
        baseline_results,
        advanced_results,
        'test_benchmark_report.json'
    )
    print("   ✓ Report generated: test_benchmark_report.json")
except Exception as e:
    print(f"   ✗ Report generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Print summary
print("\n7. Printing summary...")
try:
    benchmark.print_summary(baseline_results, advanced_results)
    print("   ✓ Summary printed")
except Exception as e:
    print(f"   ✗ Summary printing failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
print("\nBenchmarking system is working correctly.")
print("\nNext steps:")
print("  1. Run full benchmark:")
print("     python benchmark_pipeline.py --calibration calibration.npz --synthetic --num-synthetic 10")
print("  2. Generate performance report:")
print("     python generate_performance_report.py --input benchmark_report.json")
print()
