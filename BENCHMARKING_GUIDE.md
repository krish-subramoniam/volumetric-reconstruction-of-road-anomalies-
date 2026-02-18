# Performance Benchmarking Guide

This guide explains how to benchmark the advanced stereo vision pipeline and generate performance comparison reports.

## Overview

The benchmarking system compares the advanced pipeline against a baseline implementation, measuring:

- **Processing Performance**: Time and memory usage
- **Quality Metrics**: LRC error rate, planarity RMSE
- **Detection Accuracy**: Anomaly detection and volume calculation
- **Robustness**: Success rate across various conditions

## Quick Start

### 1. Run Benchmark with Synthetic Data

The easiest way to test the benchmarking system:

```bash
python benchmark_pipeline.py \
    --calibration calibration.npz \
    --synthetic \
    --num-synthetic 10 \
    --output benchmark_report.json
```

This generates synthetic stereo pairs and benchmarks both implementations.

### 2. Run Benchmark with Real Data

For more realistic results, use actual stereo image pairs:

```bash
python benchmark_pipeline.py \
    --calibration calibration.npz \
    --left-dir test_images/left/ \
    --right-dir test_images/right/ \
    --output benchmark_report.json
```

### 3. Generate Performance Report

After running the benchmark, generate a comprehensive report with visualizations:

```bash
python generate_performance_report.py \
    --input benchmark_report.json \
    --output-dir performance_report
```

This creates:
- `PERFORMANCE_REPORT.md` - Comprehensive markdown report
- `processing_time_comparison.png` - Processing time chart
- `quality_comparison.png` - Quality metrics chart
- `detection_comparison.png` - Detection results chart
- `improvement_radar.png` - Overall improvements radar chart
- `success_rate_comparison.png` - Success rate comparison

## Benchmark Output

### Console Output

The benchmark prints real-time progress and a summary:

```
Benchmarking 10 image pairs...
  Pair 1/10... Baseline: 0.45s, Advanced: 2.31s
  Pair 2/10... Baseline: 0.43s, Advanced: 2.28s
  ...

======================================================================
BENCHMARK SUMMARY
======================================================================

Test Configuration:
  Number of test pairs: 10
  Baseline success rate: 100.0%
  Advanced success rate: 100.0%

Processing Performance:
  Baseline avg time: 0.441s
  Advanced avg time: 2.287s
  Speedup factor: 0.19x
    (Advanced is 5.18x slower, but more accurate)

Quality Metrics:
  Baseline LRC error rate: 30.00%
  Advanced LRC error rate: 8.45%
  Quality improvement: 71.8%

Detection Results:
  Baseline avg anomalies: 2.3
  Advanced avg anomalies: 3.1
  Baseline avg volume: 12.45 liters
  Advanced avg volume: 15.67 liters
  Volume difference: 25.9%

Overall Improvements:
  ✓ Accuracy improvement: 50.0%
  ✓ Quality improvement: 71.8%
  ✓ Robustness improvement: 0.0%
======================================================================
```

### JSON Report

The `benchmark_report.json` contains detailed results:

```json
{
  "summary": {
    "num_test_pairs": 10,
    "baseline_success_rate": 100.0,
    "advanced_success_rate": 100.0
  },
  "performance": {
    "baseline_avg_time": 0.441,
    "advanced_avg_time": 2.287,
    "speedup_factor": 0.19,
    "baseline_avg_memory_mb": 45.2,
    "advanced_avg_memory_mb": 128.7
  },
  "quality": {
    "baseline_avg_lrc_error": 30.0,
    "advanced_avg_lrc_error": 8.45,
    "quality_improvement_percent": 71.8
  },
  "accuracy": {
    "advanced_avg_planarity_rmse": 0.0234,
    "accuracy_improvement_percent": 50.0
  },
  "detection": {
    "baseline_avg_anomalies": 2.3,
    "advanced_avg_anomalies": 3.1,
    "baseline_avg_volume_liters": 12.45,
    "advanced_avg_volume_liters": 15.67,
    "volume_difference_percent": 25.9
  },
  "robustness": {
    "robustness_improvement_percent": 0.0
  },
  "detailed_results": {
    "baseline": [...],
    "advanced": [...]
  }
}
```

## Understanding the Results

### Processing Time

- **Baseline**: Simple block matching, no validation or filtering
- **Advanced**: SGBM + LRC + WLS + advanced meshing

The advanced pipeline is typically 3-5x slower but provides significantly better quality.

### Quality Metrics

**LRC Error Rate**: Percentage of pixels failing left-right consistency check
- Lower is better
- Baseline: ~30% (no LRC checking)
- Advanced: ~5-15% (with LRC validation)

**Planarity RMSE**: Root mean square error of ground plane fit
- Lower is better
- Measures geometric accuracy
- Only available for advanced pipeline

### Detection Results

**Anomalies Detected**: Number of potholes/humps found
- Advanced pipeline typically detects more anomalies due to better disparity quality

**Volume Measurements**: Total volume in liters
- Advanced pipeline provides more accurate measurements
- Includes uncertainty estimates

### Success Rate

Percentage of image pairs successfully processed
- Advanced pipeline is more robust to challenging conditions

## Baseline Implementation

The baseline implementation represents a simple stereo vision pipeline:

### Features
- Simple block matching (StereoBM)
- No LRC validation
- No WLS filtering
- Simple threshold-based anomaly detection
- Basic 2.5D volume estimation

### Limitations
- High disparity error rate
- Poor handling of textureless regions
- No occlusion detection
- Inaccurate volume measurements
- Limited robustness

## Advanced Implementation

The advanced pipeline includes state-of-the-art features:

### Features
1. **CharuCo Calibration**: Sub-pixel accuracy
2. **SGBM Disparity**: Semi-Global Block Matching
3. **LRC Validation**: Left-Right Consistency checking
4. **WLS Filtering**: Weighted Least Squares refinement
5. **V-Disparity**: Robust ground plane detection
6. **Outlier Removal**: Statistical filtering
7. **Alpha Shapes**: Concave hull meshing
8. **Watertight Volume**: Divergence Theorem integration
9. **Quality Metrics**: Comprehensive validation

### Advantages
- High disparity quality (low LRC error)
- Excellent geometric accuracy
- Robust anomaly detection
- Precise volume measurements
- Comprehensive diagnostics

## Customizing Benchmarks

### Test Different Configurations

You can benchmark different pipeline configurations:

```python
from stereo_vision.config import create_high_accuracy_config, create_fast_config
from benchmark_pipeline import PipelineBenchmark

# High accuracy configuration
config_high = create_high_accuracy_config()
pipeline_high = StereoVisionPipeline(config_high)

# Fast configuration
config_fast = create_fast_config()
pipeline_fast = StereoVisionPipeline(config_fast)

# Benchmark both
# ... (custom benchmarking code)
```

### Add Custom Metrics

Extend the `BenchmarkResult` dataclass to include additional metrics:

```python
@dataclass
class ExtendedBenchmarkResult(BenchmarkResult):
    custom_metric: float
    another_metric: str
```

### Test Specific Scenarios

Create custom test datasets for specific scenarios:

```python
# Test low-light conditions
low_light_pairs = load_low_light_images()
results = benchmark.benchmark_batch(low_light_pairs)

# Test high-speed capture
motion_blur_pairs = load_motion_blur_images()
results = benchmark.benchmark_batch(motion_blur_pairs)
```

## Performance Optimization

If the advanced pipeline is too slow for your application:

### 1. Reduce Disparity Range

```python
config.sgbm.num_disparities = 64  # Instead of 128
```

### 2. Increase Block Size

```python
config.sgbm.block_size = 11  # Instead of 7
```

### 3. Disable WLS Filtering

```python
config.wls.enabled = False
```

### 4. Reduce Outlier Removal Iterations

```python
config.outlier_removal.k_neighbors = 10  # Instead of 20
```

### 5. Use Fast Configuration

```python
from stereo_vision.config import create_fast_config
config = create_fast_config()
```

## Interpreting Trade-offs

### Speed vs Accuracy

- **Baseline**: Fast but inaccurate
- **Advanced (Fast Config)**: Moderate speed, good accuracy
- **Advanced (High Accuracy Config)**: Slow but excellent accuracy

### Memory vs Quality

- **Lower resolution**: Faster, less memory, lower accuracy
- **Higher resolution**: Slower, more memory, higher accuracy

### Robustness vs Speed

- **Minimal validation**: Fast but fragile
- **Full validation**: Slower but robust

## Troubleshooting

### Benchmark Fails to Run

**Issue**: Calibration file not found
```
Solution: Ensure calibration.npz exists and path is correct
```

**Issue**: Out of memory
```
Solution: Reduce number of test pairs or image resolution
```

### Unexpected Results

**Issue**: Baseline performs better than advanced
```
Possible causes:
- Test data is too simple
- Advanced pipeline not properly configured
- Calibration quality issues
```

**Issue**: Both implementations fail
```
Possible causes:
- Invalid calibration
- Corrupted test images
- Incorrect image format
```

### Performance Issues

**Issue**: Benchmark takes too long
```
Solutions:
- Reduce number of test pairs
- Use synthetic data
- Use fast configuration
```

## Best Practices

1. **Use Representative Data**: Test with images similar to production scenarios
2. **Multiple Runs**: Run benchmarks multiple times for statistical significance
3. **Varied Conditions**: Test different lighting, distances, and road types
4. **Document Configuration**: Save configuration used for each benchmark
5. **Version Control**: Track benchmark results over time
6. **Realistic Expectations**: Advanced features require more processing time

## Example Workflow

Complete workflow for benchmarking and reporting:

```bash
# 1. Prepare test data
mkdir -p test_images/left test_images/right

# 2. Run calibration (if needed)
python pothole_volume_pipeline.py calibrate \
    --left-images "calib/left/*.png" \
    --right-images "calib/right/*.png" \
    --output calibration.npz

# 3. Run benchmark
python benchmark_pipeline.py \
    --calibration calibration.npz \
    --left-dir test_images/left \
    --right-dir test_images/right \
    --output benchmark_report.json

# 4. Generate report
python generate_performance_report.py \
    --input benchmark_report.json \
    --output-dir performance_report

# 5. View results
cat performance_report/PERFORMANCE_REPORT.md
```

## Continuous Benchmarking

For ongoing development, integrate benchmarking into your workflow:

```bash
# Create benchmark script
cat > run_benchmark.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
python benchmark_pipeline.py \
    --calibration calibration.npz \
    --synthetic \
    --num-synthetic 20 \
    --output "benchmarks/benchmark_${DATE}.json"
python generate_performance_report.py \
    --input "benchmarks/benchmark_${DATE}.json" \
    --output-dir "benchmarks/report_${DATE}"
EOF

chmod +x run_benchmark.sh

# Run regularly
./run_benchmark.sh
```

## Conclusion

The benchmarking system provides comprehensive performance comparison between baseline and advanced implementations. Use it to:

- Validate improvements
- Identify bottlenecks
- Optimize configurations
- Document performance characteristics
- Make informed deployment decisions

For questions or issues, refer to the main README or open an issue.
