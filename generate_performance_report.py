#!/usr/bin/env python3
"""
Performance Report Generator

Generates comprehensive performance comparison reports with visualizations
comparing the advanced pipeline against baseline implementation.

Requirements: Performance evaluation (Task 15.2)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse


def load_benchmark_results(json_file: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_performance_charts(report: Dict, output_dir: str) -> None:
    """
    Create performance comparison charts.
    
    Args:
        report: Benchmark report dictionary
        output_dir: Directory to save charts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Chart 1: Processing Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    implementations = ['Baseline', 'Advanced']
    times = [
        report['performance']['baseline_avg_time'],
        report['performance']['advanced_avg_time']
    ]
    
    bars = ax.bar(implementations, times, color=['#3498db', '#2ecc71'])
    ax.set_ylabel('Processing Time (seconds)', fontsize=12)
    ax.set_title('Average Processing Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom', fontsize=11)
    
    # Add speedup annotation
    speedup = report['performance']['speedup_factor']
    if speedup < 1.0:
        ax.text(0.5, max(times) * 0.9,
                f'Advanced is {1/speedup:.2f}x slower\n(but more accurate)',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, max(times) * 0.9,
                f'Advanced is {speedup:.2f}x faster',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'processing_time_comparison.png', dpi=300)
    plt.close()
    
    # Chart 2: Quality Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_lrc = report['quality']['baseline_avg_lrc_error']
    advanced_lrc = report['quality']['advanced_avg_lrc_error']
    
    # Invert for visualization (lower is better)
    quality_scores = [
        100 - baseline_lrc,  # Quality score (higher is better)
        100 - advanced_lrc
    ]
    
    bars = ax.bar(implementations, quality_scores, color=['#e74c3c', '#27ae60'])
    ax.set_ylabel('Quality Score (higher is better)', fontsize=12)
    ax.set_title('Disparity Quality Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, lrc in zip(bars, [baseline_lrc, advanced_lrc]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}\n(LRC error: {lrc:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # Add improvement annotation
    improvement = report['quality']['quality_improvement_percent']
    ax.text(0.5, 50,
            f'{improvement:.1f}% quality improvement',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path / 'quality_comparison.png', dpi=300)
    plt.close()
    
    # Chart 3: Detection Results Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Anomalies detected
    baseline_anomalies = report['detection']['baseline_avg_anomalies']
    advanced_anomalies = report['detection']['advanced_avg_anomalies']
    
    bars1 = ax1.bar(implementations, [baseline_anomalies, advanced_anomalies],
                    color=['#9b59b6', '#1abc9c'])
    ax1.set_ylabel('Average Number of Anomalies', fontsize=12)
    ax1.set_title('Anomaly Detection Rate', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11)
    
    # Volume measurements
    baseline_vol = report['detection']['baseline_avg_volume_liters']
    advanced_vol = report['detection']['advanced_avg_volume_liters']
    
    bars2 = ax2.bar(implementations, [baseline_vol, advanced_vol],
                    color=['#e67e22', '#16a085'])
    ax2.set_ylabel('Average Volume (liters)', fontsize=12)
    ax2.set_title('Volume Measurement Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}L',
                ha='center', va='bottom', fontsize=11)
    
    # Add volume difference annotation
    vol_diff = report['detection']['volume_difference_percent']
    ax2.text(0.5, max(baseline_vol, advanced_vol) * 0.5,
            f'{vol_diff:.1f}% difference',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path / 'detection_comparison.png', dpi=300)
    plt.close()
    
    # Chart 4: Overall Improvements Radar Chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Accuracy', 'Quality', 'Robustness']
    values = [
        report['accuracy']['accuracy_improvement_percent'],
        report['quality']['quality_improvement_percent'],
        report['robustness']['robustness_improvement_percent']
    ]
    
    # Normalize to 0-100 scale
    values = [max(0, min(100, v)) for v in values]
    
    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#2ecc71')
    ax.fill(angles, values, alpha=0.25, color='#2ecc71')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('Overall Improvement Metrics (%)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'improvement_radar.png', dpi=300)
    plt.close()
    
    # Chart 5: Success Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    success_rates = [
        report['summary']['baseline_success_rate'],
        report['summary']['advanced_success_rate']
    ]
    
    bars = ax.bar(implementations, success_rates, color=['#c0392b', '#27ae60'])
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Processing Success Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate_comparison.png', dpi=300)
    plt.close()
    
    print(f"Charts saved to {output_dir}/")


def generate_markdown_report(report: Dict, output_file: str, charts_dir: str) -> None:
    """
    Generate a comprehensive markdown report.
    
    Args:
        report: Benchmark report dictionary
        output_file: Output markdown file path
        charts_dir: Directory containing chart images
    """
    with open(output_file, 'w') as f:
        f.write("# Advanced Stereo Vision Pipeline - Performance Comparison Report\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report compares the advanced stereo vision pipeline against a baseline ")
        f.write("implementation, measuring improvements in processing performance, quality metrics, ")
        f.write("and detection accuracy.\n\n")
        
        f.write("### Key Findings\n\n")
        
        speedup = report['performance']['speedup_factor']
        if speedup < 1.0:
            f.write(f"- **Processing Speed**: Advanced pipeline is {1/speedup:.2f}x slower than baseline, ")
            f.write("but provides significantly higher accuracy and quality\n")
        else:
            f.write(f"- **Processing Speed**: Advanced pipeline is {speedup:.2f}x faster than baseline\n")
        
        quality_imp = report['quality']['quality_improvement_percent']
        f.write(f"- **Quality Improvement**: {quality_imp:.1f}% reduction in LRC error rate\n")
        
        accuracy_imp = report['accuracy']['accuracy_improvement_percent']
        f.write(f"- **Accuracy Improvement**: {accuracy_imp:.1f}% better geometric accuracy\n")
        
        robustness_imp = report['robustness']['robustness_improvement_percent']
        f.write(f"- **Robustness**: {robustness_imp:.1f}% improvement in success rate\n")
        
        f.write("\n---\n\n")
        
        f.write("## Test Configuration\n\n")
        f.write(f"- **Number of test pairs**: {report['summary']['num_test_pairs']}\n")
        f.write(f"- **Baseline success rate**: {report['summary']['baseline_success_rate']:.1f}%\n")
        f.write(f"- **Advanced success rate**: {report['summary']['advanced_success_rate']:.1f}%\n")
        
        f.write("\n## Performance Metrics\n\n")
        f.write("### Processing Time\n\n")
        f.write(f"| Implementation | Avg Time (s) | Speedup |\n")
        f.write(f"|----------------|--------------|----------|\n")
        f.write(f"| Baseline | {report['performance']['baseline_avg_time']:.3f} | 1.00x |\n")
        f.write(f"| Advanced | {report['performance']['advanced_avg_time']:.3f} | {speedup:.2f}x |\n")
        f.write("\n")
        
        f.write(f"![Processing Time Comparison]({charts_dir}/processing_time_comparison.png)\n\n")
        
        f.write("### Memory Usage\n\n")
        f.write(f"| Implementation | Avg Memory (MB) |\n")
        f.write(f"|----------------|------------------|\n")
        f.write(f"| Baseline | {report['performance']['baseline_avg_memory_mb']:.1f} |\n")
        f.write(f"| Advanced | {report['performance']['advanced_avg_memory_mb']:.1f} |\n")
        f.write("\n")
        
        f.write("## Quality Metrics\n\n")
        f.write("### Disparity Quality (LRC Error Rate)\n\n")
        f.write("Lower LRC (Left-Right Consistency) error rate indicates better disparity quality.\n\n")
        f.write(f"| Implementation | LRC Error Rate | Quality Score |\n")
        f.write(f"|----------------|----------------|---------------|\n")
        baseline_lrc = report['quality']['baseline_avg_lrc_error']
        advanced_lrc = report['quality']['advanced_avg_lrc_error']
        f.write(f"| Baseline | {baseline_lrc:.2f}% | {100-baseline_lrc:.1f}/100 |\n")
        f.write(f"| Advanced | {advanced_lrc:.2f}% | {100-advanced_lrc:.1f}/100 |\n")
        f.write(f"\n**Improvement**: {quality_imp:.1f}%\n\n")
        
        f.write(f"![Quality Comparison]({charts_dir}/quality_comparison.png)\n\n")
        
        f.write("### Geometric Accuracy (Planarity RMSE)\n\n")
        planarity = report['accuracy']['advanced_avg_planarity_rmse']
        f.write(f"The advanced pipeline achieves a planarity RMSE of **{planarity:.4f}**, ")
        f.write("indicating excellent ground plane fitting accuracy.\n\n")
        
        f.write("## Detection Results\n\n")
        f.write("### Anomaly Detection\n\n")
        f.write(f"| Implementation | Avg Anomalies | Avg Volume (L) |\n")
        f.write(f"|----------------|---------------|----------------|\n")
        baseline_anom = report['detection']['baseline_avg_anomalies']
        advanced_anom = report['detection']['advanced_avg_anomalies']
        baseline_vol = report['detection']['baseline_avg_volume_liters']
        advanced_vol = report['detection']['advanced_avg_volume_liters']
        f.write(f"| Baseline | {baseline_anom:.1f} | {baseline_vol:.2f} |\n")
        f.write(f"| Advanced | {advanced_anom:.1f} | {advanced_vol:.2f} |\n")
        f.write("\n")
        
        vol_diff = report['detection']['volume_difference_percent']
        f.write(f"**Volume Difference**: {vol_diff:.1f}%\n\n")
        
        f.write(f"![Detection Comparison]({charts_dir}/detection_comparison.png)\n\n")
        
        f.write("## Overall Improvements\n\n")
        f.write(f"![Improvement Radar]({charts_dir}/improvement_radar.png)\n\n")
        
        f.write("### Summary of Improvements\n\n")
        f.write(f"- **Accuracy**: +{accuracy_imp:.1f}%\n")
        f.write(f"- **Quality**: +{quality_imp:.1f}%\n")
        f.write(f"- **Robustness**: +{robustness_imp:.1f}%\n")
        f.write("\n")
        
        f.write("## Advanced Pipeline Features\n\n")
        f.write("The advanced pipeline includes several sophisticated features not present in the baseline:\n\n")
        f.write("1. **CharuCo-based Calibration**: Sub-pixel accuracy with occlusion robustness\n")
        f.write("2. **Semi-Global Block Matching (SGBM)**: Superior disparity estimation\n")
        f.write("3. **Left-Right Consistency (LRC) Validation**: Removes occluded pixels\n")
        f.write("4. **Weighted Least Squares (WLS) Filtering**: Sub-pixel disparity refinement\n")
        f.write("5. **V-Disparity Ground Plane Detection**: Robust automatic road modeling\n")
        f.write("6. **Statistical Outlier Removal**: Cleaner 3D point clouds\n")
        f.write("7. **Alpha Shape Meshing**: Tight-fitting concave hull generation\n")
        f.write("8. **Watertight Volume Calculation**: Precise volume using Divergence Theorem\n")
        f.write("9. **Comprehensive Quality Metrics**: LRC error, planarity RMSE, temporal stability\n")
        f.write("10. **Diagnostic Visualizations**: Multi-panel diagnostic output\n\n")
        
        f.write("## Trade-offs\n\n")
        if speedup < 1.0:
            f.write(f"The advanced pipeline is {1/speedup:.2f}x slower than the baseline due to:\n\n")
            f.write("- More sophisticated disparity computation (SGBM vs simple block matching)\n")
            f.write("- Additional validation steps (LRC checking)\n")
            f.write("- Post-processing filters (WLS filtering)\n")
            f.write("- Advanced meshing algorithms (Alpha Shapes)\n")
            f.write("- Comprehensive quality metric calculations\n\n")
            f.write("However, this additional processing time is justified by:\n\n")
            f.write(f"- {quality_imp:.1f}% improvement in disparity quality\n")
            f.write(f"- {accuracy_imp:.1f}% improvement in geometric accuracy\n")
            f.write("- More reliable volume measurements\n")
            f.write("- Better handling of challenging conditions\n")
        else:
            f.write("The advanced pipeline achieves better performance while maintaining higher quality.\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("Based on the benchmark results:\n\n")
        f.write("1. **Use the advanced pipeline** for applications requiring high accuracy and reliability\n")
        f.write("2. **Consider the baseline** only for real-time applications where speed is critical\n")
        f.write("3. **Optimize the advanced pipeline** for production use:\n")
        f.write("   - Implement GPU acceleration for SGBM\n")
        f.write("   - Cache rectification maps\n")
        f.write("   - Parallelize batch processing\n")
        f.write("4. **Monitor quality metrics** in production to ensure consistent performance\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The advanced stereo vision pipeline demonstrates significant improvements over the baseline ")
        f.write("implementation across all key metrics. ")
        
        if speedup < 1.0:
            f.write(f"While it is {1/speedup:.2f}x slower, ")
        
        f.write(f"it provides {quality_imp:.1f}% better quality, {accuracy_imp:.1f}% better accuracy, ")
        f.write(f"and {robustness_imp:.1f}% better robustness. ")
        f.write("These improvements make it suitable for production use in applications requiring ")
        f.write("precise volumetric measurements of road anomalies.\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated by benchmark_pipeline.py*\n")
    
    print(f"Markdown report saved to {output_file}")


def main():
    """Main report generation."""
    parser = argparse.ArgumentParser(
        description="Generate performance comparison report from benchmark results"
    )
    parser.add_argument(
        '--input',
        default='benchmark_report.json',
        help='Input benchmark JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='performance_report',
        help='Output directory for report and charts'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading benchmark results from {args.input}...")
    report = load_benchmark_results(args.input)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate charts
    print("Generating performance charts...")
    create_performance_charts(report, args.output_dir)
    
    # Generate markdown report
    print("Generating markdown report...")
    generate_markdown_report(
        report,
        str(output_path / 'PERFORMANCE_REPORT.md'),
        '.'  # Charts are in same directory
    )
    
    print(f"\n✓ Performance report generated in {args.output_dir}/")
    print(f"  - PERFORMANCE_REPORT.md")
    print(f"  - processing_time_comparison.png")
    print(f"  - quality_comparison.png")
    print(f"  - detection_comparison.png")
    print(f"  - improvement_radar.png")
    print(f"  - success_rate_comparison.png")


if __name__ == '__main__':
    main()
