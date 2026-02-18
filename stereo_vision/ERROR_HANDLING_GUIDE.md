# Error Handling and Logging Guide

## Overview

The stereo vision pipeline implements comprehensive error handling and logging to ensure robust operation, easy debugging, and clear error messages. This guide explains the error handling architecture and how to use it effectively.

## Architecture

### 1. Custom Exception Hierarchy

All pipeline-specific exceptions inherit from `StereoVisionError`, providing a consistent interface for error handling:

```
StereoVisionError (base)
├── CalibrationError
│   ├── CharuCoDetectionError
│   ├── InsufficientCalibrationDataError
│   ├── CalibrationQualityError
│   └── StereoCalibrationError
├── DisparityError
│   ├── InvalidDisparityMapError
│   ├── LRCValidationError
│   └── WLSFilterError
├── GroundPlaneError
│   ├── VDisparityGenerationError
│   ├── HoughLineDetectionError
│   └── GroundPlaneFittingError
├── ReconstructionError
│   ├── PointCloudGenerationError
│   ├── InsufficientPointsError
│   ├── OutlierRemovalError
│   └── DepthFilterError
├── VolumetricError
│   ├── MeshGenerationError
│   ├── AlphaShapeError
│   ├── MeshCappingError
│   ├── WatertightnessError
│   ├── VolumeCalculationError
│   └── VolumeConstraintError
├── PreprocessingError
│   ├── ImageLoadError
│   ├── ImageDimensionError
│   ├── ContrastEnhancementError
│   └── BrightnessNormalizationError
├── ConfigurationError
│   ├── InvalidParameterError
│   └── ParameterValidationError
└── PipelineError
    ├── PipelineNotCalibratedError
    ├── PipelineStageError
    └── AnomalyDetectionError
```

### 2. Logging System

The logging system provides:
- **Multiple log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Structured logging**: Context information with key-value pairs
- **Performance timing**: Automatic timing of operations
- **File and console output**: Logs to both console and file
- **Exception tracking**: Full traceback logging

## Usage

### Basic Logging

```python
from stereo_vision.logging_config import get_logger

logger = get_logger(__name__)

# Log messages at different levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error")
```

### Structured Logging with Context

```python
logger.info(
    "Processing image",
    image_shape=(640, 480),
    num_corners=25,
    processing_time="1.23s"
)
# Output: Processing image | image_shape=(640, 480) | num_corners=25 | processing_time=1.23s
```

### Performance Timing

```python
from stereo_vision.logging_config import PerformanceTimer

with PerformanceTimer(logger, "Disparity computation"):
    disparity = compute_disparity(left, right)
# Automatically logs start and completion with duration
```

### Exception Handling

```python
from stereo_vision.errors import CalibrationError, handle_error

try:
    result = calibrate_camera(images)
except CalibrationError as e:
    # Access error details
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
    
    # Log the error
    handle_error(e, logger, reraise=False)
```

### Custom Exceptions

```python
from stereo_vision.errors import StereoVisionError

# Raise with details
raise CalibrationQualityError(
    "Reprojection error too high",
    details={
        "reprojection_error": 0.8,
        "threshold": 0.5,
        "num_images": 15
    }
)
```

## Configuration

### Initialize Logging

```python
from stereo_vision.logging_config import configure_logging

# Configure with log level and directory
configure_logging(log_level="INFO", log_dir="./logs")
```

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication of potential problems
- **ERROR**: Serious problem that prevented a function from completing
- **CRITICAL**: Very serious error that may cause the program to abort

## Best Practices

### 1. Use Appropriate Exception Types

Choose the most specific exception type for the error condition:

```python
# Good - specific exception
if points.shape[0] < 4:
    raise InsufficientPointsError(
        "Need at least 4 points",
        details={"required": 4, "got": points.shape[0]}
    )

# Bad - generic exception
if points.shape[0] < 4:
    raise ValueError("Not enough points")
```

### 2. Provide Context in Error Messages

Include relevant information to help diagnose the problem:

```python
# Good - includes context
raise ImageDimensionError(
    "Image dimensions mismatch",
    details={
        "left_shape": left.shape,
        "right_shape": right.shape
    }
)

# Bad - vague message
raise ImageDimensionError("Images don't match")
```

### 3. Log at Appropriate Levels

```python
# DEBUG: Detailed diagnostic information
logger.debug("Corner detection", num_corners=25, threshold=0.5)

# INFO: Normal operation milestones
logger.info("Calibration completed", reprojection_error=0.12)

# WARNING: Potential issues that don't prevent operation
logger.warning("High reprojection error", error=0.45, threshold=0.3)

# ERROR: Operation failed
logger.error("Calibration failed", reason="Insufficient images")

# CRITICAL: System-level failure
logger.critical("Pipeline initialization failed", exc_info=True)
```

### 4. Use Performance Timers for Long Operations

```python
with PerformanceTimer(logger, "3D reconstruction"):
    point_cloud = generate_point_cloud(disparity)
    filtered_cloud = remove_outliers(point_cloud)
```

### 5. Handle Errors Gracefully

```python
try:
    result = process_image(image)
except PreprocessingError as e:
    logger.error(f"Preprocessing failed: {e.message}", **e.details)
    # Attempt fallback or recovery
    result = process_with_defaults(image)
except Exception as e:
    logger.critical("Unexpected error", exc_info=True)
    raise
```

## Error Recovery Strategies

### 1. Calibration Errors

```python
try:
    params = calibrate_intrinsics(images)
except InsufficientCalibrationDataError as e:
    logger.warning(f"Insufficient images: {e.details['found']}/{e.details['required']}")
    # Request more calibration images
    additional_images = capture_more_images()
    params = calibrate_intrinsics(images + additional_images)
```

### 2. Disparity Errors

```python
try:
    disparity = compute_disparity(left, right)
except InvalidDisparityMapError as e:
    logger.warning("Disparity computation failed, trying with adjusted parameters")
    # Adjust SGBM parameters and retry
    sgbm.configure_for_roads(baseline * 1.1, focal_length)
    disparity = compute_disparity(left, right)
```

### 3. Volume Calculation Errors

```python
try:
    volume = calculate_volume(mesh)
except WatertightnessError as e:
    logger.warning("Mesh not watertight, attempting repair")
    # Attempt mesh repair
    repaired_mesh = repair_mesh(mesh)
    volume = calculate_volume(repaired_mesh)
```

## Debugging Tips

### 1. Enable Debug Logging

```python
configure_logging(log_level="DEBUG", log_dir="./debug_logs")
```

### 2. Check Log Files

Log files are created with timestamps in the specified log directory:
```
logs/stereo_vision_20240115_143022.log
```

### 3. Use Exception Details

All custom exceptions include a `details` dictionary with diagnostic information:

```python
except CalibrationError as e:
    print(f"Error: {e.message}")
    for key, value in e.details.items():
        print(f"  {key}: {value}")
```

### 4. Enable Exception Tracebacks

```python
logger.error("Operation failed", exc_info=True)
# Logs full traceback
```

## Performance Monitoring

The logging system automatically tracks operation durations:

```python
with PerformanceTimer(logger, "Full pipeline"):
    with PerformanceTimer(logger, "Calibration"):
        calibrate()
    with PerformanceTimer(logger, "Processing"):
        process()
```

Output:
```
Starting: Calibration
Completed: Calibration | duration_seconds=2.345
Starting: Processing
Completed: Processing | duration_seconds=5.678
Completed: Full pipeline | duration_seconds=8.023
```

## Integration with Existing Code

To add error handling to existing functions:

1. Import logging and errors:
```python
from stereo_vision.logging_config import get_logger, PerformanceTimer
from stereo_vision.errors import SpecificError

logger = get_logger(__name__)
```

2. Add input validation:
```python
if invalid_input:
    raise SpecificError("Description", details={"key": "value"})
```

3. Wrap operations with try-except:
```python
try:
    result = operation()
except SpecificError:
    raise
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    raise SpecificError("Operation failed", details={"error": str(e)})
```

4. Add performance timing:
```python
with PerformanceTimer(logger, "Operation name"):
    result = operation()
```

5. Log important events:
```python
logger.info("Operation completed", result_count=len(results))
```
