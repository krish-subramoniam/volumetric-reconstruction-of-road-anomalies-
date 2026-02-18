"""
Custom exception classes for the stereo vision pipeline.

This module defines a hierarchy of custom exceptions for different error
conditions that can occur during pipeline processing. This allows for
more precise error handling and better error messages.

Requirements: Error handling across all modules
"""


class StereoVisionError(Exception):
    """Base exception for all stereo vision pipeline errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        """String representation with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# Calibration Errors
class CalibrationError(StereoVisionError):
    """Base exception for calibration-related errors."""
    pass


class CharuCoDetectionError(CalibrationError):
    """Raised when CharuCo board detection fails."""
    pass


class InsufficientCalibrationDataError(CalibrationError):
    """Raised when insufficient calibration images are provided."""
    pass


class CalibrationQualityError(CalibrationError):
    """Raised when calibration quality is below acceptable threshold."""
    pass


class StereoCalibrationError(CalibrationError):
    """Raised when stereo calibration fails."""
    pass


# Disparity Estimation Errors
class DisparityError(StereoVisionError):
    """Base exception for disparity estimation errors."""
    pass


class InvalidDisparityMapError(DisparityError):
    """Raised when disparity map is invalid or empty."""
    pass


class LRCValidationError(DisparityError):
    """Raised when Left-Right Consistency validation fails."""
    pass


class WLSFilterError(DisparityError):
    """Raised when WLS filtering fails."""
    pass


# Ground Plane Detection Errors
class GroundPlaneError(StereoVisionError):
    """Base exception for ground plane detection errors."""
    pass


class VDisparityGenerationError(GroundPlaneError):
    """Raised when V-Disparity generation fails."""
    pass


class HoughLineDetectionError(GroundPlaneError):
    """Raised when Hough line detection fails to find ground plane."""
    pass


class GroundPlaneFittingError(GroundPlaneError):
    """Raised when ground plane fitting fails."""
    pass


# 3D Reconstruction Errors
class ReconstructionError(StereoVisionError):
    """Base exception for 3D reconstruction errors."""
    pass


class PointCloudGenerationError(ReconstructionError):
    """Raised when point cloud generation fails."""
    pass


class InsufficientPointsError(ReconstructionError):
    """Raised when point cloud has insufficient points for processing."""
    pass


class OutlierRemovalError(ReconstructionError):
    """Raised when outlier removal fails."""
    pass


class DepthFilterError(ReconstructionError):
    """Raised when depth filtering fails."""
    pass


# Volumetric Analysis Errors
class VolumetricError(StereoVisionError):
    """Base exception for volumetric analysis errors."""
    pass


class MeshGenerationError(VolumetricError):
    """Raised when mesh generation fails."""
    pass


class AlphaShapeError(MeshGenerationError):
    """Raised when Alpha Shape generation fails."""
    pass


class MeshCappingError(VolumetricError):
    """Raised when mesh capping fails."""
    pass


class WatertightnessError(VolumetricError):
    """Raised when mesh is not watertight."""
    pass


class VolumeCalculationError(VolumetricError):
    """Raised when volume calculation fails."""
    pass


class VolumeConstraintError(VolumetricError):
    """Raised when calculated volume violates physical constraints."""
    pass


# Preprocessing Errors
class PreprocessingError(StereoVisionError):
    """Base exception for preprocessing errors."""
    pass


class ImageLoadError(PreprocessingError):
    """Raised when image loading fails."""
    pass


class ImageDimensionError(PreprocessingError):
    """Raised when image dimensions are invalid or mismatched."""
    pass


class ContrastEnhancementError(PreprocessingError):
    """Raised when contrast enhancement fails."""
    pass


class BrightnessNormalizationError(PreprocessingError):
    """Raised when brightness normalization fails."""
    pass


# Configuration Errors
class ConfigurationError(StereoVisionError):
    """Base exception for configuration errors."""
    pass


class InvalidParameterError(ConfigurationError):
    """Raised when a parameter value is invalid."""
    pass


class ParameterValidationError(ConfigurationError):
    """Raised when parameter validation fails."""
    pass


# Pipeline Errors
class PipelineError(StereoVisionError):
    """Base exception for pipeline execution errors."""
    pass


class PipelineNotCalibratedError(PipelineError):
    """Raised when pipeline is used without calibration."""
    pass


class PipelineStageError(PipelineError):
    """Raised when a pipeline stage fails."""
    pass


class AnomalyDetectionError(PipelineError):
    """Raised when anomaly detection fails."""
    pass


# I/O Errors
class IOError(StereoVisionError):
    """Base exception for I/O errors."""
    pass


class CalibrationFileError(IOError):
    """Raised when calibration file operations fail."""
    pass


class OutputSaveError(IOError):
    """Raised when saving output fails."""
    pass


def handle_error(error: Exception, logger=None, reraise: bool = True) -> None:
    """
    Centralized error handling function.
    
    Args:
        error: The exception to handle
        logger: Optional logger instance for logging the error
        reraise: Whether to re-raise the exception after handling
    """
    if logger is not None:
        if isinstance(error, StereoVisionError):
            logger.error(
                f"{type(error).__name__}: {error.message}",
                **error.details
            )
        else:
            logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    
    if reraise:
        raise
