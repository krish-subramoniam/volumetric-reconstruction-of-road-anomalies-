"""Advanced Stereo Vision Pipeline for Volumetric Road Anomaly Detection."""

from stereo_vision.pipeline import StereoVisionPipeline, create_pipeline
from stereo_vision.config import PipelineConfig
from stereo_vision.ground_plane import VDisparityGenerator
from stereo_vision.reconstruction import PointCloudGenerator

__version__ = "1.0.0"

__all__ = [
    'StereoVisionPipeline',
    'create_pipeline',
    'PipelineConfig',
    'VDisparityGenerator',
    'PointCloudGenerator'
]
