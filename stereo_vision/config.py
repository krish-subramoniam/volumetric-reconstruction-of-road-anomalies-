"""Parameter configuration system for stereo vision pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import json
from pathlib import Path


@dataclass
class CameraConfig:
    """Camera setup configuration parameters."""
    baseline: float = 0.12
    focal_length: float = 700.0
    image_width: int = 640
    image_height: int = 480
    min_baseline: float = 0.01
    max_baseline: float = 2.0
    
    def validate(self) -> List[str]:
        """Validate camera configuration parameters."""
        errors = []
        
        if not (self.min_baseline <= self.baseline <= self.max_baseline):
            errors.append(
                f"Baseline {self.baseline}m must be between "
                f"{self.min_baseline}m and {self.max_baseline}m"
            )
        
        if self.focal_length <= 0:
            errors.append(f"Focal length must be positive, got {self.focal_length}")
        
        if self.image_width <= 0 or self.image_height <= 0:
            errors.append(
                f"Image dimensions must be positive, got "
                f"{self.image_width}x{self.image_height}"
            )
        
        return errors


@dataclass
class SGBMConfig:
    """SGBM disparity estimation configuration."""
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    p1: Optional[int] = None
    p2: Optional[int] = None
    disp_12_max_diff: int = 1
    pre_filter_cap: int = 63
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    mode: str = "SGBM"
    
    def validate(self) -> List[str]:
        """Validate SGBM configuration parameters."""
        errors = []
        
        if self.num_disparities % 16 != 0:
            errors.append(
                f"num_disparities must be divisible by 16, got {self.num_disparities}"
            )
        
        if self.num_disparities <= 0:
            errors.append(
                f"num_disparities must be positive, got {self.num_disparities}"
            )
        
        if self.block_size % 2 == 0 or self.block_size < 1:
            errors.append(
                f"block_size must be odd and positive, got {self.block_size}"
            )
        
        if self.block_size > 11:
            errors.append(
                f"block_size should typically be <= 11, got {self.block_size}"
            )
        
        if self.p1 is not None and self.p1 < 0:
            errors.append(f"p1 must be non-negative, got {self.p1}")
        
        if self.p2 is not None and self.p2 < 0:
            errors.append(f"p2 must be non-negative, got {self.p2}")
        
        if self.p1 is not None and self.p2 is not None and self.p2 < self.p1:
            errors.append(f"p2 ({self.p2}) must be >= p1 ({self.p1})")
        
        if self.mode not in ["SGBM", "HH"]:
            errors.append(f"mode must be 'SGBM' or 'HH', got '{self.mode}'")
        
        return errors
    
    def get_p1_p2(self, block_size: Optional[int] = None) -> Tuple[int, int]:
        """Calculate P1 and P2 parameters if not explicitly set."""
        bs = block_size if block_size is not None else self.block_size
        p1 = self.p1 if self.p1 is not None else 8 * bs * bs
        p2 = self.p2 if self.p2 is not None else 32 * bs * bs
        return p1, p2


@dataclass
class LRCConfig:
    """Left-Right Consistency check configuration."""
    max_diff: int = 1
    enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate LRC configuration parameters."""
        errors = []
        
        if self.max_diff < 0:
            errors.append(f"max_diff must be non-negative, got {self.max_diff}")
        
        if self.max_diff > 10:
            errors.append(
                f"max_diff should typically be <= 10, got {self.max_diff}"
            )
        
        return errors


@dataclass
class WLSConfig:
    """Weighted Least Squares filter configuration."""
    lambda_val: float = 8000.0
    sigma_color: float = 1.5
    enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate WLS configuration parameters."""
        errors = []
        
        if self.lambda_val <= 0:
            errors.append(f"lambda_val must be positive, got {self.lambda_val}")
        
        if self.sigma_color <= 0:
            errors.append(f"sigma_color must be positive, got {self.sigma_color}")
        
        return errors


@dataclass
class DepthRangeConfig:
    """Depth range configuration for 3D reconstruction."""
    min_depth: float = 1.0
    max_depth: float = 50.0
    
    def validate(self) -> List[str]:
        """Validate depth range configuration."""
        errors = []
        
        if self.min_depth <= 0:
            errors.append(f"min_depth must be positive, got {self.min_depth}")
        
        if self.max_depth <= self.min_depth:
            errors.append(
                f"max_depth ({self.max_depth}) must be > min_depth ({self.min_depth})"
            )
        
        if self.max_depth > 1000:
            errors.append(
                f"max_depth seems unreasonably large: {self.max_depth}m"
            )
        
        return errors


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection threshold configuration."""
    threshold_factor: float = 1.5
    min_anomaly_size: int = 100
    max_anomaly_size: int = 100000
    
    def validate(self) -> List[str]:
        """Validate anomaly detection configuration."""
        errors = []
        
        if self.threshold_factor <= 0:
            errors.append(
                f"threshold_factor must be positive, got {self.threshold_factor}"
            )
        
        if self.min_anomaly_size < 0:
            errors.append(
                f"min_anomaly_size must be non-negative, got {self.min_anomaly_size}"
            )
        
        if self.max_anomaly_size <= self.min_anomaly_size:
            errors.append(
                f"max_anomaly_size ({self.max_anomaly_size}) must be > "
                f"min_anomaly_size ({self.min_anomaly_size})"
            )
        
        return errors


@dataclass
class OutlierRemovalConfig:
    """Statistical outlier removal configuration."""
    k_neighbors: int = 20
    std_ratio: float = 2.0
    enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate outlier removal configuration."""
        errors = []
        
        if self.k_neighbors < 1:
            errors.append(f"k_neighbors must be >= 1, got {self.k_neighbors}")
        
        if self.std_ratio <= 0:
            errors.append(f"std_ratio must be positive, got {self.std_ratio}")
        
        return errors


@dataclass
class VDisparityConfig:
    """V-Disparity ground plane detection configuration."""
    max_disparity: Optional[int] = None
    hough_threshold: int = 50
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    
    def validate(self) -> List[str]:
        """Validate V-Disparity configuration."""
        errors = []
        
        if self.max_disparity is not None and self.max_disparity <= 0:
            errors.append(
                f"max_disparity must be positive, got {self.max_disparity}"
            )
        
        if self.hough_threshold <= 0:
            errors.append(
                f"hough_threshold must be positive, got {self.hough_threshold}"
            )
        
        if self.hough_min_line_length <= 0:
            errors.append(
                f"hough_min_line_length must be positive, "
                f"got {self.hough_min_line_length}"
            )
        
        if self.hough_max_line_gap < 0:
            errors.append(
                f"hough_max_line_gap must be non-negative, "
                f"got {self.hough_max_line_gap}"
            )
        
        return errors


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    sgbm: SGBMConfig = field(default_factory=SGBMConfig)
    lrc: LRCConfig = field(default_factory=LRCConfig)
    wls: WLSConfig = field(default_factory=WLSConfig)
    depth_range: DepthRangeConfig = field(default_factory=DepthRangeConfig)
    anomaly_detection: AnomalyDetectionConfig = field(
        default_factory=AnomalyDetectionConfig
    )
    outlier_removal: OutlierRemovalConfig = field(
        default_factory=OutlierRemovalConfig
    )
    v_disparity: VDisparityConfig = field(default_factory=VDisparityConfig)
    
    def validate(self) -> List[str]:
        """Validate all configuration parameters."""
        errors = []
        
        # Validate individual sections
        errors.extend([f"Camera: {e}" for e in self.camera.validate()])
        errors.extend([f"SGBM: {e}" for e in self.sgbm.validate()])
        errors.extend([f"LRC: {e}" for e in self.lrc.validate()])
        errors.extend([f"WLS: {e}" for e in self.wls.validate()])
        errors.extend([f"Depth Range: {e}" for e in self.depth_range.validate()])
        errors.extend([f"Anomaly Detection: {e}" for e in self.anomaly_detection.validate()])
        errors.extend([f"Outlier Removal: {e}" for e in self.outlier_removal.validate()])
        errors.extend([f"V-Disparity: {e}" for e in self.v_disparity.validate()])
        
        # Cross-section validation
        errors.extend(self._validate_cross_section())
        
        return errors
    
    def _validate_cross_section(self) -> List[str]:
        """Validate consistency across configuration sections."""
        errors = []
        
        # Check disparity range vs depth range consistency
        max_disparity = self.sgbm.min_disparity + self.sgbm.num_disparities
        
        if max_disparity > 0:
            min_measurable_depth = (
                self.camera.baseline * self.camera.focal_length / max_disparity
            )
            if min_measurable_depth > self.depth_range.min_depth:
                errors.append(
                    f"Disparity range cannot measure depths below "
                    f"{min_measurable_depth:.2f}m, but min_depth is "
                    f"{self.depth_range.min_depth}m. Increase num_disparities."
                )
        
        if self.sgbm.min_disparity > 0:
            max_measurable_depth = (
                self.camera.baseline * self.camera.focal_length / self.sgbm.min_disparity
            )
            if max_measurable_depth < self.depth_range.max_depth:
                errors.append(
                    f"Disparity range cannot measure depths above "
                    f"{max_measurable_depth:.2f}m, but max_depth is "
                    f"{self.depth_range.max_depth}m. Decrease min_disparity."
                )
        
        # Check V-Disparity max_disparity vs SGBM num_disparities
        if self.v_disparity.max_disparity is not None:
            if self.v_disparity.max_disparity < max_disparity:
                errors.append(
                    f"V-Disparity max_disparity ({self.v_disparity.max_disparity}) "
                    f"should be >= SGBM max disparity ({max_disparity})"
                )
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(
            camera=CameraConfig(**config_dict.get('camera', {})),
            sgbm=SGBMConfig(**config_dict.get('sgbm', {})),
            lrc=LRCConfig(**config_dict.get('lrc', {})),
            wls=WLSConfig(**config_dict.get('wls', {})),
            depth_range=DepthRangeConfig(**config_dict.get('depth_range', {})),
            anomaly_detection=AnomalyDetectionConfig(
                **config_dict.get('anomaly_detection', {})
            ),
            outlier_removal=OutlierRemovalConfig(
                **config_dict.get('outlier_removal', {})
            ),
            v_disparity=VDisparityConfig(**config_dict.get('v_disparity', {}))
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls.from_dict(config_dict)
        
        # Validate loaded configuration
        errors = config.validate()
        if errors:
            raise ValueError(
                f"Invalid configuration loaded from {filepath}:\n" +
                "\n".join(errors)
            )
        
        return config


def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration."""
    return PipelineConfig()


def create_high_accuracy_config() -> PipelineConfig:
    """Create high-accuracy configuration for precise measurements."""
    return PipelineConfig(
        sgbm=SGBMConfig(
            num_disparities=256,
            block_size=7,
            speckle_window_size=200,
            uniqueness_ratio=15
        ),
        wls=WLSConfig(
            lambda_val=10000.0,
            sigma_color=1.2
        ),
        outlier_removal=OutlierRemovalConfig(
            k_neighbors=30,
            std_ratio=1.5
        )
    )


def create_fast_config() -> PipelineConfig:
    """Create fast processing configuration."""
    return PipelineConfig(
        sgbm=SGBMConfig(
            num_disparities=64,
            block_size=3,
            speckle_window_size=50,
            mode="SGBM"
        ),
        depth_range=DepthRangeConfig(
            min_depth=2.0,  # Adjusted for narrower disparity range
            max_depth=30.0
        ),
        wls=WLSConfig(
            lambda_val=5000.0,
            enabled=False
        ),
        outlier_removal=OutlierRemovalConfig(
            k_neighbors=10,
            enabled=False
        )
    )
