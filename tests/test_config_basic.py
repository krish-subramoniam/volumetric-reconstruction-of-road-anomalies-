"""Unit tests for parameter configuration system.

Tests individual configuration classes and validation logic.
"""

import pytest
import json
import tempfile
from pathlib import Path
from stereo_vision.config import (
    CameraConfig,
    SGBMConfig,
    LRCConfig,
    WLSConfig,
    DepthRangeConfig,
    AnomalyDetectionConfig,
    OutlierRemovalConfig,
    VDisparityConfig,
    PipelineConfig,
    create_default_config,
    create_high_accuracy_config,
    create_fast_config
)


class TestCameraConfig:
    """Test CameraConfig validation."""
    
    def test_valid_camera_config(self):
        """Valid camera configuration should pass validation."""
        config = CameraConfig(
            baseline=0.12,
            focal_length=700.0,
            image_width=640,
            image_height=480
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_baseline_too_small(self):
        """Baseline below minimum should fail validation."""
        config = CameraConfig(baseline=0.005)
        errors = config.validate()
        assert len(errors) > 0
        assert any("Baseline" in e for e in errors)
    
    def test_baseline_too_large(self):
        """Baseline above maximum should fail validation."""
        config = CameraConfig(baseline=3.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("Baseline" in e for e in errors)
    
    def test_negative_focal_length(self):
        """Negative focal length should fail validation."""
        config = CameraConfig(focal_length=-100.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("Focal length" in e for e in errors)
    
    def test_invalid_image_dimensions(self):
        """Invalid image dimensions should fail validation."""
        config = CameraConfig(image_width=0, image_height=-10)
        errors = config.validate()
        assert len(errors) > 0
        assert any("dimensions" in e for e in errors)


class TestSGBMConfig:
    """Test SGBMConfig validation."""
    
    def test_valid_sgbm_config(self):
        """Valid SGBM configuration should pass validation."""
        config = SGBMConfig(
            num_disparities=128,
            block_size=5,
            p1=200,
            p2=800
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_num_disparities_not_divisible_by_16(self):
        """num_disparities not divisible by 16 should fail."""
        config = SGBMConfig(num_disparities=100)
        errors = config.validate()
        assert len(errors) > 0
        assert any("divisible by 16" in e for e in errors)
    
    def test_even_block_size(self):
        """Even block size should fail validation."""
        config = SGBMConfig(block_size=4)
        errors = config.validate()
        assert len(errors) > 0
        assert any("odd" in e for e in errors)
    
    def test_p2_less_than_p1(self):
        """P2 < P1 should fail validation."""
        config = SGBMConfig(p1=1000, p2=500)
        errors = config.validate()
        assert len(errors) > 0
        assert any("p2" in e and "p1" in e for e in errors)
    
    def test_invalid_mode(self):
        """Invalid mode should fail validation."""
        config = SGBMConfig(mode="INVALID")
        errors = config.validate()
        assert len(errors) > 0
        assert any("mode" in e for e in errors)
    
    def test_auto_p1_p2_calculation(self):
        """P1 and P2 should be auto-calculated when None."""
        config = SGBMConfig(block_size=5)
        p1, p2 = config.get_p1_p2()
        assert p1 == 8 * 5 * 5
        assert p2 == 32 * 5 * 5
        assert p2 > p1
    
    def test_explicit_p1_p2(self):
        """Explicit P1 and P2 should be used when provided."""
        config = SGBMConfig(p1=100, p2=400)
        p1, p2 = config.get_p1_p2()
        assert p1 == 100
        assert p2 == 400


class TestLRCConfig:
    """Test LRCConfig validation."""
    
    def test_valid_lrc_config(self):
        """Valid LRC configuration should pass validation."""
        config = LRCConfig(max_diff=1, enabled=True)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_negative_max_diff(self):
        """Negative max_diff should fail validation."""
        config = LRCConfig(max_diff=-1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_diff" in e for e in errors)
    
    def test_large_max_diff_warning(self):
        """Very large max_diff should generate warning."""
        config = LRCConfig(max_diff=20)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_diff" in e for e in errors)


class TestWLSConfig:
    """Test WLSConfig validation."""
    
    def test_valid_wls_config(self):
        """Valid WLS configuration should pass validation."""
        config = WLSConfig(lambda_val=8000.0, sigma_color=1.5)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_negative_lambda(self):
        """Negative lambda should fail validation."""
        config = WLSConfig(lambda_val=-1000.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("lambda_val" in e for e in errors)
    
    def test_negative_sigma(self):
        """Negative sigma should fail validation."""
        config = WLSConfig(sigma_color=-0.5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("sigma_color" in e for e in errors)


class TestDepthRangeConfig:
    """Test DepthRangeConfig validation."""
    
    def test_valid_depth_range(self):
        """Valid depth range should pass validation."""
        config = DepthRangeConfig(min_depth=1.0, max_depth=50.0)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_negative_min_depth(self):
        """Negative min_depth should fail validation."""
        config = DepthRangeConfig(min_depth=-1.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("min_depth" in e for e in errors)
    
    def test_max_less_than_min(self):
        """max_depth < min_depth should fail validation."""
        config = DepthRangeConfig(min_depth=50.0, max_depth=10.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_depth" in e and "min_depth" in e for e in errors)
    
    def test_unreasonably_large_max_depth(self):
        """Unreasonably large max_depth should generate warning."""
        config = DepthRangeConfig(max_depth=2000.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("unreasonably large" in e for e in errors)


class TestAnomalyDetectionConfig:
    """Test AnomalyDetectionConfig validation."""
    
    def test_valid_anomaly_config(self):
        """Valid anomaly detection config should pass validation."""
        config = AnomalyDetectionConfig(
            threshold_factor=1.5,
            min_anomaly_size=100,
            max_anomaly_size=100000
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_negative_threshold_factor(self):
        """Negative threshold factor should fail validation."""
        config = AnomalyDetectionConfig(threshold_factor=-0.5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("threshold_factor" in e for e in errors)
    
    def test_max_size_less_than_min(self):
        """max_anomaly_size < min_anomaly_size should fail."""
        config = AnomalyDetectionConfig(
            min_anomaly_size=1000,
            max_anomaly_size=500
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_anomaly_size" in e for e in errors)


class TestOutlierRemovalConfig:
    """Test OutlierRemovalConfig validation."""
    
    def test_valid_outlier_config(self):
        """Valid outlier removal config should pass validation."""
        config = OutlierRemovalConfig(k_neighbors=20, std_ratio=2.0)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_zero_k_neighbors(self):
        """Zero k_neighbors should fail validation."""
        config = OutlierRemovalConfig(k_neighbors=0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("k_neighbors" in e for e in errors)
    
    def test_negative_std_ratio(self):
        """Negative std_ratio should fail validation."""
        config = OutlierRemovalConfig(std_ratio=-1.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("std_ratio" in e for e in errors)


class TestVDisparityConfig:
    """Test VDisparityConfig validation."""
    
    def test_valid_v_disparity_config(self):
        """Valid V-Disparity config should pass validation."""
        config = VDisparityConfig(
            max_disparity=256,
            hough_threshold=50,
            hough_min_line_length=50,
            hough_max_line_gap=10
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_negative_max_disparity(self):
        """Negative max_disparity should fail validation."""
        config = VDisparityConfig(max_disparity=-10)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_disparity" in e for e in errors)
    
    def test_negative_hough_threshold(self):
        """Negative hough_threshold should fail validation."""
        config = VDisparityConfig(hough_threshold=-5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("hough_threshold" in e for e in errors)


class TestPipelineConfig:
    """Test PipelineConfig validation and operations."""
    
    def test_valid_pipeline_config(self):
        """Valid pipeline configuration should pass validation."""
        config = PipelineConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_cross_section_validation_disparity_depth_mismatch(self):
        """Disparity range incompatible with depth range should fail."""
        config = PipelineConfig(
            camera=CameraConfig(baseline=0.12, focal_length=700.0),
            sgbm=SGBMConfig(min_disparity=0, num_disparities=32),
            depth_range=DepthRangeConfig(min_depth=1.0, max_depth=50.0)
        )
        errors = config.validate()
        # Should have error about disparity range not covering depth range
        assert any("Disparity range cannot measure" in e for e in errors)
    
    def test_to_dict_conversion(self):
        """Configuration should convert to dictionary correctly."""
        config = PipelineConfig()
        config_dict = config.to_dict()
        
        assert 'camera' in config_dict
        assert 'sgbm' in config_dict
        assert 'depth_range' in config_dict
        assert config_dict['camera']['baseline'] == 0.12
    
    def test_from_dict_conversion(self):
        """Configuration should load from dictionary correctly."""
        config_dict = {
            'camera': {'baseline': 0.15, 'focal_length': 800.0},
            'sgbm': {'num_disparities': 256, 'block_size': 7}
        }
        config = PipelineConfig.from_dict(config_dict)
        
        assert config.camera.baseline == 0.15
        assert config.camera.focal_length == 800.0
        assert config.sgbm.num_disparities == 256
        assert config.sgbm.block_size == 7
    
    def test_save_and_load_file(self):
        """Configuration should save and load from file correctly."""
        config = PipelineConfig(
            camera=CameraConfig(baseline=0.15, focal_length=800.0)
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.json"
            
            # Save configuration
            config.save_to_file(str(filepath))
            assert filepath.exists()
            
            # Load configuration
            loaded_config = PipelineConfig.load_from_file(str(filepath))
            assert loaded_config.camera.baseline == 0.15
            assert loaded_config.camera.focal_length == 800.0
    
    def test_load_invalid_config_file(self):
        """Loading invalid configuration should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid_config.json"
            
            # Create invalid configuration
            invalid_config = {
                'camera': {'baseline': -1.0},  # Invalid
                'sgbm': {'num_disparities': 100}  # Not divisible by 16
            }
            
            with open(filepath, 'w') as f:
                json.dump(invalid_config, f)
            
            with pytest.raises(ValueError) as exc_info:
                PipelineConfig.load_from_file(str(filepath))
            
            assert "Invalid configuration" in str(exc_info.value)
    
    def test_load_nonexistent_file(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.load_from_file("nonexistent_config.json")


class TestPresetConfigs:
    """Test preset configuration factories."""
    
    def test_default_config(self):
        """Default config should be valid."""
        config = create_default_config()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_high_accuracy_config(self):
        """High accuracy config should be valid."""
        config = create_high_accuracy_config()
        errors = config.validate()
        assert len(errors) == 0
        
        # Should have more disparities for accuracy
        assert config.sgbm.num_disparities >= 256
    
    def test_fast_config(self):
        """Fast config should be valid."""
        config = create_fast_config()
        errors = config.validate()
        assert len(errors) == 0
        
        # Should have fewer disparities for speed
        assert config.sgbm.num_disparities <= 64
        
        # Should have some features disabled
        assert config.wls.enabled == False or config.outlier_removal.enabled == False


class TestConfigurationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_valid_baseline(self):
        """Minimum valid baseline should pass."""
        config = CameraConfig(baseline=0.01)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_maximum_valid_baseline(self):
        """Maximum valid baseline should pass."""
        config = CameraConfig(baseline=2.0)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_minimum_num_disparities(self):
        """Minimum valid num_disparities should pass."""
        config = SGBMConfig(num_disparities=16)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_block_size_one(self):
        """Block size of 1 should pass."""
        config = SGBMConfig(block_size=1)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_zero_min_disparity(self):
        """Zero min_disparity should be valid."""
        config = SGBMConfig(min_disparity=0)
        errors = config.validate()
        assert len(errors) == 0
