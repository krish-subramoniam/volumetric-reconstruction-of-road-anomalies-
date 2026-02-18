"""Basic unit tests for image preprocessing module."""

import numpy as np
import pytest
import cv2
from stereo_vision.preprocessing import (
    ImagePreprocessor,
    calculate_contrast_metric,
    calculate_brightness_difference,
    calculate_edge_preservation
)


class TestImagePreprocessor:
    """Basic tests for ImagePreprocessor class."""
    
    def test_enhance_contrast_grayscale(self):
        """Test contrast enhancement on grayscale image."""
        # Create a low-contrast grayscale image
        image = np.ones((100, 100), dtype=np.uint8) * 128
        image[25:75, 25:75] = 140  # Add slight variation
        
        preprocessor = ImagePreprocessor()
        enhanced = preprocessor.enhance_contrast(image)
        
        # Check output properties
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8
        
        # Contrast should be improved
        original_contrast = calculate_contrast_metric(image)
        enhanced_contrast = calculate_contrast_metric(enhanced)
        assert enhanced_contrast >= original_contrast
    
    def test_enhance_contrast_color(self):
        """Test contrast enhancement on color image."""
        # Create a low-contrast color image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image[25:75, 25:75] = [140, 140, 140]
        
        preprocessor = ImagePreprocessor()
        enhanced = preprocessor.enhance_contrast(image)
        
        # Check output properties
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8
        assert len(enhanced.shape) == 3
    
    def test_normalize_brightness(self):
        """Test brightness normalization between stereo pairs."""
        # Create stereo pair with brightness difference
        left_image = np.ones((100, 100), dtype=np.uint8) * 150
        right_image = np.ones((100, 100), dtype=np.uint8) * 100
        
        preprocessor = ImagePreprocessor()
        left_norm, right_norm = preprocessor.normalize_brightness(left_image, right_image)
        
        # Check output properties
        assert left_norm.shape == left_image.shape
        assert right_norm.shape == right_image.shape
        
        # Brightness difference should be reduced
        original_diff = calculate_brightness_difference(left_image, right_image)
        normalized_diff = calculate_brightness_difference(left_norm, right_norm)
        assert normalized_diff <= original_diff
    
    def test_handle_extreme_exposure_overexposed(self):
        """Test handling of overexposed images."""
        # Create overexposed image
        image = np.ones((100, 100), dtype=np.uint8) * 250
        image[10:30, 10:30] = 255  # Saturated region
        
        preprocessor = ImagePreprocessor()
        corrected = preprocessor.handle_extreme_exposure(image)
        
        # Check output properties
        assert corrected.shape == image.shape
        assert corrected.dtype == np.uint8
        
        # Should reduce overexposure
        assert np.mean(corrected) <= np.mean(image)
    
    def test_handle_extreme_exposure_underexposed(self):
        """Test handling of underexposed images."""
        # Create underexposed image
        image = np.ones((100, 100), dtype=np.uint8) * 20
        image[10:30, 10:30] = 0  # Very dark region
        
        preprocessor = ImagePreprocessor()
        corrected = preprocessor.handle_extreme_exposure(image)
        
        # Check output properties
        assert corrected.shape == image.shape
        assert corrected.dtype == np.uint8
        
        # Should boost shadows
        assert np.mean(corrected) >= np.mean(image)
    
    def test_filter_noise(self):
        """Test edge-preserving noise filtering."""
        # Create image with noise
        base_image = np.zeros((100, 100), dtype=np.uint8)
        base_image[:, :50] = 100
        base_image[:, 50:] = 200
        
        # Add Gaussian noise
        noise = np.random.normal(0, 20, base_image.shape)
        noisy_image = np.clip(base_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        preprocessor = ImagePreprocessor()
        filtered = preprocessor.filter_noise(noisy_image)
        
        # Check output properties
        assert filtered.shape == noisy_image.shape
        assert filtered.dtype == np.uint8
        
        # Edges should be preserved
        edge_preservation = calculate_edge_preservation(base_image, filtered)
        assert edge_preservation > 0.7  # Good edge preservation
    
    def test_preprocess_stereo_pair_complete(self):
        """Test complete preprocessing pipeline."""
        # Create stereo pair
        left_image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        right_image = np.random.randint(30, 180, (100, 100), dtype=np.uint8)
        
        preprocessor = ImagePreprocessor()
        left_processed, right_processed = preprocessor.preprocess_stereo_pair(
            left_image, right_image
        )
        
        # Check output properties
        assert left_processed.shape == left_image.shape
        assert right_processed.shape == right_image.shape
        assert left_processed.dtype == np.uint8
        assert right_processed.dtype == np.uint8
    
    def test_preprocess_stereo_pair_selective(self):
        """Test preprocessing with selective operations."""
        left_image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        right_image = np.random.randint(30, 180, (100, 100), dtype=np.uint8)
        
        preprocessor = ImagePreprocessor()
        
        # Test with only contrast enhancement
        left_proc, right_proc = preprocessor.preprocess_stereo_pair(
            left_image, right_image,
            apply_contrast=True,
            apply_normalization=False,
            apply_exposure_correction=False,
            apply_denoising=False
        )
        
        assert left_proc.shape == left_image.shape
        assert right_proc.shape == right_image.shape
    
    def test_invalid_input_handling(self):
        """Test error handling for invalid inputs."""
        preprocessor = ImagePreprocessor()
        
        # Test None input
        with pytest.raises(ValueError):
            preprocessor.enhance_contrast(None)
        
        # Test empty image
        with pytest.raises(ValueError):
            preprocessor.enhance_contrast(np.array([]))
        
        # Test mismatched stereo pair
        left = np.ones((100, 100), dtype=np.uint8)
        right = np.ones((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError):
            preprocessor.normalize_brightness(left, right)


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_calculate_contrast_metric(self):
        """Test contrast metric calculation."""
        # Uniform image - low contrast
        uniform = np.ones((100, 100), dtype=np.uint8) * 128
        contrast_uniform = calculate_contrast_metric(uniform)
        assert contrast_uniform == 0.0
        
        # High contrast image
        high_contrast = np.zeros((100, 100), dtype=np.uint8)
        high_contrast[:, :50] = 0
        high_contrast[:, 50:] = 255
        contrast_high = calculate_contrast_metric(high_contrast)
        assert contrast_high > 100
    
    def test_calculate_brightness_difference(self):
        """Test brightness difference calculation."""
        image1 = np.ones((100, 100), dtype=np.uint8) * 100
        image2 = np.ones((100, 100), dtype=np.uint8) * 150
        
        diff = calculate_brightness_difference(image1, image2)
        assert diff == 50.0
    
    def test_calculate_edge_preservation(self):
        """Test edge preservation metric."""
        # Create image with edges
        original = np.zeros((100, 100), dtype=np.uint8)
        original[:, :50] = 100
        original[:, 50:] = 200
        
        # Perfect preservation
        filtered = original.copy()
        preservation = calculate_edge_preservation(original, filtered)
        assert preservation == 1.0
        
        # Uniform image - no edges preserved
        uniform = np.ones((100, 100), dtype=np.uint8) * 150
        preservation_uniform = calculate_edge_preservation(original, uniform)
        assert preservation_uniform < 0.5  # Most edges lost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
