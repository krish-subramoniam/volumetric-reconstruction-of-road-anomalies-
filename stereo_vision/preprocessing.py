"""
Image preprocessing module for robust stereo vision pipeline.

This module provides preprocessing functions for handling various lighting conditions,
contrast enhancement, brightness normalization, and noise filtering while preserving
edge details.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import numpy as np
import cv2
from typing import Tuple, Optional

from stereo_vision.logging_config import get_logger, PerformanceTimer
from stereo_vision.errors import (
    PreprocessingError, ImageDimensionError,
    ContrastEnhancementError, BrightnessNormalizationError
)

# Initialize logger
logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Handles robust image preprocessing for stereo vision pipeline.
    
    Provides contrast enhancement, brightness normalization, and noise filtering
    while handling extreme exposure conditions gracefully.
    """
    
    def __init__(self, 
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 bilateral_d: int = 9,
                 bilateral_sigma_color: float = 75.0,
                 bilateral_sigma_space: float = 75.0):
        """
        Initialize the preprocessor with configurable parameters.
        
        Args:
            clahe_clip_limit: Contrast limiting threshold for CLAHE
            clahe_tile_size: Size of grid for histogram equalization
            bilateral_d: Diameter of pixel neighborhood for bilateral filter
            bilateral_sigma_color: Filter sigma in color space
            bilateral_sigma_space: Filter sigma in coordinate space
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        
        # Create CLAHE object for contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE improves local contrast while preventing over-amplification of noise.
        Works on grayscale images or applies to each channel of color images.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Contrast-enhanced image
            
        Validates: Requirements 8.1
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        # Handle color images by converting to LAB and applying CLAHE to L channel
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_enhanced = self.clahe.apply(l_channel)
            
            # Merge channels and convert back to BGR
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = self.clahe.apply(image)
        
        return enhanced
    
    def normalize_brightness(self, 
                            left_image: np.ndarray, 
                            right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize brightness between stereo image pairs to reduce exposure differences.
        
        Uses histogram matching to align the brightness distribution of the right image
        to match the left image, ensuring consistent exposure across the stereo pair.
        
        Args:
            left_image: Left stereo image (reference)
            right_image: Right stereo image (to be normalized)
            
        Returns:
            Tuple of (left_image, normalized_right_image)
            
        Validates: Requirements 8.2
        """
        if left_image is None or right_image is None:
            raise ValueError("Input images cannot be None")
        
        if left_image.shape != right_image.shape:
            raise ValueError("Stereo images must have the same dimensions")
        
        # Convert to grayscale if color
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        # Calculate mean brightness
        left_mean = np.mean(left_gray)
        right_mean = np.mean(right_gray)
        
        # Calculate brightness adjustment factor
        if right_mean > 0:
            brightness_ratio = left_mean / right_mean
        else:
            brightness_ratio = 1.0
        
        # Apply brightness normalization
        if len(right_image.shape) == 3:
            # Color image - normalize each channel
            normalized_right = np.clip(
                right_image.astype(np.float32) * brightness_ratio,
                0, 255
            ).astype(np.uint8)
        else:
            # Grayscale image
            normalized_right = np.clip(
                right_gray.astype(np.float32) * brightness_ratio,
                0, 255
            ).astype(np.uint8)
        
        return left_image, normalized_right
    
    def handle_extreme_exposure(self, image: np.ndarray) -> np.ndarray:
        """
        Handle extreme exposure conditions (overexposed or underexposed regions).
        
        Uses adaptive techniques to recover detail in extreme exposure regions:
        - For overexposed regions: Apply tone mapping
        - For underexposed regions: Boost shadows while preserving highlights
        
        Args:
            image: Input image with potential extreme exposure
            
        Returns:
            Image with improved exposure handling
            
        Validates: Requirements 8.3
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Detect overexposed and underexposed regions
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        overexposed_mask = gray > 240
        underexposed_mask = gray < 15
        
        # Calculate exposure statistics
        overexposed_ratio = np.sum(overexposed_mask) / gray.size
        underexposed_ratio = np.sum(underexposed_mask) / gray.size
        
        # Apply correction if significant extreme exposure detected
        if overexposed_ratio > 0.05 or underexposed_ratio > 0.05:
            # Use gamma correction for exposure adjustment
            mean_intensity = np.mean(gray) / 255.0
            
            if mean_intensity > 0:
                # Calculate adaptive gamma
                if overexposed_ratio > 0.05:
                    # Compress highlights
                    gamma = 1.2
                elif underexposed_ratio > 0.05:
                    # Boost shadows
                    gamma = 0.8
                else:
                    gamma = 1.0
                
                # Apply gamma correction
                img_corrected = np.power(img_float, gamma)
                
                # Convert back to uint8
                result = np.clip(img_corrected * 255, 0, 255).astype(np.uint8)
            else:
                result = image
        else:
            result = image
        
        return result
    
    def filter_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving noise filtering using bilateral filter.
        
        The bilateral filter reduces noise while preserving edges, which is crucial
        for stereo matching. It smooths similar pixels while maintaining sharp
        transitions at edges.
        
        Args:
            image: Input noisy image
            
        Returns:
            Filtered image with reduced noise and preserved edges
            
        Validates: Requirements 8.4
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        # Apply bilateral filter for edge-preserving smoothing
        filtered = cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        
        return filtered
    
    def preprocess_stereo_pair(self,
                               left_image: np.ndarray,
                               right_image: np.ndarray,
                               apply_contrast: bool = True,
                               apply_normalization: bool = True,
                               apply_exposure_correction: bool = True,
                               apply_denoising: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply complete preprocessing pipeline to a stereo image pair.
        
        This is the main entry point for preprocessing, applying all enhancement
        steps in the optimal order for stereo vision processing.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            apply_contrast: Whether to apply contrast enhancement
            apply_normalization: Whether to normalize brightness between pairs
            apply_exposure_correction: Whether to handle extreme exposure
            apply_denoising: Whether to apply noise filtering
            
        Returns:
            Tuple of (preprocessed_left, preprocessed_right)
            
        Validates: Requirements 8.1, 8.2, 8.3, 8.4
        """
        if left_image is None or right_image is None:
            raise ValueError("Input images cannot be None")
        
        if left_image.shape != right_image.shape:
            raise ValueError("Stereo images must have the same dimensions")
        
        # Start with original images
        left_processed = left_image.copy()
        right_processed = right_image.copy()
        
        # Step 1: Handle extreme exposure conditions
        if apply_exposure_correction:
            left_processed = self.handle_extreme_exposure(left_processed)
            right_processed = self.handle_extreme_exposure(right_processed)
        
        # Step 2: Normalize brightness between stereo pairs
        if apply_normalization:
            left_processed, right_processed = self.normalize_brightness(
                left_processed, right_processed
            )
        
        # Step 3: Enhance contrast
        if apply_contrast:
            left_processed = self.enhance_contrast(left_processed)
            right_processed = self.enhance_contrast(right_processed)
        
        # Step 4: Apply noise filtering (last to avoid amplifying noise)
        if apply_denoising:
            left_processed = self.filter_noise(left_processed)
            right_processed = self.filter_noise(right_processed)
        
        return left_processed, right_processed


def calculate_contrast_metric(image: np.ndarray) -> float:
    """
    Calculate a contrast metric for an image using standard deviation.
    
    Higher values indicate better contrast. This is used to validate
    that contrast enhancement is effective.
    
    Args:
        image: Input image
        
    Returns:
        Contrast metric (standard deviation of pixel intensities)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return float(np.std(gray))


def calculate_brightness_difference(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the absolute brightness difference between two images.
    
    Used to validate brightness normalization effectiveness.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Absolute difference in mean brightness
    """
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
    
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2
    
    return float(abs(np.mean(gray1) - np.mean(gray2)))


def calculate_edge_preservation(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calculate edge preservation metric by comparing edge strength before and after filtering.
    
    Uses Sobel edge detection to measure how well edges are preserved.
    Higher values (closer to 1.0) indicate better edge preservation.
    
    Args:
        original: Original image before filtering
        filtered: Filtered image
        
    Returns:
        Edge preservation ratio (0 to 1, where 1 is perfect preservation)
    """
    if len(original.shape) == 3:
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original
    
    if len(filtered.shape) == 3:
        gray_filt = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    else:
        gray_filt = filtered
    
    # Calculate edge strength using Sobel
    sobelx_orig = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
    sobely_orig = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength_orig = np.sqrt(sobelx_orig**2 + sobely_orig**2)
    
    sobelx_filt = cv2.Sobel(gray_filt, cv2.CV_64F, 1, 0, ksize=3)
    sobely_filt = cv2.Sobel(gray_filt, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength_filt = np.sqrt(sobelx_filt**2 + sobely_filt**2)
    
    # Calculate preservation ratio
    orig_mean = np.mean(edge_strength_orig)
    filt_mean = np.mean(edge_strength_filt)
    
    if orig_mean > 0:
        preservation_ratio = min(filt_mean / orig_mean, 1.0)
    else:
        preservation_ratio = 1.0
    
    return float(preservation_ratio)
