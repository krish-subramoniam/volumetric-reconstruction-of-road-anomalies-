#!/usr/bin/env python
"""Script to create the preprocessing properties test file."""

content = '''"""Property-based tests for image preprocessing module."""

import numpy as np
import pytest
import cv2
from hypothesis import given, strategies as st, assume, settings
from stereo_vision.preprocessing import (
    ImagePreprocessor,
    calculate_contrast_metric,
    calculate_brightness_difference,
    calculate_edge_preservation
)


# Custom strategies for image generation
@st.composite
def grayscale_image_strategy(draw, min_height=20, max_height=100,
                             min_width=20, max_width=100,
                             min_intensity=0, max_intensity=255):
    """Generate random grayscale images for property testing."""
    height = draw(st.integers(min_value=min_height, max_value=max_height))
    width = draw(st.integers(min_value=min_width, max_value=max_width))
    
    # Generate random image
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)
    
    image = rng.randint(min_intensity, max_intensity + 1, (height, width), dtype=np.uint8)
    
    return image
'''

with open('tests/test_preprocessing_properties.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File created successfully")
