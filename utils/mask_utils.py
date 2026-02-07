import numpy as np
import cv2
from PIL import Image, ImageDraw

def mask_to_polygon(mask):
    """Converts a binary mask to a polygon list."""
    # ... implementation for saving/loading if needed
    pass

def polygon_to_mask(polygon_data, shape):
    """
    Converts Streamlit Canvas polygon data to a binary mask.
    polygon_data: List of objects from fabric.js
    shape: (height, width)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    
    # If using fabric.js output from streamlit-drawable-canvas
    # The 'path' or 'objects' need parsing.
    # Assuming we get a list of points or a path string.
    
    # Actually, streamlit-drawable-canvas returns an image_data numpy array!
    # If we use the polygon tool, the alpha channel of the returned image 
    # effectively acts as the mask.
    pass

def merge_masks(base_mask, new_mask, operation="add"):
    """
    Merges two binary masks.
    operation: 'add' (union) or 'subtract' (difference).
    """
    if operation == "add":
        return np.logical_or(base_mask, new_mask)
    elif operation == "subtract":
        return np.logical_and(base_mask, np.logical_not(new_mask))
    return base_mask

def smooth_mask(mask):
    """Applies morphological operations to smooth mask edges and fill holes (fixes 'bleaching' in thin regions)."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 1. Median Blur to remove salt-and-pepper noise
    mask_uint8 = cv2.medianBlur(mask_uint8, 5) # Increased blur for smoother base
    
    # 2. Closing to fill small/medium holes (Artifact reduction)
    # Reducing slightly from 11x11 to avoid merging distinct objects too aggressively
    kernel_close = np.ones((9,9), np.uint8)
    closing = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Slight Dilation (Expand) instead of Open (Erode)
    # Reduced back to 3x3 to prevent "spilling" over edges (like windows/roofs)
    kernel_dilate = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(closing, kernel_dilate, iterations=1)
    
    return dilated > 127

def feather_mask(mask, blur_radius=5):
    """
    Creates a soft-edged alpha mask using Gaussian blur.
    Returns: Float array (0.0 to 1.0)
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    if blur_radius % 2 == 0:
        blur_radius += 1
    # Feathering creates an alpha slope at the edges
    feathered = cv2.GaussianBlur(mask_uint8, (blur_radius, blur_radius), 0)
    return feathered.astype(np.float32) / 255.0

def dilate_mask(mask, kernel_size=3):
    """Dilates the mask to slightly expand the painted area."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    return dilated > 127
