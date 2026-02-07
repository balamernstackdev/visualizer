import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

def render_lasso_tool(background_image, key="lasso", canvas_width=None):
    """
    Renders the drawing canvas for Lasso selection.
    Returns the mask drawn by the user (boolean array at ORIGINAL image size).
    """
    img_w, img_h = background_image.size
    
    if canvas_width and canvas_width < img_w:
        # Scale down for display
        scale_display = canvas_width / img_w
        display_w = canvas_width
        display_h = int(img_h * scale_display)
        # Resize image to match display dimensions
        background_display = background_image.resize((display_w, display_h), Image.LANCZOS)
    else:
        # Use native size
        scale_display = 1.0
        display_w, display_h = img_w, img_h
        background_display = background_image

    stroke_color = "#ffffff"
    stroke_width = 2
    bg_color = "#000000"
    drawing_mode = "polygon"
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1.0)",  # White fill, fully opaque
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=background_display,
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode=drawing_mode,
        point_display_radius=3,
        key=key,
    )
    
    if canvas_result.image_data is not None:
        # Extract drawing mask (binary)
        mask_drawn = canvas_result.image_data[:, :, 3] > 0
        
        if scale_display != 1.0:
            # Upscale the mask back to the original image dimensions
            mask_uint8 = (mask_drawn.astype(np.uint8)) * 255
            mask_orig = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            return mask_orig > 0
            
        return mask_drawn
    
    return None

def render_click_tool(background_image, key="click_tool", canvas_width=None):
    """
    Renders canvas for simple Point clicking.
    Returns (x, y) of the SCALE-CORRECTED click (relative to original image).
    """
    img_w, img_h = background_image.size
    
    if canvas_width and canvas_width < img_w:
        # Scale down for display
        scale_display = canvas_width / img_w
        display_w = canvas_width
        display_h = int(img_h * scale_display)
        # IMPORTANT: Resize the image to match display dimensions to prevent cropping
        background_display = background_image.resize((display_w, display_h), Image.LANCZOS)
    else:
        # Use native size
        scale_display = 1.0
        display_w, display_h = img_w, img_h
        background_display = background_image

    # Point mode for clean clicking
    canvas_result = st_canvas(
        fill_color="rgba(100, 100, 100, 0.4)", # Subtle Gray Dot
        stroke_width=1, # Minimal border
        stroke_color="rgba(255, 255, 255, 0.6)", # Subtle halo
        background_image=background_display,  # Use resized image
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="point", # Clean CLICK interaction
        point_display_radius=8, # Large feedback for mobile touch users
        key=key,
        display_toolbar=False,
    )
    
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            # Get LAST added point
            last_obj = objects[-1]
            
            # Use Center of the object (works for point or line)
            x_raw = last_obj["left"] # + (last_obj["width"] / 2) # For point, left/top is center or corner? 
            # fabric.js 'circle': left/top is usually top-left of bounding box without origin set.
            # But let's stick to center calc just in case.
            x_raw = last_obj["left"] + (last_obj["width"] / 2)
            y_raw = last_obj["top"] + (last_obj["height"] / 2)
            
            # SCALE BACK (if desktop/mobile scale was applied)
            x_final = x_raw / scale_display
            y_final = y_raw / scale_display
            
            return {"x": int(x_final), "y": int(y_final)}
            
    return None
def render_box_tool(background_image, key="box_tool", canvas_width=None):
    """
    Renders canvas for Drag Box selection.
    Returns [x1, y1, x2, y2] of the scaled box.
    """
    img_w, img_h = background_image.size
    
    if canvas_width and canvas_width < img_w:
        scale_display = canvas_width / img_w
        display_w = canvas_width
        display_h = int(img_h * scale_display)
        # Resize image to prevent cropping
        background_display = background_image.resize((display_w, display_h), Image.LANCZOS)
    else:
        scale_display = 1.0
        display_w, display_h = img_w, img_h
        background_display = background_image

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)", # Subtle Orange Box
        stroke_width=2,
        stroke_color="#FFA500",
        background_image=background_display,  # Use resized image
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="rect", # Rect mode for dragging
        key=key,
        display_toolbar=False,
    )
    
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            last_obj = objects[-1]
            x1 = last_obj["left"] / scale_display
            y1 = last_obj["top"] / scale_display
            x2 = (last_obj["left"] + last_obj["width"]) / scale_display
            y2 = (last_obj["top"] + last_obj["height"]) / scale_display
            return [int(x1), int(y1), int(x2), int(y2)]
            
    return None
