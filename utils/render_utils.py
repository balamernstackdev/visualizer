import cv2
import numpy as np
from PIL import Image
from paint_ai.paint_engine import apply_realistic_paint
from utils.lighting_utils import extract_lighting_maps

def render_high_res(original_image, masks, wall_assignments):
    """
    Rerenders the final painted image at full resolution.
    
    Args:
        original_image: PIL Image at full resolution.
        masks: List of masks at lower resolution.
        wall_assignments: Dict mapping mask index to color/finish data.
    """
    # 1. Prepare Full Res Image and Lighting
    full_res_cv2 = np.array(original_image.convert("RGB"))
    h_full, w_full = full_res_cv2.shape[:2]
    
    # Extract lighting maps for the FULL resolution image
    # Note: This is computationally expensive but necessary for 4K quality
    full_res_lighting = extract_lighting_maps(full_res_cv2)
    
    result_image = full_res_cv2.copy()
    
    # 2. Sequential Application of Masks
    for m_idx, data in wall_assignments.items():
        if m_idx < len(masks):
            mask_low = masks[m_idx]
            
            # 3. Upscale Mask to Full resolution
            # Use INTER_NEAREST to preserve binary nature of mask
            mask_uint8 = (mask_low.astype(np.uint8)) * 255
            mask_full = cv2.resize(mask_uint8, (w_full, h_full), interpolation=cv2.INTER_NEAREST)
            mask_full_bool = mask_full > 0
            
            # 4. Apply Paint Engine at high resolution
            result_image = apply_realistic_paint(
                result_image, 
                mask_full_bool, 
                data['lab'], 
                finish=data['finish'], 
                reflectance=data.get('reflectance', 0.5),
                lighting_maps=full_res_lighting
            )
            
    return result_image
