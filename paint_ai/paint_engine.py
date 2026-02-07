import cv2
import numpy as np
from utils.lighting_utils import extract_lighting_maps

def hex_to_lab(hex_color):
    """Converts hex string to LAB numpy array."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    rgb_pixel = np.array([[[r, g, b]]], dtype=np.uint8)
    return cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0][0]

def apply_realistic_paint(final_image_rgb, mask, target_lab, finish="matte", reflectance=0.5, lighting_maps=None):
    """
    Applies paint using physics-based blending. Optimized with Bounding Box cropping.
    """
    if not np.any(mask):
        return final_image_rgb

    # 1. OPTIMIZATION: Work only on Bounding Box
    # cv2.boundingRect returns x, y, w, h (x=col, y=row)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    
    # Define Slices (Numpy uses [row, col])
    slice_y = slice(y, y+h)
    slice_x = slice(x, x+w)
    
    # Crop Inputs
    mask_crop = mask[slice_y, slice_x]
    
    # If lighting maps not provided, allow failure or fallback (assumed provided per system design)
    if lighting_maps is None:
        return final_image_rgb # Should handle this better, but strict requirement says use extracted maps.
        
    orig_l_norm = lighting_maps["l_norm"][slice_y, slice_x]
    shadow_map = lighting_maps["shadow_strength"][slice_y, slice_x]
    texture_detail = lighting_maps["texture_detail"][slice_y, slice_x]
    
    # 2. Prepare Target
    t_l, t_a, t_b = target_lab
    
    # 3. Simulate Paint Interaction (On Crop)
    
    # A. Base Color (Albedo)
    # Start with full target color
    paint_l = np.full_like(orig_l_norm, t_l)
    paint_a = np.full_like(orig_l_norm, t_a)
    paint_b = np.full_like(orig_l_norm, t_b)
    
    # B. Lighting Integration (Soft Light Blend Mode)
    # Simulate realistic paint interaction where pigment covers but retains light/shadow.
    # L_target is the base color luminance (paint_l).
    # L_lighting is the environment luminance (orig_l_norm).
    
    # Soft Light Logic:
    # If Lighting > 0.5: L_out = L_target * (2*Lighting + (1-2*Lighting)*L_target) ?? (Complex)
    # Simple Soft Light Approx for u8: (A * B) / 128 + A * (255 - ((255-A)*(255-B)/128)) ?? No.
    
    # Let's use standard Photoshop Soft Light formula on normalized 0-1 values:
    # if L_lighting <= 0.5: Result = L_target - (1 - 2*L_lighting) * L_target * (1 - L_target)
    # if L_lighting > 0.5:  Result = L_target + (2*L_lighting - 1) * (D(L_target) - L_target)
    # where D(x) = sqrt(x) if x <= 0.25 else ...
    
    # A simpler approach that works well for paint:
    # Overlay Blend Logic but softer.
    # New L = (Target L) * (Lighting Factor) was bleaching because Lighting Factor < 1.0 (always darkens 0-255 range essentially).
    
    # Correct Approach:
    # We normalized L_lighting (orig_l_norm) to 0-1.
    # We treat paint_l as 0-255.
    
    if finish == "gloss":
        # Gloss retains strong specular highlights (bleaching is expected on highlights)
        # Use Hard Light or Overlay
        lighting_factor = np.power(orig_l_norm, 1.2)
        simulated_l = (paint_l * lighting_factor) # Keep multiply for gloss
    else:
        # Matte/Silk: Use a "Richness Preserving" Multiply.
        # Problem with Multiply: 200 * 0.8 = 160 (Darker)
        # Real paint ADDS opacity. It hides the dark underlying wall.
        # We need to mix the lighting map with a constant "Ambient Light" to simulate pigment opacity.
        
        # Opacity Factor: How much of the original wall darkness bleeds through?
        # Increased to 0.85 to prevent "bleaching" or background showing through too much
        opacity = 0.85 if finish == "matte" else 0.75
        
        # Flatten the lighting map towards medium gray (0.5) to simulate covering the wall
        flat_lighting = orig_l_norm * (1.0 - opacity) + 0.5 * opacity
        
        # Now multiply. This prevents deep blacks from turning the paint black.
        simulated_l = paint_l * flat_lighting * 2.0 # *2.0 to normalize around 0.5->1.0
        
        # Warmth preservation for "bleaching"
        # If the result is too bright/white, clamp the max luma to the target luma?
        # No, highlights should be brighter.
        
    # C. Shadow Preservation
    # Enhance shadows slightly to ground the paint
    # Desaturate color in deep shadows (restoring missing logic)
    desat_factor = 1.0 - (shadow_map * 0.4) # Slightly less aggressive desat
    simulated_l = simulated_l * desat_factor # Apply shadow to L
    
    # Apply shadow desaturation to A/B channels (Fixing UnboundLocalError)
    simulated_a = (paint_a - 128) * desat_factor + 128
    simulated_b = (paint_b - 128) * desat_factor + 128

    # D. Texture Re-injection
    # Texture detail is (L - Blurred). Centered at 0.
    # Add it back, but scale it by reflectance.
    simulated_l = simulated_l + (texture_detail * reflectance * 1.5) # Boost texture slightly for realism
    
    # Clip L
    simulated_l = np.clip(simulated_l, 0, 255)
    
    # E. Construct LAB Image
    simulated_l = simulated_l.astype(np.uint8)
    simulated_a = simulated_a.astype(np.uint8)
    simulated_b = simulated_b.astype(np.uint8)
    
    painted_lab_crop = cv2.merge([simulated_l, simulated_a, simulated_b])
    painted_rgb_crop = cv2.cvtColor(painted_lab_crop, cv2.COLOR_LAB2RGB)
    
    # F. Composition (Alpha Blending)
    from utils.mask_utils import feather_mask
    # Reduced blur_radius to 1 to prevent "bleeding" onto adjacent objects
    alpha = feather_mask(mask_crop, blur_radius=1)
    # alpha is (H, W), we need to broadcast to (H, W, 3)
    alpha_3d = np.atleast_3d(alpha)
    
    # Get ROI from output
    output = final_image_rgb.copy()
    roi = output[slice_y, slice_x].astype(np.float32)
    painted_f = painted_rgb_crop.astype(np.float32)
    
    # Blend: (1-alpha)*orig + alpha*painted
    blended = (1.0 - alpha_3d) * roi + alpha_3d * painted_f
    
    # Paste Back
    output[slice_y, slice_x] = blended.astype(np.uint8)
    
    return output
