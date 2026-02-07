import cv2
import numpy as np

def extract_lighting_maps(image_rgb):
    """
    Extracts lighting components from the image for physics-based rendering.
    Returns:
        gray: Grayscale version (H, W)
        highlight: Highlight map (H, W)
        shadow: Shadow map (H, W)
    """
    # Convert to numpy if PIL Image
    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.array(image_rgb)

    # Convert to LAB to get Luminance
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Normalize L channel 0-1
    l_norm = l_channel.astype(np.float32) / 255.0
    
    # 1. Texture/Detail Extraction (High Pass Filter)
    # Blur to get low frequency (lighting), subtract to get texture
    blurred = cv2.GaussianBlur(l_channel, (21, 21), 0)
    texture_detail = l_channel.astype(np.float32) - blurred.astype(np.float32)
    # texture_detail is centered around 0.
    
    # 2. Shadow Map
    # Shadows are areas with significantly lower luminance than local average?
    # Or just use the inverted luminance for multiplication?
    # For multiplicative blending: Paint * (L / 255) is standard basic.
    # But real walls have ambient occlusion.
    
    # We will use the raw L channel as the "Lighting Map".
    # And we calculate a "Shadow Strength" map to desaturate paint in dark areas.
    
    # Shadow strength: 1.0 = deep shadow, 0.0 = bright light
    # Invert L
    shadow_strength = 1.0 - l_norm
    # Contrast stretch shadow map to focus on real shadows
    shadow_strength = np.clip((shadow_strength - 0.2) * 2.0, 0.0, 1.0)
    
    return {
        "luminance": l_channel,
        "l_norm": l_norm,
        "shadow_strength": shadow_strength,
        "texture_detail": texture_detail
    }

def adjust_white_balance(image_rgb):
    """
    Simple Gray World assumption white balance.
    """
    result = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    # shift a and b to be centered around 128 (neutral)
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
