from PIL import Image, ImageOps
import numpy as np
import cv2

def resize_image_max_side(image_pil, max_side=1024):
    """Resizes image so its longest side is at most max_side."""
    w, h = image_pil.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image_pil

def load_image_from_bytes(file_bytes):
    """Loads image from bytes and handles orientation."""
    image = Image.open(file_bytes)
    image = ImageOps.exif_transpose(image) # Fix mobile upload rotation
    return image.convert("RGB")

def pil_to_cv2(image_pil):
    """Converts PIL image to OpenCV format (RGB)."""
    return np.array(image_pil)

def cv2_to_pil(image_cv2):
    """Converts OpenCV image to PIL."""
    return Image.fromarray(image_cv2)
