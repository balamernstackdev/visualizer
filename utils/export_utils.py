from PIL import Image, ImageDraw, ImageFont
import io

def convert_to_downloadable(image_pil, format="PNG"):
    """Converts PIL image to bytes for download."""
    buf = io.BytesIO()
    image_pil.save(buf, format=format, quality=95)
    return buf.getvalue()

def create_comparison_image(original, painted, label_before="Before", label_after="After"):
    """Creates a side-by-side comparison image."""
    w, h = original.size
    
    # Create new image double width
    comparison = Image.new('RGB', (w * 2, h))
    comparison.paste(original, (0, 0))
    comparison.paste(painted, (w, 0))
    
    # Optional: Add text labels?
    # Simple for now.
    
    return comparison

def add_watermark(image_pil, text="AI Paint Visualizer"):
    """Adds a simple watermark to the bottom right."""
    # ... implementation can be added if needed, kept simple for now
    return image_pil
