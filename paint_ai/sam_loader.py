import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import requests
import streamlit as st

# Switched to ViT-B for Speed Optimization (with high density scan)
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

def download_model_if_needed():
    """Downloads the SAM checkpoint if it doesn't exist."""
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        st.info(f"Downloading SAM model ({MODEL_TYPE})... this is ~375MB. Please wait.")
        try:
            response = requests.get(SAM_CHECKPOINT_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8 * 1024 # 8KB
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            downloaded = 0
            
            with open(SAM_CHECKPOINT_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded / total_size
                        progress_bar.progress(min(percent, 1.0))
                        status_text.text(f"Downloaded {downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB")
            
            progress_bar.empty()
            status_text.empty()
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return False
    return True

@st.cache_resource
def load_sam_model():
    """Loads the SAM model and returns it. Cached by Streamlit."""
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        # We don't want to auto-download inside a cached function if it's large
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        # On CPU, we stay in float32 for compatibility, but we can limit threads
        if device == "cpu":
            torch.set_num_threads(1) 
        sam.to(device=device)
        return sam
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_mask_generator(sam):
    """Returns an automatic mask generator optimized for walls."""
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24, # 24 = Standard Density (Maximum Speed)
        pred_iou_thresh=0.70, # Aggressively lowered to catch ALL walls
        stability_score_thresh=0.80, # Relaxed stability to accept more valid regions
        crop_n_layers=0, # Faster without crops
        min_mask_region_area=20, # Tiny threshold to keep small slivers
    )

def get_predictor(sam):
    """Returns a predictor for prompt-based segmentation."""
    return SamPredictor(sam)

def get_sam_predictor():
    """Convenience function to load model and return predictor."""
    sam = load_sam_model()
    if sam:
        return get_predictor(sam)
    return None
