import os
import streamlit as st
import urllib.request

def ensure_sam_model_exists(model_type, model_path):
    """
    Checks if the SAM model file exists. If not, provides a download button or auto-downloads.
    """
    if os.path.exists(model_path):
        return True
    
    st.warning(f"⚠️ SAM Model file ({model_path}) not found!")
    
    # Define official URLs
    urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b31ee.pth"
    }
    
    url = urls.get(model_type)
    if not url:
        st.error("Unknown model type. Cannot download.")
        return False
    
    st.info(f"The AI model is too large for GitHub (2.5GB). You need to download it to the server.")
    
    if st.button(f"📥 Download {model_type} Model (2.5GB)"):
        with st.status(f"Downloading {model_type} model... this will take a few minutes.", expanded=True) as status:
            try:
                def report(block_num, block_size, total_size):
                    read_so_far = block_num * block_size
                    if total_size > 0:
                        percent = read_so_far * 1e2 / total_size
                        status.update(label=f"Downloading: {percent:.1f}% done", state="running")
                
                urllib.request.urlretrieve(url, model_path, reporthook=report)
                status.update(label="✅ Download Complete!", state="complete")
                st.success("Model downloaded! The app will now reload.")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")
                status.update(label="❌ Download Failed", state="error")
                return False
    
    return False
