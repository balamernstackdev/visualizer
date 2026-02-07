# AI Paint Visualizer Pro - Version 1.0.1
import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image




from utils.image_utils import resize_image_max_side, load_image_from_bytes, pil_to_cv2, cv2_to_pil
from utils.export_utils import convert_to_downloadable, create_comparison_image
from utils.lighting_utils import extract_lighting_maps
from utils.mask_utils import merge_masks, smooth_mask, dilate_mask
from ui.lasso_canvas import render_lasso_tool, render_click_tool, render_box_tool
from paint_ai.paint_engine import apply_realistic_paint
from utils.render_utils import render_high_res
from streamlit_javascript import st_javascript

# Page Config
st.set_page_config(
    page_title="AI Paint Visualizer Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for Asian Paints Look ---
st.markdown("""
<style>
    /* 1. Global Layout Fixes */
    /* Restore Sidebar Toggle: Make header transparent to keep UI clean */
    [data-testid="stHeader"] { 
        display: block !important;
        background: transparent !important;
        height: 3rem !important;
    }
    [data-testid="stDecoration"] { display: none !important; }
    
    /* Force Sidebar to be ALWAYS VISIBLE (No collapse) - DESKTOP ONLY */
    @media (min-width: 800px) {
        [data-testid="collapsedControl"] { display: none !important; }
        section[data-testid="stSidebar"] > button { display: none !important; }
    }
    
    .block-container { 
        padding-top: 2.5rem !important; /* Space for menu icon */
        padding-bottom: 0rem !important; 
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Adjust Global Title for Menu Button */
    h4 { 
        padding-left: 3rem !important; /* Space for the menu icon */
        margin-top: -5px !important;
    }

    /* 2. Responsive Layout for Mobile & Tablets (< 1100px) */
    @media (max-width: 1100px) {
        .block-container { 
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            padding-top: 3rem !important; /* Reduced from 4rem */
            max-width: 100vw !important;
            overflow-x: hidden !important;
        }
        
        /* Smaller Title for Mobile */
        h4 { 
            font-size: 1.05rem !important; 
            line-height: 2rem !important;
            margin-top: 5px !important; /* Reset negative margin */
            padding-left: 3.5rem !important; /* Ensure clears menu icon */
        }
        
        /* Ensure columns stay side-by-side but with better spacing */
        [data-testid="column"] { 
            min-width: 0 !important; 
            flex: 1 1 auto !important;
        }
        
        /* Ensure color picker stays visible and compact */
        .stColorPicker { min-width: 40px; }
        
        /* Fix vertical alignment for the header row specifically */
        [data-testid="stHorizontalBlock"] {
            align-items: center !important;
        }
    }

    /* 3. Button Styling */
    .stButton>button {
        background-color: #FF6600; color: white; border-radius: 20px; font-weight: bold; width: 100%;
    }
    
    /* 4. Canvas Styling */
    img, iframe { width: 100% !important; max-width: 100% !important; touch-action: none !important; }

    /* 5. Header Content Centering */
    [data-testid="column"]:nth-child(2) [data-testid="stVerticalBlock"] {
        align-items: center !important;
        justify-content: center !important;
    }

    /* 6. Remove ANY potential leading space */
    .stMarkdown, .stVerticalBlock { gap: 0rem !important; }
    iframe[title="streamlit_javascript.streamlit_javascript"] { display: none !important; }
    
    /* Asian Paints Header Style */
    .header-container {
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Convert Hex to LAB for Paint Engine
def hex_to_lab(hex_code):
    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    rgb_arr = np.array([[list(rgb)]], dtype=np.uint8) 
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    lab_arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2LAB)
    return lab_arr[0][0]


# --- Session State ---
if 'state' not in st.session_state:
    st.session_state.state = {
        'masks': [], # List of binary masks (H, W)
        'wall_assignments': {}, # {mask_idx: {'id': 'mm01', 'finish': 'matte'}}
        'history': [],
        'mode': 'select', # 'select', 'lasso_add', 'lasso_sub'
        'lighting_maps': None,
        'history': [],
        'mode': 'select', # 'select', 'lasso_add', 'lasso_sub'
        'lighting_maps': None,
        'image_id': None,
        'debug_logs': [],
        'cached_paint_cv2': None,
        'cached_assignments_hash': None,
        'compare_mode': False,
        'segmentation_mode': 'Walls (Default)',
        'selected_object_index': -1,
        'ai_ready': False # Lazy AI flag
    }

def add_log(msg):
    """Helper to add timestamped log."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.state['debug_logs'].insert(0, f"[{ts}] {msg}")
    # Keep log short
    if len(st.session_state.state['debug_logs']) > 50:
        st.session_state.state['debug_logs'].pop()

if 'canvas_key_id' not in st.session_state:
    st.session_state.canvas_key_id = 0

if 'last_click_coords' not in st.session_state:
    st.session_state.last_click_coords = None

def save_history():
    st.session_state.state['history'].append(st.session_state.state['wall_assignments'].copy())

# New: State for Depth Cycling
if 'selection_state' not in st.session_state:
    st.session_state.selection_state = {
        'last_click_pos': (-1, -1),
        'layer_index': 0
    }

def undo():
    if st.session_state.state['history']:
        st.session_state.state['wall_assignments'] = st.session_state.state['history'].pop()

    save_history()
    st.session_state.state['wall_assignments'] = {}

def reset_paint():
    save_history()
    st.session_state.state['wall_assignments'] = {}

# UPLOAD HANDLING: Usually in Sidebar, but we need JS screen width first
# ENSURE JS RUNS (Global Scope)
# This MUST be outside any conditional block to work immediately on first load
js_exists = st_javascript("window.innerWidth", key="screen_width_check_global")
if js_exists:
    st.session_state.screen_width = js_exists
    # Force rerun if width changed significantly to update canvas interactively?
    # Only if it was 0 before (first load)
    # Actually st_javascript triggers rerun automatically on return.
    
# UPLOAD HANDLING: Always Sidebar
uploaded_file = st.sidebar.file_uploader("Upload Room Image", type=["jpg", "png"])

@st.fragment
def render_dashboard(tool_mode, compare_mode=False, seg_mode="Walls (Default)", lasso_op="Add"):
    # --- UNIFIED CONTROL HEADER ---
    h_col1, h_col2, h_spacer = st.columns([5, 1, 2], vertical_alignment="center")
    js_width = st.session_state.get('screen_width', 0)
    is_mobile = (js_width == 0 or (js_width > 0 and js_width < 1100))
    
    # Initialize unified color state if not exists
    if 'unified_color' not in st.session_state:
        st.session_state['unified_color'] = "#E8C39E"
    
    if 'base_image' in st.session_state:
        img_w, _ = st.session_state.base_image.size
        # --- ROBUST GLOBAL MOBILE & TABLET TOUCH SUPPORT (Ver 16: Shielded Bridge) ---
        st.components.v1.html("""
            <script>
            function applyTouchToCanvas(win) {
                const elements = win.document.querySelectorAll('.upper-canvas');
                elements.forEach(canvas => {
                    if (canvas.dataset.touchFixed === 'v16') return;
                    canvas.dataset.touchFixed = 'v16';
                    
                    canvas.style.cssText += '; touch-action: none !important; user-select: none !important;';
                    
                    const dispatch = (type, e) => {
                        const rect = canvas.getBoundingClientRect();
                        const x = e.clientX - rect.left;
                        const y = e.clientY - rect.top;
                        
                        // Flag this event as ours to prevent infinite recursion
                        const params = {
                            clientX: e.clientX, clientY: e.clientY,
                            bubbles: true, cancelable: true, view: win,
                            pointerId: 99, pointerType: 'touch', isPrimary: true,
                            button: 0, buttons: (type === 'up' ? 0 : 1)
                        };

                        if (type === 'down') {
                            const dot = win.document.createElement('div');
                            dot.style.cssText = `position:absolute; left:${x-10}px; top:${y-10}px; width:20px; height:20px; background:rgba(0,255,100,0.8); border:2px solid white; border-radius:50%; z-index:99999; pointer-events:none; box-shadow:0 0 10px #00FF66;`;
                            canvas.parentElement.appendChild(dot);
                            setTimeout(() => dot.remove(), 800);

                            const ev = new PointerEvent('pointerdown', params);
                            ev._isSimulated = true;
                            canvas.dispatchEvent(ev);
                            
                            const mev = new MouseEvent('mousedown', params);
                            mev._isSimulated = true;
                            canvas.dispatchEvent(mev);
                        } else if (type === 'up') {
                            const ev = new PointerEvent('pointerup', params);
                            ev._isSimulated = true;
                            canvas.dispatchEvent(ev);
                            
                            const mev = new MouseEvent('mouseup', params);
                            mev._isSimulated = true;
                            canvas.dispatchEvent(mev);
                            
                            const cev = new MouseEvent('click', params);
                            cev._isSimulated = true;
                            canvas.dispatchEvent(cev);
                        }
                    };

                    // Only listen to "REAL" touches from the user's hand
                    canvas.addEventListener('pointerdown', e => {
                        if (e._isSimulated || e.pointerType !== 'touch') return;
                        dispatch('down', e);
                    }, true);
                    
                    canvas.addEventListener('pointerup', e => {
                        if (e._isSimulated || e.pointerType !== 'touch') return;
                        dispatch('up', e);
                    }, true);
                });
            }

            const connect = () => {
                applyTouchToCanvas(parent.window);
                parent.document.querySelectorAll('iframe').forEach(f => {
                    try { if(f.contentWindow) applyTouchToCanvas(f.contentWindow); } catch(e){}
                });
            };
            setInterval(connect, 1000);
            connect();
            </script>
        """, height=0)

        # Width Logic
        if js_width and js_width > 0:
            padding = 0 if is_mobile else 40
            target_width = js_width - padding
        else:
            target_width = 360 if is_mobile else 1600
        display_width = int(min(img_w, target_width))

        # Color and Header Row
        if is_mobile:
            st.markdown("<h4 style='margin:0 0 10px 0; padding:0; font-size: 1rem;'>🎨 AI Paint Visualizer Pro</h4>", unsafe_allow_html=True)
            def update_mobile():
                st.session_state['unified_color'] = st.session_state['unified_color_mobile']
            st.color_picker("Pick Color", st.session_state.get('unified_color', '#E8C39E'), key="unified_color_mobile", on_change=update_mobile)
            chosen_hex = st.session_state.get('unified_color', '#E8C39E')
            def update_hex_mobile():
                val = st.session_state.manual_hex_mobile.strip()
                if not val.startswith("#"): val = "#" + val
                if len(val) == 7 and all(c in "0123456789ABCDEFabcdef#" for c in val):
                    st.session_state['unified_color'] = val.upper()
                    st.rerun()
            st.text_input("Color Code", value=chosen_hex, key="manual_hex_mobile", on_change=update_hex_mobile)
        else:
            c_title, c_picker, c_code, _ = st.columns([3.5, 0.3, 0.5, 0.5], vertical_alignment="center")
            with c_title:
                st.markdown("<h4 style='margin:0; padding:0; line-height: 1.5rem; white-space: nowrap;'>🎨 AI Paint Visualizer Pro</h4>", unsafe_allow_html=True)
            with c_picker:
                def update_desktop():
                    st.session_state['unified_color'] = st.session_state['unified_color_desktop']
                st.color_picker("Color", st.session_state.get('unified_color', '#E8C39E'), key="unified_color_desktop", label_visibility="collapsed", on_change=update_desktop)
                chosen_hex = st.session_state.get('unified_color', '#E8C39E')
            with c_code:
                def update_hex_desktop():
                    val = st.session_state.manual_hex_desktop.strip()
                    if not val.startswith("#"): val = "#" + val
                    if len(val) == 7 and all(c in "0123456789ABCDEFabcdef#" for c in val):
                        st.session_state['unified_color'] = val.upper()
                        st.rerun()
                st.text_input("Hex", value=chosen_hex, key="manual_hex_desktop", label_visibility="collapsed", on_change=update_hex_desktop)
    else:
        chosen_hex = "#E8C39E"
        display_width = 800

    ui_lab = hex_to_lab(chosen_hex)
    ui_color = {'id': chosen_hex, 'hex': chosen_hex, 'lab': ui_lab}
    
    selected_finish = "Matte"
    selected_reflectance = 0.5
    
    # --- LAZY AI AND LIGHTING INITIALIZATION ---
    # Only run these if we have an image but haven't analyzed it yet
    if 'base_image' in st.session_state:
        if st.session_state.state.get('lighting_maps') is None:
            with st.spinner("🌤 Analyzing lighting..."):
                from utils.lighting_utils import extract_lighting_maps
                st.session_state.state['lighting_maps'] = extract_lighting_maps(st.session_state.base_image)
        
        # Only initialize SAM if it's the first time and we are in an AI tool mode
        if "AI" in tool_mode and not st.session_state.state.get('ai_ready'):
            from paint_ai.sam_loader import get_sam_predictor, download_model_if_needed
            if download_model_if_needed():
                with st.spinner("🧠 Connecting AI (one-time setup)..."):
                    st.session_state.predictor = get_sam_predictor()
                    if st.session_state.predictor:
                        st.session_state.predictor.set_image(np.array(st.session_state.base_image))
                        st.session_state.state['ai_ready'] = True
    # ---------------------------------------------

    if 'base_image' in st.session_state:
        preview_holder = st.empty()
        w, h = st.session_state.base_image.size
        
        # CSS for Fluid Fragment and Canvas Responsiveness
        # Ensure iframe and canvas fit within the container width on mobile
        st.markdown(f"""
            <style>
            /* Force container responsiveness */
            .main .block-container {{
                max-width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
            /* Canvas and iframe base styles */
            /* FORCE CANVAS RESPONSIVENESS */
            canvas, .upper-canvas, .lower-canvas, [data-testid="stHorizontalBlock"] > div {{
                width: 100% !important;
                height: auto !important;
                max-width: 100% !important;
            }}
            /* Specific fix for st-canvas wrapper divs */
            div[style*="width:"] canvas, div[style*="width:"] .upper-canvas {{
                width: 100% !important;
                height: auto !important;
            }}
            iframe {{
                aspect-ratio: {w}/{h} !important;
                width: 100% !important;
                height: auto !important;
            }}
            [data-testid="stDeckGlJsonChart"] {{
                width: 100% !important;
                max-width: 100% !important;
            }}
            /* Mobile-specific overrides */
            @media (max-width: 600px) {{
                body {{
                    overflow-x: hidden !important;
                    -webkit-overflow-scrolling: touch !important;
                }}
                .main .block-container {{
                    padding: 0 !important;
                    max-width: 100vw !important;
                }}
                /* Remove vertical gap between Streamlit elements on mobile */
                [data-testid="stVerticalBlock"] {{
                    gap: 0 !important;
                }}
                iframe, canvas, img, .upper-canvas, .lower-canvas {{
                    max-width: 100% !important;
                    width: 100% !important;
                    height: auto !important;
                    display: block !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    touch-action: none !important;
                }}
                canvas {{
                    transform-origin: top left !important;
                }}
                [data-testid="stDeckGlJsonChart"], 
                [data-testid="stDeckGlJsonChart"] > div,
                [data-testid="stDeckGlJsonChart"] canvas {{
                    max-width: 100% !important;
                    width: 100% !important;
                    overflow: hidden !important;
                }}
                [data-testid="stHorizontalBlock"] {{
                    flex-direction: row !important;
                    flex-wrap: nowrap !important;
                    gap: 0.5rem !important;
                }}
                [data-testid="column"] {{
                    width: auto !important;
                    flex: 1 1 auto !important;
                    min-width: 0 !important;
                }}
            }}
            </style>
        """, unsafe_allow_html=True)

        base_cv2 = pil_to_cv2(st.session_state.base_image)
        
        # Repaint logic (using caching)
        current_hash = str(st.session_state.state['wall_assignments'])
        if (st.session_state.state.get('cached_assignments_hash') == current_hash and 
            st.session_state.state.get('cached_paint_cv2') is not None):
            canvas_cv2 = st.session_state.state['cached_paint_cv2'].copy()
        else:
            canvas_cv2 = base_cv2.copy()
            for mask_idx, paint_data in st.session_state.state['wall_assignments'].items():
                if 0 <= mask_idx < len(st.session_state.state['masks']):
                    # FIX: Remove dilate_mask here. Smooth is enough, engine handles edges with alpha.
                    mask = smooth_mask(st.session_state.state['masks'][mask_idx])
                    canvas_cv2 = apply_realistic_paint(canvas_cv2, mask, paint_data['lab'], paint_data['finish'].lower(), paint_data['reflectance'], st.session_state.state['lighting_maps'])
            st.session_state.state['cached_paint_cv2'] = canvas_cv2.copy()
            st.session_state.state['cached_assignments_hash'] = current_hash

        img_disp = cv2_to_pil(canvas_cv2)

        # COMPARISON VIEW INJECTION
        if compare_mode:
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.image(st.session_state.base_image, caption="Original", use_container_width=True)
            with c_col2:
                st.image(img_disp, caption="Painted", use_container_width=True)
            # When in comparison mode, clicking/lasso is disabled or we just show them?
            # User might want to paint while comparing. Let's keep one as the "Active" one.
            st.info("💡 Comparison mode active. Use the tools below to continue painting.")

        # Key logic: Stabilize keys to prevent remounting during mobile viewport shifts
        # Only change key when an explicit reset/new image happens (canvas_key_id)
        click_key = f"click_tool_{st.session_state.canvas_key_id}"
        lasso_key = f"lasso_tool_{st.session_state.canvas_key_id}"
        box_key = f"box_tool_{st.session_state.canvas_key_id}"
        
        # Use Lasso or Click Tool based on Sidebar Mode
        # tool_mode logic
        if "AI Click" in tool_mode:
            value = render_click_tool(img_disp, key=click_key, canvas_width=display_width)
            # Use a unique signature for each click to ensure it triggers even after remounts
            if value is not None:
                click_sig = f"{value['x']}_{value['y']}_{st.session_state.canvas_key_id}"
                if click_sig != st.session_state.get('last_click_sig'):
                    st.session_state.last_click_sig = click_sig
                    x, y = value['x'], value['y']
                    st.toast(f"🎯 Point Detected at {x}, {y}!", icon="🎯")
                    add_log(f"Click Detected: {x}, {y}")
                    
                    # 1. CHECK FOR HIT ON EXISTING OBJECT (Edit Mode)
                    # Check in reverse order (top-most first)
                    hit_index = -1
                    sorted_keys = sorted(list(st.session_state.state['wall_assignments'].keys()), reverse=True)
                    for m_idx in sorted_keys:
                        # Bounds check key just in case
                        if m_idx < len(st.session_state.state['masks']):
                            mask = st.session_state.state['masks'][m_idx]
                            if mask[y, x]:
                                hit_index = m_idx
                                break
                    
                    if hit_index != -1:
                        st.session_state.state['selected_object_index'] = hit_index
                        new_color = st.session_state.state['wall_assignments'][hit_index]['hex']
                        st.session_state['unified_color'] = new_color
                        st.rerun()
                    # 2. IF NO HIT, PROCEED WITH SAM (New Object)
                    candidates = []
                    for i, mask in enumerate(st.session_state.state['masks']):
                         if mask[y, x]:
                             candidates.append({'mask': mask, 'id': f"arch_{i}"})
                    
                    if 'predictor' in st.session_state:
                        p_masks, _, _ = st.session_state.predictor.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
                        
                        # SEGMENTATION MODE LOGIC: Influence candidate selection
                        # SAM p_masks indices: 0 (Smallest/Detail), 1 (Medium/Object), 2 (Whole/Largest)
                        if seg_mode == "Small Objects":
                            indices = [0, 1, 2]
                        elif seg_mode == "Floors/Whole":
                            indices = [2, 1, 0]
                        else: # Walls (Default)
                            indices = [1, 0, 2] # Usually index 1 is best for walls
                            
                        for j in indices:
                            candidates.append({'mask': smooth_mask(p_masks[j]), 'id': f"pinpoint_{j}"})
                        add_log(f"SAM Generated {len(candidates)} candidates")
                    st.toast(f"AI found {len(candidates)} wall options", icon="🤖")

                    if candidates:
                        candidates.sort(key=lambda c: np.sum(c['mask']))
                        last_cx, last_cy = st.session_state.selection_state['last_click_pos']
                        dist = np.sqrt((x-last_cx)**2 + (y-last_cy)**2)
                        st.session_state.selection_state['layer_index'] = (st.session_state.selection_state['layer_index'] + 1) % len(candidates) if dist < 15 else 0
                        st.session_state.selection_state['last_click_pos'] = (x, y)
                        
                        selected_candidate = candidates[st.session_state.selection_state['layer_index']]
                        save_history()
                        
                        if selected_candidate['id'].startswith('arch_'):
                            chosen_idx = int(selected_candidate['id'].split('_')[1])
                        else:
                            chosen_idx = len(st.session_state.state['masks'])
                            st.session_state.state['masks'].append(selected_candidate['mask'])
                        
                        st.session_state.state['wall_assignments'][chosen_idx] = {
                            'id': ui_color['id'], 'hex': ui_color['hex'], 'lab': ui_color['lab'],
                            'finish': selected_finish, 'reflectance': selected_reflectance
                        }
                        st.toast("✅ Paint Applied!", icon="🎨")
                        st.session_state.canvas_key_id += 1
                        st.rerun()
                    else:
                        st.error("AI couldn't find a wall at that spot. Try clicking slightly differently.")
                        add_log("No SAM candidates found.")
        elif "Box" in tool_mode:
            box = render_box_tool(img_disp, key=box_key, canvas_width=display_width)
            if box is not None:
                if 'predictor' in st.session_state:
                    # SAM Box prediction
                    p_masks, _, _ = st.session_state.predictor.predict(box=np.array(box), multimask_output=True)
                    
                    # Same segmentation preference as click
                    if seg_mode == "Small Objects": indices = [0, 1, 2]
                    elif seg_mode == "Floors/Whole": indices = [2, 1, 0]
                    else: indices = [1, 0, 2]

                    # Just pick the preferred one
                    # Usually predict returns sorted by score, but let's confirm logic
                    final_mask = smooth_mask(p_masks[indices[0]]) # Use the first preferred index
                    
                    save_history()
                    idx = len(st.session_state.state['masks'])
                    st.session_state.state['masks'].append(final_mask)
                    st.session_state.state['wall_assignments'][idx] = {'id': ui_color['id'], 'hex': ui_color['hex'], 'lab': ui_color['lab'], 'finish': selected_finish, 'reflectance': selected_reflectance}
                    
                    # Auto-select the newly created object
                    st.session_state.state['selected_object_index'] = idx
                    
                    st.session_state.canvas_key_id += 1
                    st.rerun()
            else:
                st.info("💡 Click and drag on the image to select an object with a box.")
        else:
            # Lasso UI inside fragment
            lasso_key = f"lasso_tool_{st.session_state.canvas_key_id}"
            lasso_mask = render_lasso_tool(img_disp, key=lasso_key, canvas_width=display_width)
            if lasso_mask is not None and np.any(lasso_mask):
                if st.button("Apply Paint" if lasso_op == "Add" else "Apply Remove", type="primary"):
                    save_history()
                    if lasso_op == "Add":
                        idx = len(st.session_state.state['masks'])
                        st.session_state.state['masks'].append(lasso_mask)
                        st.session_state.state['wall_assignments'][idx] = {'id': ui_color['id'], 'hex': ui_color['hex'], 'lab': ui_color['lab'], 'finish': selected_finish, 'reflectance': selected_reflectance}
                    else:
                        for idx in list(st.session_state.state['wall_assignments'].keys()):
                            st.session_state.state['masks'][idx] = np.logical_and(st.session_state.state['masks'][idx], np.logical_not(lasso_mask))
                    st.session_state.canvas_key_id += 1
                    st.rerun()


        with st.expander("Export Options", expanded=False):
            if is_mobile:
                # Vertical Stack for Mobile
                if st.button("Download High Quality (4K) Image", key="dl_btn_mobile", use_container_width=True):
                    with st.spinner("Generating 4K Render..."):
                        high_res_cv2 = render_high_res(
                            st.session_state.full_res_image, 
                            st.session_state.state['masks'], 
                            st.session_state.state['wall_assignments']
                        )
                        download_bytes = convert_to_downloadable(cv2_to_pil(high_res_cv2))
                        st.download_button(
                            "Confirm 4K Download", 
                            download_bytes, 
                            "painted_room_4k.png", 
                            "image/png", 
                            use_container_width=True
                        )
                st.markdown("<br>", unsafe_allow_html=True)
                st.image(create_comparison_image(st.session_state.base_image, cv2_to_pil(canvas_cv2)), caption="Comparison", width=display_width)
            else:
                # Horizontal Columns for Desktop
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Generate High Quality (4K) Download", key="dl_btn_desktop"):
                        with st.spinner("Preparing 4K resolution image (this takes a few seconds)..."):
                            # Decompress full res image for rendering
                            import io
                            full_img = Image.open(io.BytesIO(st.session_state.full_res_bytes))
                            high_res_cv2 = render_high_res(
                                full_img, 
                                st.session_state.state['masks'], 
                                st.session_state.state['wall_assignments']
                            )
                            download_bytes = convert_to_downloadable(cv2_to_pil(high_res_cv2))
                            st.download_button("Confirm 4K Download", download_bytes, "painted_room_4k.png", "image/png")
                with c2:
                    st.image(create_comparison_image(st.session_state.base_image, cv2_to_pil(canvas_cv2)), caption="Comparison")



# 2. SIDEBAR CONTROLS (Define Inputs BEFORE Usage)
# Moving this up so tool_mode is available for render_dashboard
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🛠 Selection Tool")
    tool_mode = st.radio("Method", [
        "🎯 AI Click Object (Point)", 
        "✨ AI Object (Drag Box)", 
        "🪄 Manual Lasso (Polygon)"
    ], label_visibility="visible")

    # UNIQUE CONTROLS PER MODE
    seg_mode = "Walls (Default)"
    lasso_op = "Add"
    
    if "AI" in tool_mode:
        with st.expander("⚙️ Advanced Precision (Optional)", expanded=True):
            seg_mode = st.radio("Segmentation Mode", [
                "Walls (Default)", 
                "Small Objects", 
                "Floors/Whole"
            ], index=["Walls (Default)", "Small Objects", "Floors/Whole"].index(st.session_state.state.get('segmentation_mode', 'Walls (Default)')))
            st.session_state.state['segmentation_mode'] = seg_mode
    
    # Common Add/Remove for Lasso and Drag Box (Consolidated Logic)
    if "Lasso" in tool_mode or "Drag Box" in tool_mode:
        with st.expander("🎨 Operation", expanded=True):
            lasso_op = st.radio("Operation", ["Add", "Remove"], horizontal=True, key="lasso_op_radio")

    st.markdown("### 👁 View Settings")
    compare_mode = st.toggle("Compare Before/After", value=st.session_state.state.get('compare_mode', False))
    st.session_state.state['compare_mode'] = compare_mode

    st.markdown("---")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        if st.button("Undo", key="undo_btn"):
            undo(); st.rerun()
    with col_u2:
        if st.button("Reset", key="reset_btn"):
            reset_paint(); st.rerun()

if uploaded_file:
    # 0. ENSURE JS RUNS (Main Body) - MOVED TO TOP
    # Kept here just in case, but global is better
    # Removing duplicate call to avoid double reruns
        
    # 1. FAST IMAGE LOAD (Ghost Mode)
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.state.get('image_id') != current_file_id:
        import gc, io
        # Reset state on new file
        st.session_state.state = {
            'masks': [],
            'wall_assignments': {},
            'history': [],
            'image_id': current_file_id,
            'lighting_maps': None,
            'cached_paint_cv2': None,
            'cached_assignments_hash': "",
            'debug_logs': [],
            'compare_mode': False,
            'selected_object_index': -1,
            'ai_ready': False
        }
        st.session_state.canvas_key_id += 1
        
        # Use Ghost Load: Open, Resize, then CLOSE and clear
        image_raw = Image.open(uploaded_file).convert("RGB")
        
        # Store as BYTES to save massive RAM
        img_byte_arr = io.BytesIO()
        image_raw.save(img_byte_arr, format='JPEG', quality=90)
        st.session_state.full_res_bytes = img_byte_arr.getvalue()
        
        # Resize aggressively for cloud RAM limits
        limit = 640 if is_mobile else 900
        st.session_state.base_image = resize_image_max_side(image_raw, limit)
        
        # Clear large raw image immediately
        del image_raw
        gc.collect()
    
    # 2. RENDER DASHBOARD
    # Pass necessary state
    render_dashboard(tool_mode, compare_mode=st.session_state.state['compare_mode'], seg_mode=seg_mode, lasso_op=lasso_op)

else:
    # LANDING PAGE
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>Welcome to AI Paint Visualizer Pro</h2>
        <p>Upload a room photo from the sidebar to start designing.</p>
    </div>
    """, unsafe_allow_html=True)
    # 2. SIDEBAR CONTROLS (Define Inputs First)

# --- Painted Objects Manager (Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🏘 Painted Objects")
if not st.session_state.state['wall_assignments']:
    st.sidebar.caption("No objects painted yet. Click a wall to begin.")
else:
    # Sort assignments to render in order of creation (or reverse? Doesn't matter for list)
    # We want latest on bottom usually, or top. 
    # Let's iterate sorted keys.
    sorted_msg_keys = sorted(list(st.session_state.state['wall_assignments'].keys()))
    
    for m_idx in sorted_msg_keys:
        data = st.session_state.state['wall_assignments'][m_idx]
        
        # Auto-Expand only active one
        is_expanded = (m_idx == st.session_state.state.get('selected_object_index', -1))
        
        with st.sidebar.expander(f"Object #{m_idx}", expanded=is_expanded):
            # If expanded, we assume user interacted with it recently
            if is_expanded:
                 st.sidebar.caption("✅ Currently Selected")

            new_h = st.sidebar.color_picker(f"Color #{m_idx}", data['hex'], key=f"obj_h_{m_idx}")
            new_f = st.sidebar.selectbox(f"Finish #{m_idx}", ["Matte", "Silk", "Gloss"], 
                                    index=["matte", "silk", "gloss"].index(data['finish'].lower()), 
                                    key=f"obj_f_{m_idx}")
            
            if new_h != data['hex'] or new_f.lower() != data['finish'].lower():
                st.session_state.state['wall_assignments'][m_idx].update({
                    'hex': new_h, 'lab': hex_to_lab(new_h), 'finish': new_f
                })
                # Set this as active if changed
                st.session_state.state['selected_object_index'] = m_idx
                st.rerun() 
            
            if st.sidebar.button(f"Remove Object #{m_idx}", key=f"rem_{m_idx}"):
                save_history()
                del st.session_state.state['wall_assignments'][m_idx]
                if st.session_state.state.get('selected_object_index') == m_idx:
                    st.session_state.state['selected_object_index'] = -1
                st.rerun()



# --- Debug Sidebar ---
st.sidebar.divider()
with st.sidebar.expander("Debug Info", expanded=False):
    debug_js_width = st.session_state.get('screen_width', 0)
    is_mobile_debug = (debug_js_width == 0 or (debug_js_width > 0 and debug_js_width < 1100))
    st.write(f"Device: {'Tablet/Mobile' if is_mobile_debug else 'Desktop'}")
    st.write(f"Screen: {debug_js_width}px")
    st.write(f"Last Tap: {st.session_state.get('last_click_coords', 'None')}")
    st.write(f"Pointer Engine: Active")
    if st.button("🧪 Simulate Middle Click"):
        # Force a click at center for testing SAM
        st.session_state.last_click_coords = {"x": 800, "y": 800}
        st.rerun()
    if st.button("🔄 Reset Interaction State"):
        st.session_state.last_click_coords = None
        st.rerun()
