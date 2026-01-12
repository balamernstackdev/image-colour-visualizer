import streamlit as st
import numpy as np
import torch
import threading
import time
import pickle
from PIL import Image
import cv2
import os
import io
import logging
import requests
import gc
import warnings

# 🛡️ WARNING SHIELD: Silence technical chatter from AI libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Also hide specific logs from noisy libraries
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("mobile_sam").setLevel(logging.ERROR)

# 🚀 ULTIMATE 1GB RAM PROTECTION (Step Id 974+)
# Forcing single-threaded CPU mode prevents RAM spikes from massive stack allocations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False # Save more RAM

# 🚀 DEPLOYMENT VERSION SYNC (Step Id 1714+)
# Increment this to force Streamlit Cloud to discard old AI logic caches.
CACHE_SALT = "V1.0.9-STABLE"

from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_image_comparison import image_comparison
# 📦 DEPENDENCY GUARD
# We check these first to give the user a clean install instruction if they are missing locally.
try:
    import mobile_sam
    import timm
    from core.segmentation import SegmentationEngine, sam_model_registry
except Exception as e:
    import streamlit as st
    # TRANSIENT ERROR GUARD (Step Id 1597+)
    # Sometimes during deployment, Python's module system hits a "blip" (KeyError) 
    # while files are being moved. We catch this and show a friendly UI.
    st.error("### 🚀 Application Updating")
    st.info("The AI engine is syncing with the latest precision updates. This usually takes 2-3 seconds.")
    if "mobile_sam" in str(e) or "timm" in str(e):
        st.code("pip install timm git+https://github.com/ChaoningZhang/MobileSAM.git")
        st.write("If you are running locally, please run the command above.")
    else:
        st.write("Refreshing the environment. Please wait a moment...")
        st.button("Click to Refresh Now")
    st.stop()

# 🎯 GLOBAL AI CONFIGURATION
MODEL_TYPE = "vit_t"
CHECKPOINT_PATH = "weights/mobile_sam.pt"
from core.colorizer import ColorTransferEngine

# --- CONFIGURATION & STYLES ---
def setup_page():
    st.set_page_config(
        page_title="Color Visualizer Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_styles():
    st.markdown("""
        <style>
        /* Force Light Theme */
        :root {
            --primary-color: #ff4b4b;
            --background-color: #ffffff;
            --secondary-background-color: #f0f2f6;
            --text-color: #31333F;
            --font: "Segoe UI", sans-serif;
        }
        
        /* Main App Background */
        .stApp {
            background-color: #ffffff;
        }
        
        /* Sidebar Background & Text */
        /* Sidebar - Professional Light */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important; 
            border-right: 1px solid #e6e6e6;
            width: 350px !important;
        }
        
        /* Sidebar collapse adjustment for the main content area */
        [data-testid="stSidebarNav"] {
            width: 350px !important;
        }
        
        /* Sidebar Text Color */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label {
            color: #31333F !important;
        }

        /* Sidebar Inputs - Standardize */
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: #ffffff;
            color: #31333F;
            border-color: #dcdcdc;
        }
        
        /* Reset Global Text to Dark for Main Area Only */
        .main .block-container h1, 
        .main .block-container h2, 
        .main .block-container h3, 
        .main .block-container p, 
        .main .block-container span, 
        .main .block-container div {
            color: #333333 !important;
        }
    
        /* Buttons */
        .stButton>button {
            background-color: #ffffff;
            color: #333333; /* Default button text dark */
            border: 1px solid #dcdcdc;
            border-radius: 8px;
            transition: all 0.2s;
        }
        
        /* Sidebar Buttons Specifics */
        [data-testid="stSidebar"] .stButton>button {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #dcdcdc;
        }
        [data-testid="stSidebar"] .stButton>button:hover {
            background-color: #f0f0f0;
            border-color: #bbbbbb;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            padding: 20px;
            border: 2px dashed #cccccc;
            border-radius: 10px;
            background-color: #ffffff;
            text-align: center;
        }
        
        /* Center align main landing text */
        .landing-header {
            text-align: center;
            padding-top: 50px;
            padding-bottom: 20px;
        }
        .landing-header h1 {
            color: #333333 !important;
        }
        .landing-sub {
            text-align: center;
            color: #666666 !important;
            font-size: 1.1rem;
            margin-bottom: 40px;
        }
        /* Force main container to use full width */
        /* Force main container to use full width */
        .main .block-container {
            max-width: 95% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* RESPONSIVE: Mobile Adjustments */
        @media (max-width: 768px) {
            .main .block-container {
                max-width: 100% !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            [data-testid="stSidebar"] {
                width: 100% !important; /* Full width sidebar on mobile */
            }
        }

        /* Whitelabel - Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stHeader"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

# --- MODEL MANAGEMENT ---
@st.cache_resource
def get_sam_model(path, type_name, salt=""):
    """Load and cache the heavy model weights globally."""
    if not os.path.exists(path):
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create the model instance using the registry directly
    model = sam_model_registry[type_name](checkpoint=path)
    model.to(device=device)
    # We don't set model.device here to avoid AttributeError on cloud environments
    return model

@st.cache_resource
def get_global_lock():
    """Lock to prevent multiple AI sessions from hitting the CPU at once."""
    return threading.Lock()

@st.cache_resource
def get_sam_engine_singleton(checkpoint_path, model_type, salt=""):
    """Global engine singleton to avoid session state duplication."""
    model = get_sam_model(checkpoint_path, model_type, salt=salt)
    if model is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SegmentationEngine(model_instance=model, device=device)

def get_sam_engine(checkpoint_path, model_type):
    """Wrapped getter."""
    return get_sam_engine_singleton(checkpoint_path, model_type, salt=CACHE_SALT)

# --- HELPER LOGIC ---
def get_crop_params(image_width, image_height, zoom_level, pan_x, pan_y):
    """
    Calculate crop coordinates based on zoom and normalized pan (0.0 to 1.0).
    """
    if zoom_level <= 1.0:
        return 0, 0, image_width, image_height

    # visible width/height
    view_w = int(image_width / zoom_level)
    view_h = int(image_height / zoom_level)

    # ensure pan stays within bounds
    # pan_x=0 -> left edge, pan_x=1 -> right edge
    max_x = image_width - view_w
    max_y = image_height - view_h
    
    start_x = int(max_x * pan_x)
    start_y = int(max_y * pan_y)
    
    return start_x, start_y, view_w, view_h

def composite_image(original_rgb, masks_data):
    """
    Apply all color layers to user image using optimized multi-layer blending.
    PERFORMANCE: Uses incremental background caching to speed up tuning.
    """
    # Filter only visible layers
    visible_masks = [m for m in masks_data if m.get('visible', True)]
    
    # Selective Compositing Optimization
    # If the ONLY thing that changed is the tuning parameters of the TOP-MOST layer,
    # we can use a cached background of the previous N-1 layers.
    if len(visible_masks) > 1:
        # Check if we have a valid background cache
        cache = st.session_state.get("bg_cache")
        if cache is not None and cache.get("mask_count") == len(visible_masks) - 1:
            # SHAPE GUARD (Step Id 846+): Ensure cache matches current image resolution
            if cache['image'].shape[:2] != original_rgb.shape[:2]:
                mismatched = True
            else:
                # Check all previous masks
                mismatched = False
                for i in range(len(visible_masks) - 1):
                    curr = visible_masks[i]
                    cached = cache['masks'][i]
                    
                    # Shape check for individual mask
                    if curr['mask'].shape[:2] != original_rgb.shape[:2]:
                        mismatched = True
                        break

                    # Check critical visual parameters
                    if (curr.get('color') != cached.get('color') or 
                        curr.get('opacity') != cached.get('opacity') or
                        curr.get('finish') != cached.get('finish') or
                        curr.get('brightness') != cached.get('brightness') or
                        curr.get('contrast') != cached.get('contrast') or
                        curr.get('saturation') != cached.get('saturation') or
                        curr.get('hue') != cached.get('hue') or
                        curr.get('texture_path') != cached.get('texture_path')):
                        mismatched = True
                        break
                        
                    # Check mask geometry
                    if id(curr['mask']) != id(cached['mask']):
                        mismatched = True
                        break
            
            if not mismatched:
                # Use background cache and only composite the NEWEST layer
                background = cache['image']
                # Extra safety: check newest layer shape
                if visible_masks[-1]['mask'].shape[:2] == background.shape[:2]:
                    return ColorTransferEngine.composite_multiple_layers(background, [visible_masks[-1]])
                else:
                    mismatched = True

    # If no cache or cache invalid, do full composite
    if len(visible_masks) > 1:
        # Process N-1 layers specifically to populate cache
        bg_layers = visible_masks[:-1]
        background = ColorTransferEngine.composite_multiple_layers(original_rgb, bg_layers)
        
        # Update cache
        st.session_state["bg_cache"] = {
            "masks": [m.copy() for m in bg_layers],
            "mask_count": len(bg_layers),
            "image": background
        }
        
        # Composite final layer on top
        return ColorTransferEngine.composite_multiple_layers(background, [visible_masks[-1]])
        
    else:
        # 0 or 1 layer, just do it directly
        result = ColorTransferEngine.composite_multiple_layers(original_rgb, visible_masks)
        
        # Update cache even for single layer to prep for the second one
        if len(visible_masks) == 1:
             st.session_state["bg_cache"] = {
                "masks": [visible_masks[0].copy()],
                "mask_count": 1,
                "image": result 
            }
        return result

def initialize_session_state():
    """Initialize all session state variables with multi-layer safety."""
    defaults = {
        "image": None,          # 640px preview image
        "image_original": None, # Full resolution original
        "file_name": None,
        "masks": [],
        "zoom_level": 1.0,
        "pan_x": 0.5,
        "pan_y": 0.5,
        "last_click_global": None,
        "mask_level": None, # 0, 1, or 2 for granularity
        "bg_cache": None,   # For selective compositing performance
        "sampling_mode": False,
        "composited_cache": None,
        "render_id": 0,         # Versioning for instant UI refresh
        "picked_color": "#ff4b4b", # Default primary color
        "engine_ready": False,  # Prevent redundant loading spinners
        "selected_layer_idx": None, # Track which layer is being edited
        "last_export": None     # High-res download buffer
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Extra Safety: Ensure all current masks have the latest properties 
    if st.session_state.get("masks"):
        for m in st.session_state["masks"]:
            if 'visible' not in m: m['visible'] = True
            if 'brightness' not in m: m['brightness'] = 0.0
            if 'contrast' not in m: m['contrast'] = 1.0
            if 'saturation' not in m: m['saturation'] = 1.0
            if 'hue' not in m: m['hue'] = 0.0
            if 'opacity' not in m: m['opacity'] = 1.0
            if 'finish' not in m: m['finish'] = 'Standard'
            if 'tex_rot' not in m: m['tex_rot'] = 0
            if 'tex_scale' not in m: m['tex_scale'] = 1.0
            # CRITICAL: Clear heavy cached blur arrays to save RAM
            if 'mask_soft' in m: del m['mask_soft']

# --- UI COMPONENTS ---
def render_sidebar(sam, device_str):
    with st.sidebar:
        st.title("🎨 Visualizer Studio")
        st.caption(f"Running on: {device_str}")
        
        # Upload Section
        uploaded_file = st.file_uploader("Start Project", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Load and set image
            if st.session_state.get("image_path") != uploaded_file.name:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Store Original
                st.session_state["image_original"] = image.copy()
                
                # OPTIMIZATION: Create Work Image (Preview)
                # With MobileSAM (Step Id 993+), we can safely go back to 640px 
                # as the model itself is 10x smaller.
                max_dim = 640 
                h, w = image.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                st.session_state["image"] = image
                st.session_state["image_path"] = uploaded_file.name
                st.session_state["masks"] = []
                st.session_state["composited_cache"] = image.copy() # Init cache
                
                # Clear Colorizer LAB Cache for new image
                for k in list(st.session_state.keys()):
                    if k.startswith("base_l_"):
                        del st.session_state[k]
                
                # Reset SAM
                sam.is_image_set = False
                st.session_state["ai_ready"] = False
                # Defer sam.set_image(image) to the lazy loading block below for perceived speed
                # st.rerun() removed to smoother flow
        else:
            # Reset state if file cleared
            if st.session_state.get("image") is not None:
                st.session_state["image"] = None
                st.session_state["image_original"] = None
                st.session_state["image_path"] = None
                st.session_state["masks"] = []
                st.session_state["composited_cache"] = None
                sam.is_image_set = False
                st.rerun()
        
        # Project Persistence (NEW)
        if st.session_state.get("image") is not None:
            st.divider()
            st.subheader("💾 Project")
            
            # 1. Save Project
            project_data = {
                "masks": st.session_state["masks"],
                "image_path": st.session_state.get("image_path")
            }
            project_bytes = pickle.dumps(project_data)
            
            st.download_button(
                label="📤 Export Project (.studio)",
                data=project_bytes,
                file_name=f"{st.session_state.get('image_path', 'project')}.studio",
                mime="application/octet-stream",
                use_container_width=True,
                help="Save your current progress to a file to resume later."
            )
            
            # 2. Load Project
            loaded_proj = st.file_uploader("📥 Import Project", type=["studio"], label_visibility="collapsed")
            if loaded_proj is not None:
                if st.button("🚀 Load This Project", use_container_width=True):
                    try:
                        data = pickle.loads(loaded_proj.read())
                        if data.get("image_path") != st.session_state.get("image_path"):
                            st.warning("⚠️ Project file might not match this image! Coordinates could be wrong.")
                        
                        # Ensure all loaded masks have modern keys (back-compat)
                        loaded_masks = data["masks"]
                        for m in loaded_masks:
                            if 'visible' not in m: m['visible'] = True
                            if 'brightness' not in m: m['brightness'] = 0.0
                            if 'contrast' not in m: m['contrast'] = 1.0
                            if 'saturation' not in m: m['saturation'] = 1.0
                            if 'hue' not in m: m['hue'] = 0.0
                        
                        st.session_state["masks"] = loaded_masks
                        st.session_state["composited_cache"] = None
                        st.success("Project Loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load project: {e}")


    
        # Material Section
        st.divider()
        st.subheader("🧱 Finish Type")
        mat_type = st.radio("Style", ["Solid Paint"], index=0, horizontal=True, label_visibility="collapsed")
        
        selected_texture = None
        picked_color = "#FFFFFF" # Default
        
        if mat_type == "Solid Paint":
            st.caption("🎨 Choose Color")
            
            preset_colors = {
                "White": "#FFFFFF", "Cream": "#FFFDD0", "Sage Green": "#8FBC8F",
                "Sky Blue": "#87CEEB", "Lavender": "#E6E6FA", "Peach": "#FFDAB9",
                "Terracotta": "#E2725B", "Forest Green": "#228B22", "Royal Blue": "#4169E1",
                "Midnight Navy": "#000080", "Charcoal": "#36454F", "Black": "#000000"
            }
            
            col_cp_1, col_cp_2 = st.columns([1, 1], vertical_alignment="bottom")
            
            with col_cp_1:
                selected_preset = st.selectbox("Collections", list(preset_colors.keys()), key="preset_selector")
            
            # Intelligent Defaulting Logic
            # 1. Start with the preset value
            default_picker_val = preset_colors[selected_preset]
            
            # 2. Check if preset actually changed (user intent override)
            if "last_preset_val" not in st.session_state: 
                st.session_state["last_preset_val"] = selected_preset
            
            preset_changed = (st.session_state["last_preset_val"] != selected_preset)
            st.session_state["last_preset_val"] = selected_preset

            # 3. Smart Reset: If the user explicitly selects a NEW preset, 
            # we assume they want to paint a NEW object.
            if preset_changed:
                 st.session_state["selected_layer_idx"] = None
                 st.session_state["picked_color"] = preset_colors[selected_preset]
                 default_picker_val = preset_colors[selected_preset]
            elif st.session_state.get("picked_color"):
                 default_picker_val = st.session_state["picked_color"]

            with col_cp_2:
                # The picker defaults to strictly following the active layer (if exists) or the preset
                picked_color = st.color_picker("Custom", default_picker_val)
                
            # Handle Sampling override
            if st.session_state.get("picked_sample"):
                picked_color = st.session_state["picked_sample"]

            # Save state for next run
            st.session_state["picked_color"] = picked_color

            # --- SIMPLIFIED UI: No automatic 'Live Tweak' from main sidebar ---
            # This prevents picking a color for 'Next Object' from changing the 'Last Object'.
            st.caption("🖌️ **Paint Mode:** Picking Color for Next Object")


        
        # elif mat_type == "Library":
        #     st.caption("🧱 Choose Material")
        #     # Scan directory dynamically
        #     tex_dir = "assets/textures"
        #     if not os.path.exists(tex_dir):
        #         os.makedirs(tex_dir, exist_ok=True)
            
        #     available_tex = [f for f in os.listdir(tex_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
        #     if not available_tex:
        #         st.warning("No textures found in assets/textures/")
        #     else:
        #         tex_name = st.selectbox("Library", available_tex, format_func=lambda x: x.split('.')[0].replace('_', ' ').title())
        #         tex_path = os.path.join(tex_dir, tex_name)
                
        #         # Show preview
        #         st.image(tex_path, width=100)
                
        #         tex_img = Image.open(tex_path).convert("RGB")
        #         selected_texture = np.array(tex_img)
        
        # else: # Upload Own
        #     st.caption("📤 Upload Custom Material")
        #     custom_tex_file = st.file_uploader("Drop image here", type=["jpg", "png", "jpeg"], key="custom_tex_uploader")
        #     if custom_tex_file:
        #         tex_img = Image.open(custom_tex_file).convert("RGB")
        #         selected_texture = np.array(tex_img)
        #         st.image(tex_img, width=100)
        #     else:
        #         st.info("Upload any image (wood, fabric, stone) to use as a material.")

        # Save to state for main loop
        st.session_state["picked_color"] = picked_color
        st.session_state["selected_texture"] = selected_texture


        # View Settings
        st.divider()
        st.subheader("👁️ View")
        st.toggle("Compare Before/After", key="show_comparison")

        # Segmentation Control (Fix for Leaks)
        # HIDDEN BY DEFAULT (Step Id 1860+): User requested "Automated no manual option".
        # We put this in an expander so it doesn't confuse them.
        with st.expander("⚙️ Advanced Precision (Optional)"):
            sens_mode = st.radio(
                "Segmentation Mode", 
                ["Fine Detail (Default)", "Optimized", "Whole Object"], 
                index=0,
                horizontal=False, 
                help="🎯 **Fine Detail:** The new default. Best for walls and ceilings with precise edges.\n\n🏠 **Optimized:** Good for standard surfaces.\n\n🌐 **Whole Object:** Use for Floors/Rugs."
            )
            
            if "Fine Detail" in sens_mode:
                st.session_state["mask_level"] = 0
            elif "Whole Object" in sens_mode:
                st.session_state["mask_level"] = 2
            else:
                # Optimized falls back to the smart heuristics (None)
                st.session_state["mask_level"] = None 

        # Painting Mode Control
        st.divider()
        st.caption("🖌️ Painting Mode")
        
        # Only show Refine option if there are layers
        options = ["New Object"]
        if st.session_state["masks"]:
            options.append("Refine Last Layer")
            
        mode = st.radio("Mode", options, index=0, label_visibility="collapsed")
        st.session_state["paint_mode"] = mode
        
        if mode == "Refine Last Layer":
            refine_type = st.radio(
                "Action", 
                ["Add Area (Include)", "Subtract Area (Exclude)"], 
                index=1, # Default to subtract as per user request
                horizontal=True
            )
            st.session_state["refine_type"] = 1 if "Add" in refine_type else 0
            st.info(f"Click on the image to {refine_type.split(' ')[0]} that area.")

        # --- SYSTEM RECOVERY ---
        with st.sidebar.expander("🛠️ System Sync"):
            st.caption(f"Backend Version: {CACHE_SALT}")
            if st.button("🔄 Hard Reset AI Engine", use_container_width=True, help="Forcefully clears the AI cache and reloads the latest precision logic."):
                st.cache_resource.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        
        # Lighting & Adjustments (Hidden to avoid interruption)
        # if st.session_state["masks"]:
        #     st.divider()
        #     st.subheader("💡 Layer Tuning")
        #     
        #     # Sync with selection hub
        #     idx = st.session_state.get("selected_layer_idx")
        #     if idx is not None and 0 <= idx < len(st.session_state["masks"]):
        #         target_mask = st.session_state["masks"][idx]
        #     else:
        #         target_mask = st.session_state["masks"][-1]
        #     
        #     # Opacity
        #     op = st.slider("Opacity", 0.0, 1.0, target_mask.get("opacity", 1.0), key="tune_opacity")
        #     if op != target_mask.get("opacity", 1.0):
        #         target_mask["opacity"] = op
        #         st.session_state["bg_cache"] = None # Invalidate cache
        #         st.session_state["composited_cache"] = None
        #         st.rerun()
        # 
        #     # Finish
        #     finish = st.selectbox("Finish", ["Standard", "Matte", "Glossy"], index=["Standard", "Matte", "Glossy"].index(target_mask.get("finish", "Standard")), key="tune_finish")
        #     if finish != target_mask.get("finish", "Standard"):
        #         target_mask["finish"] = finish
        #         st.session_state["bg_cache"] = None
        #         st.session_state["composited_cache"] = None
        #         st.rerun()
        #         
        #     with st.expander("Advanced color"):
        #         # Brightness
        #         br = st.slider("Brightness", -0.5, 0.5, target_mask.get("brightness", 0.0), key="tune_bright")
        #         if br != target_mask.get("brightness", 0.0):
        #             target_mask["brightness"] = br
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        #             
        #         # Contrast
        #         ct = st.slider("Contrast", 0.5, 1.5, target_mask.get("contrast", 1.0), key="tune_contrast")
        #         if ct != target_mask.get("contrast", 1.0):
        #             target_mask["contrast"] = ct
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        # 
        #         # Saturation
        #         sat = st.slider("Saturation", 0.0, 3.0, target_mask.get("saturation", 1.0), key="tune_sat")
        #         if sat != target_mask.get("saturation", 1.0):
        #             target_mask["saturation"] = sat
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        # 
        #         # Hue
        #         hue = st.slider("Hue Shift", -180, 180, int(target_mask.get("hue", 0)), key="tune_hue")
        #         if hue != target_mask.get("hue", 0):
        #             target_mask["hue"] = float(hue)
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        
        # Layer Management Section
        st.divider()
        st.subheader("📝 Layers")
        
        if st.session_state["masks"]:
            col_undo_1, col_undo_2 = st.columns(2)
            with col_undo_1:
                if st.button("⏪ Undo", use_container_width=True, help="Revert last click/layer"):
                    if st.session_state["masks"]:
                        last_layer = st.session_state["masks"][-1]
                        if len(last_layer.get('points', [])) > 1:
                            # Undo last refinement point
                            last_layer['points'].pop()
                            last_layer['labels'].pop()
                            # Re-generate mask for this layer
                            with torch.inference_mode():
                                new_mask = sam.generate_mask(
                                    last_layer['points'],
                                    last_layer['labels'],
                                    level=st.session_state.get("mask_level"),
                                    cleanup=(len(last_layer['points']) == 1)
                                )
                                if new_mask is not None:
                                    last_layer['mask'] = new_mask
                        else:
                            # Remove entire layer
                            st.session_state["masks"].pop()
                            st.session_state["selected_layer_idx"] = None
                        
                        st.session_state["composited_cache"] = None
                        st.session_state["bg_cache"] = None
                        st.session_state["last_export"] = None
                        st.session_state["render_id"] += 1 # Force instant UI update
                        st.rerun()
            
            with col_undo_2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state["masks"] = []
                    st.session_state["selected_layer_idx"] = None
                    st.session_state["composited_cache"] = None
                    st.session_state["bg_cache"] = None
                    st.session_state["last_export"] = None
                    st.session_state["render_id"] += 1 # Force instant UI update
                    st.rerun()
            
            st.write("---")
            # Iterate backwards to show newest first
            for i in range(len(st.session_state["masks"]) - 1, -1, -1):
                mask_data = st.session_state["masks"][i]
                
                with st.expander(f"Layer {i+1}: {mask_data.get('name', 'Untitled')}", expanded=(i == len(st.session_state['masks'])-1)):
                    # Header Row: Visibility, Name, Delete
                    # Header Row: Visibility, Name, Delete
                    h_col1, h_col2, h_col3 = st.columns([1, 4, 1], vertical_alignment="center")
                    
                    with h_col1:
                        visible = st.checkbox("👁️", value=mask_data.get('visible', True), key=f"vis_{i}", label_visibility="collapsed")
                        if visible != mask_data.get('visible', True):
                            mask_data['visible'] = visible
                            st.session_state["composited_cache"] = None
                            st.rerun()
                    
                    with h_col2:
                        st.write(f"**{mask_data.get('name', f'Surface {i+1}')}**")
                    
                    with h_col3:
                        if st.button("🗑️", key=f"del_{i}", help="Delete Layer"):
                            st.session_state["masks"].pop(i)
                            st.session_state["selected_layer_idx"] = None # Reset selection
                            st.session_state["composited_cache"] = None
                            st.session_state["bg_cache"] = None
                            st.session_state["render_id"] += 1 # Force instant UI update
                            st.rerun()

                    # Control Row: Select Hub
                    c_col1, c_col2 = st.columns([1, 1])
                    with c_col1:
                        if st.button("🎯 Tweak", key=f"tweak_{i}", use_container_width=True):
                            st.session_state["selected_layer_idx"] = i
                            if mask_data.get("color"):
                                 st.session_state["picked_color"] = mask_data["color"]
                            st.rerun()
                    
                    with c_col2:
                         layer_color = st.color_picker("Color", mask_data.get('color', '#FFFFFF'), key=f"cp_{i}", label_visibility="collapsed")
                         if layer_color != mask_data.get('color'):
                             mask_data['color'] = layer_color
                             st.session_state["composited_cache"] = None
                             st.session_state["last_export"] = None 
                             st.rerun()
                    # Move Up/Down Controls
                    c_col2, c_col3 = st.columns([1, 1])
                    
                    with c_col2:
                        if i < len(st.session_state["masks"]) - 1:
                            if st.button("🔼", key=f"up_{i}", help="Move Up", use_container_width=True):
                                st.session_state["masks"][i], st.session_state["masks"][i+1] = st.session_state["masks"][i+1], st.session_state["masks"][i]
                                st.session_state["composited_cache"] = None
                                st.session_state["bg_cache"] = None
                                st.rerun()
                    with c_col3:
                        if i > 0:
                            if st.button("🔽", key=f"down_{i}", help="Move Down", use_container_width=True):
                                st.session_state["masks"][i], st.session_state["masks"][i-1] = st.session_state["masks"][i-1], st.session_state["masks"][i]
                                st.session_state["composited_cache"] = None
                                st.session_state["bg_cache"] = None
                                st.rerun()
            
            # Redundant button removed to fix DuplicateElementId error
        else:
            st.caption("No active layers.")

        # Download Section
        st.divider()
        if st.session_state["image"] is not None and st.session_state["masks"]:
            if st.button("💎 Prepare High-Res Download", use_container_width=True, help="Processes your design at original resolution for maximum quality."):
                with st.spinner("Processing 4K Export..."):
                    try:
                        # Scale Masks to Original Resolution
                        original_img = st.session_state["image_original"]
                        oh, ow = original_img.shape[:2]
                        
                        # Create High-Res Mask list
                        high_res_masks = []
                        for m_data in st.session_state["masks"]:
                            hr_m = m_data.copy()
                            # Scale the actual boolean mask
                            mask_uint8 = (m_data['mask'] * 255).astype(np.uint8)
                            hr_mask_uint8 = cv2.resize(mask_uint8, (ow, oh), interpolation=cv2.INTER_LINEAR)
                            hr_m['mask'] = hr_mask_uint8 > 127
                            hr_m['mask_soft'] = None 
                            high_res_masks.append(hr_m)
                        
                        # Use isolated Colorizer pass (Does NOT use or update bg_cache)
                        dl_comp = ColorTransferEngine.composite_multiple_layers(original_img, high_res_masks)
                        dl_pil = Image.fromarray(dl_comp)
                        dl_buf = io.BytesIO()
                        dl_pil.save(dl_buf, format="PNG")
                        st.session_state["last_export"] = dl_buf.getvalue()
                        st.success("✅ Download Ready!")
                    except Exception as e:
                        st.error(f"Export failed: {e}")

            if st.session_state.get("last_export"):
                st.download_button(
                    label="📥 Save Final Image",
                    data=st.session_state["last_export"],
                    file_name="pro_visualizer_design.png",
                    mime="image/png",
                    use_container_width=True
                )
            
        return # No longer need to return picked_color

def overlay_pan_controls(image):
    """Draws semi-transparent pan arrows on the image edges."""
    h, w, c = image.shape
    overlay = image.copy()
    
    # Define color (white with transparency) and thickness
    color = (255, 255, 255)
    thickness = 2
    
    # Margin specific to display size
    margin = 40
    center_x, center_y = w // 2, h // 2
    
    # Draw Arrows
    # Top Arrow
    cv2.arrowedLine(overlay, (center_x, margin), (center_x, 10), color, thickness, tipLength=0.5)
    
    # Bottom Arrow
    cv2.arrowedLine(overlay, (center_x, h - margin), (center_x, h - 10), color, thickness, tipLength=0.5)
    
    # Left Arrow
    cv2.arrowedLine(overlay, (margin, center_y), (10, center_y), color, thickness, tipLength=0.5)
    
    # Right Arrow
    cv2.arrowedLine(overlay, (w - margin, center_y), (w - 10, center_y), color, thickness, tipLength=0.5)
    
    # Blend overlay
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def render_zoom_controls():
    """Render zoom and pan controls below the image."""
    
    # Shared Column Ratio for Alignment
    # Spacer(5) | LeftBtn(1) | Center(1.5) | RightBtn(1) | Spacer(5)
    ratio = [5, 0.8, 1.2, 0.8, 5]

    # --- Zoom Row ---
    z_col1, z_col2, z_col3, z_col4, z_col5 = st.columns(ratio, vertical_alignment="center")
    
    def update_zoom(delta):
        st.session_state["zoom_level"] = max(1.0, min(4.0, st.session_state["zoom_level"] + delta))
    
    with z_col2:
        if st.button("➖", help="Zoom Out", use_container_width=True):
            update_zoom(-0.2)
            st.rerun()
            
    with z_col3:
        st.markdown(
            f"""
            <div style='
                text-align: center; 
                font-weight: bold; 
                background-color: #f0f2f6; 
                color: #31333F;
                padding: 6px 10px; 
                border-radius: 4px;
                border: 1px solid #dcdcdc;
            '>
                {int(st.session_state['zoom_level'] * 100)}%
            </div>
            """, 
            unsafe_allow_html=True
        )
            
    with z_col4:
        if st.button("➕", help="Zoom In", use_container_width=True):
            update_zoom(0.2)
            st.rerun()

    # --- Reset Button ---
    if st.session_state["zoom_level"] > 1.0 or st.session_state["pan_x"] != 0.5 or st.session_state["pan_y"] != 0.5:
        r_col1, r_col2, r_col3 = st.columns([4, 2, 4])
        with r_col2:
            if st.button("🎯 Reset View", use_container_width=True):
                st.session_state["zoom_level"] = 1.0
                st.session_state["pan_x"] = 0.5
                st.session_state["pan_y"] = 0.5
                st.rerun()

    # --- Pan Controls are now integrated into the image canvas ---


def main():
    setup_page()
    setup_styles()
    initialize_session_state()

    # --- AUTO-HEAL: Download weights FIRST if missing ---
    if not os.path.exists(CHECKPOINT_PATH):
        placeholder = st.empty()
        with placeholder.container():
            st.warning("⚠️ Light AI model not found. Downloading automatically... (approx 40MB)")
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 512 * 1024 
                with open(CHECKPOINT_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = min(1.0, downloaded / total_size)
                                progress_bar.progress(percent)
                            status_text.text(f"📥 Downloaded {downloaded//(1024*1024)}MB...")
                if os.path.getsize(CHECKPOINT_PATH) < 35 * 1024 * 1024:
                     st.error("Download incomplete or corrupt.")
                     os.remove(CHECKPOINT_PATH)
                     st.stop()
                st.success("✅ Model weights verified.")
                time.sleep(1)
                st.cache_resource.clear() # CRITICAL: Clear cache so it tries to load for real
                st.rerun() 
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                if os.path.exists(CHECKPOINT_PATH): os.remove(CHECKPOINT_PATH)
                st.stop()

    # --- STABILITY MIGRATION ---
    if st.session_state.get("image") is not None:
        opts_h, opts_w = st.session_state["image"].shape[:2]
        # HD RESOLUTION UNLOCK (Step Id 1779+)
        # We upped this from 640 -> 1024 to support 360-degree/Panoramic images.
        # Wide images need more pixels to resolve wall edges (straight, right, left, back).
        limit = 1024 
        if max(opts_h, opts_w) > limit:
            scale = limit / max(opts_h, opts_w)
            new_w, new_h = int(opts_w * scale), int(opts_h * scale)
            st.session_state["image"] = cv2.resize(st.session_state["image"], (new_w, new_h), interpolation=cv2.INTER_AREA)
            st.session_state["masks"] = [] 
            st.session_state["composited_cache"] = None
            st.session_state["bg_cache"] = None
            sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)
            if sam: sam.is_image_set = False 
            st.rerun()
    
    # Render Sidebar FIRST (Instant UI)
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    
    # We pass None for sam initially to sidebar, or refactor sidebar to not need it for basic UI
    # Let's adjust sidebar to take device string specifically
    
    # Load Model (Session Aware - Optimized for "Instant" feel)
    placeholder = st.empty()
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    
    # PERFORMANCE: Only show spinner on first load to avoid "Loading Interruption" on every click
    if not st.session_state.get("engine_ready", False):
        with placeholder.container():
            with st.spinner(f"🚀 Initializing AI Engine on {device_str}..."):
                sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)
                if sam:
                    st.session_state["engine_ready"] = True
    else:
        sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)
    
    if not sam:
        st.error("AI Engine could not be initialized. Please check if the model weights are downloaded correctly.")
        st.stop()
        
    placeholder.empty()
    render_sidebar(sam, device_str)

    # CENTRALIZED LOADING: Compute embeddings if a new image is present OR if the engine was reset/re-allocated
    if st.session_state["image"] is not None:
        img_id = id(st.session_state["image"])
        # CRITICAL: We also check sam.is_image_set in case the global singleton was reset
        if st.session_state.get("engine_img_id") != img_id or not getattr(sam, 'is_image_set', False):
            lock = get_global_lock()
            with placeholder.container():
                with st.spinner("🚀 Analyzing image structure..."):
                    with lock: 
                        try:
                            # RAM PROTECTION: Avoid unnecessary copies
                            # If we must process, we do it in-place or via a very tight reference
                            img_to_process = st.session_state["image"]
                            
                            with torch.inference_mode():
                                # This is the heavy part. Single-threaded to keep RAM flat.
                                sam.set_image(img_to_process)
                            
                            st.session_state["engine_img_id"] = img_id
                            
                            # Final aggressive cleanup 
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error analyzing image: {e}")
                            if "Out of memory" in str(e):
                                torch.cuda.empty_cache()
                            st.stop()
    
    # Main Workflow
    if st.session_state["image"] is not None:
        
        # 1. Image Check is already handled in stability migration above

        h, w, c = st.session_state["image"].shape
        
        # 1. Calc Geometry (Where are we looking?)
        start_x, start_y, view_w, view_h = get_crop_params(
            w, h, 
            st.session_state["zoom_level"], 
            st.session_state["pan_x"], 
            st.session_state["pan_y"]
        )
        
        # 2. Calc Display Geometry
        display_width = 1000 # Reduced safe width
        # Determine crop dimensions
        crop_w = view_w
        crop_h = view_h
        
        # Calculate scale
        scale_factor = display_width / crop_w
        
        # 3. Check for Click in Session State
        # CRITICAL: This key MUST match the rendered component key exactly to capture clicks.
        canvas_key = f"canvas_{st.session_state['render_id']}_{st.session_state['zoom_level']}_{st.session_state['pan_x']:.2f}_{st.session_state['pan_y']:.2f}"
        input_value = st.session_state.get(canvas_key)
        
        if input_value is not None:
            click_x_display, click_y_display = input_value["x"], input_value["y"]
            
            # --- Pan Logic ---
            if st.session_state["zoom_level"] > 1.0:
                # Need to deduce display height for pan zones
                new_h = int(crop_h * scale_factor)
                d_h, d_w = new_h, display_width
                
                margin = 50
                step_size = 0.1 / st.session_state["zoom_level"]
                pan_triggered = False
                
                if click_y_display < margin:
                    st.session_state["pan_y"] = max(0.0, st.session_state["pan_y"] - step_size)
                    pan_triggered = True
                elif click_y_display > d_h - margin:
                    st.session_state["pan_y"] = min(1.0, st.session_state["pan_y"] + step_size)
                    pan_triggered = True
                elif click_x_display < margin:
                    st.session_state["pan_x"] = max(0.0, st.session_state["pan_x"] - step_size)
                    pan_triggered = True
                elif click_x_display > d_w - margin:
                    st.session_state["pan_x"] = min(1.0, st.session_state["pan_x"] + step_size)
                    pan_triggered = True
                    
                if pan_triggered:
                    st.rerun()

            # --- Painting Logic ---
            # Map Display -> Global
            local_x = int(click_x_display / scale_factor)
            local_y = int(click_y_display / scale_factor)
            global_click_x = start_x + local_x
            global_click_y = start_y + local_y
            
            # Clamp
            global_click_x = max(0, min(global_click_x, w - 1))
            global_click_y = max(0, min(global_click_y, h - 1))
            
            click_tuple = (global_click_x, global_click_y)
            last_click = st.session_state.get("last_click_global")
            
            if click_tuple != last_click:
                st.session_state["last_click_global"] = click_tuple
                
                # NEW: SAMPLING MODE Logic
                if st.session_state.get("sampling_mode"):
                    pixel = st.session_state["image"][global_click_y, global_click_x]
                    hex_val = '#%02x%02x%02x' % (pixel[0], pixel[1], pixel[2])
                    st.session_state["picked_sample"] = hex_val
                    st.session_state["sampling_mode"] = False # Reset
                    st.toast(f"Sampled color: {hex_val}")
                    st.rerun()

                # PROCESS CLICK (No Spinner for speed)
                mode = st.session_state.get("paint_mode", "New Object")
                masks_state = st.session_state["masks"]
                
                if mode == "Refine Last Layer" and masks_state:
                     # REFINE Logic
                     last_layer = masks_state[-1]
                     if 'points' not in last_layer:
                         last_layer['points'] = [last_layer['point']]
                         last_layer['labels'] = [1]
                     
                     new_label = st.session_state.get("refine_type", 0)
                     last_layer['points'].append(click_tuple)
                     last_layer['labels'].append(new_label)
                     
                     # RESILIENCE: Self-healing check
                     if not sam.is_image_set:
                         with st.spinner("🔄 AI Re-syncing image..."):
                             sam.set_image(st.session_state["image"])
                             st.session_state["engine_img_id"] = id(st.session_state["image"])
                             
                     refined_mask = sam.generate_mask(
                         last_layer['points'],
                         last_layer['labels'],
                         level=st.session_state.get("mask_level", None),
                         cleanup=False
                     )
                     
                     if refined_mask is not None:
                         last_layer['mask'] = refined_mask
                         st.toast("🎯 Shape refined")
                         # Invalidate cache to force redraw in next step
                         st.session_state["composited_cache"] = None
                         st.session_state["render_id"] += 1
                         st.rerun()
                else:
                     # NEW OBJECT Logic
                     # RESILIENCE: Self-healing check if engine lost its image embeddings
                     if st.session_state.get("engine_img_id") != id(st.session_state["image"]) or not sam.is_image_set:
                         with st.spinner("🔄 AI Re-syncing image..."):
                             sam.set_image(st.session_state["image"])
                             st.session_state["engine_img_id"] = id(st.session_state["image"])

                     mask = sam.generate_mask(
                        [click_tuple],
                        [1], 
                        level=st.session_state.get("mask_level", None)
                     )
                     if mask is not None:
                        new_layer = {
                            'mask': mask,
                            'mask_soft': None, # Performance Cache
                            'color': st.session_state.get("picked_color", "#FFFFFF"),
                            'texture': st.session_state.get("selected_texture"),
                            'brightness': 0.0,
                            'contrast': 1.0,
                            'saturation': 1.0,
                            'hue': 0.0,
                            'opacity': 1.0,
                            'finish': 'Standard',
                            'tex_rot': 0,
                            'tex_scale': 1.0,
                            'point': click_tuple,
                            'points': [click_tuple],
                            'labels': [1],
                            'visible': True,
                            'name': f"Surface {len(st.session_state['masks'])+1}"
                        }
                        st.session_state["masks"].append(new_layer)
                        st.toast(f"✅ Applied color to {new_layer['name']}")
                        
                        # Set as selected immediately for tweaking
                        st.session_state["selected_layer_idx"] = len(st.session_state["masks"]) - 1
                        
                        # Invalidate cache to force redraw in next step
                        st.session_state["composited_cache"] = None
                        st.session_state["bg_cache"] = None # Reset optimization cache
                        st.session_state["render_id"] += 1
                        st.rerun() # Force instant UI update after layer addition


        # 1. Prepare Base Image (Original + Applied Colors) (Draw Phase)
        # OPTIMIZATION: Use cached result if valid
        if st.session_state.get("composited_cache") is None:
             st.session_state["composited_cache"] = composite_image(st.session_state["image"], st.session_state["masks"])
        
        full_composited = st.session_state["composited_cache"]
        
        if st.session_state.get("show_comparison", False):
            # Render Comparison Slider (Read Only)
            st.info("👀 Comparison Mode Active. Toggle off in sidebar to edit.")
            
            # Ensure images are same size/type
            img1 = st.session_state["image"] # Original
            img2 = full_composited          # Edited
            
            image_comparison(
                img1=Image.fromarray(img1),
                img2=Image.fromarray(img2),
                label1="Original",
                label2="Your Design",
                width=1000, # Match display width for consistency
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )
            return # Stop execution here for view mode

        # 2. Apply Zoom/Crop
        h, w, c = full_composited.shape
        start_x, start_y, view_w, view_h = get_crop_params(
            w, h, 
            st.session_state["zoom_level"], 
            st.session_state["pan_x"], 
            st.session_state["pan_y"]
        )
        
        cropped_view = full_composited[start_y:start_y+view_h, start_x:start_x+view_w]
        
        # 3. Interactive Display
        # Resize for display consistency
        display_width = 1000
        crop_h, crop_w, _ = cropped_view.shape
        
        # Calculate scale to fit display layout, handling both upscale (zoom) and downscale
        
        # PERFORMANCE: Cache Display Image to prevent unnecessary re-processing/flicker
        current_disp_params = (
            st.session_state["render_id"], 
            st.session_state["zoom_level"], 
            st.session_state["pan_x"], 
            st.session_state["pan_y"]
        )
        
        if st.session_state.get("last_disp_params") != current_disp_params:
            # Recompute Display Image
            scale_factor = display_width / crop_w
            new_h = int(crop_h * scale_factor)
            
            display_image = cv2.resize(cropped_view, (display_width, new_h), interpolation=cv2.INTER_LINEAR)
            
            if st.session_state["zoom_level"] > 1.0:
                display_image = overlay_pan_controls(display_image)
            
            st.session_state["cached_display_image"] = display_image
            st.session_state["last_disp_params"] = current_disp_params
            st.session_state["last_scale_factor"] = scale_factor
        
        final_display_image = st.session_state.get("cached_display_image")
        scale_factor = st.session_state.get("last_scale_factor", 1.0)

        if final_display_image is None or final_display_image.size == 0:
            st.warning("⚠️ Rendering issue: Image preview is empty. Try refreshing the page.")
            return

        # Container for the image
        with st.container():
            # Ensure explicit strict types for the component
            final_display_image = np.ascontiguousarray(final_display_image, dtype=np.uint8)
            
            # Dynamic key ensures component resets on view change, preventing infinite pan loops
            # PERFORMANCE: We must use 'render_id' in the key to force the component to reset.
            # Otherwise, clicking the same pixel twice (e.g. to re-paint) won't register.
            canvas_key = f"canvas_{st.session_state['render_id']}_{st.session_state['zoom_level']}_{st.session_state['pan_x']:.2f}_{st.session_state['pan_y']:.2f}"
            
            # OPTIMIZATION: Convert to JPEG Base64 to speed up the "flash" reload
            # PNG (default) is too slow and causes visible scanlines. JPEG is instant.
            try:
                value = streamlit_image_coordinates(
                    Image.fromarray(final_display_image),
                    key=canvas_key,
                    width=display_width 
                )
            except Exception as e:
                st.error(f"Display Error: {e}")
                value = None

        # 4. Zoom Controls
        render_zoom_controls()

        # 5. Handle Click Event
        pass

    else:
        # Landing Page
        st.markdown("""
        <div class="landing-header">
            <h1>Welcome to Color Visualizer</h1>
        </div>
        <div class="landing-sub">
            <p>Upload a photo of your room to start experimenting with colors.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.info("👈 Use the sidebar to upload an image.")

if __name__ == "__main__":
    main()
