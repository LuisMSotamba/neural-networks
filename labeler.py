from PIL import Image
import streamlit as st
import pandas as pd
import os
import json

import torchvision.transforms as T

# --- CONFIGURATION ---
CSV_PATH = "/media/luchocode/Extra vol/thesis/data/new_labels/val/labels.csv"
PROGRESS_FILE = "/media/luchocode/Extra vol/thesis/data/new_labels/val/labeling_progress.json"
BASE_DIR = "/media/luchocode/Extra vol/thesis/"
SOURCE_BASE_DIR_IMG = "/media/luchocode/Extra vol/thesis/data/selected_exoimages/val"
TARGET_BASE_DIR_IMG = "/media/luchocode/Extra vol/thesis/data/new_labels/val"

# --- LABEL MAPPING ---
LABEL_MAP = {
    "IS-T-D": "Escaleras Inclinadas a Puerta",
    "IS-T-W": "Escaleras Inclinadas a Pared",
    "IS-T-LG": "Escaleras Inclinadas a Suelo Nivelado",
    "IS-S":    "Escaleras Inclinadas (Solo)",
    "DS-T-LG": "Escaleras Descendentes a Suelo Nivelado",
    "DS-S": "Escaleras Descendentes (Solo)",
    "LG-T-D": "Suelo Nivelado a Puerta",
    "LG-T-W": "Suelo Nivelado a Pared",
    "LG-T-O":  "Suelo Nivelado a Obstáculo",
    "LG-T-IS": "Suelo Nivelado a Escaleras Inclinadas",
    "LG-T-DS": "Suelo Nivelado a Escaleras Descendentes",
    "LG-T-SE": "Suelo Nivelado a Asientos",
    "LG-S":    "Suelo Nivelado (Solo)",
    "D-S": "Puerta (Solo) ",
    "W-S": "Pared (Solo) "
}

CATEGORIES = {
    "Incline Stairs (IS)": ["IS-T-D", "IS-T-W", "IS-T-LG", "IS-S"],
    "Decline Stairs (DS)": ["DS-T-LG", "DS-S"],
    "Door / Wall (DW)":    ["D-S", "W-S"],
    "Level Ground (LG)":   ["LG-T-D", "LG-T-W", "LG-T-O", "LG-T-IS", "LG-T-DS", "LG-T-SE", "LG-S"]
}

# --- 1. HELPER FUNCTIONS FOR PROGRESS ---
def save_progress(video_name, index_val):
    """Saves the current video and frame index to a JSON file."""
    data = {"video": video_name, "index": index_val}
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f)

def load_progress():
    """Reads the last saved position."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

# --- 2. LOAD DATA ---
@st.cache_data()
def load_data():
    import sys
    import numpy.core.numeric as _numeric
    # Create a dummy module entry for the missing path
    sys.modules['numpy._core.numeric'] = _numeric

    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    
    # Read val dataframe and create a new column to store the new class assigned
    df = pd.read_pickle(os.path.join(BASE_DIR, 'pickle/df_val.pkl'))
    df['new_class'] = df['class']
    
    df.to_csv(CSV_PATH, index=False)
    return df

# Initialize DF
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df_master = st.session_state.df 

# --- 3. INITIALIZATION & RESUME LOGIC ---
# This block runs only once when the script starts or refreshes completely
if 'initialized' not in st.session_state:
    last_progress = load_progress()
    
    if last_progress and last_progress['video'] in df_master['video'].unique():
        st.session_state.selected_video = last_progress['video']
        st.session_state.index = last_progress['index']
        st.toast(f"Resumed from frame {st.session_state.index}", icon="🔄")
    else:
        st.session_state.selected_video = df_master['video'].unique()[0] if not df_master.empty else None
        st.session_state.index = 0
    
    st.session_state.initialized = True

# --- 4. FILTERING ---
all_videos = df_master['video'].unique().tolist()

# Sidebar: Video Selector
# We use index=... to force the selectbox to show the resumed video
try:
    vid_idx = all_videos.index(st.session_state.selected_video)
except (ValueError, TypeError):
    vid_idx = 0

selected_video = st.sidebar.selectbox(
    "Select Video to Label", 
    all_videos, 
    index=vid_idx
)

# Detect manual video change
if selected_video != st.session_state.selected_video:
    st.session_state.selected_video = selected_video
    st.session_state.index = 0 # Reset to 0 if user manually changes video
    save_progress(selected_video, 0) # Save the new start point

# Create Filtered View
df_filtered = df_master[df_master['video'] == selected_video].reset_index()

# --- 5. PATH HELPER ---
def get_image_path(row, root_dir):
    filename = f"['{row['video']}'] frame {row['frame']}.jpg"
    return os.path.join(root_dir, row['class'], filename)

def transform_image(img_path):
    raw_image = Image.open(img_path).convert('RGB')
    vis_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224)
    ])
    model_view_image = vis_transform(raw_image)
    return model_view_image

# --- 6. UPDATE FUNCTION ---
def update_label(new_short_code):
    row = df_filtered.iloc[st.session_state.index]
    master_idx = row['index'] 

    # Update Dataframe
    st.session_state.df.at[master_idx, 'new_class'] = new_short_code
    st.session_state.df.to_csv(CSV_PATH, index=False)
    
    # Move Next
    if st.session_state.index < len(df_filtered) - 1:
        st.session_state.index += 1
        # SAVE PROGRESS HERE
        save_progress(st.session_state.selected_video, st.session_state.index)
    else:
        st.balloons()
        st.success("Finished this video!")

# --- UI NAVIGATION ---
def prev_image():
    if st.session_state.index > 0:
        st.session_state.index -= 1
        save_progress(st.session_state.selected_video, st.session_state.index)

def next_image():
    if st.session_state.index < len(df_filtered) - 1:
        st.session_state.index += 1
        save_progress(st.session_state.selected_video, st.session_state.index)

# --- 7. MANUAL JUMP (New Feature) ---
# Allows you to type "500" to jump directly to frame index 500
with st.sidebar:
    st.write("---")
    st.write("**Navigation**")
    jump_to = st.number_input("Jump to Index", min_value=0, max_value=len(df_filtered)-1, value=st.session_state.index)
    if st.button("Go"):
        st.session_state.index = jump_to
        save_progress(st.session_state.selected_video, jump_to)
        st.rerun()

# --- 8. DISPLAY ---
st.set_page_config(layout="wide")
st.title(f"Labeling: {selected_video}")

if not df_filtered.empty:
    current_row = df_filtered.iloc[st.session_state.index]
    print(current_row)
    current_code = current_row['new_class']
    print(current_code)
    
    # Progress
    progress = (st.session_state.index + 1) / len(df_filtered)
    st.progress(progress)
    
    # Create two main columns: Left for Image, Right for Controls
    col_left, col_right = st.columns([1, 2]) 

    with col_left:
        # IMAGE SIDE        
        full_path = get_image_path(current_row, SOURCE_BASE_DIR_IMG)
        if os.path.exists(full_path):
            st.image(transform_image(full_path), use_container_width=True)
        else:
            st.warning(f"Image not found:\n{full_path}")
        
        # Navigation Buttons under the image for quick access
        c1, c2, c3 = st.columns([1, 2, 1])
        if c1.button("⬅️ Prev", use_container_width=True):
            prev_image()
            st.rerun()
        if c3.button("Next ➡️", use_container_width=True):
            next_image()
            st.rerun()

    with col_right:
        # CONTROL SIDE
        st.write(f"**Index:** {st.session_state.index} | **Frame:** {current_row['frame']}")
        st.info(f"**Current:** {LABEL_MAP.get(current_code, 'Unknown')}")
        
        # Compact Tabs instead of long lists
        # tab1, tab2, tab3, tab4 = st.tabs(["Incline", "Decline", "Door/Wall", "Level Ground"])

        c1, c2 = st.columns([1,1])
        
        # Helper to render buttons inside tabs
        def render_buttons(category_list):
            for code in category_list:
                btn_text = LABEL_MAP.get(code, code)
                is_selected = (code == current_code)
                if st.button(btn_text, key=code, type="primary" if is_selected else "secondary", use_container_width=True):
                    update_label(code)
                    st.rerun()

        with c1:
            render_buttons(CATEGORIES["Incline Stairs (IS)"])
            render_buttons(CATEGORIES["Decline Stairs (DS)"])
            render_buttons(CATEGORIES["Door / Wall (DW)"])
        with c2:
            render_buttons(CATEGORIES["Level Ground (LG)"])
else:
    st.warning("No frames found for this video.")