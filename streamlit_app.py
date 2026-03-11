import streamlit as st
from utils import inject_custom_css, get_video_id, run_analysis_and_summarize, save_to_pdf
from pathlib import Path
from io import BytesIO 
import json 
from typing import List, Dict, Any, Optional
import re 
import tempfile
import os
import google.generativeai as genai

# Call the CSS injection function (for base styling)
inject_custom_css()

# --- Configuration and Mapping ---

# Mapping friendly label -> expected JSON key
LABEL_TO_KEY = {
    'Topic Breakdown': 'topic_breakdown',
    'Key Vocabulary': 'key_vocabulary',
    'Formulas & Principles': 'formulas_and_principles',
    'Teacher Insights': 'teacher_insights',
    'Exam Focus Points': 'exam_focus_points',
    'Common Mistakes': 'common_mistakes_explained',
    'Key Points': 'key_points',
    'Short Tricks': 'short_tricks',
    'Must Remembers': 'must_remembers'
}

# Normal Settings Mappings
PAGE_WORD_COUNT_MAP = {
    "3–4": 800,
    "6–8": 1500,
    "10–12": 2200,
    "12+": 3000
}

TIME_MODE_DIVISION_MAP = {
    "Quick": 1,
    "Medium": 3,
    "Detailed": 6
}

# --- Application Setup ---
st.title("📹 AI-Powered Hyperlinked Video Notes Generator")

# --- Model Context Constants ---
WARNING_THRESHOLD_CHARS = 300000 

# Initialize session state variables
if 'analysis_data' not in st.session_state:
    st.session_state['analysis_data'] = None
if 'api_key_valid' not in st.session_state:
    st.session_state['api_key_valid'] = False
if 'output_filename_base' not in st.session_state:
    st.session_state['output_filename_base'] = "Video_Notes"
if 'chunked_results' not in st.session_state:
    st.session_state['chunked_results'] = []
if 'processing' not in st.session_state:
    st.session_state['processing'] = False

# Initialize new session states for persistence and defaults
if 'num_pages_select' not in st.session_state:
    st.session_state['num_pages_select'] = "12+"
if 'time_mode_select' not in st.session_state:
    st.session_state['time_mode_select'] = "Medium" 
if 'custom_word_count' not in st.session_state:
    st.session_state['custom_word_count'] = 1500
if 'custom_divisions' not in st.session_state:
    st.session_state['custom_divisions'] = 3
if 'settings_mode' not in st.session_state:
    st.session_state['settings_mode'] = "Normal Settings"

# --- 🔧 CORE HELPER FUNCTIONS ---

def preprocess_transcript(text):
    # (Implementation remains unchanged)
    import re
    pattern = r'\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?' 
    matches = list(re.finditer(pattern, text))
    segments = []
    
    if not matches:
         if text:
             return [{"time": "00:00", "text": text.strip()}]
         return []

    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        ts = matches[i].group(1)
        segments.append({"time": ts, "text": text[start:end].strip()})
        
    return segments

def split_transcript_by_parts(transcript: str, num_parts: int) -> List[str]:
    # (Implementation remains unchanged)
    text = transcript or ""
    length = len(text)
    
    num_parts = max(1, min(num_parts, length)) 
    part_size = length // num_parts
    
    parts = []
    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size if i < num_parts - 1 else length
        parts.append(text[start:end])
    return parts

def merge_all_json_outputs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # (Implementation remains unchanged)
    LIST_KEYS = list(LABEL_TO_KEY.values())
    
    combined: Dict[str, Any] = {"main_subject": ""}
    
    for key in LIST_KEYS:
        combined[key] = []
        
    for res in results:
        res_normalized = {LABEL_TO_KEY.get(k, k): v for k, v in res.items()}
        
        for k, v in res_normalized.items():
            if k == "main_subject":
                if not combined.get("main_subject") and v:
                    combined["main_subject"] = str(v).strip()
                continue

            # CRITICAL FIX: Explicitly check if the value is a list before extending.
            if k in LIST_KEYS and isinstance(v, list):
                combined[k].extend(v)
            # End of critical fix
    
    for k in LIST_KEYS:
        if combined[k]:
            unique_items = []
            seen_hashes = set()
            
            for item in combined[k]:
                item_hash = json.dumps(item, sort_keys=True)
                
                if item_hash not in seen_hashes:
                    unique_items.append(item)
                    seen_hashes.add(item_hash)
            
            combined[k] = unique_items
            
    return combined

# --------------------------------------------------------------------------
# --- Sidebar Setup and Conditional Logic ---
# --------------------------------------------------------------------------

# Set up the two-mode selector
with st.sidebar:
    st.header("🔑 Configuration")
    
    # 1. API Key Input (Always visible)
    api_key = st.text_input("Gemini API Key:", type="password")
    if api_key:
        st.session_state['api_key_valid'] = True
        st.success("API Key Entered.")
    else:
        st.session_state['api_key_valid'] = False
        st.warning("Please enter your Gemini API Key.")

    st.markdown("---")
    
    # Mode Selector (Always visible)
    settings_mode = st.radio(
        "⚙️ Settings Mode", 
        ["Normal Settings", "Advanced Custom Settings"], 
        key='settings_mode'
    )
    
    st.markdown("---")

    # --- Conditional Settings Panel ---
    
    final_max_words: int
    final_num_divisions: int
    
    if settings_mode == "Normal Settings":
        st.header("✨ Normal Settings")
        
        # Normal Setting 1: Number of Pages (Word Count Control)
        st.subheader("Output Size (Normal)")
        num_pages_choice = st.selectbox(
            "Target PDF Length (Pages):",
            options=list(PAGE_WORD_COUNT_MAP.keys()),
            index=list(PAGE_WORD_COUNT_MAP.keys()).index(st.session_state.get('num_pages_select', "12+")),
            key='num_pages_select', 
            help="Controls the **total length** of the extracted summary."
        )
        final_max_words = PAGE_WORD_COUNT_MAP.get(num_pages_choice, 3000)
        st.markdown(f"**Target Word Count:** `{final_max_words}`")

        st.markdown("---")
        
        # Normal Setting 2: Choose Time Mode (Transcript Division Control)
        st.subheader("Processing Speed (Normal)")
        time_mode_choice = st.selectbox(
            "Chunking Mode:",
            options=list(TIME_MODE_DIVISION_MAP.keys()),
            index=list(TIME_MODE_DIVISION_MAP.keys()).index(st.session_state.get('time_mode_select', "Medium")),
            key='time_mode_select', 
            help="Fewer divisions = quicker processing; More divisions = better contextual density."
        )
        final_num_divisions = TIME_MODE_DIVISION_MAP[time_mode_choice]
        st.markdown(f"**Transcript Divisions:** `{final_num_divisions}x`")

    else: # Advanced Custom Settings
        st.header("🔬 Advanced Custom Settings")
        
        # Advanced Setting 1: Custom Word Count (Overrides Pages)
        st.subheader("Output Size (Custom)")
        custom_word_count = st.number_input(
            'Custom Target Word Count:', 
            min_value=500, 
            max_value=10000, 
            value=st.session_state.get('custom_word_count', 1500), 
            step=100, 
            key='custom_word_count', 
            help="Set the precise word limit for the total output summary."
        )
        final_max_words = custom_word_count
        st.markdown(f"**Target Word Count:** `{final_max_words}`")
        
        st.markdown("---")

        # Advanced Setting 2: Custom Transcript Divisions (Overrides Time Mode)
        st.subheader("Processing Speed (Custom)")
        custom_divisions = st.slider(
            'Custom Transcript Divisions:',
            min_value=1,
            max_value=10,
            value=st.session_state.get('custom_divisions', 3),
            step=1,
            key='custom_divisions',
            help="The number of parts the transcript will be split into for analysis."
        )
        final_num_divisions = custom_divisions
        st.markdown(f"**Transcript Divisions:** `{final_num_divisions}x`")

    st.markdown("---")
    
    # Model Selection (Always visible)
    model_choice = st.selectbox(
        "Model Selection:",
        options=["gemini-2.5-pro", "gemini-2.5-flash"],
        index=0, 
        key='model_choice_select', 
        help="Pro = better reasoning, 1M token context. Flash = cheaper, faster, smaller context."
    )

    st.markdown("---")
    
    # 4. YouTube URL Input (Always visible)
    yt_url = st.text_input("YouTube URL (Optional):", help="Provide a URL to enable hyperlinked timestamps in the PDF.")
    video_id = get_video_id(yt_url)
    if video_id:
        st.success(f"Video ID found: {video_id}")
    elif yt_url:
        st.warning("Invalid YouTube URL format. Timestamps will not be hyperlinked.")
    
    st.markdown("---")
    st.header("⚙️ Analysis Details")
    
    # B. Checkboxes for Section Selection (Always visible)
    section_options = {
        'Topic Breakdown': True, 'Key Vocabulary': True,
        'Formulas & Principles': True, 'Teacher Insights': False, 
        'Exam Focus Points': True, 'Common Mistakes': False,
        'Key Points': True, 'Short Tricks': False, 'Must Remembers': True      
    }
    
    sections_list = []
    st.subheader("Select Output Sections")
    for label, default_val in section_options.items():
        if st.checkbox(label, value=default_val):
            sections_list.append(label)

    st.markdown("---")
    
    # G. Custom Filename Input (Always visible)
    if video_id:
        st.session_state['output_filename_base'] = f"Notes_{video_id}"
    
    output_filename_base = st.session_state['output_filename_base']
    output_filename = st.text_input(
        "Base Name for PDF file:",
        value=output_filename_base + ".pdf",
        key="output_filename_input"
    )
    
# --------------------------------------------------------------------------
# --- Main Content: Transcript Input, Button, and Output ---
# --------------------------------------------------------------------------

# 💡 Dual Format Selection Radio Button
format_choice = st.radio(
    "Choose Reading Format:",
    options=["Default (Compact)", "Easier Read (Spacious & Highlighted)"],
    index=0,
    horizontal=True,
    key='pdf_format_choice',
    help="Easier Read format adds vertical spacing between lines and enables content highlighting."
)
st.markdown("---")

st.subheader("Input Sources")
uploaded_pdf = st.file_uploader("Upload an Image-Based PDF (Optional):", type=["pdf"])

transcript_text = st.text_area(
    'Paste the video transcript here (Optional, but must include timestamps if provided):',
    height=300,
    placeholder="[00:00] Welcome to the lesson. [00:45] We start with Topic A..."
)

# Optional Transcript Warning
if len(transcript_text) > WARNING_THRESHOLD_CHARS and model_choice == "gemini-2.5-flash":
    st.warning(f"⚠️ **Long Transcript Detected!** The text is over {WARNING_THRESHOLD_CHARS} characters. We recommend selecting **Gemini 2.5 Pro** or increasing the divisions to avoid context overflow with Flash.")

user_prompt_input = st.text_area(
    'Refine AI Focus (Optional Prompt):',
    value="Ensure the output is highly condensed and only focus on practical applications and examples.",
    height=100
)

# E. The Analysis Trigger Button
can_run = (bool(transcript_text.strip()) or uploaded_pdf) and st.session_state['api_key_valid']
run_analysis = st.button(
    f"🚀 Generate Notes using {model_choice}", 
    type="primary", 
    disabled=not can_run or st.session_state['processing']
) 

is_easy_read = format_choice.startswith("Easier Read")

if run_analysis and not st.session_state['processing']:
    
    # Print the active configuration for debugging
    print(f"\n--- DEBUG RUN START ---")
    print(f"| Settings Mode: {settings_mode}")
    print(f"| Final Max Words: {final_max_words}")
    print(f"| Final Divisions: {final_num_divisions}")
    print(f"| Model: {model_choice}")
    print(f"| Video ID: {video_id}")
    print(f"--- DEBUG RUN END ---")
    
    st.session_state['processing'] = True
    st.session_state['chunked_results'] = []
    
    # *** START DEBUG OUTPUT AREA ***
    debug_placeholder = st.empty()
    debug_messages = []
    # *** END DEBUG OUTPUT AREA ***
    
    try:
        pdf_file_obj = None
        pdf_temp_path = None
        if uploaded_pdf:
            st.info("Uploading PDF to Gemini...")
            genai.configure(api_key=api_key)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getvalue())
                pdf_temp_path = tmp.name
                
            pdf_file_obj = genai.upload_file(pdf_temp_path, mime_type="application/pdf")
            st.success("PDF uploaded successfully.")

        # Determine the number of parts to use (1 for Pro, or the configured division count)
        num_parts_to_use = 1 
        if model_choice == "gemini-2.5-flash":
            num_parts_to_use = final_num_divisions
        
        transcript_parts = split_transcript_by_parts(transcript_text, num_parts_to_use)
        
        st.info(f"Analyzing in **{len(transcript_parts)}** sequential part(s) using **{model_choice}** (Divisions: {num_parts_to_use}).")

        # Chunked Execution
        status_bar = st.progress(0, text="Starting analysis...")
        
        sections_list_keys = [LABEL_TO_KEY.get(lbl, lbl) for lbl in sections_list]

        for i, part in enumerate(transcript_parts, start=1):
            status_bar.progress(
                i / len(transcript_parts), 
                text=f'Analyzing Part {i} of {len(transcript_parts)}... (Model: {model_choice})'
            )
            
            preprocessed_part = preprocess_transcript(part)

            data_json, error_msg, full_prompt = run_analysis_and_summarize(
                api_key, preprocessed_part, final_max_words, sections_list_keys, user_prompt_input, model_choice, is_easy_read, pdf_file_obj=pdf_file_obj
            )
            
            # --- DEBUG: Raw Response Inspection ---
            if error_msg:
                # If JSON parsing failed, the error_msg is the most immediate indicator of the raw output problem
                debug_messages.append(f"🛑 **DEBUG Part {i} API Error:** {error_msg}")
                debug_messages.append(f"Check the console for the full prompt/raw response text.")
            
            if data_json:
                st.session_state['chunked_results'].append(data_json)
                debug_messages.append(f"✅ **DEBUG Part {i} Success:** Extracted keys: {list(data_json.keys())}")
            else:
                st.error(f"Analysis failed for Part {i}. Error: {error_msg}")
                st.session_state['chunked_results'] = []
                break
                
            debug_placeholder.info("\n".join(debug_messages))
            # --- END DEBUG ---


        status_bar.empty()
        debug_placeholder.empty() # Clear transient debug messages on completion

        if st.session_state['chunked_results']:
            st.success(f"Analysis complete for all {len(st.session_state['chunked_results'])} parts.")
            st.session_state['pdf_ready'] = True
        else:
            st.session_state['pdf_ready'] = False
            
    finally:
        st.session_state['processing'] = False
        if 'pdf_file_obj' in locals() and pdf_file_obj:
            try:
                genai.delete_file(pdf_file_obj.name)
            except Exception as e:
                print(f"Failed to delete Gemini file: {e}")
        if 'pdf_temp_path' in locals() and pdf_temp_path and os.path.exists(pdf_temp_path):
            try:
                os.remove(pdf_temp_path)
            except Exception as e:
                print(f"Failed to delete temp file: {e}")

st.markdown("---")

# 7. Output Options and Download Section
if st.session_state['chunked_results']:
    st.subheader("🧩 Output Options")
    combine_choice = st.radio(
        "Choose how to handle analyzed chunks:",
        options=["🔗 Combine all outputs into one file", "📦 Download each part separately"],
        index=0,
        horizontal=True,
        key='combine_choice_radio',
        help="You can merge all analyzed chunks into a single hyperlinked PDF, or keep each chunk's output separate."
    )

    current_dir = Path(__file__).parent

    if combine_choice.startswith("🔗"):
        st.subheader("Single Merged PDF")
        
        combined_data = merge_all_json_outputs(st.session_state['chunked_results'])
        
        # 🧠 DEBUG 2 & 3: Final merged output check
        st.write("🧠 DEBUG: Final merged JSON keys (check for expected keys and list lengths):")
        
        data = combined_data
        # FIX: Remove invalid escape sequence \_
        st.write("**main_subject**:", data.get("main_subject"))
        
        # Check list keys
        for k, v in data.items():
            if isinstance(v, list) and k != "main_subject":
                # FIX: Remove invalid escape sequence \_
                st.write(f"**{k}**:", len(v))
        
        pdf_output = BytesIO()
        try:
            with st.spinner("Generating combined PDF..."):
                # 🧠 DEBUG 4: save_to_pdf is called here
                save_to_pdf(combined_data, video_id, current_dir, pdf_output, format_choice)
            
            st.download_button(
                label=f"⬇️ Download Merged Notes: {output_filename_base}.pdf",
                data=pdf_output,
                file_name=output_filename, 
                mime="application/pdf" 
            )
        except Exception as e:
            st.error(f"Error generating merged PDF: {e}")
            st.warning("Ensure font files (NotoSans-*.ttf) are in the main directory.")

    else:
        st.subheader("Separate PDF Downloads")
        st.info("Each part represents a section of the original transcript.")

        for i, part_data in enumerate(st.session_state['chunked_results'], start=1):
            pdf_output = BytesIO()
            
            try:
                with st.spinner(f"Preparing Part {i}..."):
                    # 🧠 DEBUG 4: save_to_pdf is called here
                    save_to_pdf(part_data, video_id, current_dir, pdf_output, format_choice)
                
                st.download_button(
                    label=f"⬇️ Download Part {i}",
                    data=pdf_output,
                    # FIX: Corrected incomplete file_name string
                    file_name=f"{output_filename_base}_part{i}.pdf",
                    mime="application/pdf",
                    key=f'download_part_{i}'
                )
            except Exception as e:
                st.error(f"Error generating Part {i} PDF: {e}")
                break

st.markdown("---")

if not st.session_state['chunked_results'] and st.session_state.get('pdf_ready'):
    st.warning("Analysis ran successfully, but no output was generated. Please verify your transcript and settings.")
