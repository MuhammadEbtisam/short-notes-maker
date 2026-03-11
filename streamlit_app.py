import streamlit as st
from utils import (
    inject_custom_css, 
    get_video_id, 
    preprocess_transcript, 
    generate_latex_document,
    compile_latex_to_pdf
)
from pathlib import Path
from io import BytesIO 
import json 
from typing import List, Dict, Any, Optional
import re 

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

# --- Application Setup ---
st.title("📹 AI-Powered LaTeX Video Notes Generator")

st.warning("**IMPORTANT:** This app now requires a full TeX Live (or MiKTeX) distribution, including the `pdflatex` command, to be installed on the server. Compilation will fail without it.")

# Initialize session state variables
if 'api_key_valid' not in st.session_state:
    st.session_state['api_key_valid'] = False
if 'output_filename_base' not in st.session_state:
    st.session_state['output_filename_base'] = "Video_Notes_LaTeX"
if 'processing' not in st.session_state:
    st.session_state['processing'] = False
if 'pdf_bytes' not in st.session_state:
    st.session_state['pdf_bytes'] = None
if 'latex_code' not in st.session_state:
    st.session_state['latex_code'] = None

# --------------------------------------------------------------------------
# --- Sidebar Setup ---
# --------------------------------------------------------------------------

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

    # Model Selection
    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Model Selection (Pro recommended):",
        options=["gemini-2.5-pro", "gemini-2.5-flash"],
        index=0, 
        key='model_choice_select', 
        help="Pro is strongly recommended for handling the large context of a full transcript and complex LaTeX generation."
    )

    st.markdown("---")
    
    # 4. YouTube URL Input
    st.subheader("Video Details")
    yt_url = st.text_input("YouTube URL (Optional):", help="Provide a URL to enable hyperlinked timestamps in the PDF.")
    video_id = get_video_id(yt_url)
    if video_id:
        st.success(f"Video ID found: {video_id}")
    elif yt_url:
        st.warning("Invalid YouTube URL format. Timestamps will not be hyperlinked.")
    
    st.markdown("---")
    st.header("⚙️ Analysis Details")
    
    # B. Checkboxes for Section Selection
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
    
    # G. Custom Filename Input
    if video_id:
        st.session_state['output_filename_base'] = f"Notes_{video_id}"
    
    output_filename_base = st.session_state['output_filename_base']
    output_filename = st.text_input(
        "Base Name for PDF/TEX files:",
        value=output_filename_base,
        key="output_filename_input"
    )
    
# --------------------------------------------------------------------------
# --- Main Content: Transcript Input, Button, and Output ---
# --------------------------------------------------------------------------

st.subheader("Transcript Input")
transcript_text = st.text_area(
    'Paste the video transcript here (must include timestamps for best results):',
    height=300,
    placeholder="[00:00] Welcome to the lesson. [00:45] We start with Topic A..."
)

user_prompt_input = st.text_area(
    'Refine AI Focus (Optional Prompt):',
    value="Ensure the output is highly condensed and only focus on practical applications and examples. Use `amsmath` for all complex equations.",
    height=100
)

# E. The Analysis Trigger Button
can_run = transcript_text and st.session_state['api_key_valid']
run_analysis = st.button(
    f"🚀 Generate PDF using {model_choice} + LaTeX", 
    type="primary", 
    disabled=not can_run or st.session_state['processing']
) 

if run_analysis and not st.session_state['processing']:
    
    # Reset state
    st.session_state['processing'] = True
    st.session_state['pdf_bytes'] = None
    st.session_state['latex_code'] = None
    
    try:
        # 1. Preprocess transcript into segments
        transcript_segments = preprocess_transcript(transcript_text)
        
        # 2. Generate the LaTeX document string from the AI
        latex_code = ""
        error_msg = ""
        with st.spinner(f"Generating LaTeX document with {model_choice}... (This may take a while)"):
            latex_code, error_msg = generate_latex_document(
                api_key=api_key,
                transcript_segments=transcript_segments,
                sections_list=sections_list,
                user_prompt=user_prompt_input,
                model_name=model_choice,
                video_id=video_id
            )

        if error_msg or not latex_code:
            st.error(f"Failed to generate LaTeX from AI: {error_msg}")
            st.session_state['processing'] = False
        else:
            st.session_state['latex_code'] = latex_code
            st.success("AI generated LaTeX code successfully.")
            
            # 3. Compile the LaTeX string to a PDF
            pdf_bytes = None
            compile_error = ""
            with st.spinner("Compiling LaTeX to PDF... (Requires `pdflatex`)"):
                pdf_bytes, compile_error = compile_latex_to_pdf(latex_code)
            
            if compile_error:
                st.error(f"PDF Compilation Failed! {compile_error}")
                st.info("The AI-generated LaTeX code below is likely broken.")
            else:
                st.success("PDF compiled successfully!")
                st.session_state['pdf_bytes'] = pdf_bytes
                
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        st.session_state['processing'] = False

st.markdown("---")

# 7. Output Options and Download Section
if st.session_state['pdf_bytes']:
    st.subheader("✅ PDF Generation Successful")
    st.download_button(
        label=f"⬇️ Download PDF: {output_filename}.pdf",
        data=st.session_state['pdf_bytes'],
        file_name=f"{output_filename}.pdf",
        mime="application/pdf"
    )

if st.session_state['latex_code']:
    st.subheader("📄 Raw LaTeX Code")
    st.download_button(
        label=f"⬇️ Download .tex: {output_filename}.tex",
        data=st.session_state['latex_code'],
        file_name=f"{output_filename}.tex",
        mime="text/plain"
    )
    with st.expander("View Generated LaTeX Code"):
        st.code(st.session_state['latex_code'], language='latex')