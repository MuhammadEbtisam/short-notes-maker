# -*- coding: utf-8 -*-
import streamlit as st
import json
import re
from pathlib import Path
import google.generativeai as genai
from io import BytesIO
import time
from typing import Optional, Tuple, Dict, Any, List

# --- NEW IMPORTS ---
import subprocess
import tempfile
import os

# --- SYSTEM PROMPT FOR LATEX GENERATION ---
# This is the new core prompt.
# THE FIX IS HERE: Note the 'r' before the """
SYSTEM_PROMPT = r"""
You are an expert academic typesetter. Your task is to convert a video transcript, provided as JSON segments, into a single, complete, and high-quality LaTeX document.

RULES:
1.  **Output Format:** Your response MUST be ONLY raw LaTeX code. It MUST start with `\documentclass{article}` and end with `\end{document}`. Do NOT use markdown code fences (```).
2.  **Preamble:** You MUST include a robust preamble. Use:
    - `\documentclass[11pt, a4paper]{article}`
    - `\usepackage[utf8]{inputenc}`
    - `\usepackage[T1]{fontenc}`
    - `\usepackage{amsmath}` (for all math)
    - `\usepackage{amssymb}`
    - `\usepackage{booktabs}` (for any tables)
    - `\usepackage[a4paper, margin=1in]{geometry}` (for page layout)
    - `\usepackage{hyperref}` (for links)
    - `\usepackage{graphicx}`
    - `\usepackage{parskip}` (for better paragraph spacing)
    - `\usepackage{titlesec}` (for section styling)
    - `\usepackage{enumitem}` (for lists)
3.  **Title:** You MUST create a suitable title for the document using the `\title{}` and `\author{}` (e.g., "Notes from Video") commands, and you MUST call `\maketitle` after `\begin{document}`.
4.  **Content:** You MUST structure the document using `\section{}`, `\subsection{}`, and `\subsubsection{}` based on the `REQUESTED SECTIONS` and the transcript content.
5.  **Math:** You MUST render all mathematical content using standard LaTeX math environments (e.g., `$..$`, `$$..$$`, `\begin{equation}`, `\begin{align}`). Use `amsmath` features for complex formulas.
6.  **Timestamps:** If a `video_id` is provided, you MUST hyperlink timestamps.
    -   Example: `\href{https://www.youtube.com/watch?v=VIDEO_ID&t=123s}{[02:03]}`
    -   If `video_id` is blank, just write the timestamp as plain text (e.g., `[02:03]`).
7.  **Lists:** Use `itemize` or `enumerate` for lists (e.g., for Key Points).
8.  **User Focus:** Pay close attention to the `USER PREFERENCES` prompt.
9.  **Completeness:** The document MUST be a complete, runnable file. Do NOT use any custom packages or external files.
"""

# --- UTILITY FUNCTIONS (KEPT) ---

def inject_custom_css():
    """Modern CSS styling"""
    st.markdown("""
        <style>
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        p, label, .stMarkdown {
            font-size: 1.05rem !important;
            line-height: 1.6;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1E88E5, #1565C0);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border-radius: 8px;
            transition: transform 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

def get_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID"""
    patterns = [
        r"(?<=v=)[^&#?]+", r"(?<=be/)[^&#?]+", r"(?<=live/)[^&#?]+",
        r"(?<=embed/)[^&#?]+", r"(?<=shorts/)[^&#?]+"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(0)
    return None

def extract_gemini_text(response) -> Optional[str]:
    """Extract text from Gemini API response"""
    if hasattr(response, 'text'):
        return response.text
    if hasattr(response, 'candidates') and response.candidates:
        try:
            return response.candidates[0].content.parts[0].text
        except (AttributeError, IndexError):
            pass
    return None

def preprocess_transcript(text: str) -> List[Dict[str, Any]]:
    """
    Parses raw transcript text (with timestamps) into a structured
    list of segments for the AI.
    """
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
        ts_str = matches[i].group(1)
        
        # Convert timestamp to seconds for the AI
        sec = 0
        try:
            parts_ts = ts_str.split(':')
            if len(parts_ts) == 3: # HH:MM:SS
                sec = int(parts_ts[0])*3600 + int(parts_ts[1])*60 + int(parts_ts[2])
            elif len(parts_ts) == 2: # MM:SS
                sec = int(parts_ts[0])*60 + int(parts_ts[1])
        except ValueError:
            pass # Keep sec = 0

        segments.append({"time": sec, "text": text[start:end].strip()})
        
    return segments

# --- NEW CORE FUNCTIONS ---

def generate_latex_document(
    api_key: str, 
    transcript_segments: List[Dict], 
    sections_list: list, 
    user_prompt: str, 
    model_name: str, 
    video_id: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls Gemini API to generate a complete LaTeX document string.
    """
    
    sections_str = ", ".join(sections_list)
    video_id_str = video_id if video_id else ""
    
    prompt_header = f"""
USER PREFERENCES: {user_prompt}
REQUESTED SECTIONS: {sections_str}
VIDEO_ID: {video_id_str}
"""
    
    transcript_json = json.dumps(transcript_segments, indent=2)
    full_prompt = f"{SYSTEM_PROMPT}\n{prompt_header}\n\nTRANSCRIPT DATA:\n{transcript_json}"
    
    if not api_key:
        return None, "API Key Missing"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(full_prompt)
        response_text = extract_gemini_text(response)
        
        if not response_text:
            return None, "Empty API response"
        
        # Basic check to see if it looks like LaTeX
        if not response_text.strip().startswith(r"\documentclass"):
            return None, f"AI did not return valid LaTeX. Response: {response_text[:200]}..."
        
        return response_text, None
        
    except Exception as e:
        return None, f"API Error: {e}"

def compile_latex_to_pdf(latex_string: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Compiles a LaTeX string into a PDF using pdflatex.
    Returns (pdf_bytes, error_message)
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = "notes"
            tex_file_path = os.path.join(temp_dir, f"{base_name}.tex")
            pdf_file_path = os.path.join(temp_dir, f"{base_name}.pdf")
            log_file_path = os.path.join(temp_dir, f"{base_name}.log")

            # Write the .tex file
            with open(tex_file_path, 'w', encoding='utf-8') as f:
                f.write(latex_string)

            cmd = ['pdflatex', '-interaction=nonstopmode', '-output-directory', temp_dir, tex_file_path]
            
            # --- Run pdflatex ---
            # We run it up to 2 times. 1st pass for content, 2nd for references (like ToC, if any)
            
            log = ""
            for i in range(2):
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                log += f"--- PASS {i+1} ---\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}\n"
                
                # If PDF doesn't exist after first pass, it failed
                if i == 0 and not os.path.exists(pdf_file_path):
                    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as log_f:
                        log_details = log_f.read()
                    return None, f"LaTeX Compilation Failed. Check .tex file.\nError Log:\n{log_details[-1000:]}"
            
            # If PDF exists, read it
            if os.path.exists(pdf_file_path):
                with open(pdf_file_path, 'rb') as f:
                    pdf_bytes = f.read()
                return pdf_bytes, None
            else:
                return None, f"Compilation finished, but no PDF was created. Log:\n{log}"

    except FileNotFoundError:
        return None, "Error: 'pdflatex' command not found. Is TeX Live or MiKTeX installed and in your system's PATH?"
    except subprocess.TimeoutExpired:
        return None, "Error: LaTeX compilation timed out (120s). The .tex file may be too complex or in an infinite loop."
    except Exception as e:
        return None, f"An unknown error occurred during compilation: {e}"