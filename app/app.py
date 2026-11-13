"""
HarmoniX | AI Music Emotion Recognition Studio
Author: Adnan Rizvi (Lead Architect & Product Engineering Lead)

Minimalist Studio Edition (No Emojis, Pure Dark Scientific Aesthetic)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline_orchestrator import predict_audio
from src.multicultural_loader import scan_multicultural_directory, create_sample_diverse_dataset

st.set_page_config(
    page_title="HarmoniX Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Pitch Black Canvas */
    .stApp {
        background-color: #000000 !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -5%, rgba(0, 229, 255, 0.06) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 95% 95%, rgba(139, 92, 246, 0.04) 0%, transparent 50%);
        color: #e2e8f0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px !important;
    }

    /* Hide Streamlit Header */
    header[data-testid="stHeader"], #MainMenu, footer {
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #030405 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    /* Equalizer Bar Animation */
    .eq-container {
        display: inline-flex;
        align-items: flex-end;
        gap: 2.5px;
        height: 18px;
        margin-left: 10px;
    }
    .eq-bar {
        width: 3px;
        background: linear-gradient(to top, #00e5ff, #00b8d4);
        border-radius: 1.5px;
        animation: eqPulse 1.2s ease-in-out infinite alternate;
    }
    .eq-bar:nth-child(1) { height: 30%; animation-delay: 0.1s; }
    .eq-bar:nth-child(2) { height: 75%; animation-delay: 0.3s; }
    .eq-bar:nth-child(3) { height: 100%; animation-delay: 0.15s; }
    .eq-bar:nth-child(4) { height: 50%; animation-delay: 0.4s; }
    .eq-bar:nth-child(5) { height: 85%; animation-delay: 0.25s; }

    @keyframes eqPulse {
        0% { height: 20%; }
        100% { height: 100%; }
    }

    /* Pulse Status Dot */
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #00e5ff;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #00e5ff, 0 0 16px rgba(0, 229, 255, 0.5);
        animation: pulseGlow 2s ease-in-out infinite;
    }
    @keyframes pulseGlow {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.7; }
    }

    /* Studio Metric Card */
    .metric-card {
        background: #060709;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 16px 18px;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(0, 229, 255, 0.3) 50%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(0, 229, 255, 0.04);
    }
    .metric-card:hover::before {
        opacity: 1;
    }
    .card-label {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #475569;
        margin-bottom: 2px;
    }
    .card-number {
        font-size: 28px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
        color: #ffffff;
        line-height: 1.1;
    }

    /* Minimal Quadrant Badges */
    .badge-quad {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-joyful {
        background: rgba(0, 230, 153, 0.1);
        color: #00e699;
        border: 1px solid rgba(0, 230, 153, 0.3);
        box-shadow: 0 0 20px rgba(0, 230, 153, 0.15), inset 0 0 12px rgba(0, 230, 153, 0.05);
    }
    .badge-tense {
        background: rgba(255, 51, 102, 0.1);
        color: #ff3366;
        border: 1px solid rgba(255, 51, 102, 0.3);
        box-shadow: 0 0 20px rgba(255, 51, 102, 0.15), inset 0 0 12px rgba(255, 51, 102, 0.05);
    }
    .badge-melancholic {
        background: rgba(168, 85, 247, 0.1);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.3);
        box-shadow: 0 0 20px rgba(192, 132, 252, 0.15), inset 0 0 12px rgba(192, 132, 252, 0.05);
    }
    .badge-calm {
        background: rgba(0, 229, 255, 0.1);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.3);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.15), inset 0 0 12px rgba(0, 229, 255, 0.05);
    }

    /* Audio Element */
    audio {
        width: 100% !important;
        border-radius: 8px;
        filter: invert(0.88) hue-rotate(180deg);
        opacity: 0.9;
        margin-top: 4px;
    }

    /* Glassmorphic Panels */
    .studio-panel {
        background: #060709;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 14px;
        transition: border-color 0.3s ease;
        position: relative;
    }
    .studio-panel:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .panel-title {
        font-size: 14px;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.01em;
    }
    .panel-subtitle {
        font-size: 11px;
        color: #475569;
        margin-top: 3px;
    }
    .card-subtext {
        font-size: 11.5px;
        color: #64748b;
        margin-top: 4px;
        font-weight: 500;
    }
    .progress-track {
        background: rgba(255, 255, 255, 0.06);
        height: 5px;
        border-radius: 3px;
        margin-top: 10px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 10px;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 34px;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid transparent;
        border-bottom: none;
        color: #64748b;
        font-size: 11px;
        font-weight: 600;
        padding: 6px 14px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background-color: rgba(255, 255, 255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 229, 255, 0.06) !important;
        color: #00e5ff !important;
        border-color: rgba(0, 229, 255, 0.2) !important;
        border-bottom: 2px solid #00e5ff !important;
    }

    /* Sidebar Form Polish */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #08090d !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 12px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 12px;
        color: #94a3b8;
    }
    .streamlit-expanderHeader {
        background: #060709 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

    /* Dataframe */
    .stDataFrame { border-radius: 10px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown("""
<div style="padding: 16px 0 20px 0;">
    <div style="font-size: 9px; font-weight: 700; letter-spacing: 2px; color: #475569; text-transform: uppercase;">Acoustic Laboratory</div>
    <div style="font-size: 19px; font-weight: 800; color: #ffffff; margin-top: 4px; letter-spacing: -0.02em;">Audio Ingestion</div>
    <div style="height: 2px; width: 28px; background: linear-gradient(90deg, #00e5ff, transparent); margin-top: 8px; border-radius: 1px;"></div>
</div>
""", unsafe_allow_html=True)

test_diverse_dir = PROJECT_ROOT / "data" / "test_diverse"
if not test_diverse_dir.exists():
    create_sample_diverse_dataset(test_diverse_dir)

diverse_tracks = scan_multicultural_directory(test_diverse_dir)

source_choice = st.sidebar.radio(
    "Select Mode",
    ["Multicultural Presets", "Upload Audio (.mp3, .wav)"],
    index=0
)

selected_audio_path = None
audio_label = ""
genre_badge = ""

if source_choice == "Multicultural Presets":
    if diverse_tracks:
        genre_options = {f"{t['genre']} / {t['title']}": (t['filepath'], t['genre']) for t in diverse_tracks}
        chosen_title = st.sidebar.selectbox("Benchmark Clip:", list(genre_options.keys()))
        selected_audio_path = Path(genre_options[chosen_title][0])
        genre_badge = genre_options[chosen_title][1]
        audio_label = chosen_title.split(" / ")[1]
    else:
        st.sidebar.warning("No pre-loaded audio tracks found.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "ogg", "flac"]
    )
    if uploaded_file is not None:
        tmp_suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp_f:
            tmp_f.write(uploaded_file.read())
            selected_audio_path = Path(tmp_f.name)
            audio_label = uploaded_file.name
            genre_badge = "Custom Input"


col_title, col_status = st.columns([2, 1])

with col_title:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size: 28px; font-weight: 800; letter-spacing: -0.04em; color:#ffffff; font-family: 'Inter', sans-serif;">HarmoniX Audio Studio</span>
        <div class="eq-container">
            <div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div>
        </div>
    </div>
    <div style="font-size: 12px; color: #475569; margin-top: 5px; font-weight: 500;">Multimodal Music Emotion Perception & 2D Spectrogram Neural Intelligence</div>
    <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top: 10px;">
        <span class="chip">Phase 2 Champion Hybrid Fusion</span>
        <span class="chip">128 Mel Bins (Slaney)</span>
        <span class="chip">22.05 kHz DSP</span>
        <span class="chip">Grad-CAM Attribution</span>
    </div>
    """, unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div style="text-align: right; padding-top: 6px;">
        <div style="display:inline-flex; align-items:center; gap:7px; background: rgba(0, 229, 255, 0.06); border: 1px solid rgba(0, 229, 255, 0.18); border-radius: 24px; padding: 6px 16px;">
            <span style="width: 7px; height: 7px; background: #00e5ff; border-radius: 50%; box-shadow: 0 0 8px #00e5ff, 0 0 16px rgba(0,229,255,0.4); display:inline-block;"></span>
            <span style="color: #00e5ff; font-size: 10.5px; font-weight: 700; letter-spacing: 0.8px; font-family: 'JetBrains Mono', monospace;">ENGINE ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

if selected_audio_path is None:
    st.info("Select a benchmark audio track or upload an audio file from the sidebar to launch the studio.")
    st.stop()


with st.container():
    with st.spinner("Analyzing audio frequencies and generating neural predictions..."):
        try:
            results = predict_audio(selected_audio_path)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    meta = results["metadata"]

    st.markdown(f"""
    <div style="background: #060709; border: 1px solid rgba(0, 229, 255, 0.15); border-radius: 16px; padding: 18px 22px 14px 22px; margin-bottom: 14px; position: relative; overflow: hidden;">
        <div style="position:absolute; top:0; left:0; right:0; height:1px; background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.3), transparent);"></div>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.04);">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span class="pulse-dot"></span>
                <div>
                    <div style="font-size: 18px; font-weight: 800; color: #ffffff; letter-spacing: -0.03em; font-family: 'Inter', sans-serif;">{audio_label}</div>
                    <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px;">
                        <span style="background: rgba(0, 229, 255, 0.1); color: #00e5ff; border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 6px; padding: 2px 9px; font-size: 9.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">{genre_badge}</span>
                        <span style="font-size: 11px; color: #475569; font-family: 'JetBrains Mono';">Slaney DSP  //  22.05 kHz Ingestion</span>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 5px 12px; text-align: center; min-width: 60px;">
                    <div style="font-size: 8.5px; color: #475569; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Duration</div>
                    <div style="font-size: 13px; font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono'; margin-top: 1px;">{meta['duration_seconds']:.1f}s</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 5px 12px; text-align: center; min-width: 60px;">
                    <div style="font-size: 8.5px; color: #475569; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Sample Rate</div>
                    <div style="font-size: 13px; font-weight: 700; color: #00e5ff; font-family: 'JetBrains Mono'; margin-top: 1px;">{meta['target_sample_rate']} Hz</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 5px 12px; text-align: center; min-width: 60px;">
                    <div style="font-size: 8.5px; color: #475569; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Signal</div>
                    <div style="font-size: 13px; font-weight: 700; color: #94a3b8; font-family: 'JetBrains Mono'; margin-top: 1px;">Mono Downmix</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 5px 12px; text-align: center; min-width: 60px;">
                    <div style="font-size: 8.5px; color: #475569; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Peak Level</div>
                    <div style="font-size: 13px; font-weight: 700; color: #00e699; font-family: 'JetBrains Mono'; margin-top: 1px;">{meta['normalized_peak']:.2f} pk</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with open(selected_audio_path, "rb") as f_audio:
        audio_bytes = f_audio.read()
    st.audio(audio_bytes, format="audio/wav")

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

v_score = results["primary_valence"]
a_score = results["estimated_arousal"]
quadrant = results["emotional_quadrant"]

badge_class = "badge-joyful" if "Q1" in quadrant else "badge-tense" if "Q2" in quadrant else "badge-melancholic" if "Q3" in quadrant else "badge-calm"

import math

def render_gauge_svg(val, min_v, max_v, col, grad_id, c1, c2, center_label, subtext, t_left, t_mid, t_right):
    p = max(0.0, min(1.0, (val - min_v) / (max_v - min_v)))
    r = 58
    total_len = math.pi * r
    offset = total_len * (1.0 - p)
    nx = 100.0 - r * math.cos(p * math.pi)
    ny = 88.0 - r * math.sin(p * math.pi)
    return f"""
    <svg viewBox="0 0 200 108" style="width:100%; height:auto; display:block; margin: 4px auto 0 auto;">
        <defs>
            <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="{c1}"/>
                <stop offset="100%" stop-color="{c2}"/>
            </linearGradient>
            <filter id="glow-{grad_id}" x="-30%" y="-30%" width="160%" height="160%">
                <feDropShadow dx="0" dy="0" stdDeviation="3.5" flood-color="{col}" flood-opacity="0.85"/>
            </filter>
        </defs>
        <path d="M 42 88 A 58 58 0 0 1 158 88" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="8" stroke-linecap="round"/>
        <path d="M 42 88 A 58 58 0 0 1 158 88" fill="none" stroke="url(#{grad_id})" stroke-width="8" stroke-linecap="round"
              stroke-dasharray="{total_len:.2f}" stroke-dashoffset="{offset:.2f}" filter="url(#glow-{grad_id})"/>
        <circle cx="{nx:.1f}" cy="{ny:.1f}" r="4.5" fill="#ffffff" stroke="{col}" stroke-width="2" filter="drop-shadow(0 0 6px {col})"/>
        <text x="100" y="74" text-anchor="middle" font-size="25" font-weight="800" font-family="'JetBrains Mono', monospace" fill="#ffffff" filter="drop-shadow(0 0 8px {col}88)">{center_label}</text>
        <text x="100" y="90" text-anchor="middle" font-size="10" font-weight="600" font-family="'Inter', sans-serif" fill="#64748b">{subtext}</text>
        <text x="42" y="103" text-anchor="middle" font-size="8.5" font-family="'JetBrains Mono', monospace" fill="#475569">{t_left}</text>
        <text x="100" y="103" text-anchor="middle" font-size="8.5" font-family="'JetBrains Mono', monospace" fill="#475569">{t_mid}</text>
        <text x="158" y="103" text-anchor="middle" font-size="8.5" font-family="'JetBrains Mono', monospace" fill="#475569">{t_right}</text>
    </svg>
    """

def render_circumplex_svg(v, a, quad_str, quad_title):
    bx = max(68.0, min(132.0, 100.0 + (v - 0.5) * 60.0))
    by = max(18.0, min(78.0, 48.0 - (a - 0.5) * 60.0))
    q_col = "#00e699" if "Q1" in quad_str else "#ff3366" if "Q2" in quad_str else "#c084fc" if "Q3" in quad_str else "#00e5ff"
    
    if "Q1" in quad_str:
        q_rect = '<rect x="100" y="14" width="34" height="34" fill="rgba(0, 230, 153, 0.12)" rx="3"/>'
    elif "Q2" in quad_str:
        q_rect = '<rect x="66" y="14" width="34" height="34" fill="rgba(255, 51, 102, 0.12)" rx="3"/>'
    elif "Q3" in quad_str:
        q_rect = '<rect x="66" y="48" width="34" height="34" fill="rgba(192, 132, 252, 0.12)" rx="3"/>'
    else:
        q_rect = '<rect x="100" y="48" width="34" height="34" fill="rgba(0, 229, 255, 0.12)" rx="3"/>'

    return f"""
    <svg viewBox="0 0 200 108" style="width:100%; height:auto; display:block; margin: 4px auto 0 auto;">
        <defs>
            <filter id="scope-glow" x="-30%" y="-30%" width="160%" height="160%">
                <feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="{q_col}" flood-opacity="0.8"/>
            </filter>
        </defs>
        {q_rect}
        <circle cx="100" cy="48" r="34" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1" stroke-dasharray="2,2"/>
        <circle cx="100" cy="48" r="18" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>
        <line x1="66" y1="48" x2="134" y2="48" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
        <line x1="100" y1="14" x2="100" y2="82" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
        <circle cx="{bx:.1f}" cy="{by:.1f}" r="7" fill="none" stroke="{q_col}" stroke-width="1" stroke-dasharray="2,2"/>
        <polygon points="{bx:.1f},{by-4:.1f} {bx+4:.1f},{by:.1f} {bx:.1f},{by+4:.1f} {bx-4:.1f},{by:.1f}" fill="#ffffff" stroke="{q_col}" stroke-width="2" filter="url(#scope-glow)"/>
        <text x="100" y="93" text-anchor="middle" font-size="12.5" font-weight="800" font-family="'Inter', sans-serif" fill="{q_col}">{quad_title}</text>
        <text x="100" y="105" text-anchor="middle" font-size="9" font-weight="600" font-family="'JetBrains Mono', monospace" fill="#64748b">V: {v:.3f} • A: {a:.3f}</text>
    </svg>
    """

m1, m2, m3, m4 = st.columns(4)

with m1:
    v_col = "#00e699" if v_score >= 0.50 else "#ff3366"
    v_grad_c1 = "#059669" if v_score >= 0.50 else "#991b1b"
    v_grad_c2 = "#00e699" if v_score >= 0.50 else "#ff3366"
    v_bias = v_score - 0.50
    bias_str = f"+{v_bias:.3f} Uplift" if v_bias >= 0 else f"{v_bias:.3f} Somber"
    tag_str = "MAJOR / UPLIFT" if v_score >= 0.50 else "SOMBER / MINOR"
    svg_v = render_gauge_svg(v_score, 0.0, 1.0, v_col, "vGrad", v_grad_c1, v_grad_c2, f"{v_score:.3f}", bias_str, "0.0", "0.5", "1.0")
    st.markdown(f"""
    <div class="metric-card" style="border-top: 2px solid {v_col};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="card-label">Valence Polarity</span>
            <span style="font-size:9.5px; font-weight:700; color:{v_col}; background:{v_col}18; border:1px solid {v_col}44; padding:2px 7px; border-radius:4px; font-family:'JetBrains Mono';">{tag_str}</span>
        </div>
        {svg_v}
    </div>
    """, unsafe_allow_html=True)

with m2:
    pct_ar = max(0.0, min(100.0, a_score * 100))
    energy_label = "HIGH KINETIC" if a_score >= 0.65 else "MODERATE" if a_score >= 0.40 else "SUBDUED / LOW"
    svg_a = render_gauge_svg(a_score, 0.0, 1.0, "#00e5ff", "aGrad", "#0284c7", "#00e5ff", f"{a_score:.3f}", f"Dynamic Energy: {pct_ar:.1f}%", "0.0", "0.5", "1.0")
    st.markdown(f"""
    <div class="metric-card" style="border-top: 2px solid #00e5ff;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="card-label">Dynamic Arousal</span>
            <span style="font-size:9.5px; font-weight:700; color:#00e5ff; background:rgba(0,229,255,0.14); border:1px solid rgba(0,229,255,0.35); padding:2px 7px; border-radius:4px; font-family:'JetBrains Mono';">{energy_label}</span>
        </div>
        {svg_a}
    </div>
    """, unsafe_allow_html=True)

with m3:
    quad_name = quadrant.split(':')[1].strip() if ':' in quadrant else quadrant
    quad_key = quadrant.split(':')[0].strip()
    svg_c = render_circumplex_svg(v_score, a_score, quadrant, quad_name)
    st.markdown(f"""
    <div class="metric-card" style="border-top: 2px solid #c084fc;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="card-label">Emotional Space</span>
            <span style="font-size:9.5px; font-weight:700; color:#c084fc; background:rgba(192,132,252,0.14); border:1px solid rgba(192,132,252,0.35); padding:2px 7px; border-radius:4px; font-family:'JetBrains Mono';">{quad_key}</span>
        </div>
        {svg_c}
    </div>
    """, unsafe_allow_html=True)

with m4:
    centroid_str = results["acoustic_highlights"].get("Brightness (Spectral Centroid)", "2150.0 Hz")
    try:
        c_val = float(centroid_str.replace(" Hz", "").replace(",", "").strip())
        warmth_tag = "DEEP SUB-BASS" if c_val < 500 else "MIDRANGE BODY" if c_val < 2000 else "BRIGHT PRESENCE"
    except Exception:
        c_val = 1500.0
        warmth_tag = "MIDRANGE"
    svg_w = render_gauge_svg(c_val, 0.0, 4000.0, "#ffb800", "wGrad", "#d97706", "#ffb800", centroid_str, "Timbral Center of Mass", "0Hz", "2kHz", "4kHz")
    st.markdown(f"""
    <div class="metric-card" style="border-top: 2px solid #ffb800;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="card-label">Spectral Warmth</span>
            <span style="font-size:9.5px; font-weight:700; color:#ffb800; background:rgba(255,184,0,0.14); border:1px solid rgba(255,184,0,0.35); padding:2px 7px; border-radius:4px; font-family:'JetBrains Mono';">{warmth_tag}</span>
        </div>
        {svg_w}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

col_radar, col_ensemble = st.columns([1.3, 1])

with col_radar:
    st.markdown(f"""
    <div class="studio-panel" style="padding-bottom: 8px;">
        <div class="panel-header">
            <div>
                <span class="panel-title">Russell's 2D Mood Compass</span>
                <div class="panel-subtitle">Circumplex Model of Affect (Valence vs Arousal)</div>
            </div>
            <span class="chip">Target: [V: {v_score:.2f} | A: {a_score:.2f}]</span>
        </div>
    """, unsafe_allow_html=True)

    fig_compass = go.Figure()

    fig_compass.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(0, 230, 153, 0.07)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(255, 51, 102, 0.07)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(168, 85, 247, 0.07)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(0, 229, 255, 0.07)", line_width=0)

    for r in [0.15, 0.30, 0.45]:
        fig_compass.add_shape(
            type="circle",
            x0=0.5 - r, y0=0.5 - r, x1=0.5 + r, y1=0.5 + r,
            line=dict(color="rgba(0, 229, 255, 0.08)", width=1, dash="dot")
        )

    fig_compass.add_shape(type="line", x0=0.0, y0=0.0, x1=1.0, y1=1.0, line=dict(color="rgba(255, 255, 255, 0.04)", width=1, dash="dot"))
    fig_compass.add_shape(type="line", x0=0.0, y0=1.0, x1=1.0, y1=0.0, line=dict(color="rgba(255, 255, 255, 0.04)", width=1, dash="dot"))

    fig_compass.add_hline(y=0.5, line_dash="solid", line_color="rgba(255, 255, 255, 0.14)", line_width=1)
    fig_compass.add_vline(x=0.5, line_dash="solid", line_color="rgba(255, 255, 255, 0.14)", line_width=1)
    fig_compass.add_shape(type="circle", x0=0.490, y0=0.490, x1=0.510, y1=0.510, fillcolor="#00e5ff", line_color="#ffffff", line_width=1)

    fig_compass.add_annotation(x=0.82, y=0.93, text="Q1 // JOYFUL (EUPHORIC)", showarrow=False, font=dict(color="#00e699", size=9.5, family="Inter, sans-serif"))
    fig_compass.add_annotation(x=0.18, y=0.93, text="Q2 // TENSE (DRAMATIC)", showarrow=False, font=dict(color="#ff3366", size=9.5, family="Inter, sans-serif"))
    fig_compass.add_annotation(x=0.18, y=0.07, text="Q3 // MELANCHOLIC (SOMBER)", showarrow=False, font=dict(color="#c084fc", size=9.5, family="Inter, sans-serif"))
    fig_compass.add_annotation(x=0.82, y=0.07, text="Q4 // CALM (SERENE)", showarrow=False, font=dict(color="#00e5ff", size=9.5, family="Inter, sans-serif"))

    # Dynamic text position to prevent clipping at plot boundaries
    if v_score >= 0.65:
        track_textpos = "middle left"
    elif v_score <= 0.35:
        track_textpos = "middle right"
    elif a_score >= 0.75:
        track_textpos = "bottom center"
    else:
        track_textpos = "top center"

    fig_compass.add_trace(go.Scatter(
        x=[v_score], y=[a_score],
        mode="markers",
        marker=dict(size=36, color="rgba(0, 229, 255, 0.12)", line=dict(color="rgba(0, 229, 255, 0.5)", width=1.5, dash="dot"), symbol="circle"),
        hoverinfo="skip"
    ))
    fig_compass.add_trace(go.Scatter(
        x=[v_score], y=[a_score],
        mode="markers+text",
        marker=dict(size=14, color="#00e5ff", line=dict(color="#ffffff", width=2.5), symbol="diamond"),
        text=[f"Current Track ({v_score:.2f}, {a_score:.2f})"],
        textposition=track_textpos,
        textfont=dict(color="#ffffff", size=11, family="Inter, sans-serif"),
        cliponaxis=False,
        name="Track Location"
    ))

    fig_compass.update_layout(
        xaxis=dict(title=dict(text="Valence (Negative to Positive)", font=dict(size=11, color="#64748b")), range=[-0.03, 1.03], gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        yaxis=dict(title=dict(text="Arousal (Calm to Energetic)", font=dict(size=11, color="#64748b")), range=[-0.03, 1.03], gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        plot_bgcolor="#040508",
        paper_bgcolor="#040508",
        margin=dict(l=45, r=55, t=15, b=40),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig_compass, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col_ensemble:
    preds = results["model_predictions"]
    vals = list(preds.values())
    spread = (max(vals) - min(vals)) if vals else 0.0

    st.markdown(f"""
    <div class="studio-panel" style="padding-bottom: 8px;">
        <div class="panel-header">
            <div>
                <span class="panel-title">Model Consensus Benchmark</span>
                <div class="panel-subtitle">Cross-Architecture Perception Agreement</div>
            </div>
            <span class="chip">Spread: ±{spread/2:.2f}</span>
        </div>
    """, unsafe_allow_html=True)

    color_map = {
        "Hybrid Fusion (Champion SOTA)": "#00e5ff",
        "2D-CNN (Deep ConvNet)": "#b85bf8",
        "Random Forest (Acoustic Ensemble)": "#00e699",
        "Ridge (Linear Baseline)": "#64748b"
    }
    models_list = []
    for arch_name, score in preds.items():
        if "Hybrid" in arch_name:
            display_name = "Hybrid Fusion (Champion SOTA)"
        elif "CNN" in arch_name:
            display_name = "2D-CNN (Deep ConvNet)"
        elif "Random" in arch_name:
            display_name = "Random Forest (Acoustic Ensemble)"
        else:
            display_name = "Ridge (Linear Baseline)"
        models_list.append({"Architecture": display_name, "Valence": float(score)})

    df_models = pd.DataFrame(models_list)

    fig_bar = px.bar(
        df_models,
        x="Valence",
        y="Architecture",
        orientation="h",
        color="Architecture",
        color_discrete_map=color_map,
        range_x=[0.0, 1.0],
        text_auto=".3f"
    )
    fig_bar.update_layout(
        plot_bgcolor="#040508",
        paper_bgcolor="#040508",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title=dict(text="Valence Score", font=dict(size=11, color="#64748b"))),
        yaxis=dict(title="", tickfont=dict(size=11)),
        margin=dict(l=10, r=20, t=10, b=40),
        height=320,
        showlegend=False
    )
    fig_bar.update_traces(marker_line_width=0, opacity=0.95)
    st.plotly_chart(fig_bar, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

spec_info = results.get("spectrogram")
if spec_info and "matrix" in spec_info:
    st.markdown("""
    <div class="studio-panel">
        <div class="panel-header">
            <div>
                <span class="panel-title">2D Log-Mel Spectrogram (Acoustic Frequency Heatmap)</span>
                <div class="panel-subtitle">Visual time-frequency representation feeding the PyTorch 2D-CNN feature extractor.</div>
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="chip">128 Mel Bins</span>
                <span class="chip">22.05 kHz</span>
                <span class="chip">Slaney Scale</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    mel_tickvals = [0, 20, 42, 68, 98, 127]
    mel_ticktext = ["0 Hz", "500 Hz", "1.1 kHz", "2.2 kHz", "5.0 kHz", "11.0 kHz"]

    matrix = np.array(spec_info["matrix"])
    n_mels, n_frames = matrix.shape
    dur = float(spec_info.get("duration_sec", 6.0))
    time_axis = np.linspace(0.0, dur, n_frames)

    fig_spec = go.Figure(data=go.Heatmap(
        z=matrix,
        x=time_axis,
        y=np.arange(n_mels),
        colorscale="Magma",
        colorbar=dict(
            title=dict(text="dB", font=dict(size=10, color="#94a3b8")),
            tickfont=dict(size=9, color="#64748b"),
            len=0.8,
            thickness=10,
            bgcolor="#040508",
            outlinewidth=0
        )
    ))
    fig_spec.update_layout(
        plot_bgcolor="#040508",
        paper_bgcolor="#040508",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(title=dict(text="Timeline (Seconds)", font=dict(size=11, color="#64748b")), gridcolor="rgba(255,255,255,0.02)"),
        yaxis=dict(title=dict(text="Frequency (Hz, Mel Scale)", font=dict(size=11, color="#64748b")), tickvals=mel_tickvals, ticktext=mel_ticktext, gridcolor="rgba(255,255,255,0.02)"),
        margin=dict(l=40, r=20, t=10, b=40),
        height=220
    )
    st.plotly_chart(fig_spec, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

trajectory = results.get("trajectory", [])
if trajectory and len(trajectory) > 1:
    st.markdown("""
    <div class="studio-panel">
        <div class="panel-header">
            <div>
                <span class="panel-title">Dynamic Emotion Trajectory Across Timeline</span>
                <div class="panel-subtitle">3-second rolling analysis windows capturing emotional progression from intro to drop.</div>
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="chip">Hop: 1.5s</span>
                <span class="chip">Baseline: 0.50</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    df_traj = pd.DataFrame(trajectory)
    min_v = float(df_traj["valence"].min())
    max_v = float(df_traj["valence"].max())
    diff = max_v - min_v
    pad = 0.03 if diff < 0.05 else 0.08
    y_range = [max(0.0, min_v - pad), min(1.0, max_v + pad)]

    fig_traj = go.Figure()

    fig_traj.add_trace(go.Scatter(
        x=df_traj["time_start_sec"],
        y=df_traj["valence"],
        mode="lines+markers",
        line=dict(color="#00e5ff", width=3, shape="spline"),
        marker=dict(size=8, color="#ffffff", line=dict(color="#00e5ff", width=2)),
        fill="tozeroy",
        fillcolor="rgba(0, 229, 255, 0.08)",
        hovertemplate="Timestamp: %{x:.1f}s<br>Valence: %{y:.4f}<extra></extra>",
        name="Valence Track"
    ))

    fig_traj.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(255, 255, 255, 0.15)",
        annotation_text="Neutral Line (0.50)",
        annotation_font=dict(color="#64748b", size=10)
    )

    fig_traj.update_layout(
        plot_bgcolor="#040508",
        paper_bgcolor="#040508",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        yaxis=dict(title=dict(text="Valence", font=dict(size=11, color="#64748b")), range=y_range, gridcolor="rgba(255,255,255,0.03)"),
        xaxis=dict(title=dict(text="Timeline (Seconds)", font=dict(size=11, color="#64748b")), gridcolor="rgba(255,255,255,0.03)"),
        height=270,
        margin=dict(l=40, r=40, t=15, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_traj, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Short audio clip: Evaluated as a single temporal frame.")

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

gradcam_res = results.get("gradcam")
if gradcam_res and "raising_map" in gradcam_res:
    st.markdown("""
    <div class="studio-panel">
        <div class="panel-header">
            <div>
                <span class="panel-title">Neural Attention & Grad-CAM Explainability</span>
                <div class="panel-subtitle">Spatial activations in the deep 2D-CNN feature extractor showing time-frequency regions that raise or lower valence.</div>
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="chip">Pre-Sigmoid Attribution</span>
                <span class="chip">Block 4 Conv Layer</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab_raising, tab_lowering, tab_bipolar, tab_summary = st.tabs([
        "Valence-Raising Regions (Positive Evidence)",
        "Valence-Lowering Regions (Negative Evidence)",
        "Bipolar Gradient Map (Net Attribution)",
        "Acoustic Band Attribution Telemetry"
    ])

    raising_mat = np.array(gradcam_res["raising_map"])
    lowering_mat = np.array(gradcam_res["lowering_map"])
    signed_mat = np.array(gradcam_res.get("signed_map", gradcam_res["raising_map"]))
    h_mels, w_frames = raising_mat.shape
    t_axis = np.linspace(0.0, 3.0, w_frames)

    cs_raise = [
        [0.0, "#040508"],
        [0.2, "#3b0707"],
        [0.5, "#dc2626"],
        [0.8, "#ff3366"],
        [1.0, "#fca5a5"]
    ]
    cs_lower = [
        [0.0, "#040508"],
        [0.2, "#082f49"],
        [0.5, "#0284c7"],
        [0.8, "#00e5ff"],
        [1.0, "#bae6fd"]
    ]
    cs_bipolar = [
        [0.0, "#00e5ff"],
        [0.35, "#082f49"],
        [0.5, "#040508"],
        [0.65, "#450a0a"],
        [1.0, "#ff3366"]
    ]

    with tab_raising:
        fig_raise = go.Figure(data=go.Heatmap(
            z=raising_mat,
            x=t_axis,
            y=np.arange(h_mels),
            colorscale=cs_raise,
            colorbar=dict(
                title=dict(text="Importance", font=dict(size=10, color="#ff3366")),
                tickfont=dict(size=9, color="#64748b"),
                len=0.8,
                thickness=10,
                bgcolor="#040508",
                outlinewidth=0
            )
        ))
        fig_raise.update_layout(
            plot_bgcolor="#040508",
            paper_bgcolor="#040508",
            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
            xaxis=dict(title=dict(text="Window Timeline (Seconds)", font=dict(size=11, color="#64748b")), gridcolor="rgba(255,255,255,0.02)"),
            yaxis=dict(title=dict(text="Frequency (Hz, Mel Scale)", font=dict(size=11, color="#64748b")), tickvals=mel_tickvals, ticktext=mel_ticktext, gridcolor="rgba(255,255,255,0.02)"),
            margin=dict(l=40, r=20, t=10, b=40),
            height=220
        )
        st.plotly_chart(fig_raise, width="stretch")

    with tab_lowering:
        fig_lower = go.Figure(data=go.Heatmap(
            z=lowering_mat,
            x=t_axis,
            y=np.arange(h_mels),
            colorscale=cs_lower,
            colorbar=dict(
                title=dict(text="Importance", font=dict(size=10, color="#00e5ff")),
                tickfont=dict(size=9, color="#64748b"),
                len=0.8,
                thickness=10,
                bgcolor="#040508",
                outlinewidth=0
            )
        ))
        fig_lower.update_layout(
            plot_bgcolor="#040508",
            paper_bgcolor="#040508",
            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
            xaxis=dict(title=dict(text="Window Timeline (Seconds)", font=dict(size=11, color="#64748b")), gridcolor="rgba(255,255,255,0.02)"),
            yaxis=dict(title=dict(text="Frequency (Hz, Mel Scale)", font=dict(size=11, color="#64748b")), tickvals=mel_tickvals, ticktext=mel_ticktext, gridcolor="rgba(255,255,255,0.02)"),
            margin=dict(l=40, r=20, t=10, b=40),
            height=220
        )
        st.plotly_chart(fig_lower, width="stretch")

    with tab_bipolar:
        max_abs = float(max(abs(signed_mat.min()), abs(signed_mat.max()), 1e-6))
        fig_bipolar = go.Figure(data=go.Heatmap(
            z=signed_mat,
            x=t_axis,
            y=np.arange(h_mels),
            colorscale=cs_bipolar,
            zmin=-max_abs,
            zmax=max_abs,
            colorbar=dict(
                title=dict(text="Net Shift", font=dict(size=10, color="#94a3b8")),
                tickfont=dict(size=9, color="#64748b"),
                len=0.8,
                thickness=10,
                bgcolor="#040508",
                outlinewidth=0
            )
        ))
        fig_bipolar.update_layout(
            plot_bgcolor="#040508",
            paper_bgcolor="#040508",
            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
            xaxis=dict(title=dict(text="Window Timeline (Seconds)", font=dict(size=11, color="#64748b")), gridcolor="rgba(255,255,255,0.02)"),
            yaxis=dict(title=dict(text="Frequency (Hz, Mel Scale)", font=dict(size=11, color="#64748b")), tickvals=mel_tickvals, ticktext=mel_ticktext, gridcolor="rgba(255,255,255,0.02)"),
            margin=dict(l=40, r=20, t=10, b=40),
            height=220
        )
        st.plotly_chart(fig_bipolar, width="stretch")

    with tab_summary:
        summary_info = gradcam_res.get("summary", {})
        summary_text = summary_info.get("text", "Grad-CAM sensitivity analysis evaluated.")
        raising_bands = summary_info.get("raising_band_shares", {})
        lowering_bands = summary_info.get("lowering_band_shares", {})

        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px 14px; margin-bottom: 14px;">
            <div style="font-size: 10.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px;">Diagnostic Inference Summary</div>
            <div style="font-size: 13px; color: #e2e8f0; line-height: 1.5; font-family:'JetBrains Mono';">{summary_text}</div>
        </div>
        """, unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("<div style='font-size:12px; font-weight:700; color:#ff3366; margin-bottom:8px;'>Positive Valence Band Distribution</div>", unsafe_allow_html=True)
            for band_name, share in raising_bands.items():
                pct = share * 100
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; font-size:11.5px; padding:2px 0;">
                    <span style="color:#94a3b8;">{band_name}</span>
                    <span style="color:#ffffff; font-family:'JetBrains Mono'; font-weight:600;">{pct:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); height: 3px; border-radius: 2px; margin-bottom: 7px; overflow: hidden;">
                    <div style="background: #ff3366; width: {pct}%; height: 100%;"></div>
                </div>
                """, unsafe_allow_html=True)

        with col_b2:
            st.markdown("<div style='font-size:12px; font-weight:700; color:#00e5ff; margin-bottom:8px;'>Negative Valence Band Distribution</div>", unsafe_allow_html=True)
            for band_name, share in lowering_bands.items():
                pct = share * 100
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; font-size:11.5px; padding:2px 0;">
                    <span style="color:#94a3b8;">{band_name}</span>
                    <span style="color:#ffffff; font-family:'JetBrains Mono'; font-weight:600;">{pct:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); height: 3px; border-radius: 2px; margin-bottom: 7px; overflow: hidden;">
                    <div style="background: #00e5ff; width: {pct}%; height: 100%;"></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 10px; font-size: 11px; color: #64748b; font-style: italic;">
            Validation Notice: Attributions describe model sensitivity in time and frequency. They do not identify individual instruments or causes of perceived human emotion.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

with st.expander("View Detailed Acoustic Physics DNA (34 Descriptors)"):
    st.markdown("##### Core Acoustic Driver Highlights")
    cols = st.columns(len(results["acoustic_highlights"]))
    for col, (k, v) in zip(cols, results["acoustic_highlights"].items()):
        col.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px;">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; font-weight: 700;">{k.split('(')[0].strip()}</div>
            <div style="font-size: 15px; color: #00e5ff; font-weight: 700; font-family: 'JetBrains Mono'; margin-top: 3px;">{v}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Extracted 34 Physics Features (Full Vector)")
    df_features = pd.DataFrame([results["features"]]).T.reset_index()
    df_features.columns = ["Acoustic Feature Name", "Value"]
    st.dataframe(df_features, height=220, width="stretch")

st.markdown("""
<div style="text-align: center; color: #475569; font-size: 11.5px; margin-top: 40px; padding: 24px 0 16px 0; border-top: 1px solid rgba(255,255,255,0.04);">
    <div style="font-weight: 700; color: #94a3b8; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">
        HarmoniX Neural Audio Intelligence
    </div>
    <div style="color: #64748b; font-size: 11px; max-width: 680px; margin: 0 auto; line-height: 1.6;">
        Bridging acoustic physics and deep neural perception to decode the affective geometry of music across diverse cultures. Powered by PyTorch 2D-CNN, hybrid multimodal fusion, and explainable AI attribution.
    </div>
</div>
""", unsafe_allow_html=True)
