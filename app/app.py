"""
HarmoniX | Interactive Music Emotion Recognition Dashboard
Author: Adnan Rizvi (Lead Architect & Product Engineering Lead)

Minimalist Pitch-Black Dark Theme (Linear / Vercel Aesthetic)
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline_orchestrator import predict_audio
from src.multicultural_loader import scan_multicultural_directory, create_sample_diverse_dataset

# 1. Page Configuration
st.set_page_config(
    page_title="HarmoniX — Music Emotion AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Pure Pitch-Black Minimalist CSS (Vercel/Linear Dark Standard)
st.markdown("""
<style>
    /* Global Pure Black Reset */
    .stApp {
        background-color: #000000 !important;
        color: #ededed !important;
    }
    
    /* Clean container spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* Sidebar - Deep Matte Black */
    [data-testid="stSidebar"] {
        background-color: #050507 !important;
        border-right: 1px solid #18181b !important;
    }
    
    /* Hide Streamlit Chrome */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    #MainMenu, footer {visibility: hidden;}

    /* Premium Minimal Dark Cards */
    .dark-card {
        background: #09090b;
        border: 1px solid #18181b;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }

    .metric-box {
        background: #09090b;
        border: 1px solid #18181b;
        border-radius: 10px;
        padding: 14px 18px;
        transition: border-color 0.2s ease;
    }
    .metric-box:hover {
        border-color: #27272a;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #71717a;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
    }

    /* Minimal Status Badges */
    .badge-quadrant {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #27272a;
        background: #121215;
    }
    .badge-q1 { color: #34d399; border-color: rgba(52, 211, 153, 0.3); }
    .badge-q2 { color: #f87171; border-color: rgba(248, 113, 113, 0.3); }
    .badge-q3 { color: #a78bfa; border-color: rgba(167, 139, 250, 0.3); }
    .badge-q4 { color: #60a5fa; border-color: rgba(96, 165, 250, 0.3); }

    /* Clean Audio Player */
    audio {
        width: 100% !important;
        border-radius: 8px;
        filter: invert(1) hue-rotate(180deg);
    }

    /* Headers */
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .hero-sub {
        font-size: 14px;
        color: #71717a;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# 3. Sidebar Configuration
st.sidebar.markdown("### 🎵 Audio Selection")

test_diverse_dir = PROJECT_ROOT / "data" / "test_diverse"
if not test_diverse_dir.exists():
    create_sample_diverse_dataset(test_diverse_dir)

diverse_tracks = scan_multicultural_directory(test_diverse_dir)

input_mode = st.sidebar.radio(
    "Source Mode",
    ["📁 Multicultural Presets", "⬆️ Upload Audio (.mp3, .wav)"],
    index=0
)

selected_audio_path = None
audio_label = ""
genre_badge = ""

if input_mode == "📁 Multicultural Presets":
    if diverse_tracks:
        genre_options = {f"{t['genre']} • {t['title']}": (t['filepath'], t['genre']) for t in diverse_tracks}
        chosen_title = st.sidebar.selectbox("Track:", list(genre_options.keys()))
        selected_audio_path = Path(genre_options[chosen_title][0])
        genre_badge = genre_options[chosen_title][1]
        audio_label = chosen_title.split(" • ")[1]
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload audio file",
        type=["mp3", "wav", "ogg", "flac"]
    )
    if uploaded_file is not None:
        tmp_suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp_f:
            tmp_f.write(uploaded_file.read())
            selected_audio_path = Path(tmp_f.name)
            audio_label = uploaded_file.name
            genre_badge = "Custom"

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11px; color: #71717a; line-height: 1.6;">
    <b style="color: #d4d4d8;">HarmoniX Phase 2</b><br/>
    Lead Architect: Adnan Rizvi<br/>
    Engine: 2D Spectrograms + Hybrid Fusion<br/>
    Target: Cross-Cultural Emotion Recognition
</div>
""", unsafe_allow_html=True)


# 4. Header Section
st.markdown('<div class="hero-title">HarmoniX — Acoustic Emotion Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Continuous human valence & arousal prediction from raw sound waves</div>', unsafe_allow_html=True)

if selected_audio_path is None:
    st.info("Select a preset or upload an audio track from the sidebar to begin.")
    st.stop()


# 5. Audio Player & Ingestion Specs
col_player, col_norm = st.columns([1.6, 1])

with col_player:
    st.markdown(f"<div style='font-size:14px; font-weight:600; color:#e4e4e7; margin-bottom:6px;'>Now Analyzing: {audio_label} <span style='font-size:11px; color:#71717a; margin-left:8px;'>({genre_badge})</span></div>", unsafe_allow_html=True)
    with open(selected_audio_path, "rb") as f_audio:
        audio_bytes = f_audio.read()
    st.audio(audio_bytes, format="audio/wav")

# Run Pipeline
with st.spinner("Processing audio telemetry..."):
    try:
        results = predict_audio(selected_audio_path)
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

meta = results["metadata"]

with col_norm:
    st.markdown(f"""
    <div class="dark-card" style="padding: 12px 16px; margin-bottom: 0;">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #71717a; letter-spacing: 0.05em; margin-bottom: 6px;">Audio Normalization</div>
        <div style="display:flex; justify-content:space-between; font-size:12px; color:#d4d4d8; margin-bottom:2px;">
            <span>Duration: <b>{meta['duration_seconds']:.2f}s</b></span>
            <span>Sample Rate: <b>{meta['target_sample_rate']} Hz</b></span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px; color:#d4d4d8;">
            <span>Channels: <b>Mono (Downmixed)</b></span>
            <span>Peak: <b>{meta['normalized_peak']:.2f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# 6. Core Metrics Row
v_score = results["primary_valence"]
a_score = results["estimated_arousal"]
quadrant = results["emotional_quadrant"]
emoji = results["emoji"]

badge_class = "badge-q1" if "Q1" in quadrant else "badge-q2" if "Q2" in quadrant else "badge-q3" if "Q3" in quadrant else "badge-q4"

m1, m2, m3, m4 = st.columns(4)

with m1:
    v_color = "#34d399" if v_score >= 0.50 else "#f87171"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Predicted Valence</div>
        <div class="metric-value" style="color: {v_color};">{v_score:.3f}</div>
        <div style="font-size: 11px; color: #71717a; margin-top: 3px;">
            {'Positive Pleasantness' if v_score >= 0.50 else 'Somber / Melancholic'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Estimated Arousal</div>
        <div class="metric-value" style="color: #60a5fa;">{a_score:.3f}</div>
        <div style="font-size: 11px; color: #71717a; margin-top: 3px;">
            {'High Energy' if a_score >= 0.50 else 'Calm / Relaxed'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Emotional Quadrant</div>
        <div style="margin-top: 5px;">
            <span class="badge-quadrant {badge_class}">{quadrant.split(':')[1].strip()} {emoji}</span>
        </div>
        <div style="font-size: 11px; color: #71717a; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
            {results["mood_description"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    centroid_str = results["acoustic_highlights"].get("Brightness (Spectral Centroid)", "2150.0 Hz")
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Spectral Centroid</div>
        <div class="metric-value" style="color: #fbbf24;">{centroid_str}</div>
        <div style="font-size: 11px; color: #71717a; margin-top: 3px;">
            Timbral Warmth (Hz)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# 7. Visual Analytics: 2D Mood Compass + Model Comparison
c_compass, c_models = st.columns([1.3, 1])

with c_compass:
    st.markdown("<div style='font-size:14px; font-weight:600; color:#e4e4e7; margin-bottom:8px;'>Russell's 2D Mood Circumplex</div>", unsafe_allow_html=True)

    fig_compass = go.Figure()

    # Minimal Quadrant Tints on Pure Black
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(52, 211, 153, 0.05)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(248, 113, 113, 0.05)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(167, 139, 250, 0.05)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(96, 165, 250, 0.05)", line_width=0)

    # Clean Grid Axis Lines
    fig_compass.add_hline(y=0.5, line_dash="solid", line_color="#27272a", line_width=1)
    fig_compass.add_vline(x=0.5, line_dash="solid", line_color="#27272a", line_width=1)

    # Subdued Quadrant Text
    fig_compass.add_annotation(x=0.82, y=0.92, text="Q1: JOYFUL", showarrow=False, font=dict(color="#34d399", size=10, family="monospace"))
    fig_compass.add_annotation(x=0.18, y=0.92, text="Q2: TENSE", showarrow=False, font=dict(color="#f87171", size=10, family="monospace"))
    fig_compass.add_annotation(x=0.18, y=0.08, text="Q3: MELANCHOLIC", showarrow=False, font=dict(color="#a78bfa", size=10, family="monospace"))
    fig_compass.add_annotation(x=0.82, y=0.08, text="Q4: CALM", showarrow=False, font=dict(color="#60a5fa", size=10, family="monospace"))

    # Track Marker
    fig_compass.add_trace(go.Scatter(
        x=[v_score], y=[a_score],
        mode="markers+text",
        marker=dict(size=14, color="#ffffff", line=dict(color="#3b82f6", width=2), symbol="diamond"),
        text=["Current Song"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11),
        name="Song Location"
    ))

    fig_compass.update_layout(
        xaxis=dict(title=dict(text="Valence", font=dict(size=11, color="#71717a")), range=[0.0, 1.0], gridcolor="#18181b", zeroline=False),
        yaxis=dict(title=dict(text="Arousal", font=dict(size=11, color="#71717a")), range=[0.0, 1.0], gridcolor="#18181b", zeroline=False),
        plot_bgcolor="#050507",
        paper_bgcolor="#09090b",
        margin=dict(l=40, r=40, t=10, b=40),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig_compass, use_container_width=True)

with c_models:
    st.markdown("<div style='font-size:14px; font-weight:600; color:#e4e4e7; margin-bottom:8px;'>Model Ensemble Comparison</div>", unsafe_allow_html=True)

    preds = results["model_predictions"]
    df_models = pd.DataFrame([
        {"Architecture": "Random Forest (Best)", "Valence": preds.get("Random_Forest", 0)},
        {"Architecture": "XGBoost Regressor", "Valence": preds.get("XGBoost", 0)},
        {"Architecture": "Ridge Baseline", "Valence": preds.get("Ridge", 0)}
    ])

    fig_bar = px.bar(
        df_models,
        x="Valence",
        y="Architecture",
        orientation="h",
        color="Architecture",
        color_discrete_map={
            "Random Forest (Best)": "#10b981",
            "XGBoost Regressor": "#6366f1",
            "Ridge Baseline": "#71717a"
        },
        range_x=[0.0, 1.0],
        text_auto=".3f"
    )
    fig_bar.update_layout(
        plot_bgcolor="#050507",
        paper_bgcolor="#09090b",
        font=dict(color="#d4d4d8"),
        xaxis=dict(gridcolor="#18181b", title=dict(text="Valence Score", font=dict(size=11, color="#71717a"))),
        yaxis=dict(title="", tickfont=dict(size=11)),
        margin=dict(l=10, r=20, t=10, b=40),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# 8. Dynamic Emotion Trajectory
st.markdown("<div style='font-size:14px; font-weight:600; color:#e4e4e7; margin-bottom:2px;'>Dynamic Emotion Trajectory (Second-by-Second)</div>", unsafe_allow_html=True)
st.caption("Rolling 3-second time windows tracking valence progression over track timeline.")

trajectory = results.get("trajectory", [])
if trajectory and len(trajectory) > 1:
    df_traj = pd.DataFrame(trajectory)

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=df_traj["time_start_sec"],
        y=df_traj["valence"],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2.5, shape="spline"),
        marker=dict(size=6, color="#ffffff", line=dict(color="#38bdf8", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.05)",
        name="Valence"
    ))

    fig_traj.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#27272a",
        annotation_text="Neutral (0.50)",
        annotation_font=dict(color="#71717a", size=10)
    )

    fig_traj.update_layout(
        plot_bgcolor="#050507",
        paper_bgcolor="#09090b",
        font=dict(color="#d4d4d8"),
        yaxis=dict(title=dict(text="Valence", font=dict(size=11, color="#71717a")), range=[0.0, 1.0], gridcolor="#18181b"),
        xaxis=dict(title=dict(text="Timeline (Seconds)", font=dict(size=11, color="#71717a")), gridcolor="#18181b"),
        height=260,
        margin=dict(l=40, r=40, t=15, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_traj, use_container_width=True)
else:
    st.info("Short audio clip: Evaluated as a single temporal frame.")


# 9. Acoustic Feature Telemetry Drawer
with st.expander("🎼 Acoustic Signal Telemetry (34 Extracted Descriptors)"):
    cols = st.columns(len(results["acoustic_highlights"]))
    for col, (k, v) in zip(cols, results["acoustic_highlights"].items()):
        col.markdown(f"""
        <div style="background: #09090b; border: 1px solid #18181b; border-radius: 8px; padding: 10px 12px;">
            <div style="font-size: 10px; color: #71717a; text-transform: uppercase; font-weight: 600;">{k.split('(')[0].strip()}</div>
            <div style="font-size: 14px; color: #f4f4f5; font-weight: 700; margin-top: 3px;">{v}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    df_features = pd.DataFrame([results["features"]]).T.reset_index()
    df_features.columns = ["Descriptor", "Value"]
    st.dataframe(df_features, height=200, use_container_width=True)

# 10. Minimalist Footer
st.markdown("""
<div style="text-align: center; color: #52525b; font-size: 11px; margin-top: 40px; padding: 20px 0; border-top: 1px solid #18181b;">
    HarmoniX Research Group • Phase 2 Implementation • Lead Architect: Adnan Rizvi
</div>
""", unsafe_allow_html=True)
