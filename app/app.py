"""
HarmoniX | Interactive Music Emotion Recognition Dashboard
Author: Adnan Rizvi (Lead Architect & Product Engineering Lead)

Phase 2 Production Web Application:
- Multi-genre audio ingestion (Bollywood, Punjabi, Classical, Western Pop).
- Real-time in-browser audio playback.
- Interactive Russell's 2D Mood Compass (Plotly).
- Dynamic second-by-second valence emotion trajectories.
- Multi-model benchmarking and acoustic physics explainability.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline_orchestrator import predict_audio, get_orchestrator
from src.multicultural_loader import scan_multicultural_directory, create_sample_diverse_dataset

# 1. Page Configuration
st.set_page_config(
    page_title="HarmoniX — AI Music Emotion Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Dark Glassmorphic Theme CSS
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0a0e14 100%);
        color: #f0f6fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Card Containers */
    .metric-card {
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    
    .badge-joyful {
        background: linear-gradient(90deg, #1f6feb 0%, #238636 100%);
        color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;
    }
    .badge-calm {
        background: linear-gradient(90deg, #238636 0%, #177245 100%);
        color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;
    }
    .badge-melancholic {
        background: linear-gradient(90deg, #388bfd 0%, #8957e5 100%);
        color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;
    }
    .badge-tense {
        background: linear-gradient(90deg, #da3633 0%, #d29922 100%);
        color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;
    }
    
    /* Title styling */
    .main-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .subtitle {
        color: #8b949e;
        font-size: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# 3. Sidebar: Multicultural Test Audio & Controls
st.sidebar.markdown("### 🎛️ Audio Input & Settings")

test_diverse_dir = PROJECT_ROOT / "data" / "test_diverse"
if not test_diverse_dir.exists():
    create_sample_diverse_dataset(test_diverse_dir)

diverse_tracks = scan_multicultural_directory(test_diverse_dir)

input_mode = st.sidebar.radio(
    "Choose Audio Source:",
    ["📁 Diverse Multi-Genre Presets", "⬆️ Upload Any Song (.mp3, .wav)"],
    index=0
)

selected_audio_path = None
audio_label = ""

if input_mode == "📁 Diverse Multi-Genre Presets":
    if diverse_tracks:
        options = {f"[{t['genre']}] {t['title']}": t['filepath'] for t in diverse_tracks}
        chosen_title = st.sidebar.selectbox("Select Benchmark Track:", list(options.keys()))
        selected_audio_path = Path(options[chosen_title])
        audio_label = chosen_title
    else:
        st.sidebar.warning("No pre-loaded diverse audio tracks found.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload audio file (MP3 / WAV / OGG)",
        type=["mp3", "wav", "ogg", "flac"]
    )
    if uploaded_file is not None:
        # Save uploaded file to temp
        tmp_suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp_f:
            tmp_f.write(uploaded_file.read())
            selected_audio_path = Path(tmp_f.name)
            audio_label = uploaded_file.name

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏛️ Phase 2 Architecture")
st.sidebar.info("""
**Lead Architect:** Adnan Rizvi  
**Engine:** Multicultural Ingestion + 3s Dynamic Windows + Hybrid Fusion  
**Target:** R² > 0.50 (Cross-Cultural Audio Generalization)
""")


# 4. Header Section
st.markdown('<div class="main-title">🎵 HarmoniX — Music Emotion Recognition & Acoustic Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Continuous Human Valence Estimation via Physics-Informed Descriptors & Deep Spectrogram Models</div>', unsafe_allow_html=True)

if selected_audio_path is None:
    st.info("👈 Please select a preset song or upload an audio file from the sidebar to begin analysis.")
    st.stop()

# 5. Audio Player & Ingestion Banner
col_audio, col_meta = st.columns([1.5, 1])

with col_audio:
    st.markdown(f"**Now Playing:** `{audio_label}`")
    with open(selected_audio_path, "rb") as f_audio:
        audio_bytes = f_audio.read()
    st.audio(audio_bytes, format="audio/wav")

# 6. Run Orchestrator Pipeline
with st.spinner("🔍 Running End-to-End Analysis (Acoustic Ingestion + Model Inference + Dynamic Trajectory)..."):
    try:
        results = predict_audio(selected_audio_path)
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.stop()

meta = results["metadata"]

with col_meta:
    st.markdown("""
    <div class="metric-card">
        <h5 style="margin-top:0; color:#58a6ff;">📊 Audio Normalization Specs</h5>
        <p style="margin:2px 0; font-size:13px;"><b>Duration:</b> {:.2f}s | <b>Sample Rate:</b> {} Hz</p>
        <p style="margin:2px 0; font-size:13px;"><b>Channels:</b> {} (Mono downmixed) | <b>Format:</b> {}</p>
        <p style="margin:2px 0; font-size:13px;"><b>Loudness Peak:</b> {:.2f} (Normalized)</p>
    </div>
    """.format(meta['duration_seconds'], meta['target_sample_rate'], meta['original_channels'], meta['original_format'], meta['normalized_peak']), unsafe_allow_html=True)

st.markdown("---")

# 7. Key Telemetry Metrics Row
v_score = results["primary_valence"]
a_score = results["estimated_arousal"]
quadrant = results["emotional_quadrant"]
emoji = results["emoji"]

badge_class = "badge-joyful" if "Q1" in quadrant else "badge-tense" if "Q2" in quadrant else "badge-melancholic" if "Q3" in quadrant else "badge-calm"

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="🎯 Predicted Valence (Pleasantness)",
        value=f"{v_score:.3f} / 1.0",
        delta="Positive" if v_score >= 0.50 else "-Negative"
    )

with m2:
    st.metric(
        label="⚡ Estimated Arousal (Energy)",
        value=f"{a_score:.3f} / 1.0",
        delta="High Energy" if a_score >= 0.50 else "-Subdued"
    )

with m3:
    st.markdown(f"**🎭 Emotional Quadrant**")
    st.markdown(f'<span class="{badge_class}">{quadrant} {emoji}</span>', unsafe_allow_html=True)
    st.caption(results["mood_description"])

with m4:
    centroid_str = results["acoustic_highlights"].get("Brightness (Spectral Centroid)", "N/A")
    st.metric(
        label="🎼 Timbral Warmth / Brightness",
        value=centroid_str
    )

st.markdown("<br/>", unsafe_allow_html=True)

# 8. Visual Analytics: 2D Mood Compass + Model Comparison
c_left, c_right = st.columns([1.2, 1])

with c_left:
    st.markdown("### 🧭 Russell's 2D Mood Compass")
    
    # Generate 2D Quadrant Plotly Figure
    fig_compass = go.Figure()
    
    # Quadrant Shading
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(35, 134, 54, 0.15)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(218, 54, 51, 0.15)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(137, 87, 229, 0.15)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(56, 139, 253, 0.15)", line_width=0)
    
    # Neutral Grid Lines
    fig_compass.add_hline(y=0.5, line_dash="dash", line_color="#8b949e", line_width=1)
    fig_compass.add_vline(x=0.5, line_dash="dash", line_color="#8b949e", line_width=1)
    
    # Quadrant Text Labels
    fig_compass.add_annotation(x=0.85, y=0.92, text="<b>Q1: JOYFUL</b><br>Energetic & Happy", showarrow=False, font=dict(color="#238636", size=11))
    fig_compass.add_annotation(x=0.15, y=0.92, text="<b>Q2: TENSE</b><br>Agitated & Intense", showarrow=False, font=dict(color="#da3633", size=11))
    fig_compass.add_annotation(x=0.15, y=0.08, text="<b>Q3: MELANCHOLIC</b><br>Sad & Somber", showarrow=False, font=dict(color="#8957e5", size=11))
    fig_compass.add_annotation(x=0.85, y=0.08, text="<b>Q4: CALM</b><br>Peaceful & Serene", showarrow=False, font=dict(color="#388bfd", size=11))
    
    # Current Track Location
    fig_compass.add_trace(go.Scatter(
        x=[v_score],
        y=[a_score],
        mode="markers+text",
        marker=dict(size=20, color="#f1e05a", line=dict(color="white", width=2), symbol="diamond"),
        name="Current Track",
        text=["Current Song"],
        textposition="top center",
        textfont=dict(color="white", size=12)
    ))
    
    fig_compass.update_layout(
        xaxis=dict(title="Valence (Negative → Positive)", range=[0.0, 1.0], gridcolor="#30363d"),
        yaxis=dict(title="Arousal (Calm → Energetic)", range=[0.0, 1.0], gridcolor="#30363d"),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        margin=dict(l=40, r=40, t=20, b=40),
        height=380,
        showlegend=False
    )
    st.plotly_chart(fig_compass, use_container_width=True)

with c_right:
    st.markdown("### 🤖 Model Benchmark Comparison")
    
    preds = results["model_predictions"]
    df_models = pd.DataFrame([
        {"Model": "Random Forest (Champion)", "Valence": preds.get("Random_Forest", 0), "Type": "Non-Linear Ensemble"},
        {"Model": "XGBoost Regressor", "Valence": preds.get("XGBoost", 0), "Type": "Gradient Boosting"},
        {"Model": "Ridge Regression", "Valence": preds.get("Ridge", 0), "Type": "Linear Baseline"}
    ])
    
    fig_bar = px.bar(
        df_models,
        x="Valence",
        y="Model",
        orientation="h",
        color="Valence",
        color_continuous_scale="Viridis",
        range_x=[0.0, 1.0],
        text_auto=".3f"
    )
    fig_bar.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font=dict(color="#f0f6fc"),
        margin=dict(l=20, r=20, t=20, b=40),
        height=380,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# 9. Dynamic Trajectory Chart (Second-by-Second Valence)
st.markdown("### 📈 Dynamic Second-by-Second Emotion Trajectory")
st.caption("Rolling 3-second analysis windows capturing how valence evolves through intros, verses, and choruses.")

trajectory = results.get("trajectory", [])
if trajectory and len(trajectory) > 1:
    df_traj = pd.DataFrame(trajectory)
    
    fig_traj = px.line(
        df_traj,
        x="time_start_sec",
        y="valence",
        markers=True,
        title="Valence Trajectory Across Track Timeline",
        labels={"time_start_sec": "Time (Seconds)", "valence": "Valence Score [0, 1]"}
    )
    
    fig_traj.add_hline(y=0.5, line_dash="dash", line_color="#8b949e", annotation_text="Neutral Baseline")
    fig_traj.update_traces(line_color="#58a6ff", line_width=3, marker_size=7, marker_color="#bc8cff")
    
    fig_traj.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font=dict(color="#f0f6fc"),
        yaxis=dict(range=[0.0, 1.0], gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"),
        height=320,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_traj, use_container_width=True)
else:
    st.info("Track is shorter than 3 seconds; whole clip evaluated as a single emotional frame.")

# 10. Acoustic Physics Telemetry Breakdown
with st.expander("🎼 View Detailed Acoustic Physics & Signal Telemetry"):
    st.markdown("#### Primary Acoustic Feature Indicators")
    cols = st.columns(len(results["acoustic_highlights"]))
    for col, (k, v) in zip(cols, results["acoustic_highlights"].items()):
        col.metric(label=k, value=v)
    
    st.markdown("---")
    st.markdown("#### Extracted 34 Physics Features (Full Descriptor Vector)")
    df_features = pd.DataFrame([results["features"]]).T.reset_index()
    df_features.columns = ["Acoustic Feature", "Value"]
    st.dataframe(df_features, height=250, use_container_width=True)

# 11. Footer
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 13px; margin-top: 40px; padding: 20px 0; border-top: 1px solid #30363d;">
    <b>HarmoniX Research Group</b> • Phase 2 Master Implementation<br/>
    Lead Architect & Product Engineering: <b>Adnan Rizvi</b> • DSP: <b>Aditya</b> • DL: <b>Bhoomi</b> • XAI: <b>Praanshu</b> • Evaluation: <b>Vani</b>
</div>
""", unsafe_allow_html=True)
