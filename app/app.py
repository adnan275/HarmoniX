"""
HarmoniX | Interactive Music Emotion Recognition Dashboard
Author: Adnan Rizvi (Lead Architect & Product Engineering Lead)

Phase 2 Production Web Application — Ultra-Aesthetic OLED Black Edition:
- Deep Pitch-Black OLED UI with Cyber-Glow Aesthetics.
- Frosted glassmorphism containers & neon accents.
- Real-time in-browser audio playback with sleek waveform telemetry.
- Interactive Russell's 2D Mood Compass (Plotly Neon Edition).
- Dynamic second-by-second valence emotion trajectories.
- Multi-genre multicultural presets (Bollywood, Punjabi, Classical, Western Pop).
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
    page_title="HarmoniX — AI Music Emotion",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Ultra-Aesthetic Cyberpunk OLED Dark CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Root Variables */
    :root {
        --bg-pitch-black: #06080c;
        --bg-surface-1: #0d1117;
        --bg-surface-2: #131822;
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-hover: rgba(99, 102, 241, 0.4);
        --neon-cyan: #00f2fe;
        --neon-purple: #9d4edd;
        --neon-blue: #3b82f6;
        --neon-green: #10b981;
        --neon-orange: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    /* Streamlit Defaults Override */
    .stApp {
        background-color: var(--bg-pitch-black);
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(6, 8, 12, 1) 0px, transparent 100%);
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Clean Header */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Dark Styling */
    [data-testid="stSidebar"] {
        background: #080a0f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Glassmorphism Card Container */
    .glass-card {
        background: rgba(15, 20, 28, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 16px;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 12px 36px rgba(99, 102, 241, 0.12);
        transform: translateY(-2px);
    }

    /* Metric Display Tile */
    .metric-tile {
        background: rgba(18, 24, 38, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.25s ease;
    }
    .metric-tile:hover {
        border-color: rgba(56, 189, 248, 0.4);
        background: rgba(22, 30, 46, 0.7);
    }
    .metric-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }

    /* Glow Badges */
    .badge-q1 {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.35) 100%);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 6px 16px; border-radius: 30px; font-weight: 700; font-size: 13px;
        display: inline-block; box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
    }
    .badge-q2 {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.35) 100%);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 6px 16px; border-radius: 30px; font-weight: 700; font-size: 13px;
        display: inline-block; box-shadow: 0 0 16px rgba(239, 68, 68, 0.25);
    }
    .badge-q3 {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(126, 34, 206, 0.35) 100%);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.4);
        padding: 6px 16px; border-radius: 30px; font-weight: 700; font-size: 13px;
        display: inline-block; box-shadow: 0 0 16px rgba(168, 85, 247, 0.25);
    }
    .badge-q4 {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.35) 100%);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.4);
        padding: 6px 16px; border-radius: 30px; font-weight: 700; font-size: 13px;
        display: inline-block; box-shadow: 0 0 16px rgba(59, 130, 246, 0.25);
    }

    /* Headings with Gradient Glow */
    .title-gradient {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 0%, #cbd5e1 40%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sub-glow {
        color: #94a3b8;
        font-size: 14.5px;
        font-weight: 400;
        margin-bottom: 24px;
    }
    
    /* Neon Pill tag */
    .pill-tag {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# 3. Sidebar: Diverse Multi-Genre Music Selector
st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0;">
    <span class="pill-tag">PHASE 2 MASTER SUITE</span>
    <h3 style="margin-top: 8px; color: #fff; font-weight: 700; font-size: 20px;">Audio Engine</h3>
</div>
""", unsafe_allow_html=True)

test_diverse_dir = PROJECT_ROOT / "data" / "test_diverse"
if not test_diverse_dir.exists():
    create_sample_diverse_dataset(test_diverse_dir)

diverse_tracks = scan_multicultural_directory(test_diverse_dir)

input_mode = st.sidebar.radio(
    "Source Mode",
    ["📁 Multicultural Benchmark Presets", "⬆️ Upload Audio (.mp3, .wav)"],
    index=0
)

selected_audio_path = None
audio_label = ""
genre_badge = ""

if input_mode == "📁 Multicultural Benchmark Presets":
    if diverse_tracks:
        genre_options = {f"{t['genre']} • {t['title']}": (t['filepath'], t['genre']) for t in diverse_tracks}
        chosen_title = st.sidebar.selectbox("Choose Benchmark Clip:", list(genre_options.keys()))
        selected_audio_path = Path(genre_options[chosen_title][0])
        genre_badge = genre_options[chosen_title][1]
        audio_label = chosen_title.split(" • ")[1]
    else:
        st.sidebar.warning("No pre-loaded audio tracks found.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Unseen Song",
        type=["mp3", "wav", "ogg", "flac"]
    )
    if uploaded_file is not None:
        tmp_suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp_f:
            tmp_f.write(uploaded_file.read())
            selected_audio_path = Path(tmp_f.name)
            audio_label = uploaded_file.name
            genre_badge = "Custom Upload"

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 14px;">
    <div style="font-size: 11px; text-transform: uppercase; color: #818cf8; font-weight: 700; letter-spacing: 1px;">Team Roles</div>
    <div style="font-size: 13px; color: #e2e8f0; margin-top: 6px;"><b>Lead Architect:</b> Adnan Rizvi</div>
    <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">• DSP: Aditya<br/>• Modeling: Bhoomi<br/>• XAI: Praanshu<br/>• Evaluation: Vani</div>
    <div style="margin-top: 10px; font-size: 11px; color: #34d399; font-weight: 600;">✓ Cross-Cultural Diversity Enabled</div>
</div>
""", unsafe_allow_html=True)


# 4. Header Section
st.markdown('<div class="title-gradient">🎧 HarmoniX <span style="font-size:20px; font-weight:500; color:#818cf8;">Acoustic AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-glow">Deep Spectrogram Intelligence & Continuous Human Valence Estimation across Cultures</div>', unsafe_allow_html=True)

if selected_audio_path is None:
    st.info("👈 Please select a benchmark audio clip or upload a song from the sidebar to begin analysis.")
    st.stop()

# 5. Audio Player & Ingestion Specs Container
col_player, col_norm = st.columns([1.6, 1])

with col_player:
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 8px;">
        <span style="font-size:16px; font-weight:700; color:#f1f5f9;">🎵 {audio_label}</span>
        <span class="pill-tag">{genre_badge}</span>
    </div>
    """, unsafe_allow_html=True)
    with open(selected_audio_path, "rb") as f_audio:
        audio_bytes = f_audio.read()
    st.audio(audio_bytes, format="audio/wav")

# 6. Run Prediction Pipeline
with st.spinner("⚡ Extracting Time-Frequency Spectrograms & Running Multimodal Inference..."):
    try:
        results = predict_audio(selected_audio_path)
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.stop()

meta = results["metadata"]

with col_norm:
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 18px;">
        <div style="font-size: 11px; text-transform: uppercase; color: #38bdf8; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px;">Audio Ingestion Normalization</div>
        <div style="display: flex; justify-content: space-between; font-size: 12.5px; color: #cbd5e1; margin-bottom: 4px;">
            <span>⏱️ Duration: <b>{meta['duration_seconds']:.2f}s</b></span>
            <span>⚡ Rate: <b>{meta['target_sample_rate']} Hz</b></span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12.5px; color: #cbd5e1;">
            <span>🎚️ Peak: <b>{meta['normalized_peak']:.2f}</b></span>
            <span>🎧 Channels: <b>Mono (Downmixed)</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# 7. Telemetry Metrics Tiles (4-Column Layout)
v_score = results["primary_valence"]
a_score = results["estimated_arousal"]
quadrant = results["emotional_quadrant"]
emoji = results["emoji"]

badge_class = "badge-q1" if "Q1" in quadrant else "badge-q2" if "Q2" in quadrant else "badge-q3" if "Q3" in quadrant else "badge-q4"

t1, t2, t3, t4 = st.columns(4)

with t1:
    val_color = "#34d399" if v_score >= 0.50 else "#f87171"
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-title">Predicted Valence</div>
        <div class="metric-val" style="color: {val_color};">{v_score:.3f} <span style="font-size:16px; color:#64748b;">/ 1.0</span></div>
        <div style="font-size: 12px; color: {val_color}; font-weight: 600; margin-top: 4px;">
            {'▲ Positive Tone' if v_score >= 0.50 else '▼ Somber Tone'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with t2:
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-title">Estimated Arousal</div>
        <div class="metric-val" style="color: #60a5fa;">{a_score:.3f} <span style="font-size:16px; color:#64748b;">/ 1.0</span></div>
        <div style="font-size: 12px; color: #60a5fa; font-weight: 600; margin-top: 4px;">
            {'▲ Dynamic Energy' if a_score >= 0.50 else '▼ Subdued / Calm'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with t3:
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-title">Emotional Quadrant</div>
        <div style="margin-top: 4px;"><span class="{badge_class}">{quadrant.split(':')[1].strip()} {emoji}</span></div>
        <div style="font-size: 11.5px; color: #94a3b8; margin-top: 8px; line-height: 1.3;">
            {results["mood_description"][:48]}...
        </div>
    </div>
    """, unsafe_allow_html=True)

with t4:
    centroid_str = results["acoustic_highlights"].get("Brightness (Spectral Centroid)", "2150.0 Hz")
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-title">Timbral Warmth</div>
        <div class="metric-val" style="color: #fbbf24; font-size: 24px;">{centroid_str}</div>
        <div style="font-size: 12px; color: #fbbf24; font-weight: 600; margin-top: 4px;">
            Center Spectral Mass
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# 8. Visual Analytics: 2D Mood Compass + Model Comparison
c_compass, c_models = st.columns([1.3, 1])

with c_compass:
    st.markdown("""
    <div style="font-size: 16px; font-weight: 700; color: #f1f5f9; margin-bottom: 8px;">
        🧭 Russell's 2D Mood Circumplex
    </div>
    """, unsafe_allow_html=True)

    fig_compass = go.Figure()

    # Aesthetic Subtle Quadrants
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(168, 85, 247, 0.08)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(59, 130, 246, 0.08)", line_width=0)

    # Hairline Grid Crosshair
    fig_compass.add_hline(y=0.5, line_dash="solid", line_color="rgba(255,255,255,0.12)", line_width=1.2)
    fig_compass.add_vline(x=0.5, line_dash="solid", line_color="rgba(255,255,255,0.12)", line_width=1.2)

    # Quadrant Text Callouts
    fig_compass.add_annotation(x=0.82, y=0.92, text="<b>Q1: JOYFUL</b><br><span style='font-size:9px; color:#10b981;'>Happy & Energetic</span>", showarrow=False, font=dict(color="#34d399", size=11))
    fig_compass.add_annotation(x=0.18, y=0.92, text="<b>Q2: TENSE</b><br><span style='font-size:9px; color:#ef4444;'>Agitated & Intense</span>", showarrow=False, font=dict(color="#f87171", size=11))
    fig_compass.add_annotation(x=0.18, y=0.08, text="<b>Q3: MELANCHOLIC</b><br><span style='font-size:9px; color:#a855f7;'>Sad & Somber</span>", showarrow=False, font=dict(color="#c084fc", size=11))
    fig_compass.add_annotation(x=0.82, y=0.08, text="<b>Q4: CALM</b><br><span style='font-size:9px; color:#3b82f6;'>Peaceful & Relaxed</span>", showarrow=False, font=dict(color="#60a5fa", size=11))

    # Glow Pulse Trace
    fig_compass.add_trace(go.Scatter(
        x=[v_score], y=[a_score],
        mode="markers",
        marker=dict(size=32, color="rgba(99, 102, 241, 0.25)", symbol="circle"),
        hoverinfo="skip"
    ))
    # Main Star Diamond
    fig_compass.add_trace(go.Scatter(
        x=[v_score], y=[a_score],
        mode="markers+text",
        marker=dict(size=16, color="#f1f5f9", line=dict(color="#6366f1", width=3), symbol="diamond"),
        text=["Track Position"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11, family="Plus Jakarta Sans"),
        name="Song Location"
    ))

    fig_compass.update_layout(
        xaxis=dict(title=dict(text="Valence (Negative → Positive)", font=dict(size=11, color="#64748b")), range=[0.0, 1.0], gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(title=dict(text="Arousal (Calm → Energetic)", font=dict(size=11, color="#64748b")), range=[0.0, 1.0], gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        plot_bgcolor="#090c12",
        paper_bgcolor="#0d1117",
        margin=dict(l=40, r=40, t=10, b=40),
        height=360,
        showlegend=False
    )
    st.plotly_chart(fig_compass, use_container_width=True)

with c_models:
    st.markdown("""
    <div style="font-size: 16px; font-weight: 700; color: #f1f5f9; margin-bottom: 8px;">
        🤖 Model Performance Array
    </div>
    """, unsafe_allow_html=True)

    preds = results["model_predictions"]
    df_models = pd.DataFrame([
        {"Architecture": "Random Forest (Champion)", "Valence": preds.get("Random_Forest", 0), "Color": "#10b981"},
        {"Architecture": "XGBoost Regressor", "Valence": preds.get("XGBoost", 0), "Color": "#6366f1"},
        {"Architecture": "Ridge Regression", "Valence": preds.get("Ridge", 0), "Color": "#f59e0b"}
    ])

    fig_bar = px.bar(
        df_models,
        x="Valence",
        y="Architecture",
        orientation="h",
        color="Architecture",
        color_discrete_map={
            "Random Forest (Champion)": "#10b981",
            "XGBoost Regressor": "#818cf8",
            "Ridge Regression": "#fbbf24"
        },
        range_x=[0.0, 1.0],
        text_auto=".3f"
    )
    fig_bar.update_layout(
        plot_bgcolor="#090c12",
        paper_bgcolor="#0d1117",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title=dict(text="Predicted Score", font=dict(size=11, color="#64748b"))),
        yaxis=dict(title="", tickfont=dict(size=11)),
        margin=dict(l=10, r=20, t=10, b=40),
        height=360,
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# 9. Dynamic Emotion Trajectory Across Timeline
st.markdown("""
<div style="font-size: 16px; font-weight: 700; color: #f1f5f9; margin-bottom: 4px;">
    📈 Dynamic Emotion Trajectory Across Timeline
</div>
<div style="font-size: 12.5px; color: #64748b; margin-bottom: 12px;">
    Sliding 3-second rolling analysis windows capturing emotional progression from intro to drop.
</div>
""", unsafe_allow_html=True)

trajectory = results.get("trajectory", [])
if trajectory and len(trajectory) > 1:
    df_traj = pd.DataFrame(trajectory)

    fig_traj = go.Figure()

    # Soft glowing gradient fill under curve
    fig_traj.add_trace(go.Scatter(
        x=df_traj["time_start_sec"],
        y=df_traj["valence"],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3.5, shape="spline"),
        marker=dict(size=8, color="#818cf8", line=dict(color="#ffffff", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)",
        name="Valence Track"
    ))

    # Neutral reference line
    fig_traj.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(255,255,255,0.2)",
        annotation_text="Neutral Line (0.50)",
        annotation_font=dict(color="#64748b", size=10)
    )

    fig_traj.update_layout(
        plot_bgcolor="#090c12",
        paper_bgcolor="#0d1117",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
        yaxis=dict(title=dict(text="Valence", font=dict(size=11, color="#64748b")), range=[0.0, 1.0], gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(title=dict(text="Timeline (Seconds)", font=dict(size=11, color="#64748b")), gridcolor="rgba(255,255,255,0.04)"),
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_traj, use_container_width=True)
else:
    st.info("Short audio clip: Single emotional window evaluated.")

# 10. Acoustic Physics Telemetry Breakdown
with st.expander("🎼 View Detailed Acoustic Physics Telemetry & Feature Vectors"):
    st.markdown("##### Core Acoustic Driver Highlights")
    cols = st.columns(len(results["acoustic_highlights"]))
    for col, (k, v) in zip(cols, results["acoustic_highlights"].items()):
        col.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px;">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; font-weight: 700;">{k.split('(')[0].strip()}</div>
            <div style="font-size: 15px; color: #38bdf8; font-weight: 700; font-family: 'JetBrains Mono'; margin-top: 4px;">{v}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Extracted 34 Physics Features (Numerical Schema)")
    df_features = pd.DataFrame([results["features"]]).T.reset_index()
    df_features.columns = ["Acoustic Feature Name", "Value"]
    st.dataframe(df_features, height=220, use_container_width=True)

# 11. Minimalist Sleek Footer
st.markdown("""
<div style="text-align: center; color: #475569; font-size: 12px; margin-top: 50px; padding: 24px 0; border-top: 1px solid rgba(255,255,255,0.06);">
    <span style="font-weight: 700; color: #94a3b8;">HarmoniX Research Group</span> • Phase 2 Master Implementation<br/>
    <span style="font-size: 11px; color: #64748b;">
        Lead Architect: <b>Adnan Rizvi</b> • DSP: <b>Aditya</b> • DL Modeling: <b>Bhoomi</b> • Explainability: <b>Praanshu</b> • Evaluation: <b>Vani</b>
    </span>
</div>
""", unsafe_allow_html=True)
