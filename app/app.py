"""
HarmoniX | AI Music Emotion Recognition Studio
Author: Adnan Rizvi (Lead Architect & Product Engineering Lead)

Minimalist Studio Edition (No Emojis, Pure Dark Scientific Aesthetic)
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

st.set_page_config(
    page_title="HarmoniX Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Pitch Black Canvas */
    .stApp {
        background-color: #050608 !important;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.04) 0%, transparent 45%),
            radial-gradient(circle at 100% 100%, rgba(139, 92, 246, 0.03) 0%, transparent 40%);
        color: #f1f5f9 !important;
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1240px !important;
    }

    /* Hide Streamlit Header */
    header[data-testid="stHeader"], #MainMenu, footer {
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Sidebar - Ultra Clean Matte */
    [data-testid="stSidebar"] {
        background: #08090d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Equalizer Bar Animation */
    .eq-container {
        display: inline-flex;
        align-items: flex-end;
        gap: 3px;
        height: 16px;
        margin-left: 8px;
    }
    .eq-bar {
        width: 3px;
        background: #38bdf8;
        border-radius: 2px;
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

    /* Studio Metric Card */
    .metric-card {
        background: rgba(15, 18, 28, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 20px;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .card-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .card-number {
        font-size: 30px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    /* Minimal Quadrant Badges */
    .badge-quad {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-joyful {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-tense {
        background: rgba(239, 68, 68, 0.1);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    .badge-melancholic {
        background: rgba(168, 85, 247, 0.1);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }
    .badge-calm {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    /* Audio Element Inversion */
    audio {
        width: 100% !important;
        border-radius: 8px;
        filter: invert(0.92) hue-rotate(180deg);
        opacity: 0.95;
    }
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown("""
<div style="padding: 12px 0 18px 0;">
    <div style="font-size: 10px; font-weight: 700; letter-spacing: 1.5px; color: #38bdf8; text-transform: uppercase;">Acoustic Laboratory</div>
    <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-top: 2px;">Audio Ingestion</div>
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

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 14px;">
    <div style="font-size: 10px; text-transform: uppercase; color: #38bdf8; font-weight: 700; letter-spacing: 1px;">Phase 2 Core Team</div>
    <div style="font-size: 13px; color: #f1f5f9; font-weight: 700; margin-top: 4px;">Adnan Rizvi</div>
    <div style="font-size: 11px; color: #94a3b8;">Lead Architect & UI Engineer</div>
    <div style="font-size: 11px; color: #64748b; margin-top: 6px; line-height: 1.4;">
        Aditya: DSP Spectrograms<br/>
        Bhoomi: PyTorch CNN & Fusion<br/>
        Praanshu: Grad-CAM Explainability<br/>
        Vani: Benchmark Evaluation
    </div>
</div>
""", unsafe_allow_html=True)


col_title, col_status = st.columns([2, 1])

with col_title:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size: 26px; font-weight: 800; letter-spacing: -0.03em; color:#ffffff;">HarmoniX Audio Studio</span>
        <div class="eq-container">
            <div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div>
        </div>
    </div>
    <div style="font-size: 13px; color: #64748b; margin-top: 2px;">Multimodal Music Emotion Recognition | 2D Spectrogram Neural Intelligence</div>
    """, unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div style="text-align: right; padding-top: 4px;">
        <span style="background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 20px; padding: 4px 12px; font-size: 11px; font-weight: 700;">
            ENGINE READY
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

if selected_audio_path is None:
    st.info("Select a benchmark audio track or upload an audio file from the sidebar to launch the studio.")
    st.stop()


with st.container():
    c_play, c_spec = st.columns([1.5, 1])

    with c_play:
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 8px;">
            <div style="font-size: 15px; font-weight: 700; color: #ffffff;">{audio_label}</div>
            <span style="background: rgba(255,255,255,0.06); color: #94a3b8; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;">{genre_badge}</span>
        </div>
        """, unsafe_allow_html=True)
        with open(selected_audio_path, "rb") as f_audio:
            audio_bytes = f_audio.read()
        st.audio(audio_bytes, format="audio/wav")

    with st.spinner("Analyzing audio frequencies and generating neural predictions..."):
        try:
            results = predict_audio(selected_audio_path)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    meta = results["metadata"]

    with c_spec:
        st.markdown(f"""
        <div style="background: rgba(10, 13, 20, 0.85); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 14px 18px;">
            <div style="font-size: 10px; text-transform: uppercase; color: #38bdf8; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px;">Audio Signal Normalization</div>
            <div style="display:flex; justify-content:space-between; font-size: 12px; color: #cbd5e1; margin-bottom: 3px;">
                <span>Duration: <b>{meta['duration_seconds']:.2f}s</b></span>
                <span>Sample Rate: <b>{meta['target_sample_rate']} Hz</b></span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size: 12px; color: #cbd5e1;">
                <span>Channels: <b>Mono (Downmixed)</b></span>
                <span>Peak Loudness: <b>{meta['normalized_peak']:.2f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

v_score = results["primary_valence"]
a_score = results["estimated_arousal"]
quadrant = results["emotional_quadrant"]

badge_class = "badge-joyful" if "Q1" in quadrant else "badge-tense" if "Q2" in quadrant else "badge-melancholic" if "Q3" in quadrant else "badge-calm"

m1, m2, m3, m4 = st.columns(4)

with m1:
    v_col = "#34d399" if v_score >= 0.50 else "#f87171"
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Predicted Valence</div>
        <div class="card-number" style="color: {v_col};">{v_score:.3f}</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 3px; font-weight: 500;">
            {'Positive / Uplifting' if v_score >= 0.50 else 'Somber / Minor Mood'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Estimated Arousal</div>
        <div class="card-number" style="color: #60a5fa;">{a_score:.3f}</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 3px; font-weight: 500;">
            {'Dynamic Energy' if a_score >= 0.50 else 'Subdued / Calm'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Emotional Space</div>
        <div style="margin-top: 6px;"><span class="badge-quad {badge_class}">{quadrant.split(':')[1].strip()}</span></div>
        <div style="font-size: 11px; color: #64748b; margin-top: 6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            {results["mood_description"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    centroid_str = results["acoustic_highlights"].get("Brightness (Spectral Centroid)", "2150.0 Hz")
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Spectral Warmth</div>
        <div class="card-number" style="color: #fbbf24; font-size: 26px;">{centroid_str}</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 3px; font-weight: 500;">
            Timbral Center of Mass
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

col_radar, col_ensemble = st.columns([1.3, 1])

with col_radar:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
        <span style="font-size: 14px; font-weight: 700; color: #ffffff;">Russell's 2D Mood Compass (Radar Display)</span>
        <span style="font-size: 11px; color: #64748b; font-family:'JetBrains Mono';">V: {v_score:.2f} | A: {a_score:.2f}</span>
    </div>
    """.format(v_score=v_score, a_score=a_score), unsafe_allow_html=True)

    fig_compass = go.Figure()

    fig_compass.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(16, 185, 129, 0.06)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(239, 68, 68, 0.06)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(168, 85, 247, 0.06)", line_width=0)
    fig_compass.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(59, 130, 246, 0.06)", line_width=0)

    for r in [0.25, 0.45]:
        fig_compass.add_shape(
            type="circle",
            x0=0.5 - r, y0=0.5 - r, x1=0.5 + r, y1=0.5 + r,
            line=dict(color="rgba(255, 255, 255, 0.04)", width=1, dash="dot")
        )

    fig_compass.add_hline(y=0.5, line_dash="solid", line_color="rgba(255, 255, 255, 0.12)", line_width=1)
    fig_compass.add_vline(x=0.5, line_dash="solid", line_color="rgba(255, 255, 255, 0.12)", line_width=1)

    fig_compass.add_annotation(x=0.82, y=0.92, text="Q1: JOYFUL", showarrow=False, font=dict(color="#34d399", size=10, family="Space Grotesk"))
    fig_compass.add_annotation(x=0.18, y=0.92, text="Q2: TENSE", showarrow=False, font=dict(color="#f87171", size=10, family="Space Grotesk"))
    fig_compass.add_annotation(x=0.18, y=0.08, text="Q3: MELANCHOLIC", showarrow=False, font=dict(color="#c084fc", size=10, family="Space Grotesk"))
    fig_compass.add_annotation(x=0.82, y=0.08, text="Q4: CALM", showarrow=False, font=dict(color="#60a5fa", size=10, family="Space Grotesk"))

    fig_compass.add_trace(go.Scatter(
        x=[v_score], y=[a_score],
        mode="markers",
        marker=dict(size=28, color="rgba(56, 189, 248, 0.2)", symbol="circle"),
        hoverinfo="skip"
    ))
    fig_compass.add_trace(go.Scatter(
        x=[v_score], y=[a_score],
        mode="markers+text",
        marker=dict(size=14, color="#ffffff", line=dict(color="#38bdf8", width=2.5), symbol="diamond"),
        text=["Current Track"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11, family="Space Grotesk"),
        name="Track Location"
    ))

    fig_compass.update_layout(
        xaxis=dict(title=dict(text="Valence (Negative to Positive)", font=dict(size=11, color="#64748b")), range=[0.0, 1.0], gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        yaxis=dict(title=dict(text="Arousal (Calm to Energetic)", font=dict(size=11, color="#64748b")), range=[0.0, 1.0], gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        plot_bgcolor="#07090e",
        paper_bgcolor="#090c12",
        margin=dict(l=40, r=40, t=10, b=40),
        height=330,
        showlegend=False
    )
    st.plotly_chart(fig_compass, use_container_width=True)

with col_ensemble:
    st.markdown("""
    <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
        Model Ensemble Agreement
    </div>
    """, unsafe_allow_html=True)

    preds = results["model_predictions"]
    df_models = pd.DataFrame([
        {"Architecture": "Random Forest (Best)", "Valence": preds.get("Random_Forest", 0), "Color": "#10b981"},
        {"Architecture": "XGBoost Regressor", "Valence": preds.get("XGBoost", 0), "Color": "#38bdf8"},
        {"Architecture": "Ridge Baseline", "Valence": preds.get("Ridge", 0), "Color": "#64748b"}
    ])

    fig_bar = px.bar(
        df_models,
        x="Valence",
        y="Architecture",
        orientation="h",
        color="Architecture",
        color_discrete_map={
            "Random Forest (Best)": "#10b981",
            "XGBoost Regressor": "#38bdf8",
            "Ridge Baseline": "#64748b"
        },
        range_x=[0.0, 1.0],
        text_auto=".3f"
    )
    fig_bar.update_layout(
        plot_bgcolor="#07090e",
        paper_bgcolor="#090c12",
        font=dict(color="#cbd5e1", family="Space Grotesk"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title=dict(text="Valence Score", font=dict(size=11, color="#64748b"))),
        yaxis=dict(title="", tickfont=dict(size=11)),
        margin=dict(l=10, r=20, t=10, b=40),
        height=330,
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 2px;">
    Dynamic Emotion Trajectory Across Timeline
</div>
<div style="font-size: 12px; color: #64748b; margin-bottom: 10px;">
    3-second rolling analysis windows capturing emotional progression from intro to drop.
</div>
""", unsafe_allow_html=True)

trajectory = results.get("trajectory", [])
if trajectory and len(trajectory) > 1:
    df_traj = pd.DataFrame(trajectory)

    fig_traj = go.Figure()

    fig_traj.add_trace(go.Scatter(
        x=df_traj["time_start_sec"],
        y=df_traj["valence"],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3, shape="spline"),
        marker=dict(size=7, color="#ffffff", line=dict(color="#38bdf8", width=2)),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.06)",
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
        plot_bgcolor="#07090e",
        paper_bgcolor="#090c12",
        font=dict(color="#cbd5e1", family="Space Grotesk"),
        yaxis=dict(title=dict(text="Valence", font=dict(size=11, color="#64748b")), range=[0.0, 1.0], gridcolor="rgba(255,255,255,0.03)"),
        xaxis=dict(title=dict(text="Timeline (Seconds)", font=dict(size=11, color="#64748b")), gridcolor="rgba(255,255,255,0.03)"),
        height=270,
        margin=dict(l=40, r=40, t=15, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_traj, use_container_width=True)
else:
    st.info("Short audio clip: Evaluated as a single temporal frame.")


with st.expander("View Detailed Acoustic Physics DNA (34 Descriptors)"):
    st.markdown("##### Core Acoustic Driver Highlights")
    cols = st.columns(len(results["acoustic_highlights"]))
    for col, (k, v) in zip(cols, results["acoustic_highlights"].items()):
        col.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px;">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; font-weight: 700;">{k.split('(')[0].strip()}</div>
            <div style="font-size: 15px; color: #38bdf8; font-weight: 700; font-family: 'JetBrains Mono'; margin-top: 3px;">{v}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Extracted 34 Physics Features (Full Vector)")
    df_features = pd.DataFrame([results["features"]]).T.reset_index()
    df_features.columns = ["Acoustic Feature Name", "Value"]
    st.dataframe(df_features, height=220, use_container_width=True)

st.markdown("""
<div style="text-align: center; color: #475569; font-size: 11.5px; margin-top: 40px; padding: 20px 0; border-top: 1px solid rgba(255,255,255,0.05);">
    <span style="font-weight: 700; color: #94a3b8;">HarmoniX Research Group</span> • Phase 2 Master Implementation<br/>
    Lead Architect & UI Engineering: <b>Adnan Rizvi</b> • DSP: <b>Aditya</b> • DL: <b>Bhoomi</b> • XAI: <b>Praanshu</b> • Benchmarks: <b>Vani</b>
</div>
""", unsafe_allow_html=True)
