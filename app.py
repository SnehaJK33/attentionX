"""
app.py - AttentionX Streamlit app.
"""

import datetime
import io
import os
import shutil
import subprocess
import tempfile
import zipfile

import streamlit as st
from dotenv import load_dotenv

def _configure_ffmpeg() -> None:
    """
    Configure ffmpeg for local Windows runs and cloud deployments.
    """
    if os.environ.get("IMAGEIO_FFMPEG_EXE") and os.path.exists(
        os.environ["IMAGEIO_FFMPEG_EXE"]
    ):
        return

    app_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_bin_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    candidate_dirs = []
    custom_dir = os.getenv("ATTENTIONX_FFMPEG_DIR")
    if custom_dir:
        candidate_dirs.append(custom_dir)

    candidate_dirs.extend(
        [
            os.path.join(app_dir, "ffmpeg", "bin"),
            os.path.join(os.path.dirname(app_dir), "ffmpeg-8.1-essentials_build", "bin"),
        ]
    )

    for ffmpeg_dir in candidate_dirs:
        ffmpeg_bin = os.path.join(ffmpeg_dir, ffmpeg_bin_name)
        if os.path.exists(ffmpeg_bin):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_bin
            return

    ffmpeg_from_path = shutil.which("ffmpeg")
    if ffmpeg_from_path:
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_from_path)


_configure_ffmpeg()

load_dotenv()

from pipeline.analyzer import analyze_transcript
from pipeline.audio_viz import generate_energy_graph
from pipeline.caption import burn_captions
from pipeline.clipper import extract_clips
from pipeline.cropper import crop_to_vertical
from pipeline.refiner import refine_clips_with_emotional_peaks
from pipeline.transcriber import build_timestamped_transcript, transcribe_video


st.set_page_config(
    page_title="AttentionX",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_emotion_badge(emotion: str) -> str:
    badge_map = {
        "shock": "badge-shock",
        "inspiration": "badge-inspiration",
        "curiosity": "badge-curiosity",
        "humor": "badge-humor",
    }
    css_class = badge_map.get((emotion or "").lower(), "badge-default")
    label = (emotion or "UNKNOWN").upper()
    return f'<span class="badge {css_class}">{label}</span>'


def get_virality_class(score: int) -> str:
    if score <= 50:
        return "virality-red"
    if score <= 75:
        return "virality-orange"
    return "virality-green"


def _get_ffprobe_bin() -> str:
    ffmpeg_bin = os.environ.get("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    if ffmpeg_dir:
        probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        candidate = os.path.join(ffmpeg_dir, probe_name)
        if os.path.exists(candidate):
            return candidate
    return "ffprobe"


def _has_audio_stream(video_path: str) -> bool:
    if not video_path or not os.path.exists(video_path):
        return False
    ffprobe_bin = _get_ffprobe_bin()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


def _get_secret_value(key: str) -> str:
    """
    Read a config value from environment first, then Streamlit secrets.
    """
    from_env = (os.getenv(key) or "").strip()
    if from_env:
        return from_env

    try:
        from_secrets = st.secrets.get(key, "")
    except Exception:
        from_secrets = ""
    return str(from_secrets).strip()


def build_content_zip(clips_data: list) -> bytes:
    zip_buffer = io.BytesIO()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hooks_lines = []
    instagram_lines = []
    youtube_lines = []
    report_lines = [
        "AttentionX Virality Report",
        f"Generated: {now}",
        "==========================",
    ]

    best_clip_index = max(
        range(len(clips_data)), key=lambda i: clips_data[i].get("virality_score", 0)
    ) + 1

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, clip in enumerate(clips_data):
            n = i + 1
            hook = clip.get("hook", "")
            why_viral = clip.get("why_viral", "")
            emotion = clip.get("emotion", "")
            score = clip.get("virality_score", 0)
            ig_caption = clip.get("instagram_caption", "")
            yt_desc = clip.get("youtube_description", "")
            final_path = clip.get("final_path", "")

            if final_path and os.path.exists(final_path):
                zf.write(final_path, arcname=f"clip_{n}_final.mp4")

            hooks_lines += [
                f"CLIP {n}",
                "------",
                f"Hook: {hook}",
                f"Why Viral: {why_viral}",
                f"Emotion: {emotion}",
                f"Virality Score: {score}/100",
                "",
            ]

            instagram_lines += [f"CLIP {n} - Ready to paste:", ig_caption, ""]
            youtube_lines += [f"CLIP {n}:", yt_desc, ""]

            report_lines += [
                f"CLIP {n}: {score}/100 - {emotion}",
                "Best for: Instagram Reels / TikTok",
                f"Hook: {hook}",
                f"Reason: {why_viral}",
                "--------------------------",
                "",
            ]

        report_lines += [
            "OVERALL RECOMMENDATION:",
            f"Post Clip {best_clip_index} first for maximum reach.",
        ]

        zf.writestr("hooks_and_captions.txt", "\n".join(hooks_lines))
        zf.writestr("instagram_captions.txt", "\n".join(instagram_lines))
        zf.writestr("youtube_descriptions.txt", "\n".join(youtube_lines))
        zf.writestr("virality_report.txt", "\n".join(report_lines))

    return zip_buffer.getvalue()


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Manrope', sans-serif;
}

.stApp {
  background:
    radial-gradient(900px 450px at 85% -10%, rgba(255, 166, 77, 0.18), transparent 60%),
    radial-gradient(900px 450px at -10% 5%, rgba(79, 156, 249, 0.16), transparent 60%),
    linear-gradient(180deg, #0f141c 0%, #121923 45%, #0d1117 100%);
}

/* Global readability */
.stApp,
.stApp p,
.stApp span,
.stApp li,
.stApp label,
.stMarkdown,
.stCaption,
.stText,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
  color: #e7edf7 !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
  color: #f3f8ff !important;
}

/* Status / alerts / expanders */
div[data-testid="stStatusWidget"] p,
div[data-testid="stStatusWidget"] span,
.stAlert p,
.stAlert span,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
  color: #e7edf7 !important;
}

/* Inputs */
textarea, input, .stTextInput input, .stTextArea textarea {
  background: #0f1622 !important;
  color: #e7edf7 !important;
  border: 1px solid rgba(173, 206, 255, 0.28) !important;
}

.hero-wrap {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 18px;
  background: linear-gradient(140deg, rgba(33, 45, 63, 0.9), rgba(21, 29, 41, 0.92));
  padding: 1.1rem 1.2rem 1rem;
  margin-bottom: 1rem;
  box-shadow: 0 14px 38px rgba(0, 0, 0, 0.28);
}

.hero-title {
  margin: 0;
  font-family: 'Sora', sans-serif;
  font-size: 2.1rem;
  letter-spacing: -0.02em;
  background: linear-gradient(90deg, #77c5ff 0%, #b6d8ff 35%, #ffd28c 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-sub {
  margin: 0.25rem 0 0.75rem;
  color: #b8c6db;
  font-size: 1rem;
}

.hero-chips {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.hero-chip {
  padding: 0.3rem 0.62rem;
  border: 1px solid rgba(173, 206, 255, 0.28);
  border-radius: 999px;
  color: #d2e5ff;
  font-size: 0.78rem;
  background: rgba(19, 28, 39, 0.75);
}

.badge {
  display: inline-block;
  padding: .22rem .72rem;
  border-radius: 999px;
  font-size: .75rem;
  font-weight: 800;
  margin-right: .4rem;
  letter-spacing: .02em;
}
.badge-shock { background:#ef4444; color:#fff; }
.badge-inspiration { background:#22c55e; color:#fff; }
.badge-curiosity { background:#3b82f6; color:#fff; }
.badge-humor { background:#eab308; color:#111; }
.badge-default { background:#6b7280; color:#fff; }

.clip-card {
  border: 1px solid rgba(255, 255, 255, 0.09);
  border-radius: 16px;
  padding: 1rem;
  margin-bottom: 1rem;
  background: linear-gradient(150deg, rgba(22, 30, 42, 0.94), rgba(18, 24, 34, 0.94));
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
}

.clip-card h3 {
  font-family: 'Sora', sans-serif;
  margin-top: 0.2rem;
}

.virality-red .stProgress > div > div > div { background:#ef4444 !important; }
.virality-orange .stProgress > div > div > div { background:#f97316 !important; }
.virality-green .stProgress > div > div > div { background:#22c55e !important; }

.stButton > button {
  border-radius: 12px;
  border: 1px solid rgba(133, 177, 255, 0.35);
  background: linear-gradient(90deg, #2f6fff 0%, #4b8dff 55%, #6eb6ff 100%);
  color: #f7fbff;
  font-weight: 700;
}

.stFileUploader {
  border: 1px dashed rgba(140, 180, 240, 0.55) !important;
  border-radius: 12px !important;
  background: rgba(16, 24, 36, 0.45) !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #111722 0%, #0e131d 100%);
  border-right: 1px solid rgba(255,255,255,0.07);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: #d7e5f8;
}

/* Improve readability on metric/progress text */
.stProgress + div,
.stProgress + p {
  color: #dfe9f8 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## AttentionX")
    st.markdown("AI-powered content repurposing")
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("1. Upload a long-form video")
    st.markdown("2. AI detects emotional peaks")
    st.markdown("3. Smart crop + timed captions")
    st.markdown("4. Download your content pack")
    st.markdown("---")

    gemini_api_key = _get_secret_value("GEMINI_API_KEY")
    if gemini_api_key:
        st.success("Gemini API key loaded")
    else:
        st.warning("Missing GEMINI_API_KEY in .env or Streamlit secrets")

st.markdown(
    """
<div class="hero-wrap">
  <h1 class="hero-title">AttentionX</h1>
  <p class="hero-sub">
    Turn long-form mentorship videos into viral-ready short clips with emotional-peak detection,
    smart vertical framing, and dynamic captions.
  </p>
  <div class="hero-chips">
    <span class="hero-chip">Whisper Transcription</span>
    <span class="hero-chip">Gemini Moment Mining</span>
    <span class="hero-chip">Audio + Sentiment Peak Scoring</span>
    <span class="hero-chip">Reels/TikTok Ready Export</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write(f"File: `{uploaded_file.name}` | Size: {uploaded_file.size/(1024*1024):.1f} MB")

    run_button = st.button(
        "Generate Viral Clips",
        type="primary",
        use_container_width=True,
        disabled=(not gemini_api_key),
    )

    if run_button and gemini_api_key:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        for old_file in os.listdir(output_dir):
            if old_file.startswith("clip_") and old_file.endswith(".mp4"):
                old_path = os.path.join(output_dir, old_file)
                if os.path.isfile(old_path):
                    try:
                        os.remove(old_path)
                    except Exception:
                        pass

        temp_video_path = None
        enriched_clips = []

        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if "transcription_cache" not in st.session_state:
            st.session_state.transcription_cache = {}
        if "video_path_cache" not in st.session_state:
            st.session_state.video_path_cache = {}

        try:
            if (
                file_key not in st.session_state.video_path_cache
                or not os.path.exists(st.session_state.video_path_cache.get(file_key, ""))
            ):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp_file:
                    uploaded_file.seek(0)
                    tmp_file.write(uploaded_file.read())
                    temp_video_path = tmp_file.name
                st.session_state.video_path_cache[file_key] = temp_video_path
            else:
                temp_video_path = st.session_state.video_path_cache[file_key]

            with st.status("Processing your video...", expanded=True) as status:
                if file_key in st.session_state.transcription_cache:
                    st.write("Transcription cached - skipping Whisper")
                    transcription = st.session_state.transcription_cache[file_key]
                    timestamped_transcript = build_timestamped_transcript(transcription)
                    st.write(
                        f"  OK {len(transcription['segments'])} segments | "
                        f"{transcription['duration']:.0f}s"
                    )
                else:
                    st.write("Transcribing with Whisper...")
                    transcription = transcribe_video(temp_video_path)
                    timestamped_transcript = build_timestamped_transcript(transcription)
                    st.session_state.transcription_cache[file_key] = transcription
                    st.write(
                        f"  OK {len(transcription['segments'])} segments | "
                        f"{transcription['duration']:.0f}s"
                    )

                st.write("Gemini is finding viral moments...")
                analysis = analyze_transcript(
                    timestamped_transcript,
                    transcription["duration"],
                    gemini_api_key,
                )
                clips_meta = analysis["clips"]
                clips_meta = refine_clips_with_emotional_peaks(
                    temp_video_path,
                    clips_meta,
                    transcription["segments"],
                    max_clips=5,
                )
                st.write(f"  OK Detected {len(clips_meta)} viral moments")
                st.write("  OK Refined clips using audio spikes + sentiment intensity")

                st.write("Cutting clips from video...")
                enriched_clips = extract_clips(temp_video_path, clips_meta, output_dir)
                st.write(f"  OK Extracted {len(enriched_clips)} clips")
                raw_with_audio = sum(
                    1 for clip in enriched_clips if _has_audio_stream(clip.get("raw_path", ""))
                )
                st.write(f"  Audio check (raw): {raw_with_audio}/{len(enriched_clips)}")

                st.write("Smart-cropping to vertical (9:16)...")
                for clip in enriched_clips:
                    vertical_path, face_detected = crop_to_vertical(
                        clip["raw_path"],
                        output_dir,
                        clip["clip_index"],
                    )
                    clip["vertical_path"] = vertical_path
                    clip["face_detected"] = face_detected
                st.write("  OK Cropped all clips")
                vertical_with_audio = sum(
                    1
                    for clip in enriched_clips
                    if _has_audio_stream(clip.get("vertical_path", ""))
                )
                st.write(f"  Audio check (vertical): {vertical_with_audio}/{len(enriched_clips)}")

                st.write("Burning captions...")
                for clip in enriched_clips:
                    final_path = burn_captions(
                        clip["vertical_path"],
                        transcription["segments"],
                        clip["actual_start"],
                        clip["actual_end"],
                        output_dir,
                        clip["clip_index"],
                    )
                    clip["final_path"] = final_path
                st.write("  OK Captions burned")
                final_with_audio = sum(
                    1 for clip in enriched_clips if _has_audio_stream(clip.get("final_path", ""))
                )
                st.write(f"  Audio check (final): {final_with_audio}/{len(enriched_clips)}")

                st.write("Generating energy graph...")
                energy_fig = generate_energy_graph(temp_video_path, enriched_clips)
                st.write("Packing content bundle...")
                zip_bytes = build_content_zip(enriched_clips)

                status.update(label="Done. Your content is ready.", state="complete")

            st.markdown("---")
            st.subheader("Detected Viral Moments")
            st.pyplot(energy_fig)

            st.markdown("---")
            st.subheader("Your Viral Clips")
            for clip in enriched_clips:
                n = clip["clip_index"]
                score = int(clip.get("virality_score", 0))
                emotion = clip.get("emotion", "inspiration")
                hook = clip.get("hook", f"Clip {n}")
                why_viral = clip.get("why_viral", "")
                ig_caption = clip.get("instagram_caption", "")
                yt_desc = clip.get("youtube_description", "")
                raw_path = clip.get("raw_path", "")
                final_path = clip.get("final_path", "")
                face_detected = clip.get("face_detected", True)

                st.markdown('<div class="clip-card">', unsafe_allow_html=True)
                st.markdown(f"### CLIP {n} - {hook}")

                badge_col, score_col = st.columns([1, 3])
                with badge_col:
                    st.markdown(get_emotion_badge(emotion), unsafe_allow_html=True)
                    if not face_detected:
                        st.caption("Center crop used")
                with score_col:
                    st.markdown(f'<div class="{get_virality_class(score)}">', unsafe_allow_html=True)
                    bar_col, txt_col = st.columns([4, 1])
                    with bar_col:
                        st.progress(score / 100)
                    with txt_col:
                        st.write(f"**{score}/100**")
                    st.markdown("</div>", unsafe_allow_html=True)

                before_col, after_col = st.columns(2)
                with before_col:
                    st.write("Before (Original)")
                    if raw_path and os.path.exists(raw_path):
                        st.video(raw_path)
                        with open(raw_path, "rb") as f:
                            st.audio(f.read(), format="audio/mp4")
                with after_col:
                    st.write("After (Final)")
                    if final_path and os.path.exists(final_path):
                        st.video(final_path)
                        with open(final_path, "rb") as f:
                            final_bytes = f.read()
                        st.audio(final_bytes, format="audio/mp4")
                        st.download_button(
                            label=f"Download Clip {n}",
                            data=final_bytes,
                            file_name=f"clip_{n}_final.mp4",
                            mime="video/mp4",
                            key=f"download_clip_{n}",
                        )

                st.info(f"Why this can go viral: {why_viral}")
                ig_col, yt_col = st.columns(2)
                with ig_col:
                    st.text_area(
                        "Instagram Caption",
                        value=ig_caption,
                        key=f"ig_caption_{n}",
                        height=100,
                    )
                with yt_col:
                    st.text_area(
                        "YouTube Description",
                        value=yt_desc,
                        key=f"yt_desc_{n}",
                        height=100,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.download_button(
                label="Download Full Content Pack",
                data=zip_bytes,
                file_name=f"attentionx_content_pack_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.exception(e)

        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception:
                    pass
else:
    st.info("Upload a video to get started.")
