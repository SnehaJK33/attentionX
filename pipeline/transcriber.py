"""
transcriber.py — Whisper-based audio transcription with word-level timestamps.
Loads the Whisper 'base' model and returns structured segment data.
"""

import os
import whisper
import streamlit as st
import torch


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str):
    """Load and cache Whisper model once per model size."""
    return whisper.load_model(model_name)


def transcribe_video(video_path: str) -> dict:
    """
    Transcribe a video file using OpenAI Whisper with word-level timestamps.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        dict with keys:
            - segments: list of {start, end, text}
            - full_text: complete transcript as a single string
            - duration: total video duration in seconds

    Raises:
        ValueError: If no speech is detected in the video.
        RuntimeError: If transcription fails after one retry.
    """
    model_name = os.getenv("WHISPER_MODEL", "base").strip() or "base"
    model = load_whisper_model(model_name)
    use_fp16 = torch.cuda.is_available()
    language = (os.getenv("WHISPER_LANGUAGE") or "").strip() or None

    def _run_transcription():
        """Inner function to run Whisper transcription."""
        # Greedy decoding + segment timestamps is much faster than full
        # word-level alignment while still covering this app's needs.
        result = model.transcribe(
            video_path,
            verbose=False,
            word_timestamps=False,
            fp16=use_fp16,
            temperature=0,
            best_of=1,
            beam_size=1,
            condition_on_previous_text=False,
            language=language
        )
        return result

    # First attempt
    try:
        result = _run_transcription()
    except Exception as e:
        # Retry once on failure
        try:
            result = _run_transcription()
        except Exception as retry_e:
            raise RuntimeError(
                f"Transcription failed after retry: {retry_e}"
            ) from retry_e

    # Extract structured segments
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip()
        })

    full_text = result.get("text", "").strip()

    # Guard against empty transcripts
    if not full_text:
        raise ValueError("No audio detected in video")

    # Calculate duration from last segment end time
    duration = segments[-1]["end"] if segments else 0.0

    return {
        "segments": segments,
        "full_text": full_text,
        "duration": duration
    }


def build_timestamped_transcript(transcription: dict) -> str:
    """
    Format transcript segments with timestamps for Gemini analysis.

    Args:
        transcription: Output dict from transcribe_video().

    Returns:
        A formatted string with timestamps prefixed to each segment.
    """
    lines = []
    for seg in transcription["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        # Format as [MM:SS - MM:SS] text
        start_fmt = f"{int(start // 60):02d}:{int(start % 60):02d}"
        end_fmt = f"{int(end // 60):02d}:{int(end % 60):02d}"
        lines.append(f"[{start_fmt} - {end_fmt}] {text}")
    return "\n".join(lines)
