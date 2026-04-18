"""
refiner.py - Blend transcript sentiment with audio energy to refine clip ranking.
"""

from __future__ import annotations

import math

import librosa
import numpy as np


POSITIVE_HIGH_AROUSAL = {
    "amazing", "incredible", "breakthrough", "success", "win", "best",
    "powerful", "transform", "growth", "confident", "achieve", "clarity",
}
NEGATIVE_HIGH_AROUSAL = {
    "fear", "mistake", "problem", "pain", "struggle", "failure",
    "wrong", "shock", "urgent", "danger", "anxiety", "stuck",
}
HUMOR_HINTS = {"funny", "joke", "laugh", "hilarious", "meme", "lol"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 50.0
    return _clamp((value - low) / (high - low) * 100.0, 0.0, 100.0)


def _build_energy_profile(video_path: str):
    """
    Return (times_sec, normalized_energy_0_to_100) or (None, None) on failure.
    """
    try:
        y, sr = librosa.load(video_path, sr=22050, mono=True)
    except Exception:
        return None, None

    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    if len(rms) == 0:
        return None, None

    # Slight smoothing makes score less jittery.
    window = max(1, len(rms) // 250)
    smooth = np.convolve(rms, np.ones(window) / window, mode="same")

    lo = float(np.percentile(smooth, 5))
    hi = float(np.percentile(smooth, 95))
    normalized = np.array([_normalize(float(v), lo, hi) for v in smooth], dtype=np.float32)
    return times, normalized


def _energy_score_for_window(times, energy, start: float, end: float) -> float:
    if times is None or energy is None:
        return 50.0
    start = max(0.0, float(start))
    end = max(start + 0.1, float(end))
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        # Use nearest point when mask is empty.
        center = (start + end) / 2.0
        idx = int(np.argmin(np.abs(times - center)))
        return float(energy[idx])
    return float(np.mean(energy[mask]))


def _sentiment_intensity_score(text: str) -> float:
    if not text:
        return 45.0

    words = [w.strip(".,!?;:\"'()[]{}").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 45.0

    pos = sum(1 for w in words if w in POSITIVE_HIGH_AROUSAL)
    neg = sum(1 for w in words if w in NEGATIVE_HIGH_AROUSAL)
    intensity_tokens = pos + neg

    punctuation_boost = text.count("!") * 1.8 + text.count("?") * 0.8
    base = 40.0 + min(25.0, intensity_tokens * 4.0) + min(18.0, punctuation_boost)

    # Strongly negative emotional language tends to create "shock"-style hooks.
    if neg > pos:
        base += 6.0
    return _clamp(base, 0.0, 100.0)


def _infer_emotion(text: str, energy_score: float, sentiment_score: float, current: str) -> str:
    if current in {"shock", "inspiration", "curiosity", "humor"}:
        return current

    lower = (text or "").lower()
    if any(token in lower for token in HUMOR_HINTS):
        return "humor"
    if sentiment_score >= 72 and energy_score >= 62:
        return "shock"
    if sentiment_score >= 60:
        return "inspiration"
    return "curiosity"


def _join_overlapping_text(transcript_segments: list, start: float, end: float) -> str:
    parts = []
    for seg in transcript_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if seg_end < start or seg_start > end:
            continue
        text = str(seg.get("text", "")).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def refine_clips_with_emotional_peaks(
    video_path: str,
    clips_data: list,
    transcript_segments: list,
    max_clips: int = 5,
) -> list:
    """
    Re-rank clips using blended model:
    - Existing virality score from AI
    - Audio energy spikes
    - Transcript emotional/sentiment intensity
    """
    if not clips_data:
        return clips_data

    times, energy = _build_energy_profile(video_path)
    refined = []

    for clip in clips_data:
        start = float(clip.get("start_time", 0.0))
        end = float(clip.get("end_time", start + 8.0))
        base_score = float(clip.get("virality_score", 70))

        merged_text = _join_overlapping_text(transcript_segments, start, end)
        energy_score = _energy_score_for_window(times, energy, start, end)
        sentiment_score = _sentiment_intensity_score(merged_text)

        blended = round(
            0.58 * base_score
            + 0.27 * energy_score
            + 0.15 * sentiment_score
        )
        blended = int(_clamp(blended, 0, 100))

        updated = dict(clip)
        updated["virality_score"] = blended
        updated["emotion"] = _infer_emotion(
            merged_text,
            energy_score,
            sentiment_score,
            str(clip.get("emotion", "")).lower().strip(),
        )

        why = str(updated.get("why_viral", "")).strip()
        signal_note = (
            f"Strong emotional language and energy peak "
            f"({math.floor(sentiment_score)}/100 text, {math.floor(energy_score)}/100 audio)."
        )
        if why:
            updated["why_viral"] = f"{why} {signal_note}"
        else:
            updated["why_viral"] = signal_note

        refined.append(updated)

    refined.sort(key=lambda c: int(c.get("virality_score", 0)), reverse=True)
    return refined[:max_clips]
