"""
audio_viz.py — Generate an audio energy graph with viral moment markers.
Uses Librosa for RMS energy extraction and Matplotlib for visualization.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure


def generate_energy_graph(video_path: str, clips_data: list) -> Figure:
    """
    Generate an audio energy visualization with vertical lines marking viral moments.

    Loads audio from the video, computes RMS energy over time,
    smooths it, and plots with vertical dashed lines for each detected clip.

    Args:
        video_path: Path to the original video file.
        clips_data: List of clip dicts with at least 'start_time' and 'hook' keys.

    Returns:
        Matplotlib Figure object ready for st.pyplot().
    """
    # Load audio from video (librosa handles video audio extraction via ffmpeg)
    try:
        y, sr = librosa.load(video_path, sr=22050, mono=True)
    except Exception as e:
        # Return a minimal placeholder figure if audio load fails
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.text(
            0.5, 0.5,
            f"Audio visualization unavailable: {e}",
            ha="center", va="center", transform=ax.transAxes,
            color="gray", fontsize=12
        )
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")
        return fig

    # Compute RMS energy with a hop length of 512 samples
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Time axis in seconds
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    times_minutes = times / 60.0

    # Smooth the energy curve for better readability
    smooth_window = max(1, len(rms) // 200)
    rms_smooth = np.convolve(
        rms,
        np.ones(smooth_window) / smooth_window,
        mode="same"
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))

    # Dark background for a modern look
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Fill under the energy curve with gradient effect
    ax.fill_between(
        times_minutes,
        rms_smooth,
        alpha=0.4,
        color="#4f9cf9",
        linewidth=0
    )
    ax.plot(
        times_minutes,
        rms_smooth,
        color="#4f9cf9",
        linewidth=1.5,
        alpha=0.9
    )

    # Draw vertical dashed lines at each clip's start time
    colors_for_clips = ["#ff4b4b", "#ffa500", "#00cc88", "#a855f7", "#f59e0b"]

    for i, clip in enumerate(clips_data):
        clip_start_min = float(clip["start_time"]) / 60.0
        color = colors_for_clips[i % len(colors_for_clips)]
        label = f"Clip {i + 1}"

        ax.axvline(
            x=clip_start_min,
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.9
        )
        # Annotate above the line
        ax.text(
            clip_start_min,
            rms_smooth.max() * 1.05,
            label,
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom"
        )

    # Axes styling
    ax.set_xlabel("Time (minutes)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Audio Energy", color="#aaaaaa", fontsize=10)
    ax.set_title(
        "Audio Energy Map — Detected Viral Moments",
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_xlim(0, times_minutes[-1] if len(times_minutes) > 0 else 1)
    ax.set_ylim(0, rms_smooth.max() * 1.2 if rms_smooth.max() > 0 else 1)

    # Legend for clip lines
    if clips_data:
        legend_patches = [
            mpatches.Patch(
                color=colors_for_clips[i % len(colors_for_clips)],
                label=f"Clip {i + 1}: {clips_data[i].get('hook', '')[:30]}"
            )
            for i in range(len(clips_data))
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            facecolor="#1a1a2e",
            edgecolor="#333333",
            labelcolor="white",
            fontsize=8
        )

    plt.tight_layout()
    return fig
