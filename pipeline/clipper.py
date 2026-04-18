"""
clipper.py — Extract video clips from the original video using MoviePy.
Adds 2-second buffers around each detected moment and saves raw clips.
"""

import os
import subprocess
from moviepy.editor import VideoFileClip


def _get_ffprobe_bin() -> str:
    """Locate ffprobe binary next to configured ffmpeg when possible."""
    ffmpeg_bin = os.getenv("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    if ffmpeg_dir:
        candidate = os.path.join(ffmpeg_dir, "ffprobe.exe")
        if os.path.exists(candidate):
            return candidate
    return "ffprobe"


def _has_audio_stream(video_path: str) -> bool:
    """Return True when file contains at least one audio stream."""
    ffprobe_bin = _get_ffprobe_bin()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


def _ffmpeg_stream_copy_cut(
    source_path: str,
    output_path: str,
    start_time: float,
    end_time: float
) -> bool:
    """
    Fast clip extraction using ffmpeg stream copy (no re-encode).

    Returns True on success, False if ffmpeg fails so caller can fallback.
    """
    ffmpeg_bin = os.getenv("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{start_time:.3f}",
        "-to", f"{end_time:.3f}",
        "-i", source_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
    except Exception:
        return False

    ok = (
        result.returncode == 0
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 0
    )
    if not ok:
        return False

    # Some source/container combinations can produce silent stream-copy outputs.
    # Only accept fast-cut output when audio stream is present.
    return _has_audio_stream(output_path)


def extract_clips(video_path: str, clips_data: list, output_dir: str) -> list:
    """
    Cut clips from the original video for each detected viral moment.

    Args:
        video_path: Path to the original uploaded video file.
        clips_data: List of clip dicts from Gemini analyzer, each with
                    start_time and end_time keys.
        output_dir: Directory where raw clips will be saved.

    Returns:
        Updated list of clip dicts with added keys:
            - raw_path: path to the cut raw clip (original aspect ratio)
            - actual_start: actual start time used (with buffer)
            - actual_end: actual end time used (with buffer)

    Raises:
        RuntimeError: If MoviePy fails to write a clip.
    """
    os.makedirs(output_dir, exist_ok=True)
    use_fast_cut = os.getenv("ATTENTIONX_USE_FAST_CUT", "0").strip().lower() in {"1", "true", "yes"}
    encode_preset = os.getenv("ATTENTIONX_ENCODE_PRESET", "veryfast")
    try:
        encode_threads = int(os.getenv("ATTENTIONX_ENCODE_THREADS", str(os.cpu_count() or 2)))
    except ValueError:
        encode_threads = os.cpu_count() or 2

    # Load the source video once
    source_video = VideoFileClip(video_path)
    total_duration = source_video.duration

    enriched_clips = []

    try:
        for i, clip in enumerate(clips_data):
            clip_index = i + 1  # 1-based for file naming

            # Apply 2-second buffers, clamped to video bounds
            actual_start = max(0.0, float(clip["start_time"]) - 2.0)
            actual_end = min(total_duration, float(clip["end_time"]) + 2.0)

            # Safety check — ensure start < end with at minimum 1 second
            if actual_end - actual_start < 1.0:
                actual_end = min(total_duration, actual_start + 10.0)

            raw_path = os.path.join(output_dir, f"clip_{clip_index}_raw.mp4")

            try:
                # Fast path: no re-encode clipping via ffmpeg stream copy.
                # If this fails for any codec/container, fallback to MoviePy.
                if not use_fast_cut or not _ffmpeg_stream_copy_cut(
                    video_path,
                    raw_path,
                    actual_start,
                    actual_end
                ):
                    # Cut the subclip from the source video
                    subclip = source_video.subclip(actual_start, actual_end)
                    if source_video.audio is not None:
                        # Keep audio track explicitly attached for reliability.
                        subclip = subclip.set_audio(
                            source_video.audio.subclip(actual_start, actual_end)
                        )

                    # Fallback path: h264 re-encode for compatibility
                    subclip.write_videofile(
                        raw_path,
                        codec="libx264",
                        audio_codec="aac",
                        audio=True,
                        preset=encode_preset,
                        threads=encode_threads,
                        logger=None,  # Suppress verbose MoviePy output
                        ffmpeg_params=["-movflags", "+faststart"],
                        temp_audiofile=os.path.join(output_dir, f"temp_audio_{clip_index}.m4a"),
                        remove_temp=True
                    )
                    subclip.close()

            except Exception as e:
                raise RuntimeError(
                    f"MoviePy failed to write clip {clip_index}: {e}\n"
                    "Try with a shorter video."
                ) from e

            # Enrich clip dict with file path and actual timing
            enriched_clip = dict(clip)
            enriched_clip["raw_path"] = raw_path
            enriched_clip["actual_start"] = actual_start
            enriched_clip["actual_end"] = actual_end
            enriched_clip["clip_index"] = clip_index
            enriched_clips.append(enriched_clip)

    finally:
        # Always release the source video resource
        source_video.close()

    return enriched_clips
