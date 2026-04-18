"""
caption.py — Burn karaoke-style timestamped captions onto video clips.
Uses MoviePy TextClip to overlay transcript segments with white text and black stroke.
"""

import os
import subprocess
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


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


def _remux_audio_from_source(video_path: str, audio_source_path: str) -> bool:
    """Attach audio from source file onto rendered video without re-encoding video."""
    ffmpeg_bin = os.getenv("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    temp_path = f"{video_path}.audiofix.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-i", audio_source_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        temp_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        if result.returncode != 0 or not os.path.exists(temp_path):
            return False
        os.replace(temp_path, video_path)
        return _has_audio_stream(video_path)
    except Exception:
        return False
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def burn_captions(
    vertical_clip_path: str,
    transcript_segments: list,
    clip_actual_start: float,
    clip_actual_end: float,
    output_dir: str,
    clip_index: int
) -> str:
    """
    Burn karaoke-style captions onto a vertical video clip.

    Finds all transcript segments that overlap with the clip's time range
    and renders them as timed TextClips centered at 75% height.

    Args:
        vertical_clip_path: Path to the 9:16 vertical clip.
        transcript_segments: List of {start, end, text} dicts from transcriber.
        clip_actual_start: Start time of this clip in the original video (seconds).
        clip_actual_end: End time of this clip in the original video (seconds).
        output_dir: Directory to write the final captioned clip.
        clip_index: Clip number for file naming.

    Returns:
        Path to the saved final clip with captions burned in.

    Raises:
        RuntimeError: If MoviePy fails to write the captioned clip.
    """
    os.makedirs(output_dir, exist_ok=True)
    encode_preset = os.getenv("ATTENTIONX_ENCODE_PRESET", "veryfast")
    try:
        encode_threads = int(os.getenv("ATTENTIONX_ENCODE_THREADS", str(os.cpu_count() or 2)))
    except ValueError:
        encode_threads = os.cpu_count() or 2
    output_path = os.path.join(output_dir, f"clip_{clip_index}_final.mp4")

    clip = VideoFileClip(vertical_clip_path)
    clip_duration = clip.duration
    clip_width = clip.w
    clip_height = clip.h

    # Caption position: centered horizontally, 75% down from top
    caption_y = int(clip_height * 0.75)

    text_clips = []

    for seg in transcript_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # Check if this segment overlaps with the current clip's time window
        if seg_end < clip_actual_start or seg_start > clip_actual_end:
            continue

        # Convert absolute timestamps to relative clip-local timestamps
        local_start = max(0.0, seg_start - clip_actual_start)
        local_end = min(clip_duration, seg_end - clip_actual_start)

        if local_end <= local_start:
            continue

        text = seg["text"].strip()
        if not text:
            continue

        try:
            # Create a TextClip for each caption segment
            txt_clip = (
                TextClip(
                    text,
                    fontsize=40,
                    font="Arial-Bold",
                    color="white",
                    stroke_color="black",
                    stroke_width=3,
                    method="caption",
                    size=(int(clip_width * 0.9), None),  # 90% of clip width
                    align="center"
                )
                .set_start(local_start)
                .set_end(local_end)
                .set_position(("center", caption_y))
            )
            text_clips.append(txt_clip)

        except Exception:
            # Skip individual captions that fail (e.g., special characters)
            continue

    try:
        if text_clips:
            # Composite all text clips onto the base video
            final_clip = CompositeVideoClip([clip] + text_clips)
            if clip.audio is not None:
                # Explicitly preserve base audio when adding text overlays.
                final_clip = final_clip.set_audio(clip.audio)
        else:
            # No captions found — output the clip as-is
            final_clip = clip

        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            audio=True,
            preset=encode_preset,
            threads=encode_threads,
            logger=None,
            ffmpeg_params=["-movflags", "+faststart"],
            temp_audiofile=os.path.join(output_dir, f"temp_cap_audio_{clip_index}.m4a"),
            remove_temp=True
        )
        # Failsafe: if MoviePy writes video without audio, remux from source clip.
        if clip.audio is not None and not _has_audio_stream(output_path):
            _remux_audio_from_source(output_path, vertical_clip_path)

    except Exception as e:
        raise RuntimeError(
            f"Caption burning failed for clip {clip_index}: {e}\n"
            "Try with a shorter video."
        ) from e
    finally:
        clip.close()
        if text_clips:
            final_clip.close()

    return output_path
