"""
cropper.py — Smart vertical crop from 16:9 to 9:16 using MediaPipe face detection.
Samples frames to find average face position and keeps the speaker centered.
"""

import os
import subprocess
import numpy as np

try:
    from moviepy.editor import VideoFileClip
except ModuleNotFoundError:
    from moviepy import VideoFileClip

try:
    import mediapipe as mp
except Exception:
    mp = None


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


def _crop_clip(clip, x1: int, x2: int):
    """MoviePy v1/v2 compatible crop helper."""
    if hasattr(clip, "crop"):
        return clip.crop(x1=x1, x2=x2)
    return clip.cropped(x1=x1, x2=x2)


def _set_audio(clip, audio_clip):
    """MoviePy v1/v2 compatible audio attach helper."""
    if hasattr(clip, "set_audio"):
        return clip.set_audio(audio_clip)
    return clip.with_audio(audio_clip)


def _detect_face_center_x(clip, num_frames: int = 5) -> float | None:
    """
    Sample frames from a video to find the average horizontal face center.

    Args:
        clip: Opened MoviePy clip.
        num_frames: Number of frames to sample for face detection.

    Returns:
        Average normalized X position (0.0–1.0) of detected faces,
        or None if no faces found.
    """
    # If MediaPipe is unavailable or incompatible, use center-crop fallback.
    if mp is None or not hasattr(mp, "solutions"):
        return None

    # Initialize MediaPipe face detection
    try:
        mp_face = mp.solutions.face_detection
        face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
    except Exception:
        return None

    duration = clip.duration
    x_positions = []

    try:
        # Sample frames evenly across the clip. Fewer samples keep crop
        # responsive while still being stable for talking-head footage.
        sample_times = np.linspace(0, duration * 0.9, num_frames)
        for t in sample_times:
            try:
                frame = clip.get_frame(t)  # Returns H x W x 3 numpy array (RGB)
                results = face_detector.process(frame)

                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        # Face center X = left edge + half of width
                        face_center_x = bbox.xmin + bbox.width / 2.0
                        x_positions.append(face_center_x)
            except Exception:
                continue  # Skip bad frames silently

    finally:
        try:
            face_detector.close()
        except Exception:
            pass

    return float(np.mean(x_positions)) if x_positions else None


def crop_to_vertical(raw_clip_path: str, output_dir: str, clip_index: int) -> tuple[str, bool]:
    """
    Crop a 16:9 clip to 9:16 vertical format, centering on the detected face.

    If no face is detected, falls back to a center crop silently.

    Args:
        raw_clip_path: Path to the raw 16:9 clip.
        output_dir: Directory to save the vertical clip.
        clip_index: Clip number for file naming.

    Returns:
        Path to the saved 9:16 vertical clip.

    Raises:
        RuntimeError: If MoviePy fails to write the cropped clip.
    """
    os.makedirs(output_dir, exist_ok=True)
    encode_preset = os.getenv("ATTENTIONX_ENCODE_PRESET", "superfast")
    try:
        encode_threads = int(os.getenv("ATTENTIONX_ENCODE_THREADS", str(os.cpu_count() or 2)))
    except ValueError:
        encode_threads = os.cpu_count() or 2
    try:
        face_samples = int(os.getenv("ATTENTIONX_FACE_SAMPLES", "5"))
    except ValueError:
        face_samples = 5
    face_samples = max(2, min(12, face_samples))

    output_path = os.path.join(output_dir, f"clip_{clip_index}_vertical.mp4")

    clip = VideoFileClip(raw_clip_path)
    original_width = clip.w
    original_height = clip.h

    # Target width for 9:16 from original height
    target_width = int(original_height * 9 / 16)

    # Ensure target_width does not exceed original width
    target_width = min(target_width, original_width)

    # Try to find face center X (in pixel units)
    face_used = False
    crop_x1 = None

    normalized_x = _detect_face_center_x(clip, num_frames=face_samples)

    if normalized_x is not None:
        # Convert normalized position to pixel position
        face_pixel_x = int(normalized_x * original_width)
        # Center the crop window on the face
        crop_x1 = face_pixel_x - target_width // 2
        face_used = True
    else:
        # Fallback: center crop
        crop_x1 = (original_width - target_width) // 2

    # Clamp crop_x1 to valid range
    crop_x1 = max(0, min(crop_x1, original_width - target_width))
    crop_x2 = crop_x1 + target_width

    try:
        # Apply crop using MoviePy
        vertical_clip = _crop_clip(clip, x1=crop_x1, x2=crop_x2)
        if clip.audio is not None:
            vertical_clip = _set_audio(vertical_clip, clip.audio)

        vertical_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            audio=True,
            preset=encode_preset,
            threads=encode_threads,
            logger=None,
            ffmpeg_params=["-movflags", "+faststart"],
            temp_audiofile=os.path.join(output_dir, f"temp_vcrop_audio_{clip_index}.m4a"),
            remove_temp=True
        )
        vertical_clip.close()
        # Failsafe: if MoviePy writes video without audio, remux from raw clip.
        if clip.audio is not None and not _has_audio_stream(output_path):
            _remux_audio_from_source(output_path, raw_clip_path)

    except Exception as e:
        raise RuntimeError(
            f"Cropper failed for clip {clip_index}: {e}"
        ) from e
    finally:
        clip.close()

    return output_path, face_used
