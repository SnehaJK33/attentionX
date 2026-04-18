"""
analyzer.py — Google Gemini analysis to find viral-worthy moments.
Truncates transcript to stay within free tier token limits.
"""

import json
import os
import re
import time
import google.generativeai as genai


VIRAL_ANALYSIS_PROMPT = """You are a viral content strategist who has studied 10,000+ viral short-form videos. Analyze this transcript and find {min_clips}-{max_clips} moments perfect for Instagram Reels.

Find moments that are:
- Counterintuitive or surprising
- Emotionally charged (pain, victory, revelation)
- Quotable and punchy
- Self-contained (make sense without context)
- Have clear tension + resolution

For each moment return ONLY valid JSON:
{{
  "clips": [
    {{
      "start_time": float,
      "end_time": float,
      "hook": "catchy 8-word headline",
      "why_viral": "one sentence explanation",
      "emotion": "shock OR inspiration OR curiosity OR humor",
      "virality_score": integer 0-100,
      "instagram_caption": "caption with hashtags",
      "youtube_description": "longer YT description"
    }}
  ]
}}

Transcript:
{full_transcript_with_timestamps}

Total Duration: {duration} seconds

Return ONLY the JSON. No markdown. No extra text."""

STRICT_FALLBACK_PROMPT = """Return ONLY a JSON object, no markdown, no backticks, no prose.
{{"clips": [{{"start_time": 0.0, "end_time": 30.0, "hook": "headline here", "why_viral": "reason", "emotion": "inspiration", "virality_score": 75, "instagram_caption": "caption #hashtag", "youtube_description": "description here"}}]}}

Transcript:
{full_transcript_with_timestamps}

Total Duration: {duration} seconds"""


def configure_gemini(api_key: str):
    """Configure the Gemini API client."""
    genai.configure(api_key=api_key)


def _compute_target_clip_count(duration: float) -> int:
    """
    Scale clip count with video duration.
    10+ minute videos should produce more clips than short videos.
    """
    override = os.getenv("ATTENTIONX_TARGET_CLIPS", "").strip()
    if override:
        try:
            return max(3, min(14, int(override)))
        except ValueError:
            pass

    duration = max(0.0, float(duration or 0.0))

    if duration < 5 * 60:
        return 5
    if duration < 10 * 60:
        return 6
    if duration < 20 * 60:
        return 8
    if duration < 35 * 60:
        return 10
    return 12


def _truncate_transcript(transcript: str, max_chars: int = 5000) -> str:
    """Truncate transcript to fit within free tier token limits."""
    if len(transcript) <= max_chars:
        return transcript
    lines = transcript.split("\n")
    total = len(lines)
    third = max_chars // 3
    first = "\n".join(lines[:total // 3])[:third]
    mid = "\n".join(lines[total // 3: 2 * total // 3])[:third]
    last = "\n".join(lines[2 * total // 3:])[:third]
    return first + "\n...\n" + mid + "\n...\n" + last


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from Gemini response."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _parse_clips_json(raw_text: str) -> dict:
    """Safely parse JSON from Gemini response."""
    cleaned = _strip_json_fences(raw_text)
    return json.loads(cleaned)


def _is_quota_or_rate_error(exc: Exception) -> bool:
    """Detect API quota/rate-limit errors from Gemini SDK exception text."""
    text = str(exc).lower()
    return (
        "429" in text
        or "quota" in text
        or "rate limit" in text
        or "resource_exhausted" in text
        or "too many requests" in text
    )


def _is_temporary_api_error(exc: Exception) -> bool:
    """Detect transient Gemini/API availability failures."""
    text = str(exc).lower()
    return (
        "timeout" in text
        or "temporarily unavailable" in text
        or "unavailable" in text
        or "internal" in text
        or "deadline exceeded" in text
        or "503" in text
        or "500" in text
    )


def _generate_with_backoff(model, prompt: str, retries: int = 2, base_delay: float = 2.5):
    """Generate Gemini response with small backoff for transient 429/limit errors."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            last_error = e
            if attempt >= retries or not (_is_quota_or_rate_error(e) or _is_temporary_api_error(e)):
                raise
            time.sleep(base_delay * (attempt + 1))
    raise last_error


def _extract_segments_from_timestamped_text(timestamped_transcript: str) -> list:
    """
    Parse lines like: [MM:SS - MM:SS] text
    Returns list of dicts with start, end, text.
    """
    pattern = re.compile(
        r"^\[(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})\]\s*(.+)$"
    )
    segments = []
    for line in timestamped_transcript.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        sm, ss, em, es, text = match.groups()
        start = int(sm) * 60 + int(ss)
        end = int(em) * 60 + int(es)
        clean_text = text.strip()
        if end > start and clean_text:
            segments.append(
                {"start": float(start), "end": float(end), "text": clean_text}
            )
    return segments


def _build_local_fallback_analysis(
    timestamped_transcript: str,
    duration: float,
    target_clip_count: int = 5,
) -> dict:
    """
    Build deterministic clips when Gemini quota is exceeded.
    Uses timestamped segments and picks long/high-signal phrases.
    """
    segments = _extract_segments_from_timestamped_text(timestamped_transcript)

    target_clip_count = max(3, min(14, int(target_clip_count or 5)))

    if not segments:
        # Hard fallback when transcript format is unknown.
        clip_len = max(12.0, min(30.0, duration / max(4.0, target_clip_count)))
        if duration > 0 and target_clip_count > 1:
            span = max(0.0, duration - clip_len)
            starts = [
                round((span * i) / (target_clip_count - 1), 2)
                for i in range(target_clip_count)
            ]
        else:
            starts = [0.0]
        clips = []
        for i, start in enumerate(starts):
            end = min(duration if duration > 0 else start + clip_len, start + clip_len)
            clips.append(
                {
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "hook": f"Key moment {i + 1} from the video",
                    "why_viral": "Short, self-contained moment suitable for reels.",
                    "emotion": "curiosity",
                    "virality_score": 70 - (i * 3),
                    "instagram_caption": "Key takeaway from this moment. #reels #viral #content",
                    "youtube_description": "A concise highlight selected from this session."
                }
            )
        return {"clips": clips, "analysis_mode": "local_fallback"}

    # Rank by text density (length / duration) to surface punchier moments.
    ranked = sorted(
        segments,
        key=lambda s: len(s["text"]) / max(1.0, s["end"] - s["start"]),
        reverse=True
    )

    chosen = []
    min_separation = max(8.0, min(18.0, duration / max(8.0, target_clip_count * 1.2)))
    for seg in ranked:
        # Keep selected clips separated to reduce overlap.
        if any(abs(seg["start"] - prev["start"]) < min_separation for prev in chosen):
            continue
        chosen.append(seg)
        if len(chosen) >= target_clip_count:
            break

    if not chosen:
        chosen = segments[:target_clip_count]

    # Top-up if we still have fewer than target clips after separation filtering.
    if len(chosen) < target_clip_count:
        for seg in ranked:
            if seg in chosen:
                continue
            chosen.append(seg)
            if len(chosen) >= target_clip_count:
                break

    clips = []
    for i, seg in enumerate(chosen, start=1):
        start_time = max(0.0, seg["start"] - 1.0)
        end_time = min(duration if duration > 0 else seg["end"], seg["end"] + 1.5)
        text_snippet = seg["text"][:120].strip()
        hook_words = text_snippet.split()[:8]
        hook = " ".join(hook_words) if hook_words else f"High-impact moment {i}"
        clips.append(
            {
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "hook": hook,
                "why_viral": "Dense statement with strong standalone replay value.",
                "emotion": "curiosity" if i % 2 else "inspiration",
                "virality_score": max(62, 82 - i * 4),
                "instagram_caption": f"{text_snippet} #reels #viral #shorts",
                "youtube_description": (
                    "Auto-selected highlight from key emotional moments. "
                    f"Transcript excerpt: {text_snippet}"
                )
            }
        )

    return {"clips": clips, "analysis_mode": "local_fallback"}


def analyze_transcript(timestamped_transcript: str, duration: float, api_key: str) -> dict:
    """Use Gemini to detect viral-worthy moments in the transcript."""
    configure_gemini(api_key)
    target_clip_count = _compute_target_clip_count(duration)
    min_clips = max(4, target_clip_count - 2)

    primary_model = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash-lite").strip()
    fallback_models_env = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-1.5-flash,gemini-1.5-flash-8b")
    model_names = [primary_model]
    for name in fallback_models_env.split(","):
        clean = name.strip()
        if clean and clean not in model_names:
            model_names.append(clean)

    short_transcript = _truncate_transcript(timestamped_transcript, max_chars=5000)

    prompt = VIRAL_ANALYSIS_PROMPT.format(
        full_transcript_with_timestamps=short_transcript,
        duration=duration,
        min_clips=min_clips,
        max_clips=target_clip_count,
    )

    fallback_prompt = STRICT_FALLBACK_PROMPT.format(
        full_transcript_with_timestamps=short_transcript,
        duration=duration
    )

    last_error = None
    last_raw = "N/A"

    for model_name in model_names:
        model = genai.GenerativeModel(model_name)
        for attempt_prompt in (prompt, fallback_prompt):
            try:
                response = _generate_with_backoff(model, attempt_prompt, retries=2)
                last_raw = getattr(response, "text", "N/A")
                result = _parse_clips_json(last_raw)
                _validate_clips(result)
                return result
            except (json.JSONDecodeError, KeyError, ValueError):
                # Try next prompt/model on malformed JSON.
                continue
            except Exception as e:
                last_error = e
                # Try next prompt/model on API failure.
                continue

    if last_error and (_is_quota_or_rate_error(last_error) or _is_temporary_api_error(last_error)):
        result = _build_local_fallback_analysis(
            short_transcript,
            duration,
            target_clip_count=target_clip_count,
        )
        _validate_clips(result)
        return result

    # Safety net: never block the pipeline if Gemini returns unusable output.
    if last_error is None:
        result = _build_local_fallback_analysis(
            short_transcript,
            duration,
            target_clip_count=target_clip_count,
        )
        _validate_clips(result)
        return result

    # Final fallback for all other Gemini errors (auth/model mismatch/etc.).
    result = _build_local_fallback_analysis(
        short_transcript,
        duration,
        target_clip_count=target_clip_count,
    )
    result["analysis_error"] = str(last_error)[:250]
    _validate_clips(result)
    return result


def _validate_clips(data: dict):
    """Validate parsed Gemini response structure."""
    if "clips" not in data:
        raise ValueError("Missing 'clips' key")
    if not isinstance(data["clips"], list) or len(data["clips"]) == 0:
        raise ValueError("'clips' must be a non-empty list")

    required_keys = {"start_time", "end_time", "hook", "why_viral", "emotion", "virality_score", "instagram_caption", "youtube_description"}
    valid_emotions = {"shock", "inspiration", "curiosity", "humor"}

    for i, clip in enumerate(data["clips"]):
        missing = required_keys - set(clip.keys())
        if missing:
            raise ValueError(f"Clip {i} missing keys: {missing}")
        clip["start_time"] = float(clip["start_time"])
        clip["end_time"] = float(clip["end_time"])
        clip["virality_score"] = int(clip["virality_score"])
        clip["emotion"] = str(clip["emotion"]).lower().strip()
        if clip["emotion"] not in valid_emotions:
            clip["emotion"] = "inspiration"
        clip["virality_score"] = max(0, min(100, clip["virality_score"]))
