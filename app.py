import os
import re
import json
import uuid
import time
import shutil
import subprocess
import smtplib
import tempfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import math

import requests
import gradio as gr
import uvicorn
from faster_whisper import WhisperModel

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    import markdown
except Exception:
    markdown = None

# NEW: OpenAI + Gemini SDKs
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None

# Optional helpers
try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None
try:
    import cv2
except Exception:
    cv2 = None
YOLO = None


def _lazy_load_yolo():
    global YOLO
    if YOLO is not None:
        return YOLO
    try:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
        return YOLO
    except Exception as e:
        print("[yolo] not available:", e)
        return None

try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

# --------------------------------------------------------------------------------
# App-level email config (hard-coded list; not exposed in UI)
APP_EMAIL_ENABLED = True
APP_EMAIL_RECIPIENTS = [
    "vibhamishra.outr@gmail.com",
    "vibhamishra0907@gmail.com",
]
# --------------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "groq": "Groq · GPT-OSS-120B",
    "scout": "Groq · Llama-4-Scout",
    "openai": "OpenAI · GPT-4o-mini",
    "gemini": "Google · Gemini",
    "or_deepseek_r1d_70b": "OpenRouter · DeepSeek R1-Distill-Llama-70B",
    "or_gemma2_9b_it": "OpenRouter · Gemma2-9B-IT",
    "ollama": "Local · Ollama",
}
MODEL_EMAIL_SECTION_KEYS = [
    "groq",
    "scout",
    "openai",
    "gemini",
    "or_deepseek_r1d_70b",
    "or_gemma2_9b_it",
]
MODEL_EMAIL_SECTION_SET = set(MODEL_EMAIL_SECTION_KEYS)

# ===================== Paths & .env loader =====================
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "workspace"
AUDIO_DIR = WORK_DIR / "audio"
UPLOAD_DIR = WORK_DIR / "uploads"
BOARD_DIR  = WORK_DIR / "boards"
for p in (WORK_DIR, AUDIO_DIR, UPLOAD_DIR, BOARD_DIR):
    p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(UPLOAD_DIR))

def _load_env():
    if load_dotenv:
        load_dotenv(BASE_DIR / ".env", override=False)
        load_dotenv(BASE_DIR / "secrets" / ".env", override=True)
    cfg = BASE_DIR / "secrets" / "config.json"
    if cfg.exists():
            data = json.loads(cfg.read_text())
            for k, v in data.items():
                os.environ.setdefault(k, str(v))

_load_env()

print("[env] MONGO_URI:", os.getenv("MONGO_URI"))
print("[env] MONGO_ENABLED:", os.getenv("MONGO_ENABLED", "1"))
print("[env] MONGO_DB_NAME:", os.getenv("MONGO_DB_NAME", "classroom"))
print("[env] MONGO_COLLECTION:", os.getenv("MONGO_COLLECTION", "classroom"))

# ===================== Mongo helpers =====================
def store_teaching_result(
    result_payload: Optional[Dict[str, Any]] = None,
    context_meta: Optional[Dict[str, Any]] = None,
):
    enabled = os.getenv("MONGO_ENABLED", "1")
    if enabled == "0":
        return None
    uri = os.getenv("MONGO_URI")
    if not uri:
        return None
    if MongoClient is None:
        print("[mongo] pymongo not installed; skipping insert.")
        return None
    db_name = os.getenv("MONGO_DB_NAME", "classroom")
    coll_name = os.getenv("MONGO_COLLECTION", "classroom")
    doc = {
        "payload": result_payload or {},
        "context": context_meta or {},
        "created_at": datetime.utcnow(),
    }
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=1000)
        result = client[db_name][coll_name].insert_one(doc)
        return str(getattr(result, "inserted_id", None))
    except Exception as e:
        print("[mongo] insert failed:", e)
        return None
    finally:
        try:
            if 'client' in locals():
                client.close()
        except Exception:
            pass

# ===================== Whisper config =====================
WHISPER_SIZE   = os.getenv("WHISPER_SIZE", "small")
COMPUTE_TYPE   = os.getenv("WHISPER_COMPUTE", "auto")
WORD_TS        = os.getenv("WORD_TS", "0") == "1"
PREFER_GST     = os.getenv("PREFER_GSTREAMER", "1") != "0"
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
VAD_MIN_SIL_MS    = int(os.getenv("WHISPER_VAD_MIN_SIL_MS", "350"))
EMIT_EVERY_SEC    = float(os.getenv("EMIT_EVERY_SEC", "1.0"))

CHUNK_SEC      = int(os.getenv("CHUNK_SEC", "900"))
MAX_DIRECT_SEC = int(os.getenv("MAX_DIRECT_SEC", "1200"))
ALWAYS_SEGMENT = os.getenv("ALWAYS_SEGMENT", "0") == "1"

TRANSLATE_DEFAULT       = os.getenv("TRANSLATE_DEFAULT", "1") == "1"
PROMPT_TRANSCRIPT_CHARS = int(os.getenv("PROMPT_TRANSCRIPT_CHARS", "1000"))

# ===== Visual analysis knobs =====
VISION_SAMPLE_SEC = float(os.getenv("VISION_SAMPLE_SEC", "1.5"))
HAND_IOU_THRESH   = float(os.getenv("HAND_IOU_THRESH", "0.35"))
MAX_BOARD_FRAMES  = int(os.getenv("MAX_BOARD_FRAMES", "8"))
BOARD_WHITE_PCT   = float(os.getenv("BOARD_WHITE_PCT", "0.22"))
BOARD_DARK_PCT    = float(os.getenv("BOARD_DARK_PCT", "0.22"))

# ===================== Gmail SMTP ENV =====================
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SMTP_DEBUG = os.getenv("SMTP_DEBUG", "0") == "1"

# ===================== Groq ENV =====================
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "900"))
GROQ_TEMP       = float(os.getenv("GROQ_TEMP", "0.2"))

# Groq Llama-4-Scout
GROQ_SCOUT_MODEL      = os.getenv("GROQ_SCOUT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_SCOUT_MAX_TOKENS = int(os.getenv("GROQ_SCOUT_MAX_TOKENS", "900"))
GROQ_SCOUT_TEMP       = float(os.getenv("GROQ_SCOUT_TEMP", "0.2"))

# ===================== OpenAI ENV (NEW) =====================
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "900"))
OPENAI_TEMP       = float(os.getenv("OPENAI_TEMP", "0.2"))

# ===================== Gemini ENV =====================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMP        = float(os.getenv("GEMINI_TEMP", "0.2"))
GEMINI_MAX_TOKENS  = int(os.getenv("GEMINI_MAX_TOKENS", "1200"))
GEMINI_CHUNK_CHARS = int(os.getenv("GEMINI_CHUNK_CHARS", "7000"))
GEMINI_OVERLAP     = int(os.getenv("GEMINI_OVERLAP", "500"))

# ===================== Ollama (feedback only) =====================
def _ollama_base_url() -> str:
    raw = (os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") or "").strip()
    raw = raw.rstrip("/")
    if raw.endswith("/api"):
        raw = raw[:-4]
    u = urlparse(raw)
    scheme = u.scheme or "http"
    host   = u.hostname or "127.0.0.1"
    port   = u.port or 11434
    return f"{scheme}://{host}:{port}"

def _ollama_probe_or_raise():
    r = requests.get(f"{_ollama_base_url()}/api/version", timeout=(5, 10))
    r.raise_for_status()

# ===================== OpenRouter ENV =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODELS = {
    "or_deepseek_r1d_70b": "deepseek/deepseek-r1-distill-llama-70b",
    "or_gemma2_9b_it":     "google/gemma-2-9b-it",
}

# Shared provider timeout
PROVIDER_TIMEOUT = int(os.getenv("PROVIDER_TIMEOUT", "120"))

# ====== local analysis helpers ======
from analyze import analyze_transcript, extract_topics

# ===================== Instantiate Whisper =====================
model = WhisperModel(
    WHISPER_SIZE,
    device="auto",
    compute_type=COMPUTE_TYPE,
    cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
    download_root=str(WORK_DIR / "models"),
)


# ===================== Text chunk util (Gemini map-reduce) =====================
def _chunk_text(text: str, target_chars: int = 7000, overlap: int = 500):
    text = text or ""
    if len(text) <= target_chars:
        return [text]
    chunks, i = [], 0
    step = max(target_chars - overlap, 1000)
    while i < len(text):
        chunks.append(text[i:i+target_chars])
        i += step
    return chunks

# ===================== Small utils =====================
def _which(bin_name: str) -> Optional[str]:
    return shutil.which(bin_name)

def _ffprobe_duration(path: Path) -> float:
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return 0.0
    res = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    try:
        return float((res.stdout or b"").decode().strip())
    except Exception:
        return 0.0

def _ffprobe_streams(path: Path) -> dict:
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return {}
    res = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_streams", "-select_streams", "a", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    try:
        return json.loads((res.stdout or b"{}").decode())
    except Exception:
        return {}

def _has_audio_stream(path: Path) -> bool:
    info = _ffprobe_streams(path)
    return bool(info.get("streams"))

def ensure_wav(src_media: Path) -> Path:
    """
    Make sure we have a 16 kHz mono WAV for streaming transcription.
    Reuses the existing extract_audio pipeline so both ffmpeg and gst fallback still apply.
    """
    out_wav = AUDIO_DIR / f"{src_media.stem}.wav"
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return out_wav
    return extract_audio(src_media)


def segment_wav(wav_path: Path, chunk_sec: int) -> List[Path]:
    """
    Convenience wrapper so streaming flow can re-use the existing segmentation logic.
    """
    return segment_media_to_wavs(wav_path, chunk_sec)


def extract_wav_slice(src_media: Path, start_sec: float, duration_sec: float, out_wav: Path) -> Path:
    """
    Extract one audio slice as 16k mono WAV so we can stream chunk 1 without
    waiting for full-media extraction/segmentation.
    """
    ffm = _which("ffmpeg")
    if not ffm:
        raise FileNotFoundError("ffmpeg not found")
    if not _has_audio_stream(src_media):
        raise RuntimeError(f"No audio stream found in file: {src_media}")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffm, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(max(0.0, float(start_sec))),
        "-i", str(src_media),
        "-t", str(max(0.1, float(duration_sec))),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or (not out_wav.exists()) or out_wav.stat().st_size == 0:
        err = (res.stderr or b"").decode(errors="ignore")[:800]
        raise RuntimeError(f"Slice extraction failed: {err}")
    return out_wav


def transcribe_wav_iter(
    wav_path: Path,
    language_hint: str,
    initial_prompt: str,
    task_mode: str,
):
    """
    Generator around WhisperModel.transcribe that yields segments incrementally.
    """
    seg_iter, info = model.transcribe(
        str(wav_path),
        task=task_mode,
        language=None if not language_hint or language_hint == "auto" else language_hint,
        beam_size=max(1, WHISPER_BEAM_SIZE),
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": VAD_MIN_SIL_MS},
        condition_on_previous_text=False,
        initial_prompt=(initial_prompt or None),
        word_timestamps=WORD_TS,
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    for s in seg_iter:
        seg = {
            "start": float(getattr(s, "start", 0.0) or 0.0),
            "end": float(getattr(s, "end", 0.0) or 0.0),
            "text": (s.text or "").strip(),
        }
        yield seg, duration

def _normalize_uploaded(vpath):
    if not vpath:
        return None
    if isinstance(vpath, str):
        return vpath
    if isinstance(vpath, (list, tuple)) and vpath:
        return vpath[0]
    if isinstance(vpath, dict):
        return vpath.get("name") or vpath.get("path")
    return None

def _mmss(sec: float) -> str:
    sec = max(float(sec or 0), 0.0)
    m = int(sec // 60); s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def _mmss_to_sec(s: str) -> float:
    try:
        mm, ss = s.strip().split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0.0

def _trim_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2, 200)
    return text[:half] + "\n...\n" + text[-half:]

# ===================== Audio extraction =====================
def _gst_extract(in_path: Path, out_wav: Path) -> None:
    gst = _which("gst-launch-1.0")
    if not gst:
        raise FileNotFoundError("gst-launch-1.0 not found")
    v = in_path.resolve().as_posix()
    a = out_wav.resolve().as_posix()
    cmd = [
        gst, "-q",
        "filesrc", f"location={v}",
        "!", "decodebin",
        "!", "audioconvert",
        "!", "audioresample",
        "!", "audio/x-raw,channels=1,rate=16000,format=S16LE",
        "!", "wavenc",
        "!", "filesink", f"location={a}",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        err = (res.stderr or b"").decode(errors="ignore")
        raise RuntimeError(f"GStreamer failed: {err[:800]}")

def _ffmpeg_extract(in_path: Path, out_wav: Path) -> None:
    ffm = _which("ffmpeg")
    if not ffm:
        raise FileNotFoundError("ffmpeg not found")
    if not _has_audio_stream(in_path):
        raise RuntimeError(f"No audio stream found in file: {in_path}")
    cmd = [
        ffm, "-hide_banner", "-nostdin", "-y",
        "-i", str(in_path),
        "-vn", "-sn", "-dn",
        "-map", "0:a:0?",
        "-c:a", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_wav),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        err = (res.stderr or b"").decode(errors="ignore")[:800]
        raise RuntimeError(err)

def extract_audio(video_path: Path) -> Path:
    out_wav = AUDIO_DIR / f"{video_path.stem}.wav"
    try:
        if PREFER_GST:
            try:
                _gst_extract(video_path, out_wav)
            except Exception as e:
                print("[extract] GStreamer failed; trying FFmpeg:", e)
                _ffmpeg_extract(video_path, out_wav)
        else:
            try:
                _ffmpeg_extract(video_path, out_wav)
            except Exception as e:
                print("[extract] FFmpeg failed; trying GStreamer:", e)
                _gst_extract(video_path, out_wav)
        return out_wav
    except Exception as e:
        print("[extract] Both extractors failed; switching to long-mode. Reason:", e)
        os.environ["ALWAYS_SEGMENT"] = "1"
        return out_wav

# ===================== URL handling =====================
def _is_http_url(s: Optional[str]) -> bool:
    if not s:
        return False
    try:
        u = urlparse(s.strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def _download_media_from_url(url: str, dest_dir: Path) -> Path:
    url = url.strip()
    file_id = uuid.uuid4().hex
    exts = (".mp4", ".mkv", ".mov", ".webm", ".mp3", ".m4a", ".wav", ".aac", ".flac")
    base_no_qs = url.split("?", 1)[0].lower()

    # Direct-file URL
    if any(base_no_qs.endswith(e) for e in exts):
        suffix = Path(base_no_qs).suffix
        dst = dest_dir / f"{file_id}{suffix}"
        with requests.get(url, stream=True, timeout=(10, 600)) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        if not dst.exists() or dst.stat().st_size == 0:
            raise gr.Error("Direct download failed or empty file.")
        return dst

    # Platforms (YouTube/Drive/etc.) via yt-dlp
    if YoutubeDL is None:
        raise gr.Error("This URL isn’t a direct media file. Install yt-dlp: pip install yt-dlp")
    outtmpl = str(dest_dir / f"{file_id}.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        # Prefer 480p or lower for much faster downloads while remaining useful for analysis
        "format": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "concurrent_fragment_downloads": 5,
        "force_ipv4": True,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "socket_timeout": 15,
        "sleep_interval": 0,
        "max_sleep_interval": 0,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
    }
    cookies_path = BASE_DIR / "secrets" / "youtube_cookies.txt"
    if cookies_path.exists():
        ydl_opts["cookiefile"] = str(cookies_path)
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            files = list(dest_dir.glob(f"{file_id}.*"))
            if not files:
                guess = Path(ydl.prepare_filename(info))
                if guess.exists():
                    return guess
                raise gr.Error("Download succeeded but cannot find the output file.")
            return files[0]
        except Exception as e:
            raise gr.Error(f"Download failed: {str(e)}")

# ===================== Long-media segmentation =====================
def segment_media_to_wavs(src_media: Path, chunk_sec: int = CHUNK_SEC) -> List[Path]:
    ffm = _which("ffmpeg")
    if not ffm:
        raise FileNotFoundError("ffmpeg not found")
    if not _has_audio_stream(src_media):
        raise RuntimeError(f"No audio stream found in file: {src_media}")

    out_dir = AUDIO_DIR / f"{src_media.stem}_chunks"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = out_dir / f"{src_media.stem}_%04d.wav"
    cmd = [
        ffm, "-hide_banner", "-loglevel", "error",
        "-i", str(src_media),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "segment", "-segment_time", str(chunk_sec),
        str(pattern),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg segmentation failed: {(res.stderr or b'').decode(errors='ignore')[:800]}")
    parts = sorted(out_dir.glob(f"{src_media.stem}_*.wav"))
    if not parts:
        raise RuntimeError("Segmentation produced no chunks.")
    return parts

# ===================== Prompt builder for feedback =====================
def _build_feedback_prompt(transcript_text: str, segments: List[dict], visual_data: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    t_snip = _trim_middle(transcript_text or "", PROMPT_TRANSCRIPT_CHARS)
    lines = []
    for s in segments[:60]:
        txt = (s.get("text","" ).strip().replace("\n"," "))[:90]
        if txt:
            lines.append(f"{_mmss(s.get('start',0))}–{_mmss(s.get('end',0))} {txt}")

    system_msg = (
        "You are “TeachCoach”. Only pedagogy-improvement advice (no audio/production; no praise). "
        "Analyze both the transcript and any visual engagement data provided. "
        "Pay special attention to whether student questions (indicated by hand raises or verbal cues) were addressed. "
        "Detect topics; then produce:\n"
        "0) Detected Topics (bullets)\n"
        "1) Top 5 Improvements (fix + why + how)\n"
        "2) Minute-by-minute Fixes “(mm:ss) → fix”\n"
        "3) Add These Examples (2–3)\n"
        "4) Ask These Questions (4)\n"
        "5) Next-Class Plan (10 steps)\n"
        "Be concrete with timestamps. Return markdown only."
    )
    user_msg = f"Transcript (truncated):\n{t_snip}\n\nSegments:\n" + "\n".join(lines)
    
    if visual_data:
        hand_events = visual_data.get("hand_raise_events", [])
        if hand_events:
            summary = "\n\nVisual Analysis (Student Engagement/Hand Raises):\n"
            # Summarize total unique or major clusters
            total_hands = visual_data.get("hand_raise_unique", 0) or len(hand_events)
            summary += f"- Estimated {total_hands} hand-raise events detected.\n"
            for ev in hand_events[:15]:
                summary += f"  * {_mmss(ev['t'])}: {ev['count']} hand(s)\n"
            user_msg += summary
        
        board_text = visual_data.get("board_text", "")
        if board_text:
            user_msg += f"\n\nBoard Content (OCR snippet):\n{board_text[:1000]}"

    return system_msg, user_msg

# ===================== Whisper transcription =====================
def transcribe_wav_chunk(path: Path, language_hint: Optional[str], initial_prompt: Optional[str],
                         prime: bool, task_mode: str = "transcribe"):
    seg_iter, info = model.transcribe(
        str(path),
        task=task_mode,
        language=None if not language_hint or language_hint == "auto" else language_hint,
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        initial_prompt=(initial_prompt if prime else None),
        word_timestamps=WORD_TS,
    )
    segs, texts = [], []
    for s in seg_iter:
        t = (s.text or "").strip()
        segs.append({"start": float(getattr(s, "start", 0.0) or 0.0),
                     "end": float(getattr(s, "end", 0.0) or 0.0),
                     "text": t})
        if t:
            texts.append(t)
    return segs, " ".join(texts), float(getattr(info, "duration", 0.0) or 0.0)

def transcribe_long(src_media: Path, language_hint: Optional[str], initial_prompt: Optional[str],
                    task_mode: str = "transcribe") -> Tuple[List[dict], str, float]:
    parts = segment_media_to_wavs(src_media, CHUNK_SEC)
    all_segments: List[Dict[str, Any]] = []
    full_text: List[str] = []
    total_off = 0.0
    for i, wav_part in enumerate(parts):
        segs, text, dur = transcribe_wav_chunk(wav_part, language_hint, initial_prompt, i == 0, task_mode)
        for s in segs:
            all_segments.append({
                "start": s["start"] + total_off,
                "end": s["end"] + total_off,
                "text": s["text"]
            })
        if text:
            full_text.append(text)
        total_off += max(dur, float(CHUNK_SEC))
    duration = all_segments[-1]["end"] if all_segments else total_off
    return all_segments, " ".join(full_text), duration

def transcribe_short(wav_path: Path, language_hint: Optional[str], initial_prompt: Optional[str],
                     task_mode: str = "transcribe") -> Tuple[List[dict], str, float]:
    segments_iter, info = model.transcribe(
        str(wav_path),
        task=task_mode,
        language=None if not language_hint or language_hint == "auto" else language_hint,
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        initial_prompt=(initial_prompt or None),
        word_timestamps=WORD_TS,
    )
    segments: List[dict] = []
    full_text = []
    for s in segments_iter:
        seg = {
            "start": float(getattr(s, "start", 0.0) or 0.0),
            "end": float(getattr(s, "end", 0.0) or 0.0),
            "text": (s.text or "").strip()
        }
        segments.append(seg)
        if seg["text"]:
            full_text.append(seg["text"])
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return segments, " ".join(full_text), duration

# ===================== Provider feedbacks =====================
def openrouter_feedback_model(transcript_text: str, segments: List[dict], model_id: str, visual_data: Optional[Dict[str, Any]] = None) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY")
    sys, usr = _build_feedback_prompt(transcript_text, segments, visual_data)
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "stream": False,
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "X-Title": "Lecture Analyzer"}, json=payload, timeout=PROVIDER_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content"," ").strip()
    if not text:
        raise RuntimeError(f"OpenRouter empty response: {j}")
    return text

def ollama_feedback(transcript_text: str, segments: List[dict], visual_data: Optional[Dict[str, Any]] = None) -> str:
    _ollama_probe_or_raise()
    base       = _ollama_base_url()
    model_id   = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
    temp       = float(os.getenv("OLLAMA_TEMP", "0.0"))
    num_ctx    = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    num_predict= int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
    timeout    = int(os.getenv("OLLAMA_TIMEOUT", "900"))

    t_snip = _trim_middle(transcript_text or "", PROMPT_TRANSCRIPT_CHARS)
    lines = []
    for s in segments[:60]:
        txt = (s.get("text","").strip().replace("\n"," "))[:90]
        if txt:
            lines.append(f"{_mmss(s.get('start',0))}–{_mmss(s.get('end',0))} {txt}")

    system_msg = (
        "You are “TeachCoach”… (pedagogy advice only). "
        "Analyze transcript and any visual data (hand raises) provided. "
        "0) Topics  1) Top 5 Improvements  2) (mm:ss)→fix  3) Examples  4) Questions(4)  5) Next-Class Plan(10)."
    )
    user_msg = f"Transcript (truncated):\n{t_snip}\n\nSegments:\n" + "\n".join(lines)
    if visual_data:
        evs = visual_data.get("hand_raise_events", [])
        if evs:
            user_msg += f"\n\nVisuals: {len(evs)} hand raise events detected."
    
    merged = f"[SYSTEM]\n{system_msg}\n[/SYSTEM]\n[USER]\n{user_msg}\n[/USER]\n"

    payload = {
        "model": model_id,
        "prompt": merged,
        "stream": False,
        "keep_alive": "2h",
        "options": {"temperature": temp, "num_ctx": num_ctx, "num_predict": num_predict, "mirostat": 0},
    }

    r = requests.post(f"{base}/api/generate", json=payload, timeout=(10, timeout))
    if r.status_code == 404:
        raise gr.Error("Ollama 404 on /api/generate. Ensure OLLAMA_URL does not include /api.")
    r.raise_for_status()
    content = (r.json().get("response") or "").strip()
    if not content:
        raise gr.Error("Ollama returned empty feedback.")
    return content

def make_groq():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    maybe_proxies = None
    if os.getenv("FORCE_GROQ_PROXIES_JSON"):
        import json as _j
        try:
            maybe_proxies = _j.loads(os.getenv("FORCE_GROQ_PROXIES_JSON"))
        except Exception:
            maybe_proxies = None
    try:
        if maybe_proxies:
            return Groq(api_key=api_key, proxies=maybe_proxies)
        return Groq(api_key=api_key)
    except TypeError:
        return Groq(api_key=api_key)

def groq_feedback(transcript_text: str, segments: List[dict], visual_data: Optional[Dict[str, Any]] = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY")
    sys, usr = _build_feedback_prompt(transcript_text, segments, visual_data)
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ],
        "temperature": GROQ_TEMP,
        "max_tokens": GROQ_MAX_TOKENS,
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=payload, timeout=60)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()

def groq_scout_feedback(transcript_text: str, segments: List[dict], visual_data: Optional[Dict[str, Any]] = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY")
    sys, usr = _build_feedback_prompt(transcript_text, segments, visual_data)
    payload = {
        "model": GROQ_SCOUT_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ],
        "temperature": GROQ_SCOUT_TEMP,
        "max_tokens": GROQ_SCOUT_MAX_TOKENS,
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=payload, timeout=60)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()

# NEW: OpenAI feedback
def openai_feedback(transcript_text: str, segments: List[dict], visual_data: Optional[Dict[str, Any]] = None) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY")
    sys_msg, usr_msg = _build_feedback_prompt(transcript_text, segments, visual_data)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg},
        ],
        "temperature": 0.2,
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, json=payload, timeout=60)
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned empty text")
    return text

# UPDATED: Gemini feedback (map-reduce for long transcripts with parallel chunking)
def gemini_feedback(transcript_text: str, segments: list[dict], visual_data: Optional[Dict[str, Any]] = None) -> str:
    if genai is None:
        raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai")
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)

    system_msg, user_msg = _build_feedback_prompt(transcript_text, segments, visual_data)
    parts = _chunk_text(user_msg, GEMINI_CHUNK_CHARS, GEMINI_OVERLAP)
    
    config = types.GenerateContentConfig(
        system_instruction=system_msg,
        temperature=GEMINI_TEMP,
        max_output_tokens=GEMINI_MAX_TOKENS,
    )

    def process_chunk(idx, ch):
        prompt = f"(Part {idx} of {len(parts)})\n{ch}"
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            return f"[Chunk {idx} failed: {e}]"

    if len(parts) > 1:
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            results = list(pool.map(lambda p: process_chunk(p[0]+1, p[1]), enumerate(parts)))
        partials = results
    else:
        partials = [process_chunk(1, parts[0])] if parts else []

    if len(partials) == 0:
        return "*No feedback generated.*"
    if len(partials) == 1:
        return partials[0]
        
    combo_prompt = "Combine and deduplicate the partial analyses below into ONE cohesive report, strictly following the same 'TeachCoach' rubric with timestamps kept where present.\n\n" + "\n\n---\n\n".join(partials)
    
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=combo_prompt,
        config=config,
    )
    return (getattr(resp, "text", None) or "").strip()

# ===================== Heuristic Q&A (local) =====================
Q_RE = re.compile(
    r"(?:^|[\s\"“])(?:why|what|how|when|where|which|can|could|should|would|is|are|do|does)\b.*?\?",
    re.IGNORECASE
)

def qna_heuristic(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    questions = []
    sid_counter = 0
    last_sid = None
    last_q_end = -1e9
    for i, s in enumerate(segments):
        txt = (s.get("text","") or "").strip()
        if not txt:
            continue
        if "?" in txt or Q_RE.search(txt):
            sid = None
            if (s["start"] - last_q_end) <= 90 and last_sid is not None:
                sid = last_sid
            if not sid:
                sid_counter += 1
                sid = f"s{sid_counter}"
            last_sid = sid
            last_q_end = s["end"]
            answered = False
            for j in range(i+1, min(i+12, len(segments))):
                s2 = segments[j]
                if (s2["start"] - s["end"]) > 120:
                    break
                txt2 = (s2.get("text","") or "").strip()
                if len(txt2.split()) >= 6 and "?" not in txt2:
                    answered = True
                    break
            questions.append({
                "t_start": float(s.get("t_start", s.get("start",0.0))),
                "t_end": float(s.get("t_end", s.get("end",0.0))),
                "question": txt,
                "student_id": sid,
                "answered": answered,
            })
    return {"questions": questions}

def _items_from_heuristic(segments: List[dict]) -> List[dict]:
    items = []
    for s in segments:
        txt = s.get("text", "").strip()
        if "?" in txt:
            items.append({
                "t": _mmss(s.get("start", 0)),
                "speaker": "student" if len(txt) < 150 else "teacher",
                "student_id": "S1" if len(txt) < 150 else "T1",
                "answered": True,
                "text": txt
            })
    return items

def extract_qna_with_ai(transcript_text: str, segments: List[dict], visual_data: Optional[Dict[str, Any]] = None) -> Tuple[List[dict], str]:
    """Uses Groq to extract Q&A items and generate a one-line insight."""
    if not GROQ_API_KEY:
        # Fallback to heuristic if no key
        return _items_from_heuristic(segments), "AI extraction skipped (no API key)."

    t_snip = _trim_middle(transcript_text or "", 3000)
    
    hand_info = ""
    if visual_data:
        evs = visual_data.get("hand_raise_events", [])
        if evs:
            hand_info = "Visual hand-raise events detected at: " + ", ".join([_mmss(e['t']) for e in evs[:10]])

    prompt = (
        "You are an expert education analyst. Extract student and teacher questions from the transcript. "
        "Also consider the visual hand-raise data provided. "
        "Return a JSON object with two fields:\n"
        "1) 'items': a list of objects with {t: 'mm:ss', speaker: 'student'|'teacher', student_id: 'S1'|'T1'|..., answered: true|false, text: '...'}\n"
        "2) 'insight': a one-sentence summary of the class engagement level.\n\n"
        f"Transcript:\n{t_snip}\n\n{hand_info}"
    )

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                          json={
                              "model": "llama-3.3-70b-versatile",
                              "messages": [{"role": "user", "content": prompt}],
                              "response_format": {"type": "json_object"},
                              "temperature": 0.1
                          }, timeout=15)
        r.raise_for_status()
        import json as json_mod
        data = json_mod.loads(r.json()["choices"][0]["message"]["content"])
        items = data.get("items", [])
        insight = data.get("insight", "Great classroom interaction!")
        return items, insight
    except Exception as e:
        print(f"[QnA AI] Failed: {e}")
        return _items_from_heuristic(segments), "Could not generate AI insight."

# ===================== Gmail SMTP helpers =====================
def send_gmail_smtp(to_email: str, subject: str, body_text: str, body_html: Optional[str] = None) -> None:
    if not (GMAIL_USER and GMAIL_APP_PASSWORD):
        raise RuntimeError("Set GMAIL_USER and GMAIL_APP_PASSWORD env vars first.")
    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body_text or "")
    if body_html:
        msg.add_alternative(body_html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        if SMTP_DEBUG:
            s.set_debuglevel(1)
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)

def send_report_to_recipients(recipients: List[str], subject: str, body_text: str, body_html: Optional[str] = None) -> None:
    for addr in (recipients or []):
        a = (addr or "").strip()
        if not a:
            continue
        try:
            send_gmail_smtp(a, subject, body_text, body_html)
            print(f"[email] sent → {a}")
        except Exception as e:
            print(f"[email] failed → {a}: {e}")


def _parse_recipient_list(raw_list: Optional[str]) -> List[str]:
    if not raw_list:
        return []
    parts = re.split(r"[;,]", raw_list)
    return [p.strip() for p in parts if p and p.strip()]


def _qna_summary_pairs(q_summary: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [
        ("Total items", q_summary.get("total_items", 0)),
        ("Student questions", q_summary.get("total_student_questions", 0)),
        ("Teacher questions", q_summary.get("total_teacher_questions", 0)),
        ("Answered (est.)", q_summary.get("answered", 0)),
        ("Unanswered (est.)", q_summary.get("unanswered", 0)),
        ("Unique students (est.)", q_summary.get("unique_students_est", 0)),
    ]


def _format_qna_summary_text(q_summary: Dict[str, Any]) -> str:
    return " | ".join(f"{label}: {value}" for label, value in _qna_summary_pairs(q_summary))


def _format_qna_summary_html(q_summary: Dict[str, Any]) -> str:
    return "<br/>".join(f"<strong>{label}:</strong> {value}" for label, value in _qna_summary_pairs(q_summary))


def _markdown_to_html(text: str) -> str:
    """Convert markdown text to HTML for email formatting."""
    if markdown is None:
        # Fallback: basic conversion if markdown library not available
        # Convert **bold** to <strong>bold</strong>
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # Convert * to • for bullets (basic)
        text = re.sub(r'^\* (.+)$', r'• \1', text, flags=re.MULTILINE)
        # Convert numbered lists
        text = re.sub(r'^\d+\. (.+)$', r'• \1', text, flags=re.MULTILINE)
        # Convert headers
        text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        return text.replace('\n', '<br>')
    else:
        # Use proper markdown conversion
        html = markdown.markdown(text, extensions=['extra'])
        # Add some basic styling
        html = html.replace('<h1>', '<h1 style="color:#2c3e50;margin-top:20px;">')
        html = html.replace('<h2>', '<h2 style="color:#34495e;margin-top:18px;">')
        html = html.replace('<h3>', '<h3 style="color:#7f8c8d;margin-top:16px;">')
        html = html.replace('<h4>', '<h4 style="color:#95a5a6;margin-top:14px;">')
        html = html.replace('<ul>', '<ul style="margin:10px 0;padding-left:20px;">')
        html = html.replace('<ol>', '<ol style="margin:10px 0;padding-left:20px;">')
        html = html.replace('<li>', '<li style="margin:5px 0;">')
        html = html.replace('<p>', '<p style="margin:10px 0;line-height:1.5;">')
        html = html.replace('<strong>', '<strong style="font-weight:bold;">')
        html = html.replace('<em>', '<em style="font-style:italic;">')
        return html


def _format_qna_details(qna_rows: List[List[Any]], limit: int = 20) -> Tuple[str, str]:
    text_lines: List[str] = []
    html_items: List[str] = []
    total = len(qna_rows or [])
    rows = qna_rows or []
    for idx, row in enumerate(rows):
        if idx >= limit:
            remaining = total - limit
            text_lines.append(f"... ({remaining} more)")
            html_items.append(f"<li>... ({remaining} more)</li>")
            break
        start_ts = _mmss(row[0] if len(row) > 0 else 0.0)
        student_id = row[2] if len(row) > 2 else "—"
        answered_flag = bool(row[3]) if len(row) > 3 else False
        question = row[4] if len(row) > 4 else ""
        answered_text = "answered" if answered_flag else "pending"
        text_lines.append(f"{start_ts} · {student_id} · {answered_text.upper()} · {question}")
        html_items.append(
            f"<li><strong>{start_ts}</strong> · {student_id} · {answered_text.title()} · {question}</li>"
        )
    if not text_lines:
        text_lines.append("No Q&A items detected.")
        html_items.append("<li>No Q&A items detected.</li>")
    return "\n".join(text_lines), "<ul>" + "".join(html_items) + "</ul>"


def send_model_feedback_email(model_key: str, recipients_text: str, state: Dict[str, Any]) -> str:
    if not APP_EMAIL_ENABLED:
        raise gr.Error("Email sending is disabled on this server.")
    if not state or not state.get("feedback_map"):
        raise gr.Error("Run Generate Feedback first to load model responses.")

    feedback_map = state.get("feedback_map") or {}
    feedback_text = (feedback_map.get(model_key) or "").strip()
    model_label = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    if not feedback_text:
        raise gr.Error(f"No {model_label} feedback available yet.")

    recipients = _parse_recipient_list(recipients_text) or APP_EMAIL_RECIPIENTS
    if not recipients:
        raise gr.Error("Provide at least one recipient email.")

    q_summary = state.get("qna_summary") or {}
    qna_rows = state.get("qna_rows") or []

    summary_text = _format_qna_summary_text(q_summary)
    summary_html = _format_qna_summary_html(q_summary)
    detail_text, detail_html = _format_qna_details(qna_rows, limit=25)

    file_id = state.get("file_id") or "lecture"
    subject = f"{model_label} feedback report — {file_id}"
    # Format text body with better spacing and structure
    text_body = (
        f"{'='*50}\n"
        f"{model_label.upper()} FEEDBACK REPORT\n"
        f"{'='*50}\n\n"
        f"FEEDBACK CONTENT:\n"
        f"{'-'*20}\n"
        f"{feedback_text}\n\n"
        f"{'='*30}\n"
        f"Q&A SUMMARY:\n"
        f"{'-'*15}\n"
        f"{summary_text or 'No Q&A summary available.'}\n\n"
        f"{'='*30}\n"
        f"Q&A ITEMS:\n"
        f"{'-'*12}\n"
        f"{detail_text}\n"
        f"{'='*50}\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    # Convert markdown feedback to HTML for better email formatting
    feedback_html = _markdown_to_html(feedback_text)

    html_body = (
        f"<h2 style='color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px;'>{model_label} Feedback</h2>"
        f"<div style='background:#f8f9fa;padding:15px;border-radius:5px;margin:10px 0;'>"
        f"{feedback_html}"
        f"</div>"
        "<h3 style='color:#34495e;margin-top:20px;'>Q&amp;A Summary</h3>"
        f"<div style='background:#ecf0f1;padding:10px;border-radius:3px;margin:10px 0;'>"
        f"{summary_html or 'No Q&amp;A summary available.'}"
        f"</div>"
        "<h3 style='color:#34495e;margin-top:20px;'>Q&amp;A Items</h3>"
        f"<div style='background:#ecf0f1;padding:10px;border-radius:3px;margin:10px 0;'>"
        f"{detail_html}"
        f"</div>"
    )

    try:
        send_report_to_recipients(recipients, subject, text_body, html_body)
    except Exception as e:
        raise gr.Error(f"Failed to send email: {e}") from e

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"✅ Sent {model_label} feedback to {', '.join(recipients)} at {ts}."

# ===================== ADAPTIVE WEIGHTING (NEW) =====================

WEIGHTS_FILE = os.environ.get("MODEL_WEIGHTS_FILE", str(BASE_DIR / "model_weights.json"))

# Provider base weights (least → most)
PROVIDER_BASE_WEIGHTS = {
    "openrouter": 1.0,   # least
    "groq":       2.0,
    "ollama":     3.0,
    "gpt":        5.0,   # most
    "gemini":     5.0,   # most (same as GPT)
}

BONUS_SCALE = 1.25        # strength of vote effect
BONUS_MIN   = -2.0
BONUS_MAX   =  2.0

def _load_weights() -> Dict[str, Any]:
    if not os.path.exists(WEIGHTS_FILE):
        return {"votes": {}, "version": 1}
    try:
        return json.loads(Path(WEIGHTS_FILE).read_text())
    except Exception:
        return {"votes": {}, "version": 1}

def _save_weights(data: Dict[str, Any]) -> None:
    Path(WEIGHTS_FILE).write_text(json.dumps(data, indent=2, ensure_ascii=False))

def _provider_for_model(model_key: str) -> str:
    # normalize key to provider bucket
    if model_key in ("openai",):
        return "gpt"
    if model_key in ("gemini",):
        return "gemini"
    if model_key in ("groq", "scout"):
        return "groq"
    if model_key.startswith("or_"):
        return "openrouter"
    if model_key in ("ollama",):
        return "ollama"
    # fallback: assume lower preference
    return "openrouter"

def _base_weight(model_key: str) -> float:
    return PROVIDER_BASE_WEIGHTS.get(_provider_for_model(model_key), 1.0)

def _vote_tuple(weights_store: Dict[str, Any], key: str) -> Tuple[int, int]:
    v = weights_store.get("votes", {}).get(key, {})
    return int(v.get("up", 0)), int(v.get("down", 0))

def _bonus_from_votes(up: int, down: int) -> float:
    # Smooth, spam-resistant; ln ratio with clamp
    import math
    bonus = math.log((up + 1) / (down + 1.0)) * BONUS_SCALE
    return float(max(BONUS_MIN, min(BONUS_MAX, bonus)))

def effective_weight(weights_store: Dict[str, Any], model_key: str) -> float:
    base = _base_weight(model_key)
    up, down = _vote_tuple(weights_store, model_key)
    bonus = _bonus_from_votes(up, down)
    return base + bonus

def register_vote(model_key: str, upvote: bool) -> Dict[str, Any]:
    store = _load_weights()
    store.setdefault("votes", {})
    rec = store["votes"].setdefault(model_key, {"up": 0, "down": 0})
    if upvote:
        rec["up"] = int(rec.get("up", 0)) + 1
    else:
        rec["down"] = int(rec.get("down", 0)) + 1
    _save_weights(store)
    return store

def ranking_md(weights_store: Dict[str, Any], available_models: List[str]) -> str:
    rows = []
    for k in available_models:
        up, down = _vote_tuple(weights_store, k)
        wt = effective_weight(weights_store, k)
        rows.append((k, wt, up, down, _provider_for_model(k), _base_weight(k)))
    rows.sort(key=lambda r: r[1], reverse=True)
    lines = ["**Model Ranking (higher is preferred)**",
             "",
             "| Rank | Model | Provider | Base | Bonus(votes) | Up | Down | Eff. Weight |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for i, (k, wt, up, down, prov, base) in enumerate(rows, 1):
        bonus = wt - base
        label = MODEL_DISPLAY_NAMES.get(k, k)
        lines.append(f"| {i} | {label} | {prov} | {base:.2f} | {bonus:+.2f} | {up} | {down} | {wt:.2f} |")
    return "\n".join(lines)

# ===================== Multi-provider selector (with ordering) =====================
# mode one of:
#   'groq' | 'scout' | 'ollama' |
#   'or_deepseek_r1d_70b' | 'or_gemma2_9b_it' |
#   'openai' | 'gemini' |
#   'both' (groq + ollama) | 'all' (all seven)

def get_feedbacks(transcript_text: str, segments: List[dict], mode: str, visual_data: Optional[Dict[str, Any]] = None) -> dict:
    m = (mode or "groq").lower().strip()
    key_map = {
        "groq": "groq",
        "groq (gpt-oss-120b)": "groq",
        "scout": "scout",
        "groq (llama-4-scout)": "scout",
        "openai": "openai",
        "openai · gpt-4o-mini": "openai",
        "gemini": "gemini",
        "google · gemini": "gemini",
        "or_deepseek_r1d_70b": "or_deepseek_r1d_70b",
        "openrouter · deepseek r1-distill-llama-70b": "or_deepseek_r1d_70b",
        "or_gemma2_9b_it": "or_gemma2_9b_it",
        "openrouter · gemma2-9b-it": "or_gemma2_9b_it",
        "ollama": "ollama",
        "local · ollama": "ollama",
    }
    
    internal_mode = key_map.get(m, m)

    if internal_mode == "all":
        want = {"groq","scout","ollama","or_deepseek_r1d_70b","or_gemma2_9b_it","openai","gemini"}
    elif internal_mode == "both":
        want = {"groq","ollama"}
    else:
        want = {internal_mode}

    calls = {}
    with ThreadPoolExecutor(max_workers=len(want) or 1) as pool:
        if "groq" in want:
            calls["groq"] = pool.submit(groq_feedback, transcript_text, segments, visual_data)
        if "scout" in want:
            calls["scout"] = pool.submit(groq_scout_feedback, transcript_text, segments, visual_data)
        if "ollama" in want:
            calls["ollama"] = pool.submit(ollama_feedback, transcript_text, segments, visual_data)
        if "openai" in want:
            calls["openai"] = pool.submit(openai_feedback, transcript_text, segments, visual_data)
        if "gemini" in want:
            calls["gemini"] = pool.submit(gemini_feedback, transcript_text, segments, visual_data)
        for key, model_id in OPENROUTER_MODELS.items():
            if key in want:
                calls[key] = pool.submit(openrouter_feedback_model, transcript_text, segments, model_id, visual_data)

        out = {}
        for name, fut in calls.items():
            try:
                out[name] = fut.result()
            except Exception as e:
                out[name] = f"[{name} failed: {e}]"

        # ORDER by effective weight (desc)
        store = _load_weights()
        ordered = sorted(out.keys(), key=lambda k: effective_weight(store, k), reverse=True)
        out["_ordered_keys"] = ordered
        return out

# ===================== Helpers for Q&A table =====================
def qna_rows_from_items(items: List[dict]) -> List[List[Any]]:
    rows = []
    for it in items:
        tsec = _mmss_to_sec(it.get("t","00:00"))
        spk  = (it.get("speaker") or "").lower()
        sid  = it.get("student_id") or ("S?" if spk == "student" else "T")
        rows.append([round(tsec,2), round(tsec,2), sid, bool(it.get("answered", False)), it.get("text",""), "", spk])
    return rows

def qna_summary_from_items(items: List[dict], visual_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    total_stu = sum(1 for it in items if (it.get("speaker") or "").lower() == "student")
    total_tea = sum(1 for it in items if (it.get("speaker") or "").lower() == "teacher")
    uniq_students = len({(it.get("student_id") or "").lower()
                         for it in items if (it.get("speaker") or "").lower()=="student" and it.get("student_id")})
    answered_est = sum(1 for it in items if (it.get("speaker") or "").lower()=="student" and bool(it.get("answered")))
    
    hand_raises = 0
    if visual_data:
        # We can use the total unique estimate or just the event count
        hand_raises = visual_data.get("hand_raise_unique", 0) or len(visual_data.get("hand_raise_events", []))

    return {
        "total_items": len(items),
        "total_student_questions": total_stu,
        "total_teacher_questions": total_tea,
        "unique_students_est": uniq_students,
        "answered": answered_est,
        "unanswered": max(total_stu - answered_est, 0),
        "hand_raises": hand_raises
    }

# ===================== Streaming helpers =====================
def pack_outputs(
    transcript_text: str,
    seg_rows: List[List[Any]],
    primary_text: str,
    engine_note: str,
    fb_groq: str,
    fb_scout: str,
    fb_openai: str,
    fb_gemini: str,
    fb_ollama: str,
    fb_or_deepseek: str,
    fb_or_gemma: str,
    qna_md: str,
    qna_rows: List[List[Any]],
    visuals_md: str,
    gallery_imgs: List[Any],
    ranking_md: str,
    vote_dd_update,
    raw_json: str,
    state_dict: Dict[str, Any],
    email_status_map: Optional[Dict[str, str]] = None,
):
    status_map = email_status_map or {}
    additional_models = gr.update(value=fb_ollama or "", visible=bool(fb_ollama))
    return (
        transcript_text or "",
        seg_rows or [],
        primary_text or "",
        engine_note or "",
        fb_groq or "",
        fb_scout or "",
        fb_openai or "",
        fb_gemini or "",
        additional_models,
        fb_or_deepseek or "",
        fb_or_gemma or "",
        qna_md or "",
        qna_rows or [],
        visuals_md or "",
        gallery_imgs or [],
        ranking_md or "—",
        vote_dd_update if vote_dd_update is not None else gr.update(),
        raw_json or "{}",
        state_dict or {},
        *(status_map.get(key, "") for key in MODEL_EMAIL_SECTION_KEYS),
    )


def transcribe_stream(
    src_file: Optional[str],
    video_url: Optional[str],
    mic_audio: Optional[str],
    video_input: Optional[str],
    language_hint: str,
    initial_prompt: str,
    translate_to_en: bool,
    progress=gr.Progress(),
):
    file_id = uuid.uuid4().hex
    state = {
        "file_id": file_id,
        "media_path": "",
        "wav_path": str(AUDIO_DIR / f"{file_id}.wav"),
        "duration_sec": 0.0,
        "segments": [],
        "transcript_text": "",
        "qna_rows": [],
        "qna_summary": {},
        "feedback_map": {},
        "ordered_keys": [],
        "ranking_md": "—",
        "available_models": [],
    }

    yield pack_outputs(
        transcript_text="",
        seg_rows=[],
        primary_text="Connecting / Preparing input…",
        engine_note="—",
        fb_openai="",
        fb_gemini="",
        fb_groq="",
        fb_scout="",
        fb_ollama="",
        fb_or_deepseek="",
        fb_or_gemma="",
        qna_md="",
        qna_rows=[],
        visuals_md="",
        gallery_imgs=[],
        ranking_md="—",
        vote_dd_update=gr.update(choices=[], value=None),
        raw_json=json.dumps({"file_id": file_id, "status": "starting"}, indent=2),
        state_dict=state,
    )

    src_file = _normalize_uploaded(src_file)
    mic_audio = _normalize_uploaded(mic_audio)
    video_input = _normalize_uploaded(video_input)
    chosen: Optional[Path] = None
    dst_media = None

    if src_file and Path(src_file).exists():
        src_path = Path(src_file)
        dst_media = WORK_DIR / f"{file_id}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst_media)
    elif video_input and Path(video_input).exists():
        src_path = Path(video_input)
        dst_media = WORK_DIR / f"{file_id}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst_media)
    elif mic_audio and Path(mic_audio).exists():
        src_path = Path(mic_audio)
        dst_media = WORK_DIR / f"{file_id}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst_media)
    elif _is_http_url(video_url):
        progress(0.05, desc="Downloading URL…")
        dst_media = _download_media_from_url(video_url.strip(), WORK_DIR)
        if dst_media and dst_media.exists():
            new_dst = WORK_DIR / f"{file_id}{dst_media.suffix.lower()}"
            dst_media.rename(new_dst)
            dst_media = new_dst

    if not dst_media or not dst_media.exists():
        raise gr.Error("No valid media provided or download failed.")

    state["media_path"] = str(dst_media)
    audio_len = _ffprobe_duration(dst_media)
    state["duration_sec"] = float(audio_len or 0.0)

    use_long = ALWAYS_SEGMENT or (audio_len and audio_len > MAX_DIRECT_SEC)
    task_mode = "translate" if translate_to_en else "transcribe"

    import threading
    def _bg_store():
        mongo_id = store_teaching_result(
            result_payload={
                "type": "transcription",
                "file_id": file_id,
                "transcript": "",
                "segments": [],
                "qna_summary": {},
                "duration_sec": state["duration_sec"],
            },
            context_meta={"media_path": state["media_path"]},
        )
        state["mongo_saved"] = bool(mongo_id)
        state["mongo_id"] = mongo_id
    
    threading.Thread(target=_bg_store, daemon=True).start()

    yield pack_outputs(
        transcript_text="",
        seg_rows=[],
        primary_text="Transcribing…",
        engine_note="—",
        fb_openai="",
        fb_gemini="",
        fb_groq="",
        fb_scout="",
        fb_ollama="",
        fb_or_deepseek="",
        fb_or_gemma="",
        qna_md="",
        qna_rows=[],
        visuals_md="",
        gallery_imgs=[],
        ranking_md="—",
        vote_dd_update=gr.update(choices=[], value=None),
        raw_json=json.dumps({"file_id": file_id, "status": "starting"}, indent=2),
        state_dict=state,
    )

    all_segments: List[dict] = []
    transcript_parts: List[str] = []
    last_emit = time.time()
    total_off = 0.0

    def emit(status: str):
        nonlocal last_emit
        now = time.time()
        if now - last_emit < EMIT_EVERY_SEC:
            return None
        last_emit = now
        state["segments"] = all_segments
        state["transcript_text"] = " ".join(transcript_parts)
        seg_rows = [[round(s["start"], 2), round(s["end"], 2), s.get("text", "")] for s in all_segments]
        return pack_outputs(
            transcript_text=state["transcript_text"],
            seg_rows=seg_rows,
            primary_text="Transcribing…",
            engine_note="—",
            fb_openai="",
            fb_gemini="",
            fb_groq="",
            fb_scout="",
            fb_ollama="",
            fb_or_deepseek="",
            fb_or_gemma="",
            qna_md="",
            qna_rows=[],
            visuals_md="",
            gallery_imgs=[],
            ranking_md="—",
            vote_dd_update=gr.update(choices=[], value=None),
            raw_json=json.dumps({"file_id": file_id, "status": status, "segments": len(all_segments)}, indent=2),
            state_dict=state,
        )

    if use_long:
        if audio_len and audio_len > 0:
            total_parts = max(1, int(math.ceil(float(audio_len) / float(CHUNK_SEC))))
            progress(0.08, desc=f"Starting chunked streaming ({total_parts} chunks)…")
            
            # PIONEER CHUNK: Use a smaller first chunk (e.g. 15s) for instant feedback
            pioneer_sec = min(15.0, float(CHUNK_SEC))
            
            for idx in range(1, total_parts + 2): # +1 potential extra if pioneer shift happens
                if idx == 1:
                    start_sec = 0.0
                    dur_sec = min(pioneer_sec, float(audio_len))
                else:
                    # Adjust subsequent chunks to skip the pioneer part
                    start_sec = pioneer_sec + (idx - 2) * CHUNK_SEC
                    dur_sec = min(float(CHUNK_SEC), float(audio_len) - start_sec)
                
                if dur_sec <= 0:
                    break
                    
                part = AUDIO_DIR / f"{file_id}_part_{idx:04d}.wav"
                progress(0.10 + 0.70 * (min(idx, total_parts) / total_parts), desc=f"Chunk {idx}")
                try:
                    extract_wav_slice(dst_media, start_sec, dur_sec, part)
                    first_prompt = initial_prompt if idx == 1 else ""
                    for seg, _dur in transcribe_wav_iter(part, language_hint, first_prompt, task_mode):
                        seg2 = {"start": seg["start"] + start_sec, "end": seg["end"] + start_sec, "text": seg["text"]}
                        all_segments.append(seg2)
                        if seg2["text"]:
                            transcript_parts.append(seg2["text"])
                        packed = emit(status=f"chunk {idx}")
                        if packed is not None:
                            yield packed
                finally:
                    try:
                        if part.exists():
                            part.unlink()
                    except Exception:
                        pass
        else:
            progress(0.08, desc=f"Segmenting audio into {CHUNK_SEC}s parts…")
            wav = ensure_wav(dst_media)
            parts = segment_wav(wav, CHUNK_SEC)
            for idx, part in enumerate(parts, 1):
                progress(0.10 + 0.70 * (idx / max(1, len(parts))), desc=f"Chunk {idx}/{len(parts)}")
                first_prompt = initial_prompt if idx == 1 else ""
                for seg, _dur in transcribe_wav_iter(part, language_hint, first_prompt, task_mode):
                    seg2 = {"start": seg["start"] + total_off, "end": seg["end"] + total_off, "text": seg["text"]}
                    all_segments.append(seg2)
                    if seg2["text"]:
                        transcript_parts.append(seg2["text"])
                    packed = emit(status=f"chunk {idx}/{len(parts)}")
                    if packed is not None:
                        yield packed
                total_off += float(CHUNK_SEC)
    else:
        progress(0.10, desc="Transcribing (streaming)…")
        try:
            for seg, _dur in transcribe_wav_iter(dst_media, language_hint, initial_prompt, task_mode):
                all_segments.append(seg)
                if seg["text"]:
                    transcript_parts.append(seg["text"])
                packed = emit(status="streaming")
                if packed is not None:
                    yield packed
        except Exception:
            wav = ensure_wav(dst_media)
            for seg, _dur in transcribe_wav_iter(wav, language_hint, initial_prompt, task_mode):
                all_segments.append(seg)
                if seg["text"]:
                    transcript_parts.append(seg["text"])
                packed = emit(status="streaming")
                if packed is not None:
                    yield packed

    progress(0.90, desc="Deriving Q&A…")
    transcript_text = " ".join(transcript_parts)
    items = _items_from_heuristic(all_segments)
    q_rows = qna_rows_from_items(items)
    q_summary = qna_summary_from_items(items)

    state["segments"] = all_segments
    state["transcript_text"] = transcript_text
    state["qna_rows"] = q_rows
    state["qna_summary"] = q_summary

    try:
        analyze_transcript(transcript_text, all_segments, float(audio_len or 0.0))
    except Exception as e:
        print("[analyze_transcript] failed:", e)
    try:
        extract_topics(transcript_text)
    except Exception as e:
        print("[extract_topics] failed:", e)

    qna_md = (
        f"*Total items:* {q_summary.get('total_items',0)}  |  "
        f"*Student questions:* {q_summary.get('total_student_questions',0)}  |  "
        f"*Teacher questions:* {q_summary.get('total_teacher_questions',0)}  |  "
        f"*Answered (est):* {q_summary.get('answered',0)}  |  "
        f"*Unanswered (est):* {q_summary.get('unanswered',0)}  |  "
        f"*Unique students (est):* {q_summary.get('unique_students_est',0)}"
    )

    seg_rows = [[round(s["start"], 2), round(s["end"], 2), s.get("text", "")] for s in all_segments]

    yield pack_outputs(
        transcript_text=transcript_text,
        seg_rows=seg_rows,
        primary_text="Transcript ready ✅ (click Generate Feedback if needed)",
        engine_note="—",
        fb_openai="",
        fb_gemini="",
        fb_groq="",
        fb_scout="",
        fb_ollama="",
        fb_or_deepseek="",
        fb_or_gemma="",
        qna_md=qna_md,
        qna_rows=q_rows,
        visuals_md="",
        gallery_imgs=[],
        ranking_md=state.get("ranking_md", "—"),
        vote_dd_update=gr.update(choices=[], value=None),
        raw_json=json.dumps(state, ensure_ascii=False, indent=2),
        state_dict=state,
    )


def generate_feedback(
    state: Dict[str, Any],
    feedback_engine_choice: str,
    progress=gr.Progress(),
):
    if not state or not state.get("file_id"):
        raise gr.Error("First click Transcribe.")

    progress(0.05, desc="Generating feedback…")
    transcript_text = state.get("transcript_text", "")
    segments = state.get("segments", [])

    mode_map = {
        "Groq (gpt-oss-120b)": "groq",
        "Groq (Llama-4-Scout)": "scout",
        "OpenAI · GPT-4o-mini": "openai",
        "Google · Gemini": "gemini",
        "OpenRouter · DeepSeek R1-Distill-Llama-70B": "or_deepseek_r1d_70b",
        "OpenRouter · Gemma2-9B-IT": "or_gemma2_9b_it",
        "All (compare)": "all",
    }
    mode = mode_map.get(feedback_engine_choice, "groq")

    # Assuming visual data might be retrieved from state if analyze_visuals was run
    visual_data = state.get("visuals")
    new_fb_map = get_feedbacks(transcript_text, segments, mode, visual_data=visual_data)
    new_ordered_keys = new_fb_map.pop("_ordered_keys", [])
    
    fb_map = state.get("feedback_map", {})
    for k, v in new_fb_map.items():
        fb_map[k] = v
        
    ordered_keys = state.get("ordered_keys", [])
    for k in new_ordered_keys:
        if k not in ordered_keys:
            ordered_keys.append(k)
    # If no keys existed yet, add from dictionary keys
    for k in fb_map.keys():
        if k not in ordered_keys:
            ordered_keys.append(k)

    primary = "*No feedback produced.*"
    for k in ordered_keys:
        t = fb_map.get(k)
        if t and not str(t).startswith("["):
            primary = t
            break

    store = _load_weights()
    available_models = ordered_keys or list(fb_map.keys())
    rank_md = ranking_md(store, available_models)

    state["feedback_map"] = fb_map
    state["ordered_keys"] = ordered_keys
    state["available_models"] = available_models
    state["ranking_md"] = rank_md

    seg_rows = [[round(s["start"], 2), round(s["end"], 2), s.get("text", "")] for s in (segments or [])]

    q_summary = state.get("qna_summary") or {}
    qna_md = (
        f"*Total items:* {q_summary.get('total_items',0)}  |  "
        f"*Student questions:* {q_summary.get('total_student_questions',0)}  |  "
        f"*Teacher questions:* {q_summary.get('total_teacher_questions',0)}  |  "
        f"*Answered (est):* {q_summary.get('answered',0)}  |  "
        f"*Unanswered (est):* {q_summary.get('unanswered',0)}  |  "
        f"*Unique students (est):* {q_summary.get('unique_students_est',0)}"
    )

    raw_json = json.dumps(state, ensure_ascii=False, indent=2)
    dd_update = gr.update(choices=available_models, value=(available_models[0] if available_models else None))

    mongo_id = store_teaching_result(
        result_payload={
            "type": "feedback",
            "file_id": state.get("file_id"),
            "feedback_map": state.get("feedback_map"),
            "ranking_md": state.get("ranking_md"),
            "ordered_models": ordered_keys,
        },
        context_meta={"available_models": available_models},
    )
    state["mongo_feedback_saved"] = bool(mongo_id)
    state["mongo_feedback_id"] = mongo_id

    extra_sections: List[str] = []
    ordered_for_display = ordered_keys or list(fb_map.keys())
    for key in ordered_for_display:
        if key in MODEL_EMAIL_SECTION_SET:
            continue
        text = fb_map.get(key)
        if text:
            extra_sections.append(f"### {MODEL_DISPLAY_NAMES.get(key, key)}\n{text}")
    if not extra_sections:
        for key, text in fb_map.items():
            if key in MODEL_EMAIL_SECTION_SET:
                continue
            if text:
                extra_sections.append(f"### {MODEL_DISPLAY_NAMES.get(key, key)}\n{text}")
    extras_md = "\n\n".join(extra_sections)

    return pack_outputs(
        transcript_text=state.get("transcript_text", ""),
        seg_rows=seg_rows,
        primary_text=primary,
        engine_note=f"*Feedback engines (ordered):* {ordered_keys}",
        fb_groq=fb_map.get("groq", ""),
        fb_scout=fb_map.get("scout", ""),
        fb_openai=fb_map.get("openai", ""),
        fb_gemini=fb_map.get("gemini", ""),
        fb_ollama=extras_md,
        fb_or_deepseek=fb_map.get("or_deepseek_r1d_70b", ""),
        fb_or_gemma=fb_map.get("or_gemma2_9b_it", ""),
        qna_md=qna_md,
        qna_rows=state.get("qna_rows") or [],
        visuals_md="",
        gallery_imgs=[],
        ranking_md=rank_md,
        vote_dd_update=dd_update,
        raw_json=raw_json,
        state_dict=state,
    )


def do_vote(selected_key: str, up: bool, state: Dict[str, Any]):
    if not state or not state.get("available_models"):
        return ("Run feedback first to load models.", "—")
    available_models = state.get("available_models") or []
    if not selected_key:
        return ("Select a model first.", ranking_md(_load_weights(), available_models))
    store = register_vote(selected_key, upvote=up)
    msg = f"Recorded {'👍 upvote' if up else '👎 downvote'} for **{selected_key}**."
    return (msg, ranking_md(store, available_models))

# ===================== Pipeline =====================
def process(
     src_file: Optional[str],
     video_url: Optional[str],
     language_hint: str,
     initial_prompt: str,
     translate_to_en: bool,
     analyze_visuals_flag: bool,
     feedback_mode: str,
 ) -> Dict[str, Any]:

    def _normalize(v):
        if isinstance(v, list) and v:
            return v[0]
        return v

    src_file = _normalize(_normalize_uploaded(src_file))
    chosen: Optional[Path] = None

    if src_file and Path(src_file).exists():
        chosen = Path(src_file)
    elif _is_http_url(video_url):
        try:
            chosen = _download_media_from_url(video_url.strip(), WORK_DIR)
        except Exception as e:
            raise gr.Error(f"Could not fetch URL: {e}")

    if not chosen:
        raise gr.Error("No valid media provided (upload a file or paste a video URL).")

    file_id = uuid.uuid4().hex
    dst_vid = WORK_DIR / f"{file_id}{chosen.suffix.lower()}"
    for _ in range(20):
        try:
            with open(chosen, "rb") as r, open(dst_vid, "wb") as w:
                shutil.copyfileobj(r, w, 1024 * 1024)
            break
        except PermissionError:
            time.sleep(0.25)
    else:
        raise gr.Error("Could not read the file (Windows locked it). Close players and try again.")

    media_len = _ffprobe_duration(dst_vid)
    use_long = ALWAYS_SEGMENT or (media_len and media_len > MAX_DIRECT_SEC)
    task_mode = "translate" if translate_to_en else "transcribe"

    if use_long:
        segments, transcript_text, duration_sec = transcribe_long(dst_vid, language_hint, initial_prompt, task_mode)
    else:
        wav = extract_audio(dst_vid)
        if os.getenv("ALWAYS_SEGMENT", "0") == "1":
            segments, transcript_text, duration_sec = transcribe_long(dst_vid, language_hint, initial_prompt, task_mode)
        else:
            segments, transcript_text, duration_sec = transcribe_short(wav, language_hint, initial_prompt, task_mode)

    _ = analyze_transcript(transcript_text, segments, duration_sec)
    _ = extract_topics(transcript_text)

    items = _items_from_heuristic(segments)
    q_rows = qna_rows_from_items(items)
    q_summary = qna_summary_from_items(items)

    m = (feedback_mode or "").lower()
    if m == "both":
        mode = "both"
    elif m in {"scout","ollama","groq","or_deepseek_r1d_70b","or_gemma2_9b_it","all","openai","gemini"}:
        mode = m
    else:
        mode = "groq"

    # visuals
    visuals: Dict[str, Any] = {}
    if analyze_visuals_flag:
        visuals = analyze_visuals(dst_vid, file_id)

    feedback_map = get_feedbacks(transcript_text, segments, mode, visual_data=visuals)

    # ORDERED primary by adaptive weight
    ordered_keys = feedback_map.pop("_ordered_keys", [])
    store = _load_weights()

    primary = "*No feedback produced.*"
    for k in ordered_keys:
        text = feedback_map.get(k)
        if text and not str(text).startswith("[") :
            primary = text
            break
    used_engine = ", ".join(ordered_keys) if ordered_keys else ", ".join(sorted(feedback_map.keys()))

    # visuals: append to each feedback
    if analyze_visuals_flag:
        vr = visuals
        hand_line = f"- *Students raised hands (unique est.)*: {vr.get('hand_raise_unique', 0)}"
        board_line = f"- *Board snapshots*: {len(vr.get('board_snapshots', []))}"
        board_text = vr.get("board_text", "")
        add_section = "\n\n### Classroom Visuals\n" + hand_line + "\n" + board_line
        if board_text:
            add_section += "\n- *Board OCR (condensed)*:\n\n" + (board_text[:2000]) + ("\n..." if len(board_text)>2000 else "\n")
        for k in list(feedback_map.keys()):
            feedback_map[k] = (feedback_map[k] or "") + add_section
        primary = (primary or "") + add_section

    # Email
    if APP_EMAIL_ENABLED and APP_EMAIL_RECIPIENTS:
        parts = []
        titles = {
            "openai": "OpenAI · GPT",
            "gemini": "Google · Gemini",
            "groq": "Groq (GPT-OSS-120B)",
            "scout": "Groq (Llama-4-Scout)",
            "or_deepseek_r1d_70b": "OpenRouter · DeepSeek R1-Distill-Llama-70B",
            "or_gemma2_9b_it": "OpenRouter · Gemma2-9B-IT",
            "ollama": "Ollama (local)",
        }
        for k in ordered_keys:
            if k in feedback_map and feedback_map[k]:
                parts.append(f"### {titles.get(k,k)}\n{feedback_map[k]}")
        if not parts and primary:
            parts = [primary]
        subject = f"Lecture report — {file_id}"
        body_join = "\n\n".join(parts)
        text_body = (
            "Lecture Feedback\n================\n" + body_join +
            "\n\nQ&A Summary\n-----------\n" +
            f"Total items: {q_summary.get('total_items',0)} | "
            f"Student: {q_summary.get('total_student_questions',0)} | "
            f"Teacher: {q_summary.get('total_teacher_questions',0)} | "
            f"Answered(est): {q_summary.get('answered',0)} | "
            f"Unanswered(est): {q_summary.get('unanswered',0)} | "
            f"Unique students(est): {q_summary.get('unique_students_est',0)}\n"
        )
        html_body = (
            "<h2>Lecture Feedback</h2>"
            f"<div style='white-space:pre-wrap'>{body_join}</div>"
            "<h3>Q&amp;A Summary</h3><p>"
            f"Total items: {q_summary.get('total_items',0)} | "
            f"Student: {q_summary.get('total_student_questions',0)} | "
            f"Teacher: {q_summary.get('total_teacher_questions',0)} | "
            f"Answered(est): {q_summary.get('answered',0)} | "
            f"Unanswered(est): {q_summary.get('unanswered',0)} | "
            f"Unique students(est): {q_summary.get('unique_students_est',0)}</p>"
        )
        try:
            send_report_to_recipients(APP_EMAIL_RECIPIENTS, subject, text_body, html_body)
        except Exception as e:
            print("[email] batch send failed:", e)

    # Build ranking markdown and voting model list
    available_models = ordered_keys or list(feedback_map.keys())
    rank_md = ranking_md(store, available_models)

    return {
        "file_id": file_id,
        "duration_sec": duration_sec,
        "transcript_text": transcript_text,
        "segments": segments,
        "paragraph_primary": primary,
        "feedback_map": feedback_map,
        "ordered_keys": ordered_keys,
        "feedback_engine": used_engine,
        "qna_items": items,
        "qna_rows": q_rows,
        "qna_summary": q_summary,
        "visuals": visuals,
        "ranking_md": rank_md,
        "available_models": available_models,
    }

# ===================== Visual analysis (hand-raise + board) =====================
def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    iw = max(0, xB - xA); ih = max(0, yB - yA)
    inter = iw * ih
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def _is_board_like(frame_bgr) -> bool:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]; s = hsv[:,:,1]
    white_mask = (v > 225) & (s < 30)
    dark_mask  = (v < 50)
    white_pct = white_mask.mean()
    dark_pct  = dark_mask.mean()
    return (white_pct >= BOARD_WHITE_PCT) or (dark_pct >= BOARD_DARK_PCT)

def _ocr_text(frame_bgr) -> str:
    if pytesseract is None:
        return ""
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        txt = pytesseract.image_to_string(gray, lang="eng")
        return txt.strip()
    except Exception:
        return ""

def analyze_visuals(video_path: Path, file_id: str) -> Dict[str, Any]:
    out = {"hand_raise_unique": 0, "hand_raise_events": [], "board_snapshots": [], "board_text": ""}
    if cv2 is None:
        out["board_text"] = "[cv2 not installed — visuals skipped]"
        return out
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        out["board_text"] = "[could not open video — visuals skipped]"
        return out

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps * VISION_SAMPLE_SEC), 1)

    pose = None
    if YOLO is not None:
        try:
            pose = YOLO("yolov8n-pose.pt")
        except Exception as e:
            print("[visuals] YOLO load failed:", e)

    next_id = 1
    tracks = []
    raised_ids = set()
    saved_boards = 0
    board_texts = []

    frame_idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        ret, frame = (False, None)
        if frame_idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                break

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            curr = []
            if pose is not None:
                try:
                    res = pose(frame, verbose=False)
                    for r in res:
                        if getattr(r, "keypoints", None) is None:
                            continue
                        for kp, box in zip(r.keypoints.xy, r.boxes.xyxy):
                            k = kp.cpu().numpy()
                            x1,y1,x2,y2 = [float(v) for v in box.cpu().numpy()]
                            bbox = (x1,y1,x2,y2)
                            def get(i):
                                if i < k.shape[0]:
                                    return float(k[i,0]), float(k[i,1])
                                return None
                            Ls, Lw, Rs, Rw = get(5), get(7), get(6), get(8)
                            raised = False
                            def above(a,b): return (a is not None and b is not None) and (a[1] < b[1] - 8)
                            if above(Lw, Ls) or above(Rw, Rs):
                                raised = True
                            curr.append({"bbox":bbox, "raised":raised})
                except Exception as e:
                    print("[visuals] pose failed:", e)

            if curr:
                for c in curr:
                    best_iou, best_j = 0.0, -1
                    for j, tr in enumerate(tracks):
                        iou = _iou(c["bbox"], tr["bbox"])
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= HAND_IOU_THRESH and best_j >= 0:
                        tracks[best_j]["bbox"] = c["bbox"]
                        if c["raised"]:
                            tracks[best_j]["raised"] = True
                            raised_ids.add(tracks[best_j]["id"])
                    else:
                        tid = next_id; next_id += 1
                        tracks.append({"id":tid, "bbox":c["bbox"], "raised":c["raised"]})
                        if c["raised"]:
                            raised_ids.add(tid)

                if len(tracks) > 256:
                    tracks = tracks[-128:]

                hand_cnt = sum(1 for c in curr if c["raised"])
                if hand_cnt > 0:
                    out["hand_raise_events"].append({"t": ts, "count": hand_cnt})

            took = False
            if saved_boards < MAX_BOARD_FRAMES and _is_board_like(frame):
                snap_path = BOARD_DIR / f"{file_id}_board_{saved_boards+1:02d}.jpg"
                try:
                    cv2.imwrite(str(snap_path), frame)
                    out["board_snapshots"].append(str(snap_path))
                    saved_boards += 1
                    took = True
                except Exception as e:
                    print("[visuals] save board failed:", e)

            if took:
                text = _ocr_text(frame)
                if text:
                    board_texts.append(text)

        frame_idx += 1

    cap.release()
    out["hand_raise_unique"] = int(len(raised_ids))
    out["board_text"] = ("\n\n".join(board_texts)).strip()
    return out

# ===================== UI =====================
LANG_OPTIONS = ["auto", "en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]

# Custom CSS for Premium Look (Forced Light Mode)
CUSTOM_CSS = """
:root {
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #fdfdfd !important;
    --border-color-primary: #e5e7eb !important;
    --body-text-color: #111827 !important;
    --primary-500: #2563eb !important;
    --block-border-color: #e5e7eb !important;
    --input-background-fill: #ffffff !important;
}

/* Base Body Styles */
body { 
    background-color: #f9fafb !important; 
    color: #111827 !important; 
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

.gradio-container { 
    background-color: #f9fafb !important; 
    border: none !important; 
    padding: 20px !important;
}

/* Force light mode for all main containers */
.login-container, .app-container, .mode-box, .gr-box, .gr-form, .gr-panel, .gr-group, .gr-block { 
    background-color: #ffffff !important; 
    color: #111827 !important; 
    border: 1px solid #e5e7eb !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03) !important;
    margin-bottom: 20px !important;
}

/* Login Card refinement */
.login-container { 
    max-width: 450px !important; 
    margin: 80px auto !important; 
    padding: 3rem !important; 
    border: 1px solid #e2e8f0 !important; 
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.08) !important;
}

/* Mode Buttons styling */
.mode-btn { 
    background-color: #ffffff !important; 
    border: 1px solid #e5e7eb !important; 
    border-radius: 14px !important; 
    padding: 2rem !important; 
    height: auto !important;
    transition: all 0.3s ease !important; 
    color: #1e293b !important;
}

.mode-btn:hover { 
    border-color: #2563eb !important; 
    background-color: #f0f7ff !important; 
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1) !important;
}

/* Typography Headings */
h1, h2, h3, h4, .gr-button, .gr-label { 
    color: #0f172a !important; 
    font-weight: 600 !important;
}

h1 { font-weight: 800 !important; letter-spacing: -0.025em !important; }

/* Dashboard Container */
.app-container { 
    background: #ffffff !important; 
    padding: 2.5rem !important;
    border: 1px solid #e5e7eb !important;
}

/* Fix visibility for inputs and textareas */
input, textarea, select, .gr-input, .gr-textbox {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #111827 !important;
    padding: 10px 14px !important;
}

input:focus, textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* Accordions and Tabs */
.gr-accordion { border: 1px solid #e5e7eb !important; margin-bottom: 10px !important; }
.gr-tab-button { font-weight: 600 !important; }
.gr-tab-button-active { border-bottom: 2px solid #2563eb !important; color: #2563eb !important; }

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
}

.gr-button-secondary {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    color: #475569 !important;
}

/* Hide Graduate/Gradio footer */
footer { display: none !important; }
"""

with gr.Blocks(title="Lecture Analyzer") as demo:
    # State for login status
    login_state = gr.State(False)

    # --- PAGE 1: LOGIN ---
    with gr.Column(visible=True, elem_classes="login-container") as login_box:
        gr.Markdown("# 🔐 Sign In\nWelcome back! Please enter your credentials to access the analyzer.")
        user_input = gr.Textbox(label="Username", placeholder="Enter username...", lines=1)
        pass_input = gr.Textbox(label="Password", placeholder="Enter password...", type="password", lines=1)
        login_btn = gr.Button("Login to Dashboard", variant="primary")
        login_error = gr.Markdown(visible=False)

    # --- PAGE 2: MODE SELECTION ---
    with gr.Column(visible=False) as mode_box:
        with gr.Column(elem_classes="main-header"):
            gr.Markdown("# 🎓 Select Study Mode\nHow would you like to provide the lecture content today?")
        
        with gr.Row():
            with gr.Column(scale=1):
                btn_mode_upload = gr.Button("📂 Upload Media\nLocal file from your computer", variant="secondary", elem_classes="mode-btn")
            with gr.Column(scale=1):
                btn_mode_link = gr.Button("🔗 Paste Link\nYouTube, Drive, or Direct URL", variant="secondary", elem_classes="mode-btn")
            with gr.Column(scale=1):
                btn_mode_live = gr.Button("🎙️ Record Live\nMicrophone or Webcam stream", variant="secondary", elem_classes="mode-btn")

    # --- PAGE 3: MAIN APP ---
    with gr.Column(visible=False, elem_classes="app-container") as app_box:
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# 📘 Lecture Analyzer · Dashboard")
            with gr.Column(scale=1):
                logout_btn = gr.Button("Back to Modes", variant="secondary")

        with gr.Row():
            with gr.Column(scale=1):
                # Wrapped Inputs
                with gr.Group() as input_group:
                    video_url_tb = gr.Textbox(
                        label="Video URL (http/https)",
                        placeholder="https://... (YouTube, Drive, direct media, etc.)",
                        visible=False
                    )
                    inp = gr.File(label="Upload lecture video/audio", file_types=["video", "audio"], type="filepath", visible=False)
                    mic_input = gr.Audio(label="Record audio with microphone", sources=["microphone"], type="filepath", visible=False)
                    video_input = gr.Video(label="Record video with webcam", sources=["webcam"], format="mp4", include_audio=True, visible=False)
                
                with gr.Row():
                    lang = gr.Dropdown(LANG_OPTIONS, value="auto", label="Language (hint)")
                translate_toggle = gr.Checkbox(label="Translate non-English → English", value=TRANSLATE_DEFAULT)
                iprompt = gr.Textbox(label="Initial prompt / topic hint (optional)", lines=2, placeholder="Topic hint...")
                btn_transcribe = gr.Button("🚀 Start Analyzing", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 🤖 Feedback Engine")
                feedback_engine_dd = gr.Radio(
                    choices=[
                        "Groq (gpt-oss-120b)",
                        "Groq (Llama-4-Scout)",
                        "OpenAI · GPT-4o-mini",
                        "Google · Gemini",
                        "OpenRouter · DeepSeek R1-Distill-Llama-70B",
                        "OpenRouter · Gemma2-9B-IT",
                        "All (compare)",
                    ],
                    value="Groq (gpt-oss-120b)",
                    label="Choose AI for analysis",
                )
                btn_feedback = gr.Button("Generate Insights", variant="secondary")

        st = gr.State({})

        with gr.Tabs():
         with gr.Tab("Transcript"):
            transcript_box = gr.Textbox(label="Transcript", lines=12)

         with gr.Tab("Segments"):
            segtbl = gr.Dataframe(
                headers=["start", "end", "text"],
                datatype=["number", "number", "str"],
                wrap=True,
                value=[],
            )

         with gr.Tab("Teaching Feedback"):
            gr.Markdown("### Teaching Feedback Overview")
            feedback_primary_box = gr.Markdown()
            engine_md = gr.Markdown()
            email_recipient_tb = gr.Textbox(
                label="Email recipients (optional)",
                placeholder="teacher@example.com, mentor@example.com",
                lines=1,
            )
            gr.Markdown(
                "Leave the recipient box empty to use the default mailing list. "
                "Each model section below includes a button to send its feedback plus the captured Q&A summary."
            )

            with gr.Accordion(MODEL_DISPLAY_NAMES["groq"], open=False):
                fb_groq = gr.Markdown()
                with gr.Row():
                    gen_groq_btn = gr.Button("Generate Groq feedback", variant="secondary")
                    send_groq_btn = gr.Button("Send Groq feedback to email", variant="primary")
                groq_email_status = gr.Markdown()

            with gr.Accordion(MODEL_DISPLAY_NAMES["scout"], open=False):
                fb_scout = gr.Markdown()
                with gr.Row():
                    gen_scout_btn = gr.Button("Generate Scout feedback", variant="secondary")
                    send_scout_btn = gr.Button("Send Scout feedback to email", variant="primary")
                scout_email_status = gr.Markdown()

            with gr.Accordion(MODEL_DISPLAY_NAMES["openai"], open=False):
                fb_openai = gr.Markdown()
                with gr.Row():
                    gen_openai_btn = gr.Button("Generate OpenAI feedback", variant="secondary")
                    send_openai_btn = gr.Button("Send OpenAI feedback to email", variant="primary")
                openai_email_status = gr.Markdown()

            with gr.Accordion(MODEL_DISPLAY_NAMES["gemini"], open=False):
                fb_gemini = gr.Markdown()
                with gr.Row():
                    gen_gemini_btn = gr.Button("Generate Gemini feedback", variant="secondary")
                    send_gemini_btn = gr.Button("Send Gemini feedback to email", variant="primary")
                gemini_email_status = gr.Markdown()

            with gr.Accordion(MODEL_DISPLAY_NAMES["or_deepseek_r1d_70b"], open=False):
                fb_or_deepseek = gr.Markdown()
                with gr.Row():
                    gen_deepseek_btn = gr.Button("Generate DeepSeek feedback", variant="secondary")
                    send_deepseek_btn = gr.Button("Send DeepSeek feedback to email", variant="primary")
                deepseek_email_status = gr.Markdown()

            with gr.Accordion(MODEL_DISPLAY_NAMES["or_gemma2_9b_it"], open=False):
                fb_or_gemma = gr.Markdown()
                with gr.Row():
                    gen_gemma_btn = gr.Button("Generate Gemma feedback", variant="secondary")
                    send_gemma_btn = gr.Button("Send Gemma feedback to email", variant="primary")
                gemma_email_status = gr.Markdown()

            gr.Markdown("#### Additional models (auto)")
            gr.Markdown(
                "When you run `All (compare)`, any extra model outputs (for example, Local Ollama) will appear below."
            )
            fb_ollama = gr.Markdown(visible=False)

         with gr.Tab("Q&A Stats"):
            qna_summary_md = gr.Markdown()
            qna_tbl = gr.Dataframe(
                headers=["t_start", "t_end", "student_id", "answered", "question", "answer_span", "notes"],
                datatype=["number", "number", "str", "bool", "str", "str", "str"],
                wrap=True,
                value=[],
            )

         with gr.Tab("Visuals"):
            visuals_md = gr.Markdown()
            board_gallery = gr.Gallery(label="Board snapshots", columns=3, height=300)

         with gr.Tab("Model Ranking & Votes"):
            ranking_md_box = gr.Markdown(value="Run feedback to see ranking…")
            vote_model_dd = gr.Dropdown(choices=[], label="Model to vote", interactive=True)
            with gr.Row():
             up_btn = gr.Button("👍 Thumbs Up", variant="primary")
            down_btn = gr.Button("👎 Thumbs Down", variant="secondary")
            vote_result_md = gr.Markdown()

         with gr.Tab("Raw JSON"):
            rawjson_box = gr.Code(label="JSON", language="json")

    outputs = [
        transcript_box,
        segtbl,
        feedback_primary_box,
        engine_md,
        fb_groq,
        fb_scout,
        fb_openai,
        fb_gemini,
        fb_ollama,
        fb_or_deepseek,
        fb_or_gemma,
        qna_summary_md,
        qna_tbl,
        visuals_md,
        board_gallery,
        ranking_md_box,
        vote_model_dd,
        rawjson_box,
        st,
        groq_email_status,
        scout_email_status,
        openai_email_status,
        gemini_email_status,
        deepseek_email_status,
        gemma_email_status,
    ]

    # ===================== LOGIC & TRANSITIONS =====================
    
    def attempt_login(u, p):
        if u == "adminuser" and p == "admin@1234":
            return gr.update(visible=False), gr.update(visible=True), True, ""
        else:
            return gr.update(), gr.update(), False, "❌ Invalid credentials. Try 'adminuser' / 'admin@1234'"

    login_btn.click(
        attempt_login, 
        inputs=[user_input, pass_input], 
        outputs=[login_box, mode_box, login_state, login_error]
    )

    def show_app(mode):
        return {
            mode_box: gr.update(visible=False),
            app_box: gr.update(visible=True),
            video_url_tb: gr.update(visible=(mode == "link")),
            inp: gr.update(visible=(mode == "upload")),
            mic_input: gr.update(visible=(mode == "live")),
            video_input: gr.update(visible=(mode == "live")),
        }

    btn_mode_upload.click(lambda: show_app("upload"), None, [mode_box, app_box, video_url_tb, inp, mic_input, video_input])
    btn_mode_link.click(lambda: show_app("link"), None, [mode_box, app_box, video_url_tb, inp, mic_input, video_input])
    btn_mode_live.click(lambda: show_app("live"), None, [mode_box, app_box, video_url_tb, inp, mic_input, video_input])

    logout_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        None,
        [mode_box, app_box]
    )

    def make_gen_fb(choice):
        def fn(state, progress=gr.Progress()):
            return generate_feedback(state, choice, progress=progress)
        return fn

    gen_groq_btn.click(make_gen_fb("Groq (gpt-oss-120b)"), inputs=[st], outputs=outputs)
    gen_scout_btn.click(make_gen_fb("Groq (Llama-4-Scout)"), inputs=[st], outputs=outputs)
    gen_openai_btn.click(make_gen_fb("OpenAI · GPT-4o-mini"), inputs=[st], outputs=outputs)
    gen_gemini_btn.click(make_gen_fb("Google · Gemini"), inputs=[st], outputs=outputs)
    gen_deepseek_btn.click(make_gen_fb("OpenRouter · DeepSeek R1-Distill-Llama-70B"), inputs=[st], outputs=outputs)
    gen_gemma_btn.click(make_gen_fb("OpenRouter · Gemma2-9B-IT"), inputs=[st], outputs=outputs)

    send_groq_btn.click(
        lambda recipients, s: send_model_feedback_email("groq", recipients, s),
        inputs=[email_recipient_tb, st],
        outputs=[groq_email_status],
    )
    send_scout_btn.click(
        lambda recipients, st: send_model_feedback_email("scout", recipients, st),
        inputs=[email_recipient_tb, st],
        outputs=[scout_email_status],
    )
    send_openai_btn.click(
        lambda recipients, st: send_model_feedback_email("openai", recipients, st),
        inputs=[email_recipient_tb, st],
        outputs=[openai_email_status],
    )
    send_gemini_btn.click(
        lambda recipients, st: send_model_feedback_email("gemini", recipients, st),
        inputs=[email_recipient_tb, st],
        outputs=[gemini_email_status],
    )
    send_deepseek_btn.click(
        lambda recipients, st: send_model_feedback_email("or_deepseek_r1d_70b", recipients, st),
        inputs=[email_recipient_tb, st],
        outputs=[deepseek_email_status],
    )
    send_gemma_btn.click(
        lambda recipients, st: send_model_feedback_email("or_gemma2_9b_it", recipients, st),
        inputs=[email_recipient_tb, st],
        outputs=[gemma_email_status],
    )

    btn_transcribe.click(
        transcribe_stream,
        inputs=[inp, video_url_tb, mic_input, video_input, lang, iprompt, translate_toggle],
        outputs=outputs,
        api_name=False,
    )

    btn_feedback.click(
        generate_feedback,
        inputs=[st, feedback_engine_dd],
        outputs=outputs,
        api_name=False,
    )

    up_btn.click(
        lambda k, s: do_vote(k, True, s),
        inputs=[vote_model_dd, st],
        outputs=[vote_result_md, ranking_md_box],
    )
    down_btn.click(
        lambda k, s: do_vote(k, False, s),
        inputs=[vote_model_dd, st],
        outputs=[vote_result_md, ranking_md_box],
    )

# ===================== Gradio launch =====================
def pick_free_port(preferred: Optional[str] = None, start: int = 7860, end: int  = 7890) -> Optional[int]:
    def free(p: int) -> bool:
        import socket as _s
        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) != 0
    if preferred:
        try:
            pp = int(preferred)
            if free(pp):
                return pp
        except Exception:
            pass
    for p in range(start, end + 1):
        if free(p):
            return p
    return None

# Gradio launch is handled via FastAPI mounting below
# ============================================================
#  FastAPI JSON API for the React frontend
# ============================================================
app = FastAPI(title="Lecture Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev only — lock down for prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_full_pipeline(media_path: Path, language: str, prompt: str, translate: bool) -> Dict[str, Any]:
    """Run transcription + Q&A + feedback. Returns the React-shaped payload."""
    # 1. transcribe (auto chooses long vs short)
    duration_guess = _ffprobe_duration(media_path)
    task_mode = "translate" if translate else "transcribe"

    if ALWAYS_SEGMENT or duration_guess > MAX_DIRECT_SEC:
        segments, transcript_text, duration = transcribe_long(media_path, language, prompt, task_mode)
    else:
        wav = extract_audio(media_path)
        segments, transcript_text, duration = transcribe_short(wav, language, prompt, task_mode)

    # 2. Visuals
    visuals = analyze_visuals(media_path, uuid.uuid4().hex)

    # 3. Q&A (AI-powered)
    items, qna_insight = extract_qna_with_ai(transcript_text, segments, visual_data=visuals)
    qna_summary = qna_summary_from_items(items, visual_data=visuals)
    qna_payload = [
        {
            "t": it.get("t", "00:00"),
            "speaker": it.get("speaker", "student"),
            "student_id": it.get("student_id", "s?"),
            "answered": bool(it.get("answered", False)),
            "text": it.get("text", ""),
        }
        for it in items
    ]
    
    # 4. Feedback from all providers
    fb = get_feedbacks(transcript_text, segments, mode="all", visual_data=visuals)
    fb.pop("_ordered_keys", None)

    # 4. Ranking
    store = _load_weights()
    ranking = []
    for k in fb.keys():
        up, down = _vote_tuple(store, k)
        ranking.append({
            "key": k,
            "weight": effective_weight(store, k),
            "up": up,
            "down": down,
        })
    ranking.sort(key=lambda r: r["weight"], reverse=True)

    return {
        "fileId": uuid.uuid4().hex,
        "durationSec": duration,
        "transcript": transcript_text,
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in segments
        ],
        "qna": qna_payload,
        "qnaSummary": qna_summary,
        "qnaInsight": qna_insight,
        "feedback": fb,
        "ranking": ranking,
        "visuals": visuals,
    }


@app.post("/api/analyze")
async def api_analyze(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: Optional[str] = Form("auto"),
    prompt: Optional[str] = Form(""),
    translate: bool = Form(False),
):
    # 1. obtain a local media file
    if file is not None:
        suffix = Path(file.filename or "upload.bin").suffix or ".bin"
        local_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        with open(local_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif url:
        local_path = _download_media_from_url(url, UPLOAD_DIR)
    else:
        return JSONResponse({"error": "Provide either a file or a url."}, status_code=400)

    try:
        payload = _run_full_pipeline(local_path, language or "auto", prompt or "", bool(translate))
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/analyze-stream")
async def api_analyze_stream(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: Optional[str] = Form("auto"),
    prompt: Optional[str] = Form(""),
    translate: bool = Form(False),
):
    # 1. obtain a local media file
    if file is not None:
        suffix = Path(file.filename or "upload.bin").suffix or ".bin"
        local_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        with open(local_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif url:
        local_path = _download_media_from_url(url, UPLOAD_DIR)
    else:
        return JSONResponse({"error": "Provide either a file or a url."}, status_code=400)

    def generate():
        import json
        file_id = uuid.uuid4().hex
        audio_len = _ffprobe_duration(local_path)
        yield f"data: {json.dumps({'type': 'progress', 'status': 'Transcribing...', 'durationSec': audio_len})}\n\n"
        
        task_mode = "translate" if translate else "transcribe"
        use_long = ALWAYS_SEGMENT or (audio_len and audio_len > MAX_DIRECT_SEC)
        
        all_segments = []
        transcript_parts = []
        
        try:
            if use_long:
                wav = ensure_wav(local_path)
                parts = segment_wav(wav, CHUNK_SEC)
                total_off = 0.0
                for idx, part in enumerate(parts, 1):
                    first_prompt = prompt if idx == 1 else ""
                    yield f"data: {json.dumps({'type': 'progress', 'status': f'Transcribing chunk {idx}/{len(parts)}'})}\n\n"
                    for seg, _dur in transcribe_wav_iter(part, language or "auto", first_prompt, task_mode):
                        seg2 = {"start": seg["start"] + total_off, "end": seg["end"] + total_off, "text": seg["text"]}
                        all_segments.append(seg2)
                        if seg2["text"]:
                            transcript_parts.append(seg2["text"])
                            yield f"data: {json.dumps({'type': 'segment', 'segment': seg2})}\n\n"
                    total_off += float(CHUNK_SEC)
            else:
                wav = extract_audio(local_path)
                for seg, _dur in transcribe_wav_iter(wav, language or "auto", prompt, task_mode):
                    all_segments.append(seg)
                    if seg["text"]:
                        transcript_parts.append(seg["text"])
                        yield f"data: {json.dumps({'type': 'segment', 'segment': seg})}\n\n"
            
            yield f"data: {json.dumps({'type': 'progress', 'status': 'Analyzing visuals...'})}\n\n"
            visuals = analyze_visuals(local_path, file_id)

            yield f"data: {json.dumps({'type': 'progress', 'status': 'Generating insights...'})}\n\n"
            
            transcript_text = " ".join(transcript_parts)
            items, qna_insight = extract_qna_with_ai(transcript_text, all_segments, visual_data=visuals)
            qna_summary = qna_summary_from_items(items, visual_data=visuals)
            qna_payload = [
                {
                    "t": it.get("t", "00:00"),
                    "speaker": it.get("speaker", "student"),
                    "student_id": it.get("student_id", "s?"),
                    "answered": bool(it.get("answered", False)),
                    "text": it.get("text", ""),
                }
                for it in items
            ]
            
            fb = {}
            
            store = _load_weights()
            ranking = []
            # We initialize ranking with all available models so the frontend can display them.
            for k in MODEL_DISPLAY_NAMES.keys():
                up, down = _vote_tuple(store, k)
                ranking.append({
                    "key": k,
                    "weight": effective_weight(store, k),
                    "up": up,
                    "down": down,
                })
            ranking.sort(key=lambda r: r["weight"], reverse=True)
            
            final_result = {
                "type": "result",
                "fileId": file_id,
                "durationSec": audio_len,
                "transcript": transcript_text,
                "segments": all_segments,
                "qna": qna_payload,
                "qnaSummary": qna_summary,
                "qnaInsight": qna_insight,
                "feedback": fb,
                "ranking": ranking,
                "visuals": visuals,
            }
            
            yield f"data: {json.dumps(final_result)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/feedback")
async def api_feedback(payload: Dict[str, Any]):
    transcript_text = payload.get("transcript", "")
    segments = payload.get("segments", [])
    mode = payload.get("mode", "all")
    visual_data = payload.get("visual_data")
    
    if not transcript_text:
        return JSONResponse({"error": "Missing 'transcript'"}, status_code=400)
        
    fb = get_feedbacks(transcript_text, segments, mode=mode, visual_data=visual_data)
    fb.pop("_ordered_keys", None)
    
    return {"feedback": fb}


@app.post("/api/vote")
async def api_vote(payload: Dict[str, Any]):
    model_key = payload.get("model")
    upvote = bool(payload.get("up", True))
    if not model_key:
        return JSONResponse({"error": "Missing 'model'"}, status_code=400)
    register_vote(model_key, upvote)
    return {"ok": True}


@app.get("/api/vote/link")
async def api_vote_link(model: str, up: int = 1):
    if not model:
        return HTMLResponse("<h1>Error: Missing model parameter</h1>", status_code=400)
    
    register_vote(model, bool(up))
    action = "Upvoted" if up else "Downvoted"
    color = "#2ecc71" if up else "#e74c3c"
    
    html_content = f"""
    <html>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1 style="color: {color};">{action} successfully!</h1>
            <p>Thank you for voting on the feedback.</p>
            <p>You can close this window now.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/email")
async def api_email(payload: Dict[str, Any]):
    model_key = payload.get("model")
    recipients = payload.get("recipients", "")
    feedback_text = payload.get("feedback_text", "")
    
    if not model_key:
        return JSONResponse({"error": "Missing 'model'"}, status_code=400)
    
    model_label = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    subject = f"{model_label} feedback report"
    
    text_body = (
        f"{'='*50}\n"
        f"{model_label.upper()} FEEDBACK REPORT\n"
        f"{'='*50}\n\n"
        f"{feedback_text}\n"
    )
    
    feedback_html = _markdown_to_html(feedback_text)
    
    # Assuming local deployment for the API
    base_url = "http://localhost:7860"
    vote_url_up = f"{base_url}/api/vote/link?model={model_key}&up=1"
    vote_url_down = f"{base_url}/api/vote/link?model={model_key}&up=0"
    
    html_body = (
        f"<h2 style='color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px;'>{model_label} Feedback</h2>"
        f"<div style='background:#f8f9fa;padding:15px;border-radius:5px;margin:10px 0;'>"
        f"{feedback_html}"
        f"</div>"
        f"<div style='margin-top:30px; padding-top:20px; border-top:1px solid #ddd; text-align:center;'>"
        f"  <p style='margin-bottom:15px; font-weight:bold; color:#2c3e50; font-size:16px;'>Was this feedback helpful?</p>"
        f"  <a href='{vote_url_up}' style='display:inline-block; padding:12px 24px; background:#2ecc71; color:white; text-decoration:none; border-radius:6px; font-weight:bold; margin-right:15px;'>👍 Yes, Upvote</a>"
        f"  <a href='{vote_url_down}' style='display:inline-block; padding:12px 24px; background:#e74c3c; color:white; text-decoration:none; border-radius:6px; font-weight:bold;'>👎 No, Downvote</a>"
        f"</div>"
    )

    try:
        send_report_to_recipients(
            _parse_recipient_list(recipients) or APP_EMAIL_RECIPIENTS,
            subject=subject,
            body_text=text_body,
            body_html=html_body,
        )
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/health")
async def api_health():
    return {"ok": True, "service": "lecture-analyzer"}


# Mount the existing Gradio UI at /gradio
try:
    app = gr.mount_gradio_app(app, demo, path="/gradio")
except NameError:
    pass

# Serve the React frontend (if built)
# Note: TanStack Start apps are SSR-based; if you want a pure SPA, you'd need SSG.
# For now, we point to the built client assets.
FRONTEND_DIST = Path(__file__).parent / "insight-engine" / "dist" / "client"
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/")
    async def root_fallback():
        return {
            "message": "Classroom Analyzer API is running.",
            "frontend_status": "Built but missing index.html (common for TanStack Start SSR)." if FRONTEND_DIST.exists() else "Not built.",
            "instructions": "For development, run 'npm run dev' in insight-engine and visit http://localhost:8080.",
            "api_health": "/api/health",
            "gradio_fallback": "/gradio"
        }

if __name__ == "__main__":
    env_port = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT") or "7860"
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    uvicorn.run(
        app, 
        host=host, 
        port=int(env_port)
    )
