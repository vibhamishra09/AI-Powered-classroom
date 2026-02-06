---
project: Lecture Analyzer
version: 1.0
status: active
created: 2026-02-06
---

# Design Document: Lecture Analyzer

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                     │
│  (File Upload, URL Input, Mic/Webcam Recording, Controls)   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Application Layer (app.py)                 │
│  • Media ingestion & validation                              │
│  • Workflow orchestration                                    │
│  • State management                                          │
│  • Progress tracking & streaming                             │
└─────┬──────────┬──────────┬──────────┬──────────┬──────────┘
      │          │          │          │          │
┌─────▼──────┐ ┌▼─────────┐ ┌▼────────┐ ┌▼───────┐ ┌▼────────┐
│  Audio     │ │Transcript│ │ LLM     │ │Visual  │ │Storage  │
│ Processing │ │ Analysis │ │Feedback │ │Analysis│ │& Email  │
│            │ │          │ │         │ │        │ │         │
│• FFmpeg    │ │• Whisper │ │• Groq   │ │• YOLO  │ │• Mongo  │
│• GStreamer │ │• BERTopic│ │• OpenAI │ │• OCR   │ │• SMTP   │
│• Segment   │ │• Heuristic│ │• Gemini │ │• Board │ │• JSON   │
└────────────┘ └──────────┘ └─────────┘ └────────┘ └─────────┘
```

### 1.2 Component Overview

**Frontend Layer (Gradio)**
- Web-based UI with tabbed interface
- Real-time progress updates via streaming
- File upload, URL input, and recording widgets
- Interactive tables and markdown rendering

**Application Layer**
- Request routing and workflow orchestration
- State management across processing stages
- Parallel execution coordination
- Error handling and recovery

**Processing Layer**
- Audio extraction and segmentation
- Speech-to-text transcription
- Local analysis (metrics, topics, Q&A)
- Visual analysis (pose detection, OCR)

**Integration Layer**
- Multi-provider LLM feedback
- Email notification system
- Database persistence
- Adaptive ranking system



## 2. Data Flow

### 2.1 Primary Processing Pipeline

```
1. Media Input
   ├─ Upload file → WORK_DIR/{uuid}.{ext}
   ├─ URL download → yt-dlp/requests → WORK_DIR/{uuid}.{ext}
   └─ Recording → Gradio temp → WORK_DIR/{uuid}.{ext}
   
2. Audio Extraction
   ├─ Check audio stream (ffprobe)
   ├─ Extract 16kHz mono WAV → AUDIO_DIR/{uuid}.wav
   └─ Fallback: GStreamer ↔ FFmpeg
   
3. Transcription Decision
   ├─ Duration check (ffprobe)
   ├─ If > MAX_DIRECT_SEC → Segment mode
   └─ Else → Direct mode
   
4. Segment Mode (Long Media)
   ├─ FFmpeg segment → AUDIO_DIR/{uuid}_chunks/{uuid}_NNNN.wav
   ├─ For each chunk:
   │  ├─ Whisper transcribe (streaming)
   │  ├─ Adjust timestamps (+offset)
   │  └─ Yield progress update
   └─ Merge segments
   
5. Direct Mode (Short Media)
   ├─ Whisper transcribe (streaming)
   ├─ Yield segments as they arrive
   └─ Complete when done
   
6. Local Analysis
   ├─ analyze_transcript() → metrics, rubric
   ├─ extract_topics() → BERTopic keywords
   └─ qna_heuristic() → Q&A detection
   
7. LLM Feedback (Optional)
   ├─ Build prompt (system + user messages)
   ├─ Parallel execution (ThreadPoolExecutor)
   ├─ Provider-specific API calls
   └─ Adaptive ranking by effective weight
   
8. Visual Analysis (Optional)
   ├─ Sample frames (every VISION_SAMPLE_SEC)
   ├─ YOLO pose → hand raise detection
   ├─ Board detection → OCR text extraction
   └─ Save snapshots → BOARD_DIR
   
9. Persistence & Notification
   ├─ MongoDB insert (if enabled)
   ├─ Email reports (if configured)
   └─ Return results to UI
```

### 2.2 State Management

**Session State Dictionary**
```python
{
    "file_id": str,              # Unique identifier
    "media_path": str,           # Original media path
    "wav_path": str,             # Extracted WAV path
    "duration_sec": float,       # Media duration
    "segments": List[dict],      # Timestamped segments
    "transcript_text": str,      # Full transcript
    "qna_rows": List[List],      # Q&A table data
    "qna_summary": dict,         # Q&A statistics
    "feedback_map": dict,        # Model → feedback text
    "ordered_keys": List[str],   # Models ordered by weight
    "available_models": List[str], # Models that ran
    "ranking_md": str,           # Ranking table markdown
    "mongo_id": str,             # MongoDB document ID
}
```



## 3. Module Design

### 3.1 app.py (Main Application)

**Responsibilities**
- Gradio UI definition and event handlers
- Media ingestion and validation
- Workflow orchestration
- Streaming progress updates
- Email report generation

**Key Functions**

`transcribe_stream()`
- Generator function for streaming transcription
- Handles all input sources (file, URL, mic, webcam)
- Yields progress updates every EMIT_EVERY_SEC
- Manages state updates incrementally

`generate_feedback()`
- Orchestrates multi-provider feedback generation
- Executes providers in parallel via ThreadPoolExecutor
- Orders results by adaptive ranking
- Updates state with feedback map

`process()` (Legacy)
- Non-streaming batch processing
- Kept for backward compatibility
- Includes visual analysis integration

`send_model_feedback_email()`
- Per-model email sending
- Formats markdown as HTML
- Includes Q&A summary and details

**Adaptive Ranking System**

`effective_weight(model_key) = base_weight + vote_bonus`

Where:
- `base_weight`: Provider tier (openrouter=1.0, groq=2.0, ollama=3.0, gpt/gemini=5.0)
- `vote_bonus`: ln((upvotes+1)/(downvotes+1)) * BONUS_SCALE, clamped to [-2.0, +2.0]

Stored in `model_weights.json`:
```json
{
  "version": 1,
  "votes": {
    "groq": {"up": 15, "down": 3},
    "openai": {"up": 22, "down": 1}
  }
}
```

### 3.2 analyze.py (Local Analysis)

**Responsibilities**
- Transcript metrics calculation
- Topic extraction via BERTopic
- Teaching rubric scoring
- Actionable feedback generation

**Key Functions**

`analyze_transcript(text, segments, duration_sec)`
- Returns: WPM, filler stats, structure score, sentence length, advice

`extract_topics(text, top_k=8)`
- Uses BERTopic with HDBSCAN clustering
- Fallback to frequency-based extraction
- Returns: List of top keywords

`teaching_feedback(text, sections, duration_sec)`
- Scores 8 teaching dimensions (1-5 scale)
- Generates "quick wins" recommendations
- Returns: Rubric dict + quick wins list

`sectioning(segments)`
- Naive section detection based on pauses (>2.5s) and segment length
- Returns: List of section dicts with start/end/text

**Teaching Rubric Dimensions**
1. Clarity (jargon density)
2. Examples & analogies (marker count)
3. Engagement & questions (questions per minute)
4. Definitions & terminology (definition cues)
5. Structure & signposting (signpost count)
6. Pacing (WPM in 120-160 range)
7. Contrast & misconceptions (contrast markers)
8. Recap (presence of recap)



### 3.3 llm_local.py (Ollama Integration)

**Responsibilities**
- Local LLM feedback via Ollama
- Dual endpoint support (/api/chat, /api/generate)
- Model warmup and keep-alive

**Key Functions**

`ollama_feedback(transcript_text, topics, feedback)`
- Tries /api/chat first (newer Ollama versions)
- Falls back to /api/generate (older versions)
- Returns: (feedback_text, endpoint_used)
- Handles connection errors gracefully

**Ollama Configuration**
- `OLLAMA_URL`: Base URL (default http://127.0.0.1:11434)
- `OLLAMA_MODEL`: Model name (default qwen2.5:7b-instruct)
- `OLLAMA_NUM_CTX`: Context window (default 8192)
- `OLLAMA_NUM_PREDICT`: Max output tokens (default 800)
- `OLLAMA_TEMP`: Temperature (default 0.0)
- `OLLAMA_TIMEOUT`: Request timeout (default 600s)

## 4. External Integrations

### 4.1 LLM Providers

**Groq**
- Endpoint: https://api.groq.com/openai/v1/chat/completions
- Models: openai/gpt-oss-120b, meta-llama/llama-4-scout-17b-16e-instruct
- SDK: groq Python client
- Rate limits: Handled by SDK

**OpenAI**
- Endpoint: https://api.openai.com/v1/chat/completions
- Models: gpt-4o-mini (configurable)
- SDK: openai Python client
- Rate limits: Handled by SDK

**Google Gemini**
- SDK: google-generativeai
- Models: gemini-2.0-flash (configurable)
- Special handling: Map-reduce for long transcripts
  - Chunks text into GEMINI_CHUNK_CHARS (default 7000)
  - Processes chunks in parallel
  - Combines partial results with final synthesis call

**OpenRouter**
- Endpoint: https://openrouter.ai/api/v1/chat/completions
- Models: deepseek/deepseek-r1-distill-llama-70b, google/gemma-2-9b-it
- Direct HTTP requests (no SDK)
- Custom headers: X-Title for app identification

**Ollama (Local)**
- Endpoint: http://127.0.0.1:11434/api/{chat|generate}
- Models: User-configured (qwen2.5:7b-instruct default)
- Direct HTTP requests
- Warmup call to avoid cold start latency

### 4.2 Whisper (faster-whisper)

**Configuration**
- Model sizes: tiny (39M), base (74M), small (244M), medium (769M), large (1550M)
- Compute types: auto, int8, float16, float32
- Device: auto (CPU/GPU detection)
- Download location: workspace/models/

**Transcription Parameters**
- `task`: "transcribe" or "translate"
- `language`: ISO code or None for auto-detect
- `beam_size`: Search beam width (default 5)
- `temperature`: Sampling temperature (0.0 for greedy)
- `vad_filter`: Voice activity detection (enabled)
- `vad_parameters`: {"min_silence_duration_ms": 350}
- `condition_on_previous_text`: False (prevents hallucination)
- `initial_prompt`: Domain-specific vocabulary hint
- `word_timestamps`: Optional word-level timing

**Streaming Interface**
```python
segments_iter, info = model.transcribe(audio_path, ...)
for segment in segments_iter:
    yield {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    }
```



### 4.3 Visual Analysis

**YOLOv8 Pose Estimation**
- Model: yolov8n-pose.pt (nano size for speed)
- Keypoints: 17 body landmarks (COCO format)
- Hand raise detection logic:
  - Left wrist (7) above left shoulder (5), OR
  - Right wrist (8) above right shoulder (6)
  - Threshold: 8 pixels vertical difference

**Tracking Algorithm**
- IoU-based tracking across frames
- Threshold: HAND_IOU_THRESH (default 0.35)
- Assigns unique IDs to detected persons
- Maintains track history (max 256 tracks)

**Board Detection**
- HSV color space analysis
- White board: V > 225, S < 30, coverage > 22%
- Black board: V < 50, coverage > 22%
- Saves up to MAX_BOARD_FRAMES snapshots

**OCR (Tesseract)**
- Preprocessing: Grayscale + bilateral filter
- Language: English (configurable)
- Output: Raw text extraction
- Aggregation: Concatenate all board texts

### 4.4 Email (Gmail SMTP)

**Configuration**
- Server: smtp.gmail.com:465 (SSL)
- Authentication: App password (not account password)
- Debug mode: SMTP_DEBUG=1 for troubleshooting

**Email Structure**
```python
EmailMessage:
  From: GMAIL_USER
  To: recipient(s)
  Subject: "{Model} feedback report — {file_id}"
  Body (text/plain): Formatted feedback + Q&A summary
  Body (text/html): Styled HTML with markdown rendering
```

**HTML Formatting**
- Markdown → HTML conversion via markdown library
- Fallback: Regex-based conversion for basic formatting
- Styling: Inline CSS for email client compatibility
- Sections: Feedback content, Q&A summary, Q&A items (top 25)

### 4.5 MongoDB Persistence

**Connection**
- URI: MongoDB Atlas connection string
- Timeout: 4 seconds for server selection
- Auto-close: Connection closed after each operation

**Document Schema**
```json
{
  "payload": {
    "type": "transcription" | "feedback",
    "file_id": "uuid",
    "transcript": "...",
    "segments": [...],
    "feedback_map": {...},
    "qna_summary": {...}
  },
  "context": {
    "media_path": "...",
    "available_models": [...]
  },
  "created_at": ISODate("...")
}
```

**Error Handling**
- Graceful failure: Logs error, continues operation
- No retry logic: Single attempt per operation
- Optional: Controlled by MONGO_ENABLED flag



## 5. User Interface Design

### 5.1 Gradio Interface Structure

**Layout**
```
┌─────────────────────────────────────────────────────────┐
│  Lecture Analyzer · Fast CPU Streaming                  │
├─────────────────────────────────────────────────────────┤
│  Input Column              │  Feedback Column           │
│  • Video URL               │  • Engine Selection        │
│  • File Upload             │    - Groq (GPT-OSS-120B)  │
│  • Mic Recording           │    - Groq (Llama-4-Scout) │
│  • Webcam Recording        │    - OpenAI GPT-4o-mini   │
│  • Language Hint           │    - Google Gemini        │
│  • Translate Toggle        │    - OpenRouter DeepSeek  │
│  • Initial Prompt          │    - OpenRouter Gemma     │
│  • [Transcribe Button]     │    - All (compare)        │
│                            │  • [Generate Feedback]     │
└────────────────────────────┴────────────────────────────┘
│  Tabs:                                                   │
│  ┌─ Transcript ──────────────────────────────────────┐  │
│  │  Full text with live updates                      │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌─ Segments ────────────────────────────────────────┐  │
│  │  Table: start | end | text                        │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌─ Teaching Feedback ───────────────────────────────┐  │
│  │  • Overview (primary feedback)                    │  │
│  │  • Email Recipients Input                         │  │
│  │  • Accordions per model:                          │  │
│  │    - Groq [Send Email]                            │  │
│  │    - Scout [Send Email]                           │  │
│  │    - OpenAI [Send Email]                          │  │
│  │    - Gemini [Send Email]                          │  │
│  │    - DeepSeek [Send Email]                        │  │
│  │    - Gemma [Send Email]                           │  │
│  │  • Additional Models (auto-shown)                 │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌─ Q&A Stats ───────────────────────────────────────┐  │
│  │  • Summary markdown                               │  │
│  │  • Table: t_start | t_end | student_id |          │  │
│  │           answered | question | answer_span       │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌─ Visuals ─────────────────────────────────────────┐  │
│  │  • Hand raise summary                             │  │
│  │  • Board snapshots gallery                        │  │
│  │  • OCR text                                       │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌─ Model Ranking & Votes ───────────────────────────┐  │
│  │  • Ranking table (markdown)                       │  │
│  │  • Model dropdown                                 │  │
│  │  • [👍 Thumbs Up] [👎 Thumbs Down]                │  │
│  │  • Vote result message                            │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌─ Raw JSON ────────────────────────────────────────┐  │
│  │  Complete state as formatted JSON                 │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Interaction Flows

**Flow 1: Basic Transcription**
1. User uploads file or pastes URL
2. User clicks "Transcribe"
3. UI shows progress: "Preparing input…" → "Extracting audio…" → "Transcribing…"
4. Transcript tab updates in real-time (every 1 second)
5. Segments table populates incrementally
6. Q&A Stats tab shows detected questions
7. Status: "Transcript ready ✅"

**Flow 2: Multi-Model Feedback**
1. After transcription completes
2. User selects feedback engine (or "All")
3. User clicks "Generate Feedback"
4. UI shows progress: "Generating feedback…"
5. Teaching Feedback tab populates with primary feedback
6. Model accordions show individual results
7. Ranking tab updates with current weights
8. Vote dropdown populates with available models

**Flow 3: Email Report**
1. User enters recipient email(s) or leaves blank for defaults
2. User expands desired model accordion
3. User clicks "Send {Model} feedback to email"
4. Status message appears: "✅ Sent {Model} feedback to {recipients} at {timestamp}"
5. Email includes formatted feedback + Q&A summary

**Flow 4: Model Voting**
1. User reviews feedback from multiple models
2. User switches to "Model Ranking & Votes" tab
3. User selects model from dropdown
4. User clicks 👍 or 👎
5. Vote result message: "Recorded upvote/downvote for {model}"
6. Ranking table updates with new weights
7. Future feedback runs use updated ranking



## 6. Algorithms & Logic

### 6.1 Q&A Heuristic Detection

**Question Detection**
```python
Q_RE = r"(?:^|[\s\""])(?:why|what|how|when|where|which|can|could|should|would|is|are|do|does)\b.*?\?"

For each segment:
  If "?" in text OR Q_RE matches:
    Classify as question
    
    # Student grouping
    If gap < 90s from last question:
      Assign same student_id
    Else:
      Increment student_id counter
    
    # Answer detection
    For next 12 segments (max 120s ahead):
      If segment has 6+ words and no "?":
        Mark as answered
        Break
```

**Limitations**
- No speaker diarization (relies on timing heuristics)
- False positives on rhetorical questions
- False negatives on implicit questions
- Student ID is estimated, not actual

### 6.2 Adaptive Ranking Algorithm

**Weight Calculation**
```python
def effective_weight(model_key):
    # Base weight by provider tier
    base = {
        "openrouter": 1.0,
        "groq": 2.0,
        "ollama": 3.0,
        "gpt": 5.0,
        "gemini": 5.0
    }[provider_for_model(model_key)]
    
    # Vote bonus (log ratio with spam resistance)
    up, down = get_votes(model_key)
    bonus = log((up + 1) / (down + 1)) * BONUS_SCALE
    bonus = clamp(bonus, BONUS_MIN, BONUS_MAX)
    
    return base + bonus

def order_models(models):
    return sorted(models, key=effective_weight, reverse=True)
```

**Properties**
- Base weights reflect provider quality tiers
- Vote bonus is logarithmic (diminishing returns)
- Spam resistance: +1 smoothing prevents division by zero
- Bounded: Bonus clamped to [-2.0, +2.0]
- Persistent: Votes stored in model_weights.json

**Example**
```
Model: groq
Base: 2.0
Votes: 15 up, 3 down
Bonus: ln(16/4) * 1.25 = ln(4) * 1.25 ≈ 1.73
Effective: 2.0 + 1.73 = 3.73

Model: openai
Base: 5.0
Votes: 22 up, 1 down
Bonus: ln(23/2) * 1.25 = ln(11.5) * 1.25 ≈ 3.01 → clamped to 2.0
Effective: 5.0 + 2.0 = 7.0

Ranking: openai (7.0) > groq (3.73)
```

### 6.3 Gemini Map-Reduce

**Problem**: Gemini has token limits; long transcripts exceed context window

**Solution**: Chunk → Process → Combine

```python
def gemini_feedback(transcript, segments):
    chunks = chunk_text(transcript, GEMINI_CHUNK_CHARS, GEMINI_OVERLAP)
    
    # Map phase
    partials = []
    for i, chunk in enumerate(chunks):
        prompt = f"(Part {i+1} of {len(chunks)})\n{chunk}"
        response = gemini_model.generate(system_msg + prompt)
        partials.append(response.text)
    
    # Reduce phase (if multiple chunks)
    if len(partials) > 1:
        combined_prompt = "Combine and deduplicate:\n" + "\n---\n".join(partials)
        final = gemini_model.generate(combined_prompt)
        return final.text
    
    return partials[0]
```

**Parameters**
- `GEMINI_CHUNK_CHARS`: 7000 (safe margin below token limit)
- `GEMINI_OVERLAP`: 500 (context continuity between chunks)
- `GEMINI_MAX_TOKENS`: 1200 (per response)



### 6.4 Streaming Transcription

**Challenge**: Provide real-time updates without blocking UI

**Solution**: Generator pattern with timed yields

```python
def transcribe_stream(...):
    # Setup
    state = initialize_state()
    yield initial_update(state)
    
    # Transcription loop
    last_emit = time.time()
    for segment in whisper_segments:
        all_segments.append(segment)
        
        # Throttled updates
        if time.time() - last_emit >= EMIT_EVERY_SEC:
            state["segments"] = all_segments
            state["transcript_text"] = " ".join(texts)
            yield pack_outputs(state)
            last_emit = time.time()
    
    # Final update
    state["qna_rows"] = detect_qna(all_segments)
    yield pack_outputs(state)
```

**Benefits**
- Non-blocking: UI remains responsive
- Progressive: User sees results as they arrive
- Efficient: Updates throttled to avoid UI thrashing
- Stateful: Each yield includes complete state snapshot

### 6.5 Parallel Feedback Generation

**Challenge**: Multiple LLM providers with varying latencies

**Solution**: ThreadPoolExecutor with futures

```python
def get_feedbacks(transcript, segments, mode):
    want = determine_models(mode)  # {"groq", "openai", ...}
    
    calls = {}
    with ThreadPoolExecutor(max_workers=len(want)) as pool:
        if "groq" in want:
            calls["groq"] = pool.submit(groq_feedback, transcript, segments)
        if "openai" in want:
            calls["openai"] = pool.submit(openai_feedback, transcript, segments)
        # ... other providers
        
        results = {}
        for name, future in calls.items():
            try:
                results[name] = future.result()  # Blocks until complete
            except Exception as e:
                results[name] = f"[{name} failed: {e}]"
    
    # Order by adaptive ranking
    ordered = sorted(results.keys(), key=effective_weight, reverse=True)
    results["_ordered_keys"] = ordered
    return results
```

**Benefits**
- Parallel execution: All providers run simultaneously
- Fault tolerance: Individual failures don't block others
- Timeout handling: Each provider has independent timeout
- Ordered results: Primary feedback from highest-ranked model



## 7. Configuration Management

### 7.1 Configuration Hierarchy

**Priority Order** (highest to lowest)
1. Environment variables (runtime)
2. `secrets/.env` (overrides)
3. `secrets/config.json` (overrides)
4. `.env` (base)
5. Default values (hardcoded)

**Loading Sequence**
```python
def _load_env():
    # 1. Load base .env
    load_dotenv(BASE_DIR / ".env", override=False)
    
    # 2. Load secrets .env (overrides)
    load_dotenv(BASE_DIR / "secrets" / ".env", override=True)
    
    # 3. Load secrets config.json
    config_json = BASE_DIR / "secrets" / "config.json"
    if config_json.exists():
        data = json.loads(config_json.read_text())
        for key, value in data.items():
            os.environ.setdefault(key, str(value))
```

### 7.2 Configuration Categories

**Whisper Settings**
```bash
WHISPER_SIZE=small              # Model size
WHISPER_COMPUTE=auto            # Compute type
WORD_TS=0                       # Word timestamps (0/1)
WHISPER_BEAM_SIZE=5             # Beam search width
WHISPER_VAD_MIN_SIL_MS=350      # VAD silence threshold
```

**Media Processing**
```bash
PREFER_GSTREAMER=1              # Prefer GStreamer over FFmpeg
CHUNK_SEC=900                   # Segment duration (15 min)
MAX_DIRECT_SEC=1200             # Direct mode threshold (20 min)
ALWAYS_SEGMENT=0                # Force segmentation (0/1)
TRANSLATE_DEFAULT=1             # Default translate toggle
PROMPT_TRANSCRIPT_CHARS=1000    # Prompt context length
```

**Visual Analysis**
```bash
VISION_SAMPLE_SEC=1.5           # Frame sampling rate
HAND_IOU_THRESH=0.35            # Tracking IoU threshold
MAX_BOARD_FRAMES=8              # Max board snapshots
BOARD_WHITE_PCT=0.22            # White board detection threshold
BOARD_DARK_PCT=0.22             # Black board detection threshold
```

**Provider Timeouts**
```bash
PROVIDER_TIMEOUT=120            # LLM request timeout (seconds)
OLLAMA_TIMEOUT=600              # Ollama-specific timeout
```

**Gradio Server**
```bash
GRADIO_SERVER_PORT=7860         # Server port
GRADIO_HOST=127.0.0.1           # Bind address
GRADIO_SHARE=0                  # Public sharing (0/1)
GRADIO_TEMP_DIR=workspace/uploads  # Temp file location
```

### 7.3 Secrets Management

**Security Best Practices**
- Never commit `secrets/` directory
- Use `.gitignore` to exclude secrets
- Use environment-specific `.env` files
- Rotate API keys regularly
- Use Gmail app passwords (not account passwords)

**secrets/.env Template**
```bash
# LLM Providers
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...

# Email
GMAIL_USER=your-email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# MongoDB (optional)
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGO_ENABLED=1

# Ollama (local)
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```



## 8. Error Handling & Recovery

### 8.1 Error Handling Strategy

**Graceful Degradation**
- Optional features fail silently with logging
- Core features raise user-friendly errors
- Partial results returned when possible

**Error Categories**

| Category | Strategy | Example |
|----------|----------|---------|
| Configuration | Raise on startup | Missing API key for required provider |
| Media Input | Raise with guidance | Invalid file format, URL unreachable |
| Processing | Retry with fallback | FFmpeg fails → try GStreamer |
| Provider | Continue with others | Groq timeout → show other models |
| Optional | Log and skip | MongoDB unavailable → skip persistence |

### 8.2 Specific Error Handlers

**Audio Extraction Failure**
```python
try:
    if PREFER_GSTREAMER:
        _gst_extract(video, wav)
    else:
        _ffmpeg_extract(video, wav)
except Exception as e:
    print(f"[extract] Primary failed: {e}")
    try:
        # Fallback to alternate
        if PREFER_GSTREAMER:
            _ffmpeg_extract(video, wav)
        else:
            _gst_extract(video, wav)
    except Exception as e2:
        print(f"[extract] Both failed: {e2}")
        # Switch to segmentation mode
        os.environ["ALWAYS_SEGMENT"] = "1"
```

**Provider Timeout**
```python
with ThreadPoolExecutor() as pool:
    future = pool.submit(provider_feedback, ...)
    try:
        result = future.result(timeout=PROVIDER_TIMEOUT)
    except TimeoutError:
        result = f"[{provider} timed out after {PROVIDER_TIMEOUT}s]"
    except Exception as e:
        result = f"[{provider} failed: {e}]"
```

**MongoDB Connection**
```python
def store_teaching_result(...):
    if not MONGO_ENABLED or not MONGO_URI:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        result = client[db][coll].insert_one(doc)
        return str(result.inserted_id)
    except Exception as e:
        print(f"[mongo] insert failed: {e}")
        return None  # Continue without persistence
    finally:
        client.close()
```

**File Locking (Windows)**
```python
for attempt in range(20):
    try:
        with open(src, "rb") as r, open(dst, "wb") as w:
            shutil.copyfileobj(r, w)
        break
    except PermissionError:
        time.sleep(0.25)
else:
    raise gr.Error("File locked by another process")
```

### 8.3 User-Facing Error Messages

**Good Error Messages**
- Specific: "No audio stream found in file: video.mp4"
- Actionable: "Install yt-dlp: pip install yt-dlp"
- Contextual: "Groq API key not set. Add GROQ_API_KEY to secrets/.env"

**Bad Error Messages**
- Vague: "Processing failed"
- Technical: "AttributeError: 'NoneType' object has no attribute 'text'"
- Unhelpful: "Error occurred"

**Implementation**
```python
# Bad
raise Exception("Error")

# Good
raise gr.Error("No valid media provided. Upload a file or paste a URL.")

# Better
if not chosen:
    debug_info = f"src_file={src_file}, url={video_url}"
    raise gr.Error(
        f"No valid media provided. "
        f"Upload a file or paste a URL. "
        f"Debug: {debug_info}"
    )
```



## 9. Performance Optimization

### 9.1 Transcription Performance

**CPU Optimization**
- Use `faster-whisper` (CTranslate2 backend) instead of OpenAI Whisper
- Speedup: 4-5x faster on CPU
- Model size selection: `small` balances speed/accuracy
- Compute type: `auto` selects optimal precision

**Streaming Benefits**
- User sees results immediately (no waiting for full transcription)
- Memory efficient: Processes segments incrementally
- Perceived performance: Progress updates every 1 second

**Segmentation Strategy**
- Long videos (>20 min): Segment into 15-min chunks
- Parallel potential: Could process chunks in parallel (not implemented)
- Trade-off: Slight accuracy loss at chunk boundaries

### 9.2 LLM Feedback Performance

**Parallel Execution**
- All providers run simultaneously via ThreadPoolExecutor
- Wall time = max(provider_times), not sum(provider_times)
- Example: 7 providers × 30s each = 30s total (not 210s)

**Prompt Optimization**
- Truncate transcript to PROMPT_TRANSCRIPT_CHARS (default 1000)
- Include only first 60 segments with timestamps
- Reduces token count → faster responses, lower cost

**Caching Opportunities** (not implemented)
- Cache transcripts by file hash
- Cache feedback by (transcript_hash, model, prompt)
- Potential speedup: Instant results for repeated analyses

### 9.3 Visual Analysis Performance

**Frame Sampling**
- Sample every VISION_SAMPLE_SEC (default 1.5s)
- 1-hour video: 2400 frames instead of 90,000 (at 25fps)
- Speedup: 37.5x fewer frames to process

**Model Selection**
- YOLOv8n-pose (nano): Fastest variant
- Trade-off: Slightly lower accuracy vs. larger models
- Sufficient for classroom scenarios

**Early Termination**
- Board snapshots: Stop after MAX_BOARD_FRAMES
- Reduces unnecessary processing for long videos

### 9.4 Memory Management

**Workspace Cleanup**
- Temporary files accumulate in workspace/
- Manual cleanup required (not automated)
- Recommendation: Periodic cleanup script or cron job

**Segment Chunking**
- Long videos: Process in chunks to avoid memory spikes
- Each chunk processed independently
- Memory usage: O(chunk_size), not O(total_size)

**Model Loading**
- Whisper model: Loaded once at startup
- YOLO model: Lazy loaded on first visual analysis
- Reused across requests (singleton pattern)



## 10. Testing Strategy

### 10.1 Unit Testing

**Core Functions to Test**
```python
# analyze.py
test_compute_wpm()
test_filler_stats()
test_sectioning()
test_extract_topics()
test_teaching_feedback()

# app.py
test_normalize_uploaded()
test_mmss_conversion()
test_chunk_text()
test_parse_recipient_list()
test_effective_weight()
test_vote_bonus_calculation()

# llm_local.py
test_ollama_url_parsing()
```

**Test Data**
- Sample transcripts (short, medium, long)
- Edge cases: Empty text, single word, no punctuation
- Mock segments with various timestamp patterns

### 10.2 Integration Testing

**Media Processing Pipeline**
```python
def test_audio_extraction():
    # Given: Sample video file
    # When: extract_audio() called
    # Then: WAV file created with correct format (16kHz, mono)

def test_transcription_short():
    # Given: Short audio file (<20 min)
    # When: transcribe_short() called
    # Then: Segments returned with timestamps

def test_transcription_long():
    # Given: Long audio file (>20 min)
    # When: transcribe_long() called
    # Then: Segments merged correctly with offset timestamps
```

**LLM Provider Integration**
```python
def test_groq_feedback():
    # Given: Sample transcript
    # When: groq_feedback() called
    # Then: Structured feedback returned

def test_provider_failure_handling():
    # Given: Invalid API key
    # When: get_feedbacks() called
    # Then: Error message in results, other providers succeed
```

### 10.3 End-to-End Testing

**Complete Workflow**
```python
def test_full_pipeline():
    # 1. Upload sample video
    # 2. Transcribe
    # 3. Generate feedback (all models)
    # 4. Verify Q&A detection
    # 5. Check ranking table
    # 6. Vote on model
    # 7. Verify weight update
```

**UI Testing**
- Manual testing via Gradio interface
- Verify streaming updates appear
- Check tab switching works
- Test email sending (with test recipients)

### 10.4 Performance Testing

**Load Testing**
```python
def test_concurrent_requests():
    # Simulate 10 concurrent users
    # Each uploads different video
    # Verify: All complete successfully
    # Measure: Average response time
```

**Stress Testing**
```python
def test_large_file():
    # Given: 2-hour lecture video
    # When: Process through pipeline
    # Then: Completes without memory errors
    # Measure: Peak memory usage, total time
```

### 10.5 Test Environment Setup

**Dependencies**
```bash
pip install pytest pytest-cov pytest-mock
```

**Mock Services**
```python
@pytest.fixture
def mock_whisper():
    # Mock WhisperModel to avoid loading actual model
    pass

@pytest.fixture
def mock_llm_providers():
    # Mock API calls to avoid rate limits
    pass
```

**Test Data**
```
tests/
  fixtures/
    sample_short.mp4      # 2-min lecture
    sample_long.mp4       # 25-min lecture
    sample_transcript.txt # Known-good transcript
    sample_segments.json  # Known-good segments
```



## 11. Deployment

### 11.1 Local Deployment

**Prerequisites**
```bash
# System dependencies
- Python 3.8+
- FFmpeg or GStreamer
- Tesseract OCR (optional, for board text)

# For Ollama (optional)
- Ollama server running on localhost:11434
```

**Installation Steps**
```bash
# 1. Clone repository
git clone <repo-url>
cd classroom

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
mkdir -p secrets
cp secrets/.env.example secrets/.env
# Edit secrets/.env with your API keys

# 5. Run application
python app.py
```

**Access**
- Local: http://127.0.0.1:7860
- Network: Set `GRADIO_HOST=0.0.0.0` for LAN access

### 11.2 Hugging Face Spaces Deployment

**Configuration**
```yaml
# spaces.yaml
title: Lecture Analyzer
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
```

**Secrets Management**
- Add secrets via Spaces Settings → Repository secrets
- Required: GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
- Optional: GMAIL_USER, GMAIL_APP_PASSWORD, MONGO_URI

**Limitations**
- CPU-only (no GPU)
- Disk space limited (recommend cleanup strategy)
- Network egress limits (for LLM API calls)
- Concurrent user limits (queue helps)

### 11.3 Docker Deployment

**Dockerfile**
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create workspace directories
RUN mkdir -p workspace/audio workspace/uploads workspace/boards

# Expose port
EXPOSE 7860

# Run application
CMD ["python", "app.py"]
```

**Docker Compose**
```yaml
version: '3.8'

services:
  lecture-analyzer:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./workspace:/app/workspace
      - ./secrets:/app/secrets
    environment:
      - GRADIO_SERVER_PORT=7860
      - GRADIO_HOST=0.0.0.0
    restart: unless-stopped
```

**Build & Run**
```bash
docker-compose up -d
```

### 11.4 Production Considerations

**Scaling**
- Horizontal: Multiple instances behind load balancer
- Vertical: Increase CPU/RAM for faster processing
- Queue: Gradio queue handles concurrent requests

**Monitoring**
- Log aggregation: Collect stdout/stderr
- Metrics: Track processing times, error rates
- Alerts: Notify on provider failures, disk space

**Security**
- HTTPS: Use reverse proxy (nginx, Caddy)
- Authentication: Add Gradio auth or OAuth
- Rate limiting: Prevent abuse
- Input validation: Sanitize file uploads

**Backup & Recovery**
- Workspace: Periodic cleanup or archival
- MongoDB: Regular backups if using persistence
- Secrets: Secure backup of .env files



## 12. Future Enhancements

### 12.1 Planned Features

**Speaker Diarization**
- Identify individual speakers (teacher vs. students)
- Attribute questions to specific students
- Track participation per student
- Libraries: pyannote.audio, resemblyzer

**Real-Time Analysis**
- WebSocket-based live streaming
- Process audio as lecture happens
- Instant feedback during class
- Requires: Low-latency transcription, incremental analysis

**Custom Rubrics**
- User-defined teaching dimensions
- Configurable scoring criteria
- Domain-specific feedback templates
- Storage: JSON or database

**LMS Integration**
- Canvas, Moodle, Blackboard connectors
- Auto-import lecture recordings
- Push feedback to gradebook
- OAuth authentication

**Mobile App**
- Native iOS/Android apps
- In-app recording
- Offline processing queue
- Push notifications for results

### 12.2 Technical Improvements

**Caching Layer**
- Redis for transcript/feedback caching
- File hash-based cache keys
- TTL-based expiration
- Reduces redundant processing

**Async Processing**
- FastAPI + Celery for background jobs
- Job queue for long-running tasks
- Status polling endpoint
- Email notification on completion

**Database Schema**
- Relational schema for structured queries
- User accounts and history
- Feedback comparison across lectures
- Analytics dashboard

**Model Fine-Tuning**
- Fine-tune Whisper on lecture domain
- Custom Q&A detection model
- Personalized feedback style
- Requires: Training data, GPU resources

### 12.3 Research Opportunities

**Engagement Scoring**
- Combine visual cues (hand raises, attention)
- Audio features (tone, energy)
- Transcript features (questions, participation)
- ML model: Engagement score 0-100

**Slide Synchronization**
- Extract slides from video
- OCR slide content
- Align slides with transcript timestamps
- Generate slide-aware feedback

**Automated Action Items**
- Extract actionable tasks from feedback
- Categorize by priority
- Track completion status
- Reminder system

**Multi-Language Support**
- Translate feedback to teacher's language
- Support non-English lectures
- Cross-lingual topic extraction
- Requires: Translation API, multilingual models

**Comparative Analysis**
- Compare lecture to previous lectures
- Track improvement over time
- Benchmark against peer teachers
- Longitudinal analytics

### 12.4 Community Contributions

**Plugin System**
- Custom feedback providers
- Custom analysis modules
- Custom export formats
- Plugin marketplace

**Open Dataset**
- Anonymized lecture transcripts
- Feedback examples
- Q&A annotations
- Research collaboration

**Model Marketplace**
- Share fine-tuned models
- Rate and review models
- Community-contributed prompts
- Monetization options



## 13. Appendices

### 13.1 Glossary

**Terms**
- **ASR**: Automatic Speech Recognition (speech-to-text)
- **BERTopic**: Topic modeling algorithm using BERT embeddings
- **CTranslate2**: Optimized inference engine for Transformer models
- **Filler words**: Non-semantic utterances (um, uh, like)
- **Gradio**: Python library for building ML web interfaces
- **HDBSCAN**: Hierarchical density-based clustering algorithm
- **IoU**: Intersection over Union (object tracking metric)
- **LLM**: Large Language Model
- **Map-Reduce**: Parallel processing pattern (divide, process, combine)
- **OCR**: Optical Character Recognition (image-to-text)
- **Rubric**: Scoring framework with defined criteria
- **Signposting**: Explicit structural markers in speech ("First...", "Next...")
- **VAD**: Voice Activity Detection (silence removal)
- **WPM**: Words Per Minute (speech rate metric)
- **YOLO**: You Only Look Once (real-time object detection)

### 13.2 File Structure

```
classroom/
├── app.py                      # Main application
├── analyze.py                  # Local analysis module
├── llm_local.py                # Ollama integration
├── requirements.txt            # Python dependencies
├── README.md                   # User documentation
├── model_weights.json          # Adaptive ranking votes
├── .env                        # Base configuration
├── .gitignore                  # Git exclusions
├── secrets/                    # Secrets (not in git)
│   ├── .env                    # Secret overrides
│   └── config.json             # JSON config
├── workspace/                  # Working directory
│   ├── {uuid}.mp4              # Uploaded media
│   ├── audio/                  # Extracted audio
│   │   ├── {uuid}.wav
│   │   └── {uuid}_chunks/      # Segmented audio
│   ├── uploads/                # Gradio temp files
│   ├── boards/                 # Board snapshots
│   └── models/                 # Whisper model cache
├── .kiro/                      # Kiro specs (this document)
│   └── specs/
│       ├── requirements.md
│       └── design.md
└── __pycache__/                # Python bytecode
```

### 13.3 API Reference

**Environment Variables**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| WHISPER_SIZE | str | small | Whisper model size |
| WHISPER_COMPUTE | str | auto | Compute type |
| CHUNK_SEC | int | 900 | Segment duration |
| MAX_DIRECT_SEC | int | 1200 | Segmentation threshold |
| GROQ_API_KEY | str | - | Groq API key |
| OPENAI_API_KEY | str | - | OpenAI API key |
| GEMINI_API_KEY | str | - | Gemini API key |
| GMAIL_USER | str | - | Gmail address |
| GMAIL_APP_PASSWORD | str | - | Gmail app password |
| MONGO_URI | str | - | MongoDB connection string |
| MONGO_ENABLED | str | 1 | Enable persistence (0/1) |
| GRADIO_SERVER_PORT | int | 7860 | Server port |
| GRADIO_HOST | str | 127.0.0.1 | Bind address |

**Key Functions**

`transcribe_stream(src_file, video_url, mic_audio, video_input, language_hint, initial_prompt, translate_to_en)`
- Generator function for streaming transcription
- Yields: Tuple of UI component updates
- Returns: Final state with complete transcript

`generate_feedback(state, feedback_engine_choice)`
- Generates LLM feedback for transcribed lecture
- Returns: Tuple of UI component updates with feedback

`analyze_transcript(text, segments, duration_sec)`
- Performs local analysis (metrics, rubric)
- Returns: Dict with WPM, fillers, structure, teaching feedback

`extract_topics(text, top_k=8)`
- Extracts key topics using BERTopic
- Returns: List of top keywords

`qna_heuristic(segments)`
- Detects questions using regex patterns
- Returns: Dict with questions list

`analyze_visuals(video_path, file_id)`
- Performs visual analysis (hand raises, board)
- Returns: Dict with hand_raise_unique, board_snapshots, board_text

### 13.4 References

**Libraries**
- [Gradio](https://gradio.app/) - Web UI framework
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized Whisper
- [BERTopic](https://maartengr.github.io/BERTopic/) - Topic modeling
- [Ultralytics YOLO](https://docs.ultralytics.com/) - Object detection
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Text recognition

**APIs**
- [Groq API](https://console.groq.com/docs) - Fast LLM inference
- [OpenAI API](https://platform.openai.com/docs) - GPT models
- [Google Gemini API](https://ai.google.dev/docs) - Gemini models
- [OpenRouter API](https://openrouter.ai/docs) - Multi-model gateway

**Research Papers**
- Whisper: Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)
- BERTopic: Neural topic modeling with a class-based TF-IDF procedure (Grootendorst, 2022)
- YOLOv8: Ultralytics YOLO (Jocher et al., 2023)

### 13.5 Change Log

**Version 1.0 (Current)**
- Initial release
- Multi-provider LLM feedback
- Adaptive ranking system
- Streaming transcription
- Visual analysis (hand raises, board OCR)
- Email reports
- MongoDB persistence
- Q&A heuristic detection

**Planned for Version 1.1**
- Speaker diarization
- Custom rubrics
- Caching layer
- Performance dashboard

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-06  
**Authors**: Kiro AI Assistant  
**Status**: Active
