---
project: Lecture Analyzer
version: 1.0
status: active
created: 2026-02-06
---

# Requirements Document: Lecture Analyzer

## Executive Summary

The Lecture Analyzer is an AI-powered teaching analysis platform that processes recorded lectures (video or audio) to generate accurate transcripts, detect student questions, provide pedagogical feedback from multiple LLMs, and optionally analyze classroom visuals. The system operates entirely server-side as a Gradio web application.

## 1. Functional Requirements

### 1.1 Media Input & Processing

**FR-1.1.1: Multi-Source Media Input**
- System SHALL accept video/audio from multiple sources:
  - Direct file upload (mp4, mkv, mov, webm, mp3, m4a, wav, aac, flac)
  - HTTP/HTTPS URLs (direct media files)
  - YouTube and platform URLs via yt-dlp
  - Microphone recording (audio only)
  - Webcam recording (video with audio)
- System SHALL generate unique file IDs for each processed media
- System SHALL store uploaded media in workspace directory

**FR-1.1.2: Audio Extraction**
- System SHALL extract 16kHz mono WAV audio from video files
- System SHALL support both FFmpeg and GStreamer extraction methods
- System SHALL automatically fallback between extractors on failure
- System SHALL validate audio stream presence before processing

**FR-1.1.3: Long Media Handling**
- System SHALL segment media longer than MAX_DIRECT_SEC (default 1200s) into chunks
- System SHALL use configurable chunk size (default 900s)
- System SHALL maintain timestamp continuity across chunks

### 1.2 Transcription

**FR-1.2.1: Speech-to-Text**
- System SHALL use faster-whisper for transcription
- System SHALL support configurable Whisper model sizes (tiny, base, small, medium, large)
- System SHALL provide language detection and hint support
- System SHALL support translation to English
- System SHALL generate timestamped segments with start/end times
- System SHALL support streaming transcription with live updates

**FR-1.2.2: Transcription Configuration**
- System SHALL support configurable beam size (default 5)
- System SHALL support VAD filtering with configurable silence duration
- System SHALL support optional word-level timestamps
- System SHALL accept initial prompts for domain-specific vocabulary

### 1.3 Question & Answer Detection

**FR-1.3.1: Heuristic Q&A Detection**
- System SHALL detect questions using regex patterns and punctuation
- System SHALL estimate student vs teacher questions
- System SHALL group related questions by student ID
- System SHALL estimate answer status (answered/unanswered)
- System SHALL track unique student count estimates

**FR-1.3.2: Q&A Statistics**
- System SHALL provide summary statistics:
  - Total Q&A items
  - Student questions count
  - Teacher questions count
  - Answered questions estimate
  - Unanswered questions estimate
  - Unique students estimate

### 1.4 Pedagogical Feedback

**FR-1.4.1: Multi-Provider LLM Feedback**
- System SHALL support multiple LLM providers:
  - Groq (GPT-OSS-120B, Llama-4-Scout)
  - OpenAI (GPT-4o-mini)
  - Google Gemini (2.0-flash)
  - OpenRouter (DeepSeek R1-Distill-Llama-70B, Gemma2-9B-IT)
  - Local Ollama
- System SHALL execute feedback generation in parallel for multiple providers
- System SHALL handle provider failures gracefully with error messages

**FR-1.4.2: Feedback Content**
- System SHALL generate structured feedback including:
  - Detected topics
  - Top 5 teaching improvements
  - Minute-by-minute fixes with timestamps
  - Suggested examples to add
  - Check-for-understanding questions
  - Next-class action plan (10 steps)

**FR-1.4.3: Adaptive Model Ranking**
- System SHALL maintain persistent vote counts (upvotes/downvotes) per model
- System SHALL calculate effective weights combining base weights and vote bonuses
- System SHALL order feedback results by effective weight (highest first)
- System SHALL display ranking table with provider, votes, and weights
- System SHALL allow users to vote on model quality

### 1.5 Local Analysis

**FR-1.5.1: Transcript Analysis**
- System SHALL calculate words per minute (WPM)
- System SHALL detect filler words and calculate filler ratio
- System SHALL perform sentence segmentation and calculate average sentence length
- System SHALL score structure and signposting (0-10 scale)
- System SHALL provide actionable advice based on metrics

**FR-1.5.2: Topic Extraction**
- System SHALL extract key topics using BERTopic
- System SHALL fallback to frequency-based extraction if BERTopic unavailable
- System SHALL return configurable number of top keywords (default 8)

**FR-1.5.3: Teaching Rubric**
- System SHALL score teaching on 8 dimensions (1-5 scale):
  - Clarity
  - Examples & analogies
  - Engagement & questions
  - Definitions & terminology
  - Structure & signposting
  - Pacing
  - Contrast & misconceptions
  - Recap
- System SHALL provide "quick wins" recommendations

### 1.6 Visual Analysis (Optional)

**FR-1.6.1: Hand Raise Detection**
- System SHALL detect raised hands using YOLOv8 pose estimation
- System SHALL track unique students raising hands
- System SHALL record hand raise events with timestamps
- System SHALL use configurable sampling rate (default 1.5s)

**FR-1.6.2: Board Content Capture**
- System SHALL detect whiteboard/blackboard frames
- System SHALL capture up to MAX_BOARD_FRAMES snapshots (default 8)
- System SHALL perform OCR on board content using Tesseract
- System SHALL aggregate board text across snapshots

### 1.7 Reporting & Notifications

**FR-1.7.1: Email Reports**
- System SHALL send email reports via Gmail SMTP
- System SHALL support multiple recipients (comma/semicolon separated)
- System SHALL include feedback content in both text and HTML formats
- System SHALL include Q&A summary in reports
- System SHALL support per-model email sending
- System SHALL format markdown feedback as HTML for emails

**FR-1.7.2: Data Export**
- System SHALL export complete results as JSON
- System SHALL include all segments, feedback, Q&A, and metadata
- System SHALL provide downloadable transcript text

### 1.8 Data Persistence

**FR-1.8.1: MongoDB Storage (Optional)**
- System SHALL optionally store results in MongoDB
- System SHALL store transcription results with metadata
- System SHALL store feedback results with model rankings
- System SHALL support toggling persistence via configuration
- System SHALL handle MongoDB connection failures gracefully

## 2. Non-Functional Requirements

### 2.1 Performance

**NFR-2.1.1: Transcription Speed**
- System SHALL provide streaming transcription with updates every EMIT_EVERY_SEC (default 1.0s)
- System SHALL process audio in near real-time on CPU
- System SHALL handle videos up to 2 hours without memory issues

**NFR-2.1.2: Feedback Generation**
- System SHALL generate feedback from multiple providers in parallel
- System SHALL complete feedback generation within PROVIDER_TIMEOUT (default 120s) per provider
- System SHALL not block UI during long-running operations

### 2.2 Reliability

**NFR-2.2.1: Error Handling**
- System SHALL handle missing dependencies gracefully
- System SHALL provide clear error messages for configuration issues
- System SHALL continue operation when optional features fail
- System SHALL retry failed operations with fallback strategies

**NFR-2.2.2: Data Integrity**
- System SHALL validate media files before processing
- System SHALL preserve timestamp accuracy across chunks
- System SHALL handle file locking issues on Windows

### 2.3 Usability

**NFR-2.3.1: User Interface**
- System SHALL provide web-based Gradio interface
- System SHALL display live transcription progress
- System SHALL organize results in tabbed interface
- System SHALL provide clear status messages and progress indicators

**NFR-2.3.2: Configuration**
- System SHALL support configuration via environment variables
- System SHALL support configuration via .env files
- System SHALL support configuration via JSON config files
- System SHALL provide sensible defaults for all settings

### 2.4 Security

**NFR-2.4.1: Credentials Management**
- System SHALL load API keys from secrets directory
- System SHALL never commit secrets to version control
- System SHALL support separate .env files for different environments

**NFR-2.4.2: Email Security**
- System SHALL use Gmail app passwords (not plain passwords)
- System SHALL support SMTP SSL/TLS
- System SHALL validate recipient email addresses

### 2.5 Scalability

**NFR-2.5.1: Concurrent Processing**
- System SHALL support queue-based request handling
- System SHALL limit concurrent requests (default max_size=4)
- System SHALL handle multiple users via Gradio queue

**NFR-2.5.2: Resource Management**
- System SHALL clean up temporary files after processing
- System SHALL manage workspace directory size
- System SHALL support configurable model download locations

### 2.6 Compatibility

**NFR-2.6.1: Platform Support**
- System SHALL run on Linux, macOS, and Windows
- System SHALL support both FFmpeg and GStreamer
- System SHALL handle platform-specific file locking

**NFR-2.6.2: Deployment**
- System SHALL support local deployment
- System SHALL support Hugging Face Spaces deployment
- System SHALL support Docker containerization
- System SHALL support configurable host and port binding

## 3. User Stories

### 3.1 Core Workflows

**US-3.1.1: Basic Transcription**
```
As a teacher
I want to upload a lecture recording
So that I can get an accurate transcript with timestamps
```

**US-3.1.2: Multi-Model Feedback**
```
As a teacher
I want to compare feedback from multiple AI models
So that I can get diverse pedagogical insights
```

**US-3.1.3: Question Tracking**
```
As a teacher
I want to see which student questions were answered
So that I can follow up on unanswered questions
```

**US-3.1.4: Email Reports**
```
As a teacher
I want to receive feedback reports via email
So that I can review them offline and share with colleagues
```

**US-3.1.5: Model Voting**
```
As a teacher
I want to vote on which AI models provide the best feedback
So that the system learns my preferences over time
```

### 3.2 Advanced Workflows

**US-3.2.1: Visual Analysis**
```
As a teacher
I want to see when students raised their hands
So that I can assess engagement and participation
```

**US-3.2.2: Board Content Review**
```
As a teacher
I want to extract text from whiteboard snapshots
So that I can review what was written during class
```

**US-3.2.3: YouTube Processing**
```
As a teacher
I want to analyze YouTube lecture videos
So that I can get feedback on recorded online classes
```

**US-3.2.4: Live Recording**
```
As a teacher
I want to record directly through the interface
So that I don't need separate recording software
```

## 4. Acceptance Criteria

### 4.1 Transcription Quality
- Transcription accuracy > 90% for clear audio
- Timestamp accuracy within ±0.5 seconds
- Language detection accuracy > 95% for supported languages
- Streaming updates delivered within 2 seconds of processing

### 4.2 Feedback Quality
- All feedback includes timestamps for specific issues
- Feedback covers all 6 required sections (topics, improvements, fixes, examples, questions, plan)
- At least 3 models successfully generate feedback when "All" is selected
- Adaptive ranking correctly orders models by effective weight

### 4.3 Q&A Detection
- Question detection recall > 80% for explicit questions
- Student/teacher classification accuracy > 70%
- Answer status estimation accuracy > 60%

### 4.4 System Reliability
- System handles 2-hour videos without crashes
- System recovers from single provider failures
- System processes at least 10 concurrent requests
- Uptime > 99% for deployed instances

### 4.5 User Experience
- Interface loads within 3 seconds
- Transcription starts within 10 seconds of upload
- All tabs remain responsive during processing
- Error messages are actionable and clear

## 5. Constraints & Assumptions

### 5.1 Technical Constraints
- Requires Python 3.8+
- Requires FFmpeg or GStreamer for audio extraction
- Requires API keys for cloud LLM providers
- Requires sufficient disk space for workspace (recommend 10GB+)

### 5.2 Assumptions
- Users have stable internet connection for cloud LLMs
- Audio quality is sufficient for transcription (>16kHz, clear speech)
- Lectures are primarily in supported languages
- Users have valid API credentials for desired providers

### 5.3 Out of Scope
- Real-time live lecture analysis during class
- Speaker diarization (identifying individual speakers)
- Automatic grading or assessment
- Student privacy compliance (FERPA, GDPR) - user responsibility
- Multi-language feedback (feedback is English-only)

## 6. Dependencies

### 6.1 Required Dependencies
- gradio >= 5.49.1
- faster-whisper == 1.0.3
- requests >= 2.31.0
- numpy >= 1.24, < 2.0
- pandas == 2.2.2

### 6.2 Optional Dependencies
- yt-dlp >= 2025.10.22 (for URL downloads)
- opencv-python-headless == 4.10.0.84 (for visual analysis)
- pytesseract == 0.3.10 (for OCR)
- ultralytics (for YOLO pose detection)
- pymongo >= 4.9.0 (for persistence)
- bertopic (for topic extraction)

### 6.3 External Services
- Groq API (optional)
- OpenAI API (optional)
- Google Gemini API (optional)
- OpenRouter API (optional)
- Ollama server (optional, local)
- MongoDB Atlas (optional)
- Gmail SMTP (optional)

## 7. Configuration Reference

### 7.1 Core Settings
- `WHISPER_SIZE`: Model size (tiny/base/small/medium/large)
- `WHISPER_COMPUTE`: Compute type (auto/int8/float16/float32)
- `CHUNK_SEC`: Segment duration for long media (default 900)
- `MAX_DIRECT_SEC`: Threshold for segmentation (default 1200)

### 7.2 Provider Settings
- `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_MAX_TOKENS`, `GROQ_TEMP`
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_MAX_TOKENS`, `OPENAI_TEMP`
- `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_MAX_TOKENS`, `GEMINI_TEMP`
- `OPENROUTER_API_KEY`
- `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PREDICT`

### 7.3 Email Settings
- `GMAIL_USER`: Gmail address
- `GMAIL_APP_PASSWORD`: Gmail app password
- `SMTP_DEBUG`: Enable SMTP debugging (0/1)

### 7.4 Visual Analysis Settings
- `VISION_SAMPLE_SEC`: Frame sampling rate (default 1.5)
- `HAND_IOU_THRESH`: IoU threshold for tracking (default 0.35)
- `MAX_BOARD_FRAMES`: Max board snapshots (default 8)

### 7.5 MongoDB Settings
- `MONGO_URI`: MongoDB connection string
- `MONGO_ENABLED`: Enable persistence (0/1)
- `MONGO_DB_NAME`: Database name (default "classroom")
- `MONGO_COLLECTION`: Collection name (default "classroom")

## 8. Success Metrics

### 8.1 Usage Metrics
- Number of lectures processed per week
- Average processing time per lecture
- User retention rate (weekly active users)

### 8.2 Quality Metrics
- User satisfaction score (via feedback votes)
- Model ranking convergence (stable top 3 models)
- Email report open rate

### 8.3 Technical Metrics
- System uptime percentage
- Average response time
- Error rate per provider
- Storage utilization

## 9. Future Enhancements

### 9.1 Planned Features
- Speaker diarization for multi-speaker lectures
- Real-time streaming analysis during live classes
- Custom feedback templates and rubrics
- Integration with LMS platforms (Canvas, Moodle)
- Mobile app for recording and analysis

### 9.2 Research Areas
- Automatic slide extraction and synchronization
- Student engagement scoring from visual cues
- Personalized feedback based on teaching style
- Multi-language feedback generation
- Automated action item tracking
