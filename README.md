🧠 CLASSROOM — AI-Powered Speech Processing Pipeline

An intelligent system that extracts transcripts, topics, and insights from classroom video/audio recordings. This helps educators analyze teaching effectiveness and receive actionable feedback.

The pipeline supports:

🎥 YouTube video ingestion (via yt-dlp)

🎙️ Speech-to-text transcription

🧠 Topic & insight extraction using LLMs

📊 Optional storage & reporting

Supports both local (Ollama) and cloud LLM providers like Gemini, OpenAI, Groq, OpenRouter.

📁 Project Structure
speech2/
│
├── app.py
├── requirements.txt
│
├── secrets/
│   ├── .env
│   └── config.json
│
├── workspace/
├── outputs/
└── README.md

⚠️ Important: Never commit the secrets/ folder to GitHub. It is already ignored via .gitignore.

🎥 Demo

👉 https://drive.google.com/file/d/1-Rn6TaNa3KP38D1wWbyxIlVad2kEVoez/view?usp=sharing

🔐 Secrets Setup
1. Create .env file
mkdir -p secrets
touch secrets/.env
Example .env
# API Keys
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key

# Gmail (for sending reports)
GMAIL_USER=your_email
GMAIL_APP_PASSWORD=your_app_password

SMTP_DEBUG=0

# Whisper Settings
WHISPER_SIZE=small
WHISPER_COMPUTE=auto
WORD_TS=0

# Processing Config
CHUNK_SEC=900
MAX_DIRECT_SEC=1200
TRANSLATE_DEFAULT=1

# Vision Sampling
VISION_SAMPLE_SEC=1.5

# Board Detection
BOARD_WHITE_PCT=0.22
BOARD_DARK_PCT=0.22

# Provider Timeout
PROVIDER_TIMEOUT=120

# Gradio Config
GRADIO_SERVER_PORT=7860
GRADIO_HOST=127.0.0.1
GRADIO_SHARE=0

# MongoDB (Optional)
MONGO_URI=mongodb+srv://username:password@cluster0.mongodb.net
MONGO_DB_NAME=classroom
MONGO_COLLECTION=teaching_results
MONGO_ENABLED=1
MONGO_TLS_ALLOW_INVALID_CERTS=0
2. Create config.json
{
  "GROQ_API_KEY": "",
  "OPENROUTER_API_KEY": "",
  "GEMINI_API_KEY": "",
  "OPENAI_API_KEY": "",
  "GMAIL_USER": "",
  "GMAIL_APP_PASSWORD": "",
  "OLLAMA_URL": "http://127.0.0.1:11434",
  "OLLAMA_MODEL": "qwen2.5:7b-instruct",
  "OLLAMA_TEMP": "0.7",
  "OLLAMA_NUM_CTX": "4096",
  "OLLAMA_NUM_PREDICT": "512",
  "OLLAMA_TIMEOUT": "120"
}
⚙️ Ollama Setup (Local LLM)

Use this if you prefer running models locally instead of cloud APIs.

Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
Pull Model
ollama pull qwen2.5:7b-instruct
Start Server
ollama serve

Runs at:

http://127.0.0.1:11434
🧩 Installation
1. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
2. Install Dependencies
pip install -r requirements.txt

If you face build issues:

pip install --no-cache-dir -r requirements.txt
▶️ Run the Project
python app.py

Open in browser:

http://127.0.0.1:7860
🚀 Features

🎥 YouTube video download (yt-dlp)

🎙️ Speech-to-text transcription (Whisper)

🧠 Topic extraction using LLMs

🔄 Multi-provider LLM support:

OpenAI

Groq

Gemini

OpenRouter

Ollama (local)

📊 Teaching insights & feedback

📧 Email reporting

🗄️ Optional MongoDB storage

🧠 Use Cases

Classroom lecture analysis

Teacher performance feedback

Student engagement insights

AI-powered academic analytics

If you want, I can also:

Make this GitHub-ready with badges + screenshots

Add architecture diagram

Or convert it into a proper project documentation (for internship/report use)
