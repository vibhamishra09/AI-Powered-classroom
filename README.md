CLASSROOM — AI-Powered Speech Processing Pipeline

An intelligent system that extracts transcripts, topics, and insights from classroom video/audio recordings to help teachers analyze and improve their teaching.

🎥 Demo

https://drive.google.com/file/d/1-Rn6TaNa3KP38D1wWbyxIlVad2kEVoez/view?usp=sharing
## 📁 Project Structure

```
speech2/
│
├── app.py
├── requirements.txt
├── secrets/
│   ├── .env
│   └── config.json
├── workspace/
├── outputs/
└── README.md
```
⚠️ Important: Never commit the secrets/ folder. It is ignored via .gitignore.

🔐 Secrets Setup
### Create `.env`

```bash
mkdir -p secrets
touch secrets/.env
```
Example .env
# API Keys
## 🔐 API Keys

```env
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key

//Gmail
GMAIL_USER=your_email
GMAIL_APP_PASSWORD=your_app_password

//SMTP_DEBUG=0
Whisper
WHISPER_SIZE=small
WHISPER_COMPUTE=auto

 //Processing
CHUNK_SEC=900
MAX_DIRECT_SEC=1200
TRANSLATE_DEFAULT=1

//Gradio
GRADIO_SERVER_PORT=7860
GRADIO_HOST=127.0.0.1
GRADIO_SHARE=0

//MongoDB (optional)
MONGO_URI=mongodb+srv://username:password@cluster0.mongodb.net
MONGO_DB_NAME=classroom
MONGO_COLLECTION=teaching_results
MONGO_ENABLED=1
```
Create config.json
```
{
  "GROQ_API_KEY": "",
  "OPENROUTER_API_KEY": "",
  "GEMINI_API_KEY": "",
  "OPENAI_API_KEY": "",
  "GMAIL_USER": "",
  "GMAIL_APP_PASSWORD": "",
  "OLLAMA_URL": "http://127.0.0.1:11434",
  "OLLAMA_MODEL": "qwen2.5:7b-instruct"
}
```
⚙️ Ollama Setup (Local LLM)

Install
```
curl -fsSL https://ollama.com/install.sh | sh
```
Pull Model
```
ollama pull qwen2.5:7b-instruct
```
Run Server
```
ollama serve
```
Runs on: http://127.0.0.1:11434

🧩 Installation

Create Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
Install Requirements
pip install -r requirements.txt
```
If errors occur:

pip install --no-cache-dir -r requirements.txt
▶️ Run the Project
python app.py

Open:

http://127.0.0.1:7860


🚀 Features

🎥 YouTube video processing (yt-dlp)

🎙️ Speech-to-text (Whisper)

🧠 Topic extraction using LLMs

🔄 Multi-provider support (OpenAI, Groq, Gemini, Ollama)

📊 Teaching insights & feedback

📧 Email reports

🗄️ MongoDB storage (optional)

🧠 Use Cases

Classroom lecture analysis

Teacher performance feedback

Student engagement insights

❗ Common Mistakes (why formatting breaks)

❌ Using ``` id="something" → REMOVE id

❌ Missing blank line before/after code blocks

❌ Wrong indentation in folders

❌ Mixing tabs and spaces
