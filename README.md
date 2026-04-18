# ⚡ AttentionX — Automated Content Repurposing Engine

> Turn hours of long-form video into viral short clips in one click using AI

## 🔗 Live Demo: [PASTE STREAMLIT URL HERE]
## 🎥 Demo Video: [PASTE GOOGLE DRIVE LINK HERE]

---

## Problem Statement

Educators, mentors and creators spend hours producing long-form content that modern audiences never fully watch. Valuable insights are buried inside 60-minute videos while audiences consume content in 60-second bursts.

## Solution

AttentionX uses GenAI and Multimodal models to automatically detect the most powerful moments in any video, crop them to vertical format, add captions, and export a full content pack — in minutes.

---

## ✅ Features

| Feature | Description |
|---------|-------------|
| 🧠 AI Emotional Peak Detection | Gemini + audio energy + transcript sentiment refine the most viral moments |
| 🎙️ Auto Transcription | OpenAI Whisper with timestamped segments |
| 📱 Smart Vertical Crop | MediaPipe face tracking keeps the speaker centered |
| 💬 Karaoke Captions | Timed captions burned directly onto clips |
| 📊 Energy Visualization | Audio energy graph with clip markers |
| 🎯 Virality Scoring | 0–100 score per clip with color-coded bars |
| 🔴 Emotion Tagging | shock / inspiration / curiosity / humor |
| 📦 One-Click Export | Complete ZIP with clips + captions + reports |
| 📝 Ready-to-post copy | Instagram captions + YouTube descriptions generated |
| 🆚 Before/After Preview | Side-by-side 16:9 vs 9:16 comparison |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Transcription | OpenAI Whisper (configurable, tiny default) |
| AI Analysis | Google Gemini (configurable model) |
| Face Tracking | MediaPipe Face Detection |
| Video Processing | MoviePy |
| Audio Analysis | Librosa |
| Visualization | Matplotlib |
| Packaging | Python zipfile |

---

## 📁 Project Structure

```
attentionx/
├── app.py                  ← Main Streamlit app
├── pipeline/
│   ├── __init__.py
│   ├── transcriber.py      ← Whisper transcription
│   ├── analyzer.py         ← Gemini AI analysis
│   ├── clipper.py          ← Video clip extraction
│   ├── cropper.py          ← 16:9 to 9:16 face crop
│   ├── caption.py          ← Burn captions onto video
│   └── audio_viz.py        ← Audio energy graph
├── output/                 ← Generated clips saved here
├── .env                    ← API keys (never commit!)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/attentionx
cd attentionx
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Whisper requires ffmpeg. Install it with:
> - macOS: `brew install ffmpeg`
> - Ubuntu: `sudo apt install ffmpeg`
> - Windows: [Download from ffmpeg.org](https://ffmpeg.org/download.html)

### 3. Add your API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash-lite
GEMINI_FALLBACK_MODELS=gemini-1.5-flash,gemini-1.5-flash-8b
WHISPER_MODEL=tiny
WHISPER_LANGUAGE=
ATTENTIONX_ENCODE_PRESET=superfast
ATTENTIONX_ENCODE_THREADS=8
ATTENTIONX_USE_FAST_CUT=1
ATTENTIONX_FACE_SAMPLES=5
ATTENTIONX_AUDIO_DIAGNOSTICS=0
ATTENTIONX_TARGET_CLIPS=
```


### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 🔄 How It Works

```
Upload Video
     ↓
Whisper transcribes audio with timestamps
     ↓
Gemini analyzes transcript → finds 3-5 viral moments
     ↓
MoviePy cuts each moment (+ 2s buffers)
     ↓
MediaPipe detects face → centers vertical crop (9:16)
     ↓
Captions burned karaoke-style onto each clip
     ↓
Audio energy chart shows clip positions
     ↓
Download ZIP with all clips + copy
```

---

## 🚀 Deploy to Streamlit Cloud

1. Push code to GitHub (see commands below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set main file: `app.py`
5. Go to **Advanced Settings → Secrets** and add:
   ```
   GEMINI_API_KEY = "your_key_here"
   ```
6. Click **Deploy**
7. Copy the public URL and update this README

---

## 🗂️ Git Commands

```bash
git init
git add .
git commit -m "AttentionX MVP - AI Content Repurposing Engine"
git branch -M main
git remote add origin https://github.com/USERNAME/attentionx.git
git push -u origin main
```

After getting live URL:
```bash
# Update README with your live demo URL, then:
git add README.md
git commit -m "Add live demo link"
git push
```

---

## 📦 ZIP Content Pack Contents

| File | Contents |
|------|----------|
| `clip_N_final.mp4` | Vertical 9:16 clip with captions |
| `hooks_and_captions.txt` | Hook headlines + virality scores |
| `instagram_captions.txt` | Ready-to-paste Instagram captions |
| `youtube_descriptions.txt` | Full YouTube descriptions |
| `virality_report.txt` | Scores, emotions, recommendations |

---

## 🔑 Getting a Free Gemini API Key

1. Go to [aistudio.google.com](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API key**
4. Copy and paste into the sidebar or `.env` file

---

## 👤 Team

Solo Project — Built for the AttentionX AI Hackathon by **UnsaidTalks**

---

## 📄 License

MIT License — free to use, modify, and distribute.

