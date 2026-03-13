# 🎤 Voice Cloner — Pashto / Urdu / Any Language

Clone anyone's voice using just a **short audio sample (10-30 seconds)**. Works with **Pashto**, **Urdu**, **English**, or **any language**!

Powered by [OpenVoice v2](https://github.com/myshell-ai/OpenVoice) — zero-shot voice cloning.

---

## ✨ Features

- 🌍 **Language-agnostic** — Works with Pashto, Urdu, English, or any language
- ⚡ **Zero-shot cloning** — No training needed, just a short reference clip
- 🔄 **Cross-lingual** — Person A can speak Pashto, Person B can speak Urdu
- 🖥️ **Web UI** — Beautiful, easy-to-use browser interface
- 🔒 **100% Local** — Everything runs on your machine, no cloud
- 🚀 **GPU accelerated** — Fast processing on NVIDIA GPUs

## 🎯 How It Works

```
Person A audio (10-30s, any language)  ──┐
                                          ├──→  🎯 Output: Person B's words in Person A's voice
Person B audio (speech to convert)     ──┘
```

1. Upload **Person A's audio** — the voice you want to clone (10-30 seconds)
2. Upload **Person B's audio** — the speech content you want converted
3. Click **"Clone Voice"** — get Person B's words spoken in Person A's voice!

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA (recommended) — also works on CPU (slower)
- **Git**

### Setup (One-time)

```bash
# Clone the repository
git clone https://github.com/syedkazim110/check-kazim.git pashto-voice-cloner
cd pashto-voice-cloner

# Run the setup script (installs everything)
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install OpenVoice v2
- Download model checkpoints (~500MB)
- Install the Gradio web UI

### Run the App

```bash
# Activate the virtual environment
source venv/bin/activate

# Start the web UI
python app.py
```

Then open **http://localhost:7860** in your browser.

---

## 🔌 SSH Remote Access

If running on a remote server (e.g., office PC), connect with port forwarding:

```bash
# From your local machine:
ssh -L 7860:localhost:7860 user@remote-server-ip

# Then on the remote server:
cd pashto-voice-cloner
source venv/bin/activate
python app.py

# Open on your LOCAL browser:
# http://localhost:7860
```

---

## 📖 Usage Options

```bash
# Default (localhost:7860)
python app.py

# Custom port
python app.py --port 8080

# Public Gradio link (accessible from anywhere)
python app.py --share

# Listen on all network interfaces
python app.py --host 0.0.0.0
```

---

## 💡 Tips for Best Results

### Person A (Reference Voice):
- Use **10-30 seconds** of clear speech
- Avoid background noise or music
- One speaker only — no overlapping voices
- Any language works

### Person B (Source Speech):
- Clear audio gives better results
- Can be in any language (doesn't need to match Person A)
- Longer clips work fine too

---

## 📁 Project Structure

```
pashto-voice-cloner/
├── app.py              # Gradio Web UI application
├── setup.sh            # One-click setup script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── samples/            # Place sample audio files here
├── outputs/            # Generated audio outputs
├── checkpoints_v2/     # Model checkpoints (downloaded by setup.sh)
├── OpenVoice/          # OpenVoice repository (cloned by setup.sh)
└── venv/               # Python virtual environment (created by setup.sh)
```

---

## 🛠️ Tech Stack

- **[OpenVoice v2](https://github.com/myshell-ai/OpenVoice)** — Zero-shot voice cloning engine
- **[Gradio](https://gradio.app)** — Web UI framework
- **[PyTorch](https://pytorch.org)** — Deep learning framework with CUDA
- **Python 3.10+** — Runtime

---

## ⚠️ Disclaimer

This tool is for **demonstration and educational purposes only**. Please use responsibly and respect privacy and consent when cloning voices. Do not use this tool to impersonate others or create misleading content.

---

## 📄 License

This project uses OpenVoice v2 which is licensed under the MIT License.
