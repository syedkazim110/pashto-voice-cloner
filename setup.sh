#!/bin/bash
set -e

echo ""
echo "🎤 Voice Cloner Setup (OpenVoice v2)"
echo "====================================="
echo "Pashto / Urdu / Any Language"
echo "====================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION found"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "✅ GPU detected: $GPU_NAME"
else
    echo "⚠️  No NVIDIA GPU detected. Will use CPU (slower but works)"
fi

echo ""

# Install system dependencies (FFmpeg libs needed for PyAV/OpenVoice)
echo "📦 Installing system dependencies (FFmpeg, etc.)..."
echo "   (May ask for sudo password)"
sudo apt-get update -qq 2>/dev/null || true
sudo apt-get install -y -qq \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    2>/dev/null
echo "✅ System dependencies installed"

echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment already exists"
fi

source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch with CUDA
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
echo "   (This may take a few minutes)"
pip install torch torchaudio --quiet

# Verify CUDA
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ PyTorch CUDA is available — GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  PyTorch CUDA not available — will use CPU')
"

# Pre-install PyAV binary wheel (avoids Cython compilation issues)
echo ""
echo "📥 Installing PyAV (audio/video processing)..."
pip install av --only-binary=:all: --quiet
echo "✅ PyAV installed"

# Clone and install OpenVoice
echo ""
echo "📥 Installing OpenVoice v2..."
if [ ! -d "OpenVoice" ]; then
    git clone https://github.com/myshell-ai/OpenVoice.git
    echo "✅ OpenVoice repository cloned"
else
    echo "✅ OpenVoice repository already exists"
fi

cd OpenVoice
pip install -e . --quiet
cd ..
echo "✅ OpenVoice installed"

# Install additional dependencies
echo ""
echo "📥 Installing Web UI dependencies..."
pip install gradio huggingface_hub --quiet
echo "✅ Gradio and dependencies installed"

# Download model checkpoints
echo ""
echo "📥 Downloading model checkpoints from HuggingFace..."
echo "   (This may take a few minutes on first run)"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='checkpoints_v2')
print('✅ Model checkpoints downloaded!')
"

# Create directories
mkdir -p samples outputs

echo ""
echo "====================================="
echo "✅ Setup complete!"
echo "====================================="
echo ""
echo "To run the Voice Cloner:"
echo ""
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:7860"
echo ""
echo "If connecting via SSH, use:"
echo "  ssh -L 7860:localhost:7860 user@office-pc"
echo "  Then open http://localhost:7860 on your local browser"
echo ""
