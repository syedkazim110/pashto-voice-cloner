#!/bin/bash
set -e

echo ""
echo "🎤 RVC Training Setup"
echo "====================="
echo "Sets up RVC WebUI for voice model training"
echo "====================="
echo ""

# Ensure we're in the project directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup.sh first!"
    exit 1
fi

# Clone RVC WebUI
echo ""
echo "📥 Setting up RVC WebUI for training..."
if [ ! -d "Retrieval-based-Voice-Conversion-WebUI" ]; then
    git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
    echo "✅ RVC WebUI cloned"
else
    echo "✅ RVC WebUI already exists"
fi

cd Retrieval-based-Voice-Conversion-WebUI

# Install RVC dependencies
echo ""
echo "📥 Installing RVC dependencies..."
pip install -r requirements.txt --quiet 2>/dev/null || true

# Download pre-trained models needed for training
echo ""
echo "📥 Downloading pre-trained models for RVC training..."
python3 -c "
import os
from huggingface_hub import hf_hub_download

# Create directories
os.makedirs('assets/pretrained_v2', exist_ok=True)
os.makedirs('assets/hubert', exist_ok=True)
os.makedirs('assets/rmvpe', exist_ok=True)

# Download HuBERT base model
print('  Downloading HuBERT model...')
hf_hub_download(
    repo_id='lj1995/VoiceConversionWebUI',
    filename='hubert_base.pt',
    local_dir='assets/hubert',
)

# Download RMVPE model (pitch extraction)
print('  Downloading RMVPE model...')
hf_hub_download(
    repo_id='lj1995/VoiceConversionWebUI',
    filename='rmvpe.pt',
    local_dir='assets/rmvpe',
)

# Download pretrained v2 models
pretrained_files = [
    'f0D48k.pth', 'f0G48k.pth',
    'f0D40k.pth', 'f0G40k.pth',
]
for f in pretrained_files:
    print(f'  Downloading {f}...')
    try:
        hf_hub_download(
            repo_id='lj1995/VoiceConversionWebUI',
            filename=f'pretrained_v2/{f}',
            local_dir='assets',
        )
    except Exception as e:
        print(f'  Warning: Could not download {f}: {e}')

print('✅ Pre-trained models downloaded!')
" 2>/dev/null || echo "⚠️  Some model downloads may have failed. Training may still work."

cd "$SCRIPT_DIR"

# Create models directory for trained models
mkdir -p models

echo ""
echo "====================="
echo "✅ RVC Training Setup Complete!"
echo "====================="
echo ""
echo "To train a voice model:"
echo ""
echo "  1. First, prepare your audio:"
echo "     python preprocess.py your_40min_audio.wav -o training_data/"
echo ""
echo "  2. Then start the RVC WebUI for training:"
echo "     cd Retrieval-based-Voice-Conversion-WebUI"
echo "     python infer-web.py"
echo ""
echo "  3. In the RVC WebUI:"
echo "     - Go to 'Train' tab"
echo "     - Set experiment name (e.g., 'person_a')"
echo "     - Point training data to: $SCRIPT_DIR/training_data/"
echo "     - Click 'Process data' → 'Feature extraction' → 'Train'"
echo "     - Training takes ~20-30 min on RTX 5090"
echo ""
echo "  4. Copy the trained model:"
echo "     cp Retrieval-based-Voice-Conversion-WebUI/assets/weights/*.pth models/"
echo "     cp Retrieval-based-Voice-Conversion-WebUI/logs/*/added_*.index models/"
echo ""
echo "  5. Use in our app:"
echo "     python app.py"
echo "     → Go to 'Trained Voice' tab"
echo ""
