#!/usr/bin/env python3
"""
🎤 Voice Cloner — Pashto / Urdu / Any Language
================================================
Clone anyone's voice using just a short audio sample (10-30 seconds).
Uses OpenVoice v2 for zero-shot voice cloning.

Usage:
    python app.py                    # Default: localhost:7860
    python app.py --port 8080        # Custom port
    python app.py --share            # Create public Gradio link
    python app.py --host 0.0.0.0     # Listen on all interfaces
"""

import os
import time
import argparse
import torch
import numpy as np
import gradio as gr

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints_v2/converter"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load Model ───────────────────────────────────────────────────────────────
def load_model():
    """Load the OpenVoice ToneColorConverter model."""
    from openvoice.api import ToneColorConverter

    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model config not found at {config_path}. "
            "Please run setup.sh first to download checkpoints."
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {ckpt_path}. "
            "Please run setup.sh first to download checkpoints."
        )

    converter = ToneColorConverter(config_path, device=DEVICE)
    converter.load_ckpt(ckpt_path)
    return converter


print(f"🔧 Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("📦 Loading OpenVoice v2 model...")
converter = load_model()
print("✅ Model loaded and ready!\n")


# ── Voice Cloning Function ───────────────────────────────────────────────────
def clone_voice(reference_audio, source_audio, progress=gr.Progress()):
    """
    Clone voice: Take source audio content and make it sound like the reference voice.

    Args:
        reference_audio: Path to Person A's audio (voice to clone — 10-30 sec)
        source_audio: Path to Person B's audio (content/speech to convert)

    Returns:
        Path to the output audio file
    """
    from openvoice import se_extractor

    if reference_audio is None:
        raise gr.Error("❌ Please upload Person A's reference audio!")
    if source_audio is None:
        raise gr.Error("❌ Please upload Person B's source audio!")

    try:
        progress(0.1, desc="Extracting Person A's voice characteristics...")
        # Extract speaker embedding from reference audio (Person A)
        target_se, _ = se_extractor.get_se(
            reference_audio, converter, vad=True
        )

        progress(0.4, desc="Analyzing Person B's speech...")
        # Extract speaker embedding from source audio (Person B)
        source_se, _ = se_extractor.get_se(
            source_audio, converter, vad=True
        )

        progress(0.6, desc="Cloning voice... 🎤")
        # Generate output: Person B's words in Person A's voice
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f"cloned_{timestamp}.wav")

        converter.convert(
            audio_src_path=source_audio,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
        )

        progress(1.0, desc="Done! ✅")
        return output_path

    except Exception as e:
        error_msg = str(e)
        if "no speech" in error_msg.lower() or "too short" in error_msg.lower():
            raise gr.Error(
                "❌ Could not detect speech in the audio. "
                "Please make sure the audio contains clear speech (at least 5 seconds)."
            )
        raise gr.Error(f"❌ Error during voice cloning: {error_msg}")


# ── Gradio Web UI ────────────────────────────────────────────────────────────
CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5em;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 1.5em;
}
.tips-box {
    background: #f0f7ff;
    border: 1px solid #d0e3ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 8px;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(
    title="🎤 Voice Cloner — Pashto / Urdu",
    theme=gr.themes.Soft(),
    css=CSS,
) as app:

    gr.HTML("""
        <h1 class="main-title">🎤 Voice Cloner</h1>
        <p class="subtitle">
            Clone anyone's voice using just a short audio sample — 
            works with <strong>Pashto</strong>, <strong>Urdu</strong>, 
            <strong>English</strong>, or <strong>any language</strong>!
        </p>
    """)

    # ── How it works ──
    with gr.Accordion("ℹ️ How does it work?", open=False):
        gr.Markdown("""
        ### How it works:
        1. **Upload Person A's audio** — A short clip (10-30 seconds) of the voice you want to clone
        2. **Upload Person B's audio** — The speech you want converted to Person A's voice
        3. **Click "Clone Voice"** — The AI will output Person B's words in Person A's voice!

        ### Key points:
        - 🌍 **Language doesn't matter** — Person A can speak Pashto and Person B can speak Urdu (or any other combination)
        - 🎯 The output will have **Person B's words** spoken in **Person A's voice**
        - 🔒 Everything runs **locally** on this machine — no data sent to the cloud
        - ⚡ Processing takes just a few seconds on GPU
        """)

    gr.Markdown("---")

    # ── Audio Inputs ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Person A — Voice to Clone")
            ref_audio = gr.Audio(
                label="Reference Voice (10-30 seconds)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            gr.HTML("""
                <div class="tips-box">
                    <strong>💡 Tips for best results:</strong><br>
                    • Use <strong>10-30 seconds</strong> of clear speech<br>
                    • Avoid background noise or music<br>
                    • One speaker only — no overlapping voices<br>
                    • Any language works (Pashto, Urdu, English, etc.)
                </div>
            """)

        with gr.Column(scale=1):
            gr.Markdown("### 🗣️ Person B — Speech to Convert")
            src_audio = gr.Audio(
                label="Source Speech",
                type="filepath",
                sources=["upload", "microphone"],
            )
            gr.HTML("""
                <div class="tips-box">
                    <strong>💡 Tips:</strong><br>
                    • This is the speech whose <strong>words</strong> will be kept<br>
                    • But the <strong>voice</strong> will change to Person A's<br>
                    • Can be in <strong>any language</strong> — doesn't need to match Person A<br>
                    • Clear audio gives better results
                </div>
            """)

    # ── Clone Button ──
    gr.Markdown("")
    clone_btn = gr.Button(
        "🔄 Clone Voice",
        variant="primary",
        size="lg",
    )

    # ── Output ──
    gr.Markdown("---")
    gr.Markdown("### ✅ Result — Person B's words in Person A's voice")
    output_audio = gr.Audio(
        label="Cloned Audio Output",
        type="filepath",
        interactive=False,
    )

    # ── Wire up the button ──
    clone_btn.click(
        fn=clone_voice,
        inputs=[ref_audio, src_audio],
        outputs=[output_audio],
    )

    # ── Footer info ──
    gr.Markdown("---")
    gr.Markdown(
        "<center><small>Powered by <strong>OpenVoice v2</strong> | "
        "Runs locally on your machine | "
        f"Device: <strong>{DEVICE}</strong>"
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")
        + "</small></center>"
    )


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Cloner Web UI")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for network access)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run on (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link"
    )
    args = parser.parse_args()

    print(f"🚀 Starting Voice Cloner at http://{args.host}:{args.port}")
    if args.host == "127.0.0.1":
        print(f"🌐 Open in browser: http://localhost:{args.port}")
        print(f"💡 For SSH access: ssh -L {args.port}:localhost:{args.port} user@this-machine")

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
