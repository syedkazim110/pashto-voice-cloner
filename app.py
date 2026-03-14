#!/usr/bin/env python3
"""
🎤 Voice Cloner — Pashto / Urdu / Any Language
================================================
Clone anyone's voice using just a short audio sample OR a trained voice model.

Features:
- Tab 1: Quick Clone (OpenVoice v2 — zero-shot, any voice)
- Tab 2: Trained Voice (RVC — best quality with trained model)
- Tab 3: Audio Preparation (clean & split audio for training)

Usage:
    python app.py                    # Default: localhost:7860
    python app.py --port 8080        # Custom port
    python app.py --share            # Create public Gradio link
"""

import os
import sys
import glob
import time
import argparse
import torch
import numpy as np
import gradio as gr

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
CHECKPOINT_DIR = "checkpoints_v2/converter"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"🔧 Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 VRAM: {vram:.1f} GB")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OPENVOICE (Quick Clone)
# ══════════════════════════════════════════════════════════════════════════════

openvoice_converter = None


def load_openvoice():
    """Load OpenVoice ToneColorConverter (lazy load)."""
    global openvoice_converter
    if openvoice_converter is not None:
        return openvoice_converter

    from openvoice.api import ToneColorConverter

    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "OpenVoice checkpoints not found. Run setup.sh first."
        )

    openvoice_converter = ToneColorConverter(config_path, device=DEVICE)
    openvoice_converter.load_ckpt(ckpt_path)
    return openvoice_converter


def clone_voice_openvoice(reference_audio, source_audio, progress=gr.Progress()):
    """Zero-shot voice cloning with OpenVoice v2."""
    from openvoice import se_extractor

    if reference_audio is None:
        raise gr.Error("❌ Please upload Person A's reference audio!")
    if source_audio is None:
        raise gr.Error("❌ Please upload Person B's source audio!")

    try:
        converter = load_openvoice()

        progress(0.1, desc="Extracting Person A's voice characteristics...")
        target_se, _ = se_extractor.get_se(reference_audio, converter, vad=True)

        progress(0.4, desc="Analyzing Person B's speech...")
        source_se, _ = se_extractor.get_se(source_audio, converter, vad=True)

        progress(0.6, desc="Cloning voice... 🎤")
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f"openvoice_{timestamp}.wav")

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
                "Make sure the audio has clear speech (at least 5 seconds)."
            )
        raise gr.Error(f"❌ Error: {error_msg}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RVC (Trained Voice)
# ══════════════════════════════════════════════════════════════════════════════

rvc_model = None
rvc_available = False

# Check if rvc-python is available
try:
    from rvc_python.infer import RVCInference
    rvc_available = True
    print("✅ RVC inference available")
except ImportError:
    print("⚠️  RVC inference not available (install with: pip install rvc-python)")


def get_available_models():
    """Get list of trained RVC models in models/ directory."""
    models = []
    for ext in ["*.pth"]:
        models.extend(glob.glob(os.path.join(MODELS_DIR, ext)))
    # Filter out index files
    models = [m for m in models if "index" not in os.path.basename(m).lower()]
    return sorted(models)


def get_model_choices():
    """Get model choices for dropdown."""
    models = get_available_models()
    if not models:
        return ["No models found — train one first!"]
    return [os.path.basename(m) for m in models]


def find_index_file(model_path):
    """Find the matching .index file for a model."""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # Look for matching index files
    for pattern in [f"{model_name}*.index", "*.index"]:
        matches = glob.glob(os.path.join(MODELS_DIR, pattern))
        if matches:
            return matches[0]
    return ""


def clone_voice_rvc(
    source_audio,
    model_name,
    pitch_shift,
    index_rate,
    progress=gr.Progress(),
):
    """Voice conversion using trained RVC model."""
    if not rvc_available:
        raise gr.Error(
            "❌ RVC is not installed. Run: pip install rvc-python"
        )
    if source_audio is None:
        raise gr.Error("❌ Please upload source audio!")
    if "No models found" in model_name:
        raise gr.Error("❌ No trained models found in models/ directory!")

    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        index_path = find_index_file(model_path)

        progress(0.2, desc="Loading RVC model...")
        rvc = RVCInference(device=DEVICE)
        rvc.load_model(model_path)

        if index_path and os.path.exists(index_path):
            rvc.set_params(
                f0up_key=int(pitch_shift),
                index_rate=index_rate,
                index_path=index_path,
            )
        else:
            rvc.set_params(
                f0up_key=int(pitch_shift),
                index_rate=0,
            )

        progress(0.5, desc="Converting voice... 🎤")
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f"rvc_{timestamp}.wav")

        rvc.infer_file(source_audio, output_path)

        progress(1.0, desc="Done! ✅")
        return output_path

    except Exception as e:
        raise gr.Error(f"❌ RVC Error: {str(e)}")


def refresh_models():
    """Refresh the list of available models."""
    return gr.Dropdown(choices=get_model_choices(), value=get_model_choices()[0])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AUDIO PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_audio(
    audio_file,
    denoise_strength,
    min_segment,
    max_segment,
    progress=gr.Progress(),
):
    """Clean and split audio for RVC training."""
    if audio_file is None:
        raise gr.Error("❌ Please upload an audio file!")

    from preprocess import clean_and_split_audio

    output_dir = "training_data"

    def progress_cb(val, desc):
        progress(val, desc=desc)

    try:
        segments, info = clean_and_split_audio(
            input_path=audio_file,
            output_dir=output_dir,
            denoise_strength=denoise_strength,
            min_segment_sec=min_segment,
            max_segment_sec=max_segment,
            progress_callback=progress_cb,
        )

        summary = (
            f"### ✅ Audio Preprocessing Complete!\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Input duration | {info['input_duration']:.1f} seconds |\n"
            f"| Output segments | {info['output_segments']} files |\n"
            f"| Output duration | {info['output_duration']:.1f} seconds |\n"
            f"| Saved to | `{info['output_dir']}/` |\n\n"
            f"**Next step:** Use these segments to train an RVC model.\n"
            f"Run `./setup_rvc.sh` to set up training, then follow the instructions."
        )

        # Return the cleaned full file for preview
        clean_full = os.path.join(output_dir, "cleaned_full.wav")
        if os.path.exists(clean_full):
            return summary, clean_full
        return summary, None

    except Exception as e:
        raise gr.Error(f"❌ Preprocessing error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO WEB UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
.main-title { text-align: center; margin-bottom: 0.2em; }
.subtitle { text-align: center; color: #666; margin-bottom: 1em; font-size: 1.1em; }
.tips-box {
    background: #f0f7ff; border: 1px solid #d0e3ff;
    border-radius: 8px; padding: 12px 16px; margin-top: 8px;
}
.warn-box {
    background: #fff8e6; border: 1px solid #ffe0a0;
    border-radius: 8px; padding: 12px 16px; margin-top: 8px;
}
footer { display: none !important; }
"""

# Load OpenVoice at startup
print("📦 Loading OpenVoice v2 model...")
try:
    load_openvoice()
    print("✅ OpenVoice model loaded!\n")
except Exception as e:
    print(f"⚠️  OpenVoice load failed: {e}\n")

with gr.Blocks(
    title="🎤 Voice Cloner — Pashto / Urdu",
    theme=gr.themes.Soft(),
    css=CSS,
) as app:

    gr.HTML("""
        <h1 class="main-title">🎤 Voice Cloner</h1>
        <p class="subtitle">
            Clone voices in <strong>Pashto</strong>, <strong>Urdu</strong>,
            <strong>English</strong>, or <strong>any language</strong>
        </p>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # TAB 1: QUICK CLONE (OpenVoice)
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("🚀 Quick Clone (Zero-Shot)"):
            gr.Markdown("""
            **Zero-shot voice cloning** — just upload a short audio sample (10-30 sec).
            No training needed! Great for quick demos with any voice.
            """)

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("### 🎯 Person A — Voice to Clone")
                    ov_ref_audio = gr.Audio(
                        label="Reference Voice (10-30 seconds)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    gr.HTML("""<div class="tips-box">
                        <strong>💡 Tips:</strong><br>
                        • 10-30 seconds of clear speech<br>
                        • No background noise<br>
                        • Any language works
                    </div>""")

                with gr.Column():
                    gr.Markdown("### 🗣️ Person B — Speech to Convert")
                    ov_src_audio = gr.Audio(
                        label="Source Speech",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    gr.HTML("""<div class="tips-box">
                        <strong>💡 Tips:</strong><br>
                        • Words from this audio will be kept<br>
                        • Voice will change to Person A's<br>
                        • Any language — doesn't need to match
                    </div>""")

            ov_btn = gr.Button("🔄 Clone Voice", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### ✅ Result")
            ov_output = gr.Audio(
                label="Cloned Audio Output",
                type="filepath",
                interactive=False,
            )

            ov_btn.click(
                fn=clone_voice_openvoice,
                inputs=[ov_ref_audio, ov_src_audio],
                outputs=[ov_output],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2: TRAINED VOICE (RVC)
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("🎯 Trained Voice (Best Quality)"):
            if rvc_available:
                gr.Markdown("""
                **Trained voice model** — Best quality! Uses an RVC model trained on
                Person A's voice data. Much more natural and accurate than zero-shot.
                """)
            else:
                gr.HTML("""<div class="warn-box">
                    <strong>⚠️ RVC not installed yet.</strong><br>
                    To use trained voice models, run:<br>
                    <code>pip install rvc-python</code><br>
                    Then restart the app.
                </div>""")

            with gr.Row():
                with gr.Column(scale=2):
                    rvc_src_audio = gr.Audio(
                        label="🗣️ Source Speech (Person B)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    rvc_model_dropdown = gr.Dropdown(
                        choices=get_model_choices(),
                        value=get_model_choices()[0],
                        label="Voice Model",
                        interactive=True,
                    )
                    rvc_refresh_btn = gr.Button("🔄 Refresh Models", size="sm")

                    rvc_pitch = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        label="Pitch Shift (semitones)",
                        info="0 = no change, +12 = one octave up, -12 = one octave down",
                    )
                    rvc_index_rate = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.75,
                        step=0.05,
                        label="Index Rate",
                        info="Higher = more like training voice, Lower = more like source",
                    )

            rvc_btn = gr.Button(
                "🔄 Convert Voice",
                variant="primary",
                size="lg",
                interactive=rvc_available,
            )

            gr.Markdown("---")
            gr.Markdown("### ✅ Result — Converted to trained voice")
            rvc_output = gr.Audio(
                label="Converted Audio Output",
                type="filepath",
                interactive=False,
            )

            gr.HTML("""<div class="tips-box">
                <strong>📁 How to add trained models:</strong><br>
                1. Place your trained <code>.pth</code> model file in the <code>models/</code> directory<br>
                2. Optionally place the <code>.index</code> file in <code>models/</code> too<br>
                3. Click "Refresh Models" to see it in the dropdown<br>
                4. Train models using: <code>./setup_rvc.sh</code> then follow instructions
            </div>""")

            rvc_btn.click(
                fn=clone_voice_rvc,
                inputs=[rvc_src_audio, rvc_model_dropdown, rvc_pitch, rvc_index_rate],
                outputs=[rvc_output],
            )
            rvc_refresh_btn.click(
                fn=refresh_models,
                outputs=[rvc_model_dropdown],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3: AUDIO PREPARATION
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("🔧 Prepare Audio for Training"):
            gr.Markdown("""
            **Clean and prepare audio** for RVC training.
            Upload your long audio file (e.g., 40 minutes) and this tool will:
            - 🔇 Remove background noise
            - 📊 Normalize volume
            - ✂️ Split into clean training segments
            """)

            prep_audio = gr.Audio(
                label="📂 Upload Long Audio File",
                type="filepath",
                sources=["upload"],
            )

            with gr.Row():
                prep_denoise = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Denoise Strength",
                    info="0 = no denoising, 1.0 = maximum denoising",
                )
                prep_min_seg = gr.Slider(
                    minimum=3.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.5,
                    label="Min Segment (seconds)",
                )
                prep_max_seg = gr.Slider(
                    minimum=10.0,
                    maximum=30.0,
                    value=15.0,
                    step=1.0,
                    label="Max Segment (seconds)",
                )

            prep_btn = gr.Button(
                "🧹 Clean & Split Audio",
                variant="primary",
                size="lg",
            )

            gr.Markdown("---")
            prep_summary = gr.Markdown("### Results will appear here...")
            prep_preview = gr.Audio(
                label="🔊 Preview — Cleaned Audio",
                type="filepath",
                interactive=False,
            )

            gr.HTML("""<div class="tips-box">
                <strong>📋 After preprocessing:</strong><br>
                1. Cleaned segments will be in <code>training_data/</code><br>
                2. Run <code>./setup_rvc.sh</code> to set up RVC training<br>
                3. Train using the RVC WebUI (instructions shown after setup)<br>
                4. Place trained model in <code>models/</code> and use "Trained Voice" tab
            </div>""")

            prep_btn.click(
                fn=prepare_audio,
                inputs=[prep_audio, prep_denoise, prep_min_seg, prep_max_seg],
                outputs=[prep_summary, prep_preview],
            )

    # ── Footer ──
    gr.Markdown("---")
    gr.Markdown(
        "<center><small>"
        "Powered by <strong>OpenVoice v2</strong> + <strong>RVC</strong> | "
        "Runs locally | "
        f"Device: <strong>{DEVICE}</strong>"
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")
        + "</small></center>"
    )


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Cloner Web UI")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run on (default: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link",
    )
    args = parser.parse_args()

    print(f"🚀 Starting Voice Cloner at http://{args.host}:{args.port}")
    if args.host == "127.0.0.1":
        print(f"🌐 Open in browser: http://localhost:{args.port}")
        print(f"💡 SSH access: ssh -L {args.port}:localhost:{args.port} user@machine")

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
