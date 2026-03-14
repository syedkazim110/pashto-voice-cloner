#!/usr/bin/env python3
"""
🧹 Audio Preprocessing Pipeline
=================================
Cleans, denoises, splits, and normalizes audio files for RVC training.

Usage:
    python preprocess.py input_audio.wav --output training_data/
    python preprocess.py input_audio.wav --output training_data/ --denoise-strength 0.7
"""

import os
import argparse
import numpy as np


def clean_and_split_audio(
    input_path: str,
    output_dir: str,
    target_sr: int = 44100,
    denoise_strength: float = 0.5,
    min_segment_sec: float = 5.0,
    max_segment_sec: float = 15.0,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 400,
    normalize: bool = True,
    progress_callback=None,
):
    """
    Clean a long audio file: denoise, normalize, and split into segments.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save cleaned segments
        target_sr: Target sample rate
        denoise_strength: Noise reduction strength (0.0-1.0)
        min_segment_sec: Minimum segment duration in seconds
        max_segment_sec: Maximum segment duration in seconds
        silence_thresh_db: Silence threshold in dB
        min_silence_ms: Minimum silence duration for splitting
        normalize: Whether to normalize audio volume
        progress_callback: Optional callback for progress updates

    Returns:
        List of output file paths
    """
    import librosa
    import soundfile as sf
    import noisereduce as nr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    os.makedirs(output_dir, exist_ok=True)

    if progress_callback:
        progress_callback(0.05, "Loading audio file...")

    # ── Step 1: Load audio ────────────────────────────────────────────────
    print(f"📂 Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    duration = len(audio) / sr
    print(f"   Duration: {duration:.1f}s | Sample rate: {sr}Hz")

    if progress_callback:
        progress_callback(0.15, f"Loaded {duration:.0f}s of audio. Denoising...")

    # ── Step 2: Denoise ───────────────────────────────────────────────────
    if denoise_strength > 0:
        print(f"🔇 Denoising (strength: {denoise_strength})...")

        # Use stationary noise reduction
        # Process in chunks to handle long audio
        chunk_size = sr * 60  # 1 minute chunks
        denoised_chunks = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            denoised = nr.reduce_noise(
                y=chunk,
                sr=sr,
                prop_decrease=denoise_strength,
                stationary=True,
            )
            denoised_chunks.append(denoised)

        audio = np.concatenate(denoised_chunks)
        print("   ✅ Denoising complete")

    if progress_callback:
        progress_callback(0.35, "Normalizing volume...")

    # ── Step 3: Normalize ─────────────────────────────────────────────────
    if normalize:
        print("📊 Normalizing volume...")
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        print("   ✅ Normalization complete")

    if progress_callback:
        progress_callback(0.45, "Saving cleaned audio...")

    # ── Step 4: Save cleaned full file ────────────────────────────────────
    clean_full_path = os.path.join(output_dir, "cleaned_full.wav")
    sf.write(clean_full_path, audio, sr)
    print(f"💾 Saved cleaned full audio: {clean_full_path}")

    if progress_callback:
        progress_callback(0.55, "Splitting into segments...")

    # ── Step 5: Split on silence ──────────────────────────────────────────
    print(f"✂️  Splitting into {min_segment_sec}-{max_segment_sec}s segments...")
    audio_segment = AudioSegment.from_wav(clean_full_path)

    chunks = split_on_silence(
        audio_segment,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=200,  # Keep 200ms of silence at edges
    )

    print(f"   Found {len(chunks)} speech chunks")

    # ── Step 6: Merge small chunks, split large ones ──────────────────────
    segment_paths = []
    current_chunk = AudioSegment.empty()
    segment_idx = 0
    min_ms = min_segment_sec * 1000
    max_ms = max_segment_sec * 1000

    for chunk in chunks:
        # If adding this chunk keeps us under max, add it
        if len(current_chunk) + len(chunk) <= max_ms:
            current_chunk += chunk
        else:
            # Save current chunk if it's long enough
            if len(current_chunk) >= min_ms:
                segment_path = os.path.join(
                    output_dir, f"segment_{segment_idx:04d}.wav"
                )
                current_chunk.export(segment_path, format="wav")
                segment_paths.append(segment_path)
                segment_idx += 1

            # Start new chunk
            # If the individual chunk is too long, split it
            if len(chunk) > max_ms:
                # Split large chunk into max_ms pieces
                for j in range(0, len(chunk), int(max_ms)):
                    sub = chunk[j : j + int(max_ms)]
                    if len(sub) >= min_ms:
                        segment_path = os.path.join(
                            output_dir, f"segment_{segment_idx:04d}.wav"
                        )
                        sub.export(segment_path, format="wav")
                        segment_paths.append(segment_path)
                        segment_idx += 1
                current_chunk = AudioSegment.empty()
            else:
                current_chunk = chunk

    # Save last chunk
    if len(current_chunk) >= min_ms:
        segment_path = os.path.join(output_dir, f"segment_{segment_idx:04d}.wav")
        current_chunk.export(segment_path, format="wav")
        segment_paths.append(segment_path)

    if progress_callback:
        progress_callback(0.9, "Finalizing...")

    # ── Summary ───────────────────────────────────────────────────────────
    total_duration = sum(
        len(AudioSegment.from_wav(p)) for p in segment_paths
    ) / 1000

    print(f"\n📊 Preprocessing Summary:")
    print(f"   Input: {duration:.1f}s")
    print(f"   Output: {len(segment_paths)} segments ({total_duration:.1f}s total)")
    print(f"   Saved to: {output_dir}/")

    if progress_callback:
        progress_callback(1.0, "Done!")

    return segment_paths, {
        "input_duration": duration,
        "output_segments": len(segment_paths),
        "output_duration": total_duration,
        "output_dir": output_dir,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and split audio for RVC training"
    )
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument(
        "--output",
        "-o",
        default="training_data",
        help="Output directory (default: training_data/)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Target sample rate (default: 44100)",
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=0.5,
        help="Denoise strength 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=5.0,
        help="Min segment duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-segment",
        type=float,
        default=15.0,
        help="Max segment duration in seconds (default: 15.0)",
    )

    args = parser.parse_args()

    clean_and_split_audio(
        input_path=args.input,
        output_dir=args.output,
        target_sr=args.sr,
        denoise_strength=args.denoise,
        min_segment_sec=args.min_segment,
        max_segment_sec=args.max_segment,
    )
