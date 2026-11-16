#!/usr/bin/env python3
"""
tag_audio.py - Standalone CLI for Multi-Label Environmental Sound Tagging

Usage examples:
  python tag_audio.py --audio outputs/dog_bark_car_passing.wav
  python tag_audio.py --audio outputs/dog_bark_people_talking_car.wav --threshold 0.45 --benchmark
  python tag_audio.py --audio outputs/people_talking_siren.wav --output-json outputs/tags.json
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import soundfile as sf
import torchaudio.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    EnvironmentalTaggingResNet,
    LogMelSpectrogramExtractor,
    SOUND_TAGS,
    SAMPLE_RATE,
    DURATION,
    NUM_TAGS,
    MODEL_CHECKPOINT_PATH
)


def load_and_preprocess(audio_path: str, target_sr: int = SAMPLE_RATE, target_duration: float = DURATION):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    waveform = torch.from_numpy(data)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    req_len = int(target_sr * target_duration)
    if waveform.shape[0] < req_len:
        waveform = torch.nn.functional.pad(waveform, (0, req_len - waveform.shape[0]))
    elif waveform.shape[0] > req_len:
        waveform = waveform[:req_len]

    peak = torch.max(torch.abs(waveform))
    if peak > 1e-4:
        waveform = waveform / peak * 0.90

    return waveform


def tag_audio_clip(
    audio_path: str,
    checkpoint_path: str = MODEL_CHECKPOINT_PATH,
    threshold: float = 0.50,
    benchmark: bool = False,
    output_json: str = None
):
    print("=" * 72)
    print(" ENVIRONMENTAL SOUND TAGGER - MULTI-LABEL RESNET-SE INFERENCE")
    print("=" * 72)
    print(f"Target Audio   : {audio_path}")
    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"Decision Thresh: {threshold * 100.0:.1f}%")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Run main.py first.")

    # 1. Load Audio
    waveform = load_and_preprocess(audio_path)
    audio_dur = len(waveform) / SAMPLE_RATE
    print(f"Loaded Clip    : {audio_dur:.2f}s ({len(waveform)} samples @ {SAMPLE_RATE} Hz)")

    # 2. Extract Log-Mel Spectrogram
    extractor = LogMelSpectrogramExtractor()
    with torch.no_grad():
        spec = extractor(waveform).unsqueeze(0)  # [1, 1, 128, T]
    print(f"Spectrogram Dim: {list(spec.shape)}")

    # 3. Model Inference
    model = EnvironmentalTaggingResNet(num_tags=NUM_TAGS)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(spec)
        probs = torch.sigmoid(logits).squeeze(0).numpy()

    # Sort tags by descending probability
    sorted_indices = np.argsort(probs)[::-1]
    active_tags = []

    print("\n" + "-" * 72)
    print(f"{'Rank':<6}{'Sound Event Tag':<22}{'Probability':<16}{'Status':<12}{'Visual Probability Bar'}")
    print("-" * 72)

    all_tag_results = []
    for rank, idx in enumerate(sorted_indices, start=1):
        tag_name = SOUND_TAGS[idx]
        p_pct = float(probs[idx]) * 100.0
        is_active = float(probs[idx]) >= threshold
        status_str = "[ACTIVE]" if is_active else "inactive"
        if is_active:
            active_tags.append(tag_name)

        bar_len = int(p_pct / 4.0)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f" #{rank:<5}{tag_name.replace('_', ' ').title():<22}{p_pct:>6.2f}%         {status_str:<12}{bar}")

        all_tag_results.append({
            "tag": tag_name,
            "probability_pct": round(p_pct, 2),
            "is_active": is_active
        })

    print("-" * 72)
    clean_active = [t.replace("_", " ").title() for t in active_tags]
    if len(active_tags) > 0:
        print(f" DETECTED SOUND TAGS ({len(active_tags)}): [ {', '.join(clean_active)} ]")
    else:
        print(" NO ACTIVE SOUND TAGS DETECTED (All below decision threshold)")
    print("-" * 72)

    # 4. Optional Benchmark
    bench_data = None
    if benchmark:
        print("\n[Benchmarking] Running 50-trial multi-label inference latency benchmark...")
        trials = 50
        for _ in range(5):
            with torch.no_grad():
                _ = model(spec)

        t0 = time.perf_counter()
        for _ in range(trials):
            with torch.no_grad():
                _ = model(spec)
        t1 = time.perf_counter()

        avg_lat_s = (t1 - t0) / trials
        avg_lat_ms = avg_lat_s * 1000.0
        rtf = avg_lat_s / audio_dur
        speedup = 1.0 / rtf if rtf > 0 else 0
        headroom = max(0.0, (1.0 - rtf) * 100.0)

        print(f"  -> Mean Inference Latency : {avg_lat_ms:.2f} ms")
        print(f"  -> Real-Time Factor (RTF) : {rtf:.4f}x")
        print(f"  -> Processing Speedup     : {speedup:.1f}x Faster than Real Time")
        print(f"  -> CPU Processing Headroom: {headroom:.1f}%")

        bench_data = {
            "mean_latency_ms": round(avg_lat_ms, 2),
            "real_time_factor": round(rtf, 4),
            "speedup_factor": round(speedup, 1),
            "cpu_headroom_pct": round(headroom, 1)
        }

    payload = {
        "audio_path": audio_path,
        "clip_duration_sec": round(audio_dur, 2),
        "decision_threshold_pct": round(threshold * 100.0, 1),
        "detected_active_tags": active_tags,
        "tag_probabilities": all_tag_results,
        "benchmark": bench_data
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Saved] Output saved to: {output_json}")

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Label Environmental Sound Tagging CLI."
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--checkpoint", type=str, default=MODEL_CHECKPOINT_PATH, help="Path to model weights (.pth)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Sigmoid probability threshold (default: 0.50)")
    parser.add_argument("--benchmark", action="store_true", help="Run 50-trial latency & RTF benchmark")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to output JSON file")

    args = parser.parse_args()

    tag_audio_clip(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        benchmark=args.benchmark,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()

