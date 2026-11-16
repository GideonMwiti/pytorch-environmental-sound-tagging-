#!/usr/bin/env python3
"""
main.py - Environmental Sound Tagging (Multi-Label Audio Classification)

Project 93: PyTorch 100 Projects Challenge
Detects multiple simultaneous, overlapping environmental sound events:
[dog_bark, car_passing, people_talking, siren, bird_singing,
 rain_shower, footsteps, wind_rustle]
using Log-Mel Spectrograms, Deep Residual SE-CNN, and Sigmoid-based BCE loss.
"""

import os
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio.transforms as T
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    hamming_loss,
    accuracy_score
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

SOUND_TAGS = [
    "dog_bark",
    "car_passing",
    "people_talking",
    "siren",
    "bird_singing",
    "rain_shower",
    "footsteps",
    "wind_rustle"
]
TAG_TO_IDX = {tag: i for i, tag in enumerate(SOUND_TAGS)}
NUM_TAGS = len(SOUND_TAGS)
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sound_tagging_resnet_best.pth")


class MultiLabelAudioSynthesizer:
    """
    Synthesizes individual environmental acoustic events and mixes them
    polyphonically to generate realistic multi-label soundscapes.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sr = sample_rate
        self.duration = duration
        self.n = int(self.sr * self.duration)
        self.t = torch.linspace(0, self.duration, self.n)

    def synthesize_single_source(self, tag: str, seed: int = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t = self.t
        audio = torch.zeros(self.n)

        if tag == "dog_bark":
            for bp in [0.4, 1.2, 2.1]:
                idx = int(bp * self.sr)
                blen = int(0.25 * self.sr)
                bt = torch.linspace(0, 0.25, blen)
                f0 = 460.0 * torch.exp(-bt * 6.0)
                bark = (torch.sin(2 * np.pi * f0 * bt) +
                        0.4 * torch.sin(4 * np.pi * f0 * bt) +
                        torch.randn(blen) * 0.3) * torch.exp(-bt * 14.0) * 0.8
                audio[idx : idx + blen] += bark

        elif tag == "car_passing":
            # Doppler frequency shift (rising then falling engine drone) + tire hiss
            doppler = 1.0 - 0.35 * torch.tanh(2.5 * (t - 1.5))
            f_eng = (70.0 + 15.0 * torch.sin(2 * np.pi * 0.5 * t)) * doppler
            engine = (torch.sin(2 * np.pi * f_eng * t) + 0.4 * torch.sin(4 * np.pi * f_eng * t)) * 0.55
            hiss = (torch.randn(self.n) * 0.22) * (0.6 + 0.4 * torch.sin(2 * np.pi * 1.0 * t))
            env = torch.exp(-((t - 1.5) ** 2) / 1.2)
            audio = (engine + hiss) * env

        elif tag == "people_talking":
            # Overlapping conversational formants
            f1 = 280.0 + 60.0 * torch.sin(2 * np.pi * 3.0 * t)
            f2 = 520.0 + 80.0 * torch.sin(2 * np.pi * 2.2 * t)
            speech = (torch.sin(2 * np.pi * f1 * t) * 0.35 +
                      torch.sin(2 * np.pi * f2 * t) * 0.30 +
                      torch.randn(self.n) * 0.12) * (0.6 + 0.4 * torch.sin(2 * np.pi * 1.5 * t))
            audio = speech * 0.70

        elif tag == "siren":
            # Wailing siren between 700 Hz and 1150 Hz
            f_siren = 920.0 + 230.0 * torch.sin(2 * np.pi * 1.2 * t)
            phase = 2 * np.pi * torch.cumsum(f_siren, dim=0) / self.sr
            siren_tone = (torch.sin(phase) + 0.25 * torch.sin(2 * phase)) * 0.65
            audio = siren_tone

        elif tag == "bird_singing":
            for b_pos in [0.3, 1.0, 1.8, 2.4]:
                b_idx = int(b_pos * self.sr)
                b_dur = 0.22
                bt = torch.linspace(0, 0.22, int(self.sr * 0.22))
                f_chirp = 3900.0 + 1400.0 * torch.sin(2 * np.pi * 18.0 * bt)
                chirp = torch.sin(2 * np.pi * f_chirp * bt) * torch.exp(-bt * 11.0) * 0.65
                audio[b_idx : b_idx + len(chirp)] += chirp

        elif tag == "rain_shower":
            # Granular wideband noise with gentle intensity modulation
            rain_noise = torch.randn(self.n) * 0.28
            swell = 0.85 + 0.15 * torch.sin(2 * np.pi * 0.4 * t)
            audio = rain_noise * swell

        elif tag == "footsteps":
            # Rhythmic heel-toe impacts at 1.8 Hz
            step_rate = 1.8
            n_steps = int(self.duration * step_rate)
            for s in range(n_steps):
                s_idx = int((s / step_rate + 0.2) * self.sr)
                if s_idx + int(0.06 * self.sr) < self.n:
                    st = torch.linspace(0, 0.06, int(self.sr * 0.06))
                    impact = (torch.sin(2 * np.pi * 90.0 * st) + torch.randn(len(st)) * 0.3) * torch.exp(-st * 45.0) * 0.75
                    audio[s_idx : s_idx + len(impact)] += impact

        elif tag == "wind_rustle":
            # Low-frequency atmospheric rustle
            wind = (torch.randn(self.n) * 0.20) * (0.7 + 0.3 * torch.sin(2 * np.pi * 0.25 * t))
            audio = wind

        return audio

    def mix_polyphonic_soundscape(self, active_tags: list, seed: int = None) -> tuple:
        """
        Mixes multiple active acoustic sources into a single composite soundscape.
        Returns (audio_tensor, multi_hot_tensor).
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        mixed_audio = torch.zeros(self.n)
        multi_hot = torch.zeros(NUM_TAGS, dtype=torch.float32)

        for tag in active_tags:
            idx = TAG_TO_IDX[tag]
            multi_hot[idx] = 1.0
            source_audio = self.synthesize_single_source(tag, seed=seed)
            gain = float(np.random.uniform(0.65, 1.0))
            mixed_audio += source_audio * gain

        # Add gentle ambient floor
        mixed_audio += torch.randn(self.n) * 0.025

        # Normalize
        peak = torch.max(torch.abs(mixed_audio))
        if peak > 0.89:
            mixed_audio = mixed_audio / peak * 0.89

        return mixed_audio.clamp(-1.0, 1.0), multi_hot


class LogMelSpectrogramExtractor:
    """Computes normalized Log-Mel spectrograms [1, 128, T]."""
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.mel_tf = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate // 2
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        mel = torch.log(self.mel_tf(audio) + 1e-6)
        mean = mel.mean(dim=[-2, -1], keepdim=True)
        std = mel.std(dim=[-2, -1], keepdim=True) + 1e-6
        return (mel - mean) / std


class SpecAugment(nn.Module):
    """Multi-label SpecAugment: frequency and time masking."""
    def __init__(self, max_freq_mask=16, max_time_mask=24):
        super().__init__()
        self.max_f = max_freq_mask
        self.max_t = max_time_mask

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        out = spec.clone()
        b, c, f, t = out.shape
        f_len = np.random.randint(0, self.max_f)
        f_pos = np.random.randint(0, max(1, f - f_len))
        out[:, :, f_pos : f_pos + f_len, :] = 0.0
        t_len = np.random.randint(0, self.max_t)
        t_pos = np.random.randint(0, max(1, t - t_len))
        out[:, :, :, t_pos : t_pos + t_len] = 0.0
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(4, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SqueezeExcitation(out_ch)
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class EnvironmentalTaggingResNet(nn.Module):
    """
    Deep Residual SE-CNN with Dual Pooling for Multi-Label Sound Tagging.
    Outputs raw logits for independent binary cross-entropy sigmoid activation.
    """
    def __init__(self, num_tags: int = NUM_TAGS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ResidualSEBlock(32, 64, stride=(2, 2))   # 128x259 -> 64x130
        self.stage2 = ResidualSEBlock(64, 128, stride=(2, 2))  # 64x130 -> 32x65
        self.stage3 = ResidualSEBlock(128, 256, stride=(2, 2)) # 32x65 -> 16x33

        # Dual Global Pooling across Frequency & Time
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_tags)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        avg_f = self.avg_pool(x).flatten(1)
        max_f = self.max_pool(x).flatten(1)
        feat = torch.cat([avg_f, max_f], dim=1)

        logits = self.classifier(feat)
        return logits


class SoundTaggingDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_dataset(synthesizer: MultiLabelAudioSynthesizer, extractor: LogMelSpectrogramExtractor, n_train=320, n_val=80):
    """
    Generates polyphonic multi-label soundscapes with co-occurring acoustic events.
    """
    train_samples = []
    val_samples = []

    # Pre-generate 6 canonical multi-sound reference WAVs
    canonical_configs = [
        {"name": "dog_bark_car_passing.wav", "tags": ["dog_bark", "car_passing"]},
        {"name": "people_talking_siren.wav", "tags": ["people_talking", "siren"]},
        {"name": "bird_singing_wind_rustle.wav", "tags": ["bird_singing", "wind_rustle"]},
        {"name": "rain_shower_footsteps.wav", "tags": ["rain_shower", "footsteps"]},
        {"name": "dog_bark_people_talking_car.wav", "tags": ["dog_bark", "car_passing", "people_talking"]},
        {"name": "siren_rain_wind.wav", "tags": ["siren", "rain_shower", "wind_rustle"]}
    ]
    canonical_audios = {}
    for c in canonical_configs:
        audio, multi_hot = synthesizer.mix_polyphonic_soundscape(c["tags"], seed=42)
        canonical_audios[c["name"]] = (audio, multi_hot, c["tags"])
        sf.write(os.path.join(OUTPUTS_DIR, c["name"]), audio.numpy(), SAMPLE_RATE)

    # Generate Train Dataset
    for i in range(n_train):
        # Pick 1 to 4 random co-occurring tags
        k = np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
        tags = list(np.random.choice(SOUND_TAGS, size=k, replace=False))
        audio, multi_hot = synthesizer.mix_polyphonic_soundscape(tags, seed=1000 + i)
        spec = extractor(audio).squeeze(0)  # [128, T]
        train_samples.append((spec, multi_hot))

    # Generate Val Dataset
    for i in range(n_val):
        k = np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
        tags = list(np.random.choice(SOUND_TAGS, size=k, replace=False))
        audio, multi_hot = synthesizer.mix_polyphonic_soundscape(tags, seed=5000 + i)
        spec = extractor(audio).squeeze(0)
        val_samples.append((spec, multi_hot))

    return train_samples, val_samples, canonical_audios


def train_model(model, train_loader, val_loader, epochs=10):
    print(f"\n[Training] Training EnvironmentalTaggingResNet for {epochs} epochs with BCEWithLogitsLoss...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Balanced BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0] * NUM_TAGS))
    augment = SpecAugment()

    best_f1 = 0.0
    history = {"train_loss": [], "val_f1": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for specs, labels in train_loader:
            specs = augment(specs.unsqueeze(1))
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)

        ep_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(ep_loss)
        scheduler.step()

        # Validation evaluation
        model.eval()
        val_targets, val_preds = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                logits = model(specs.unsqueeze(1))
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.50).float()
                val_targets.extend(labels.numpy())
                val_preds.extend(preds.numpy())

        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        _, _, macro_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
        val_f1_pct = float(macro_f1) * 100.0
        history["val_f1"].append(val_f1_pct)

        if val_f1_pct > best_f1:
            best_f1 = val_f1_pct
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)

        if ep % 2 == 0 or ep == epochs:
            print(f"  Epoch {ep:02d}/{epochs:02d} | Loss: {ep_loss:.4f} | Val Macro F1: {val_f1_pct:.1f}% (Best: {best_f1:.1f}%)")

    print(f"[Training Complete] Checkpoint saved: {MODEL_CHECKPOINT_PATH} (Best Macro F1: {best_f1:.1f}%)")
    return history


def evaluate_and_benchmark(model, val_loader, extractor, canonical_audios):
    print("\n[Evaluation] Calculating multi-label classification metrics & latency benchmark...")
    model.eval()
    all_targets, all_probs = [], []

    with torch.no_grad():
        for specs, labels in val_loader:
            logits = model(specs.unsqueeze(1))
            probs = torch.sigmoid(logits)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.50).astype(np.float32)

    # Multi-label evaluation metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

    # Mean Average Precision (mAP)
    map_score = float(average_precision_score(y_true, y_prob, average="macro")) * 100.0
    h_loss = float(hamming_loss(y_true, y_pred))
    exact_match = float(accuracy_score(y_true, y_pred)) * 100.0

    # Per-class breakdown
    p_c, r_c, f_c, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    for idx, tag in enumerate(SOUND_TAGS):
        ap = float(average_precision_score(y_true[:, idx], y_prob[:, idx])) * 100.0
        per_class[tag] = {
            "precision": round(float(p_c[idx]) * 100.0, 2),
            "recall": round(float(r_c[idx]) * 100.0, 2),
            "f1_score": round(float(f_c[idx]) * 100.0, 2),
            "average_precision": round(ap, 2)
        }

    # Latency Benchmark (50 trials)
    sample_wav = canonical_audios["dog_bark_car_passing.wav"][0]
    sample_spec = extractor(sample_wav).unsqueeze(0)
    for _ in range(5):
        with torch.no_grad():
            _ = model(sample_spec)

    trials = 50
    t0 = time.perf_counter()
    for _ in range(trials):
        with torch.no_grad():
            _ = model(sample_spec)
    t1 = time.perf_counter()

    avg_lat_s = (t1 - t0) / trials
    avg_lat_ms = avg_lat_s * 1000.0
    rtf = avg_lat_s / DURATION
    speedup = 1.0 / rtf if rtf > 0 else 0
    headroom = max(0.0, (1.0 - rtf) * 100.0)

    metrics = {
        "project": "Project 93: Environmental Sound Tagging (Multi-Label)",
        "taxonomy": SOUND_TAGS,
        "classification": {
            "mean_average_precision_map": round(map_score, 2),
            "macro_f1_pct": round(f1_macro * 100.0, 2),
            "micro_f1_pct": round(f1_micro * 100.0, 2),
            "macro_precision_pct": round(prec_macro * 100.0, 2),
            "macro_recall_pct": round(rec_macro * 100.0, 2),
            "exact_match_ratio_pct": round(exact_match, 2),
            "hamming_loss": round(h_loss, 4),
            "per_class": per_class
        },
        "latency_benchmark": {
            "mean_latency_ms": round(avg_lat_ms, 2),
            "real_time_factor_rtf": round(rtf, 4),
            "speedup_factor": round(speedup, 1),
            "cpu_headroom_pct": round(headroom, 1)
        }
    }

    metrics_path = os.path.join(OUTPUTS_DIR, "multi_label_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, y_true, y_prob


def generate_diagnostic_report(canonical_audios, metrics, extractor, model, y_true):
    """Renders 6-panel visual diagnostic analytics dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)

    # Panel 1: Polyphonic Mixed Waveforms
    ax1 = fig.add_subplot(gs[0, 0])
    t_axis = np.linspace(0, DURATION, NUM_SAMPLES)
    sample_keys = [
        "dog_bark_car_passing.wav",
        "people_talking_siren.wav",
        "bird_singing_wind_rustle.wav",
        "rain_shower_footsteps.wav"
    ]
    colors = ["#3182CE", "#E53E3E", "#38A169", "#805AD5"]
    for i, k in enumerate(sample_keys):
        wav = canonical_audios[k][0].numpy()
        offset = (len(sample_keys) - 1 - i) * 1.5
        ax1.plot(t_axis, wav * 0.65 + offset, color=colors[i], alpha=0.85)
        clean_title = " + ".join([t.replace("_", " ").title() for t in canonical_audios[k][2]])
        ax1.text(0.05, offset + 0.55, clean_title, fontsize=8, fontweight="bold", color="#2D3748")
    ax1.set_yticks([])
    ax1.set_xlabel("Time (seconds)")
    ax1.set_title("Panel 1: Polyphonic Environmental Soundscapes", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: Log-Mel Spectrogram of Polyphonic Clip
    ax2 = fig.add_subplot(gs[0, 1])
    test_key = "dog_bark_people_talking_car.wav"
    test_wav = canonical_audios[test_key][0]
    spec = extractor(test_wav).squeeze(0).numpy()
    im2 = ax2.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
    ax2.set_xlabel("Time Frames (11.6ms)")
    ax2.set_ylabel("Mel Frequency Bins (128)")
    ax2.set_title(f"Panel 2: Spectrogram: {canonical_audios[test_key][2]}", fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=ax2, pad=0.02, label="Normalized Log-Mel")

    # Panel 3: Label Co-occurrence Correlation Matrix Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    co_matrix = np.dot(y_true.T, y_true).astype(np.float32)
    diag = np.diag(co_matrix)
    norm_co = co_matrix / np.sqrt(np.outer(diag, diag) + 1e-6)
    im3 = ax3.imshow(norm_co, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = [t.replace("_", "\n").title() for t in SOUND_TAGS]
    ax3.set_xticks(range(NUM_TAGS))
    ax3.set_yticks(range(NUM_TAGS))
    ax3.set_xticklabels(ticks, fontsize=7)
    ax3.set_yticklabels(ticks, fontsize=7)
    ax3.set_title("Panel 3: Tag Co-occurrence Correlation Heatmap", fontsize=11, fontweight="bold")
    fig.colorbar(im3, ax=ax3, pad=0.02)

    # Panel 4: Per-Class F1 and Average Precision
    ax4 = fig.add_subplot(gs[1, 0])
    per_class = metrics["classification"]["per_class"]
    x_pos = np.arange(NUM_TAGS)
    f1_vals = [per_class[t]["f1_score"] for t in SOUND_TAGS]
    ap_vals = [per_class[t]["average_precision"] for t in SOUND_TAGS]
    w = 0.35
    b1 = ax4.bar(x_pos - w/2, f1_vals, w, label="F1-Score (%)", color="#3182CE")
    b2 = ax4.bar(x_pos + w/2, ap_vals, w, label="Average Precision (%)", color="#48BB78")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([t.replace("_", "\n").title() for t in SOUND_TAGS], fontsize=7)
    ax4.set_ylim(0, 115)
    ax4.set_ylabel("Metric Score (%)")
    ax4.set_title("Panel 4: Per-Tag Multi-Label Performance", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Panel 5: Multi-Tag Sigmoid Probabilities vs Ground Truth
    ax5 = fig.add_subplot(gs[1, 1])
    spec_in = extractor(test_wav).unsqueeze(0)
    with torch.no_grad():
        logits = model(spec_in)
        probs = torch.sigmoid(logits).squeeze(0).numpy() * 100.0

    gt_tags = canonical_audios[test_key][2]
    colors_bar = ["#E53E3E" if t in gt_tags else "#718096" for t in SOUND_TAGS]
    y_pos = np.arange(NUM_TAGS)
    bars = ax5.barh(y_pos, probs, color=colors_bar)
    ax5.axvline(50.0, color="#E53E3E", linestyle="--", linewidth=1.5, label="Decision Threshold (50%)")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([t.replace("_", " ").title() for t in SOUND_TAGS], fontsize=8, fontweight="bold")
    ax5.set_xlim(0, 115)
    ax5.set_xlabel("Sigmoid Posterior Probability (%)")
    ax5.set_title("Panel 5: Sigmoid Multi-Tag Probabilities (Red = Ground Truth)", fontsize=11, fontweight="bold")
    ax5.legend(loc="lower right", fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.3, axis="x")

    # Panel 6: Radar Profile
    ax6 = fig.add_subplot(gs[1, 2], polar=True)
    categories = ["mAP\n(%)", "Macro F1\n(%)", "Micro F1\n(%)", "Speedup\n(x10)", "Exact Match\n(%)"]
    map_v = metrics["classification"]["mean_average_precision_map"]
    f1_m = metrics["classification"]["macro_f1_pct"]
    f1_u = metrics["classification"]["micro_f1_pct"]
    spd_v = min(100.0, metrics["latency_benchmark"]["speedup_factor"] * 0.5)
    ex_v = metrics["classification"]["exact_match_ratio_pct"]

    vals = [map_v, f1_m, f1_u, spd_v, ex_v]
    vals += vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax6.plot(angles, vals, color="#805AD5", linewidth=2)
    ax6.fill(angles, vals, color="#805AD5", alpha=0.25)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=8, fontweight="bold")
    ax6.set_ylim(0, 105)
    ax6.set_title("Panel 6: Multi-Label System Radar", fontsize=11, fontweight="bold", pad=15)

    report_path = os.path.join(OUTPUTS_DIR, "multi_label_tagging_report.png")
    plt.savefig(report_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Report] Diagnostic report saved to: {report_path}")


def main():
    print("=" * 70)
    print(" PROJECT 93: ENVIRONMENTAL SOUND TAGGING (MULTI-LABEL CLASSIFICATION)")
    print("=" * 70)

    synthesizer = MultiLabelAudioSynthesizer()
    extractor = LogMelSpectrogramExtractor()

    print("\n[Stage 1] Synthesizing Multi-Label Polyphonic Soundscapes...")
    train_data, val_data, canonical_audios = build_dataset(synthesizer, extractor, n_train=320, n_val=80)
    print(f"  Dataset: {len(train_data)} Train samples, {len(val_data)} Validation samples")

    train_loader = DataLoader(SoundTaggingDataset(train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(SoundTaggingDataset(val_data), batch_size=32, shuffle=False)

    print("\n[Stage 2] Building Deep Residual SE-CNN with Dual Pooling...")
    model = EnvironmentalTaggingResNet(num_tags=NUM_TAGS)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {param_count:,}")

    history = train_model(model, train_loader, val_loader, epochs=10)

    # Load best checkpoint
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))

    print("\n[Stage 3] Evaluating Metrics & Generating Benchmark Dashboard...")
    metrics, y_true, y_prob = evaluate_and_benchmark(model, val_loader, extractor, canonical_audios)
    generate_diagnostic_report(canonical_audios, metrics, extractor, model, y_true)

    print("\n" + "=" * 70)
    print(" MULTI-LABEL SOUND TAGGING BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Mean Average Precision (mAP) : {metrics['classification']['mean_average_precision_map']}%")
    print(f"  Macro F1-Score               : {metrics['classification']['macro_f1_pct']}%")
    print(f"  Micro F1-Score               : {metrics['classification']['micro_f1_pct']}%")
    print(f"  Exact Match Ratio (Subset)   : {metrics['classification']['exact_match_ratio_pct']}%")
    print(f"  Hamming Loss                 : {metrics['classification']['hamming_loss']}")
    print(f"  Mean Latency / Clip          : {metrics['latency_benchmark']['mean_latency_ms']} ms")
    print(f"  Real-Time Factor (RTF)       : {metrics['latency_benchmark']['real_time_factor_rtf']}x")
    print(f"  Processing Speedup           : {metrics['latency_benchmark']['speedup_factor']}x")
    print(f"  CPU Idle Headroom            : {metrics['latency_benchmark']['cpu_headroom_pct']}%")
    print("=" * 70)
    print(" PROJECT 93 PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
