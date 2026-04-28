#!/usr/bin/env python3
"""Comprehensive upgraded pipeline for VE216 Topic 2.

Upgrades over the first version:
1. speaker-independent split using Speech Commands speaker hashes;
2. a clear classical baseline and a classical delta-MFCC improvement;
3. an AI MLP feature classifier and a CUDA-enabled CNN on log-mel spectrograms;
4. accuracy / macro-F1 / primary-consonant-F1 metrics;
5. confusion matrices, ablation bars, per-class F1, and noise robustness curves.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import get_window, stft
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

RNG = 216
WORDS = ["down", "go", "left", "no", "right", "yes", "stop", "up"]
WORD_TO_CONSONANT = {
    "down": "d",
    "go": "g",
    "left": "l",
    "no": "n",
    "right": "r",
    "yes": "j",
    "stop": "other",
    "up": "other",
}
ALL_CLASSES = ["d", "g", "l", "n", "r", "j", "other"]
TARGETS_PRIMARY = ["d", "g", "l", "n"]
CLASS_TO_ID = {c: i for i, c in enumerate(ALL_CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}


@dataclass(frozen=True)
class AudioItem:
    path: Path
    word: str
    consonant: str
    speaker: str


def seed_all(seed: int = RNG) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / (right - center)
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    return fb * enorm[:, None]


def load_wav(path: Path, target_sr: int = 16000) -> Tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    n = target_sr
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    else:
        x = x[:n]
    return sr, x


def augment_wave(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    shift = int(rng.integers(-1600, 1601))
    y = np.roll(y, shift)
    if shift > 0:
        y[:shift] = 0
    elif shift < 0:
        y[shift:] = 0
    y = y * float(rng.uniform(0.75, 1.25))
    if rng.random() < 0.7:
        snr_db = float(rng.uniform(10, 30))
        sig_pow = float(np.mean(y**2) + 1e-12)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        y = y + rng.normal(0, math.sqrt(noise_pow), size=len(y)).astype(np.float32)
    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx
    return y.astype(np.float32)


def add_noise_at_snr(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_pow = float(np.mean(x**2) + 1e-12)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    y = x + rng.normal(0, math.sqrt(noise_pow), size=len(x)).astype(np.float32)
    mx = np.max(np.abs(y))
    return (y / mx).astype(np.float32) if mx > 0 else y.astype(np.float32)


def frame_signal(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> Tuple[np.ndarray, int, int]:
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    n_frames = 1 + (len(x) - frame_len) // hop
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, frame_len), strides=strides).copy()
    frames *= get_window("hamming", frame_len, fftbins=True)
    return frames, frame_len, hop


def compute_representations(x: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    pre = np.append(x[0], x[1:] - 0.97 * x[:-1]).astype(np.float32)
    frames, _, _ = frame_signal(pre, sr)
    n_fft = 512
    mag = np.abs(np.fft.rfft(frames, n=n_fft)) + 1e-12
    power = (mag**2) / n_fft
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = mel_filterbank(sr, n_fft, n_mels=40)
    mel_energy = np.maximum(power @ fb.T, 1e-12)
    log_mel = np.log(mel_energy).astype(np.float32)  # frames x mels
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13].astype(np.float32)
    delta = np.gradient(mfcc, axis=0).astype(np.float32)
    delta2 = np.gradient(delta, axis=0).astype(np.float32)

    total = np.maximum(np.sum(mag, axis=1), 1e-12)
    centroid = (np.sum(mag * freqs, axis=1) / total)[:, None]
    bandwidth = np.sqrt(np.sum(mag * (freqs - centroid) ** 2, axis=1) / total)[:, None]
    cumsum = np.cumsum(mag, axis=1)
    rolloff_idx = np.argmax(cumsum >= 0.85 * cumsum[:, [-1]], axis=1)
    rolloff = freqs[rolloff_idx][:, None]
    rms = np.sqrt(np.mean(frames**2, axis=1))[:, None]
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)[:, None]
    spectral = np.hstack([centroid, bandwidth, rolloff, rms, zcr]).astype(np.float32)

    def stats(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a.mean(axis=0), a.std(axis=0)]).astype(np.float32)

    base = stats(mfcc)
    improved = np.concatenate([stats(mfcc), stats(delta), stats(delta2), stats(spectral)]).astype(np.float32)
    mlp = np.concatenate([improved, mfcc.reshape(-1), delta.reshape(-1), delta2.reshape(-1)]).astype(np.float32)
    return {"base": base, "improved": improved, "mlp": mlp, "logmel": log_mel}


def collect_items(data_root: Path, max_per_word: int | None = None) -> List[AudioItem]:
    items: List[AudioItem] = []
    rng = random.Random(RNG)
    for word in WORDS:
        files = sorted((data_root / word).glob("*.wav"))
        rng.shuffle(files)
        if max_per_word:
            files = files[:max_per_word]
        for path in files:
            speaker = path.name.split("_nohash_")[0]
            items.append(AudioItem(path, word, WORD_TO_CONSONANT[word], speaker))
    rng.shuffle(items)
    return items


def build_cache(items: List[AudioItem], cache_path: Path, augment_train: bool = False) -> Dict[str, np.ndarray]:
    if cache_path.exists():
        return dict(np.load(cache_path, allow_pickle=True))
    X_base, X_improved, X_mlp, X_cnn = [], [], [], []
    y, words, speakers, paths = [], [], [], []
    examples = {}
    for item in items:
        sr, x = load_wav(item.path)
        rep = compute_representations(x, sr)
        X_base.append(rep["base"])
        X_improved.append(rep["improved"])
        X_mlp.append(rep["mlp"])
        X_cnn.append(rep["logmel"])
        y.append(CLASS_TO_ID[item.consonant])
        words.append(item.word)
        speakers.append(item.speaker)
        paths.append(str(item.path))
        if item.consonant not in examples:
            f, t, Z = stft(x, fs=sr, window="hann", nperseg=400, noverlap=240, nfft=512, boundary=None)
            examples[item.consonant] = {
                "word": item.word,
                "path": str(item.path),
                "sr": sr,
                "x": x,
                "stft_f": f,
                "stft_t": t,
                "stft_db": 20 * np.log10(np.abs(Z) + 1e-6),
                "freqs": np.fft.rfftfreq(512, d=1.0 / sr),
                "mean_log_spectrum": np.log10(np.abs(np.fft.rfft(frame_signal(np.append(x[0], x[1:] - 0.97 * x[:-1]), sr)[0], n=512)).mean(axis=0) + 1e-12),
            }
    data = {
        "X_base": np.vstack(X_base),
        "X_improved": np.vstack(X_improved),
        "X_mlp": np.vstack(X_mlp),
        "X_cnn": np.stack(X_cnn),
        "y": np.asarray(y, dtype=np.int64),
        "words": np.asarray(words),
        "speakers": np.asarray(speakers),
        "paths": np.asarray(paths),
        "examples": examples,
    }
    np.savez_compressed(cache_path, **data)
    return data


def split_by_speaker(y: np.ndarray, speakers: np.ndarray, test_size: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RNG)
    idx = np.arange(len(y))
    train_idx, test_idx = next(gss.split(idx, y, groups=speakers))
    return train_idx, test_idx


def nearest_centroid(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    centroids = {c: Xtr[y_train == c].mean(axis=0) for c in sorted(set(y_train))}
    preds = []
    for x in Xte:
        sims = {c: float(np.dot(x, mu) / ((np.linalg.norm(x) + 1e-12) * (np.linalg.norm(mu) + 1e-12))) for c, mu in centroids.items()}
        preds.append(max(sims, key=sims.get))
    return np.asarray(preds, dtype=np.int64), scaler, centroids


def nearest_centroid_predict_with_model(X, scaler, centroids):
    Xte = scaler.transform(X)
    preds = []
    for x in Xte:
        sims = {c: float(np.dot(x, mu) / ((np.linalg.norm(x) + 1e-12) * (np.linalg.norm(mu) + 1e-12))) for c, mu in centroids.items()}
        preds.append(max(sims, key=sims.get))
    return np.asarray(preds, dtype=np.int64)


class SmallKwsCnn(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_cnn(X_train, y_train, X_test, y_test, device: str, epochs: int = 24, batch_size: int = 128):
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_IMPORT_ERROR}")
    train_mean = float(X_train.mean())
    train_std = float(X_train.std() + 1e-6)
    Xtr = ((X_train - train_mean) / train_std)[:, None, :, :].astype(np.float32)
    Xte = ((X_test - train_mean) / train_std)[:, None, :, :].astype(np.float32)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_train.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = SmallKwsCnn(len(ALL_CLASSES)).to(device)
    counts = np.bincount(y_train, minlength=len(ALL_CLASSES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            pred = cnn_predict(model, X_test, train_mean, train_std, device)
            test_acc = accuracy_score(y_test, pred)
        else:
            test_acc = np.nan
        history.append({"epoch": epoch, "train_loss": total_loss / total, "train_acc": correct / total, "test_acc_snapshot": test_acc})
    pred = cnn_predict(model, X_test, train_mean, train_std, device)
    return model, pred, train_mean, train_std, pd.DataFrame(history)


def cnn_predict(model, X, mean, std, device: str, batch_size: int = 256):
    Xn = ((X - mean) / std)[:, None, :, :].astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xn)), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | str]:
    primary_ids = [CLASS_TO_ID[c] for c in TARGETS_PRIMARY]
    per = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(ALL_CLASSES))), zero_division=0)
    f1_by_class = per[2]
    row: Dict[str, float | str] = {
        "method": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "primary_mean_f1": float(np.mean([f1_by_class[i] for i in primary_ids])),
    }
    for c, i in CLASS_TO_ID.items():
        row[f"f1_{c}"] = float(f1_by_class[i])
    return row


def binary_metrics(y_true, preds_by_method: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for method, pred in preds_by_method.items():
        for target in TARGETS_PRIMARY:
            tid = CLASS_TO_ID[target]
            yt = (y_true == tid).astype(int)
            yp = (pred == tid).astype(int)
            p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
            rows.append({"method": method, "target": target, "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows)


def plot_spectral_overview(examples: Dict[str, dict], out_path: Path) -> None:
    selected = ["d", "g", "l", "n"]
    fig, axes = plt.subplots(len(selected), 2, figsize=(11, 10), constrained_layout=True)
    for i, c in enumerate(selected):
        ex = examples[c].item() if hasattr(examples[c], "item") else examples[c]
        t = np.arange(len(ex["x"])) / ex["sr"]
        axes[i, 0].plot(t, ex["x"], lw=0.7)
        axes[i, 0].set_title(f"/{c}/ example: '{ex['word']}' waveform")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        im = axes[i, 1].pcolormesh(ex["stft_t"], ex["stft_f"] / 1000, ex["stft_db"], shading="gouraud", cmap="magma", vmin=-80, vmax=0)
        axes[i, 1].set_ylim(0, 8)
        axes[i, 1].set_title(f"/{c}/ spectrogram")
        axes[i, 1].set_xlabel("Time (s)")
        axes[i, 1].set_ylabel("Frequency (kHz)")
    fig.colorbar(im, ax=axes[:, 1], shrink=0.8, label="Magnitude (dB)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title: str, out_path: Path) -> None:
    labels = list(range(len(ALL_CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(7.2, 5.9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=ALL_CLASSES, yticklabels=ALL_CLASSES, cbar_kws={"label": "row-normalized"})
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_metric_bars(summary: pd.DataFrame, out_path: Path) -> None:
    order = summary["method"].tolist()
    m = summary.melt(id_vars="method", value_vars=["accuracy", "macro_f1", "primary_mean_f1"], var_name="metric", value_name="score")
    plt.figure(figsize=(9.5, 5.2))
    sns.barplot(data=m, x="method", y="score", hue="metric", order=order)
    plt.ylim(0, 1)
    plt.xticks(rotation=18, ha="right")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("Model comparison under speaker-independent split")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_primary_f1(binary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9.2, 5.2))
    sns.barplot(data=binary, x="target", y="f1", hue="method")
    plt.ylim(0, 1)
    plt.xlabel("Primary voiced consonant")
    plt.ylabel("One-vs-rest F1")
    plt.title("Detection F1 for required voiced consonants")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_noise(noise_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.2))
    sns.lineplot(data=noise_df, x="snr_db", y="accuracy", hue="method", marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB); left is cleaner")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Noise robustness on the held-out speaker test set")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_training(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    ax1.plot(history["epoch"], history["train_loss"], marker="o", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["train_acc"], marker="s", color="tab:orange", label="train acc")
    snap = history.dropna(subset=["test_acc_snapshot"])
    ax2.plot(snap["epoch"], snap["test_acc_snapshot"], marker="^", color="tab:green", label="test acc snapshot")
    ax2.set_ylabel("Accuracy")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    plt.title("CNN training trace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def recompute_for_paths(paths: np.ndarray, snr_db: float | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG + (0 if snr_db is None else int(snr_db * 10)))
    xb, xi, xm, xc = [], [], [], []
    for p in paths:
        sr, x = load_wav(Path(str(p)))
        if snr_db is not None:
            x = add_noise_at_snr(x, snr_db, rng)
        rep = compute_representations(x, sr)
        xb.append(rep["base"]); xi.append(rep["improved"]); xm.append(rep["mlp"]); xc.append(rep["logmel"])
    return np.vstack(xb), np.vstack(xi), np.vstack(xm), np.stack(xc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mini_speech_commands"))
    parser.add_argument("--out-dir", type=Path, default=Path("build/upgrade_results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/upgrade"))
    parser.add_argument("--max-per-word", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--skip-noise", action="store_true")
    args = parser.parse_args()
    seed_all()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    items = collect_items(args.data_root, args.max_per_word)
    data = build_cache(items, args.out_dir / "upgrade_features_v1.npz")
    y = data["y"].astype(np.int64)
    speakers = data["speakers"]
    paths = data["paths"]
    train_idx, test_idx = split_by_speaker(y, speakers)

    # Build one augmented CNN copy for training only.  This keeps augmentation reproducible
    # and avoids hiding it inside the evaluation data.
    rng = np.random.default_rng(RNG)
    aug_cache = args.out_dir / "cnn_train_aug_v1.npz"
    if aug_cache.exists():
        aug = dict(np.load(aug_cache))
        X_cnn_train_aug = aug["X_cnn_train_aug"]
        y_train_aug = aug["y_train_aug"]
    else:
        X_aug = []
        for p in paths[train_idx]:
            sr, x = load_wav(Path(str(p)))
            rep = compute_representations(augment_wave(x, rng), sr)
            X_aug.append(rep["logmel"])
        X_cnn_train_aug = np.concatenate([data["X_cnn"][train_idx], np.stack(X_aug)], axis=0)
        y_train_aug = np.concatenate([y[train_idx], y[train_idx]], axis=0)
        np.savez_compressed(aug_cache, X_cnn_train_aug=X_cnn_train_aug, y_train_aug=y_train_aug)

    Xb_tr, Xb_te = data["X_base"][train_idx], data["X_base"][test_idx]
    Xi_tr, Xi_te = data["X_improved"][train_idx], data["X_improved"][test_idx]
    Xm_tr, Xm_te = data["X_mlp"][train_idx], data["X_mlp"][test_idx]
    Xc_te = data["X_cnn"][test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    pred_base, base_scaler, base_centroids = nearest_centroid(Xb_tr, y_tr, Xb_te)
    pred_improved, imp_scaler, imp_centroids = nearest_centroid(Xi_tr, y_tr, Xi_te)
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, max_iter=360, random_state=RNG))
    mlp.fit(Xm_tr, y_tr)
    pred_mlp = mlp.predict(Xm_te).astype(np.int64)

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    model, pred_cnn, cnn_mean, cnn_std, history = train_cnn(X_cnn_train_aug, y_train_aug, Xc_te, y_te, device=device, epochs=args.epochs)

    preds = {
        "baseline_centroid": pred_base,
        "delta_centroid": pred_improved,
        "mlp_features": pred_mlp,
        "cnn_logmel_aug": pred_cnn,
    }
    summary = pd.DataFrame([metrics_row(name, y_te, pred) for name, pred in preds.items()])
    binary = binary_metrics(y_te, preds)
    summary.to_csv(args.out_dir / "model_summary.csv", index=False)
    binary.to_csv(args.out_dir / "primary_binary_metrics.csv", index=False)
    history.to_csv(args.out_dir / "cnn_training_history.csv", index=False)
    pd.DataFrame({"path": paths, "word": data["words"], "speaker": speakers, "class_id": y, "class": [ID_TO_CLASS[int(i)] for i in y], "split": np.where(np.isin(np.arange(len(y)), train_idx), "train", "test")}).to_csv(args.out_dir / "dataset_split.csv", index=False)

    config = {
        "seed": RNG,
        "device": device,
        "torch_version": None if torch is None else torch.__version__,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device": None if torch is None or not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        "n_files": int(len(y)),
        "n_speakers": int(len(set(map(str, speakers)))),
        "train_files": int(len(train_idx)),
        "test_files": int(len(test_idx)),
        "train_speakers": int(len(set(map(str, speakers[train_idx])))),
        "test_speakers": int(len(set(map(str, speakers[test_idx])))),
        "classes": ALL_CLASSES,
        "word_to_consonant": WORD_TO_CONSONANT,
        "class_counts_total": {c: int((y == CLASS_TO_ID[c]).sum()) for c in ALL_CLASSES},
        "class_counts_test": {c: int((y_te == CLASS_TO_ID[c]).sum()) for c in ALL_CLASSES},
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    examples = data["examples"].item() if hasattr(data["examples"], "item") else data["examples"]
    plot_spectral_overview(examples, args.fig_dir / "spectral_overview.png")
    plot_confusion(y_te, pred_base, "Baseline: MFCC centroid", args.fig_dir / "confusion_baseline_centroid.png")
    plot_confusion(y_te, pred_improved, "Traditional upgrade: delta-MFCC centroid", args.fig_dir / "confusion_delta_centroid.png")
    plot_confusion(y_te, pred_cnn, "AI upgrade: augmented log-mel CNN", args.fig_dir / "confusion_cnn_logmel_aug.png")
    plot_metric_bars(summary, args.fig_dir / "model_comparison.png")
    plot_primary_f1(binary, args.fig_dir / "primary_f1.png")
    plot_training(history, args.fig_dir / "cnn_training_trace.png")

    if not args.skip_noise:
        noise_rows = []
        for snr in [None, 20.0, 10.0, 5.0, 0.0]:
            xb, xi, xm, xc = recompute_for_paths(paths[test_idx], snr)
            noise_preds = {
                "baseline_centroid": nearest_centroid_predict_with_model(xb, base_scaler, base_centroids),
                "delta_centroid": nearest_centroid_predict_with_model(xi, imp_scaler, imp_centroids),
                "mlp_features": mlp.predict(xm).astype(np.int64),
                "cnn_logmel_aug": cnn_predict(model, xc, cnn_mean, cnn_std, device),
            }
            for method, pred in noise_preds.items():
                noise_rows.append({
                    "method": method,
                    "snr_db": 40.0 if snr is None else snr,
                    "condition": "clean" if snr is None else f"{snr:g}dB",
                    "accuracy": accuracy_score(y_te, pred),
                    "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
                })
        noise = pd.DataFrame(noise_rows)
        noise.to_csv(args.out_dir / "noise_robustness.csv", index=False)
        plot_noise(noise, args.fig_dir / "noise_robustness.png")

    print("Run config:")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPrimary binary metrics:")
    print(binary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
