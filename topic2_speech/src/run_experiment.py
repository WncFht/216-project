#!/usr/bin/env python3
"""VE216 Topic 2: voiced-consonant spectral analysis and detection.

This script uses the TensorFlow mini Speech Commands data set as a real-data
stand-in for the MATLAB workflow requested by the project statement.  It
extracts classical short-time spectral descriptors and MFCCs, then compares an
interpretable nearest-centroid detector with a small neural-network detector.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import stft, get_window
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = 216
WORDS = ["down", "go", "left", "no", "right", "yes", "stop", "up"]
# We use the command word's initial consonant as the target.  "other" is kept
# as a negative class for words without a target voiced initial consonant.
WORD_TO_CONSONANT = {
    "down": "d",
    "go": "g",
    "left": "l",
    "no": "n",
    "right": "r",
    "yes": "j",  # palatal approximant, included as a voiced consonantal glide
    "stop": "other",
    "up": "other",
}
TARGETS_PRIMARY = ["d", "g", "l", "n"]
ALL_CLASSES = ["d", "g", "l", "n", "r", "j", "other"]


@dataclass
class AudioItem:
    path: Path
    word: str
    consonant: str


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=float)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / (right - center)
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


def load_wav(path: Path, target_sr: int = 16000) -> Tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    # The mini Speech Commands data are already 16 kHz, but keep a guardrail.
    if sr != target_sr:
        from scipy.signal import resample_poly

        g = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    # Pad / crop to 1 s for comparable features.
    n = target_sr
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    else:
        x = x[:n]
    return sr, x


def frame_signal(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> Tuple[np.ndarray, int, int]:
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, frame_len), strides=strides).copy()
    frames *= get_window("hamming", frame_len, fftbins=True)
    return frames, frame_len, hop


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)


def spectral_features(x: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    pre = np.append(x[0], x[1:] - 0.97 * x[:-1])
    frames, frame_len, _ = frame_signal(pre, sr)
    n_fft = 512
    mag = np.abs(np.fft.rfft(frames, n=n_fft)) + 1e-12
    power = (mag**2) / n_fft
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    fb = mel_filterbank(sr, n_fft, n_mels=40)
    mel_energy = np.maximum(power @ fb.T, 1e-12)
    log_mel = np.log(mel_energy)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]

    total = np.maximum(np.sum(mag, axis=1), 1e-12)
    centroid = np.sum(mag * freqs, axis=1) / total
    bandwidth = np.sqrt(np.sum(mag * (freqs - centroid[:, None]) ** 2, axis=1) / total)
    cumsum = np.cumsum(mag, axis=1)
    rolloff_idx = np.argmax(cumsum >= 0.85 * cumsum[:, [-1]], axis=1)
    rolloff = freqs[rolloff_idx]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = zero_crossing_rate(frames)

    # Summary vector: mean and std of MFCCs plus classical spectral statistics.
    stats = []
    for arr in [mfcc, centroid[:, None], bandwidth[:, None], rolloff[:, None], rms[:, None], zcr[:, None]]:
        stats.append(arr.mean(axis=0))
        stats.append(arr.std(axis=0))
    # Keep coarse temporal order for the AI model and for a fairer detector:
    # 1 s commands produce a fixed number of frames after padding/cropping.
    temporal = mfcc.reshape(-1)
    feat = np.concatenate(stats + [temporal])
    framewise = {
        "mfcc": mfcc,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "rms": rms,
        "zcr": zcr,
        "freqs": freqs,
        "mean_log_spectrum": np.log10(mag.mean(axis=0)),
    }
    params = {"frame_len": frame_len, "n_fft": n_fft}
    return feat, framewise, params


def collect_items(data_root: Path, max_per_word: int | None = None) -> List[AudioItem]:
    items: List[AudioItem] = []
    rng = random.Random(RNG)
    for word in WORDS:
        files = sorted((data_root / word).glob("*.wav"))
        rng.shuffle(files)
        if max_per_word:
            files = files[:max_per_word]
        for path in files:
            items.append(AudioItem(path=path, word=word, consonant=WORD_TO_CONSONANT[word]))
    rng.shuffle(items)
    return items


def extract_dataset(items: List[AudioItem], cache_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, dict]]:
    if cache_npz.exists():
        data = np.load(cache_npz, allow_pickle=True)
        return data["X"], data["y"], data["words"], list(data["paths"]), data["examples"].item()
    X, y, words, paths = [], [], [], []
    examples: Dict[str, dict] = {}
    for item in items:
        sr, x = load_wav(item.path)
        feat, framewise, params = spectral_features(x, sr)
        X.append(feat)
        y.append(item.consonant)
        words.append(item.word)
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
                "freqs": framewise["freqs"],
                "mean_log_spectrum": framewise["mean_log_spectrum"],
            }
    X = np.vstack(X)
    y = np.asarray(y)
    words = np.asarray(words)
    np.savez_compressed(cache_npz, X=X, y=y, words=words, paths=np.asarray(paths), examples=examples)
    return X, y, words, paths, examples


def nearest_centroid_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    centroids = {c: Xtr[y_train == c].mean(axis=0) for c in ALL_CLASSES if np.any(y_train == c)}
    preds = []
    for x in Xte:
        sims = {c: float(np.dot(x, mu) / ((np.linalg.norm(x) + 1e-12) * (np.linalg.norm(mu) + 1e-12))) for c, mu in centroids.items()}
        preds.append(max(sims, key=sims.get))
    return np.asarray(preds)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        yt = (y_true == target).astype(int)
        yp = (y_pred == target).astype(int)
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        rows.append({"target": target, "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows)


def plot_spectral_overview(examples: Dict[str, dict], out_path: Path) -> None:
    selected = ["d", "g", "l", "n"]
    fig, axes = plt.subplots(len(selected), 2, figsize=(11, 10), constrained_layout=True)
    for i, c in enumerate(selected):
        ex = examples[c]
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


def plot_mean_spectra(examples: Dict[str, dict], out_path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for c in ["d", "g", "l", "n", "r", "j"]:
        ex = examples[c]
        plt.plot(ex["freqs"] / 1000, ex["mean_log_spectrum"], label=f"/{c}/: {ex['word']}", lw=1.4)
    plt.xlim(0, 8)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Log mean magnitude")
    plt.title("Representative short-time spectra for voiced consonant targets")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    labels = [c for c in ALL_CLASSES if np.any(y_true == c) or np.any(y_pred == c)]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(7, 5.8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar_kws={"label": "row-normalized"})
    plt.xlabel("Predicted consonant")
    plt.ylabel("True consonant")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mini_speech_commands"))
    parser.add_argument("--out-dir", type=Path, default=Path("build/results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    parser.add_argument("--max-per-word", type=int, default=1000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    items = collect_items(args.data_root, args.max_per_word)
    X, y, words, paths, examples = extract_dataset(items, args.out_dir / "features_cache_v2.npz")

    # Split at file level with stratification by consonant class.
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=RNG, stratify=y)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    y_centroid = nearest_centroid_predict(X_train, y_train, X_test)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, max_iter=400, random_state=RNG, early_stopping=False),
    )
    mlp.fit(X_train, y_train)
    y_mlp = mlp.predict(X_test)

    summary = {
        "n_files": int(len(y)),
        "train_files": int(len(train_idx)),
        "test_files": int(len(test_idx)),
        "classes": ALL_CLASSES,
        "word_to_consonant": WORD_TO_CONSONANT,
        "class_counts": {c: int(np.sum(y == c)) for c in ALL_CLASSES},
        "centroid_accuracy": float(accuracy_score(y_test, y_centroid)),
        "mlp_accuracy": float(accuracy_score(y_test, y_mlp)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.DataFrame({"path": paths, "word": words, "consonant": y}).to_csv(args.out_dir / "dataset_index.csv", index=False)
    binary = pd.concat(
        [
            binary_metrics(y_test, y_centroid, TARGETS_PRIMARY).assign(method="nearest_centroid"),
            binary_metrics(y_test, y_mlp, TARGETS_PRIMARY).assign(method="mlp_ai"),
        ],
        ignore_index=True,
    )
    binary.to_csv(args.out_dir / "binary_detection_metrics.csv", index=False)

    with (args.out_dir / "classification_report_centroid.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_centroid, labels=ALL_CLASSES, zero_division=0))
    with (args.out_dir / "classification_report_mlp.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_mlp, labels=ALL_CLASSES, zero_division=0))

    plot_spectral_overview(examples, args.fig_dir / "spectral_overview.png")
    plot_mean_spectra(examples, args.fig_dir / "mean_spectra.png")
    plot_confusion(y_test, y_centroid, "Classical MFCC nearest-centroid detector", args.fig_dir / "confusion_centroid.png")
    plot_confusion(y_test, y_mlp, "AI detector: MLP on MFCC and spectral descriptors", args.fig_dir / "confusion_mlp.png")

    print(json.dumps(summary, indent=2))
    print("\nBinary detection metrics:")
    print(binary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
