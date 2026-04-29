#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from scipy.fft import dct
from scipy.signal import get_window, resample_poly, stft
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support
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
    DataLoader = TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

RNG = 216
DEFAULT_LABELS = ["g", "b", "d", "z"]
DEFAULT_EXAMPLE_WORDS = {
    "g": "get",
    "b": "boy",
    "d": "did",
    "z": "zero",
}
BAND_EDGES = np.array([0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5500, 8000], dtype=float)
FILTERBANK_MELS = 40
COURSE_AUTOCORR_LAGS = 24
COURSE_WINDOW_FRAMES = 16
COURSE_PRE_FRAMES = 2
COURSE_SEGMENTS = 3
COURSE_ENERGY_SMOOTH_FRAMES = 5
COURSE_ONSET_MARGIN = 0.22
COURSE_CANDIDATE_SHIFTS = (-2, 0, 2)
COURSE_CROSS_WEIGHT = 0.65
COURSE_AUTOCORR_WEIGHT = 0.35
STRICT_FRAME_MS = 20
STRICT_HOP_MS = 10
STRICT_BLOCK_FRAMES = 4
STRICT_NFFT = 512
STRICT_TEMPLATE_CASES = {
    "g": "case_g_get.wav",
    "b": "case_b_boy.wav",
    "d": "case_d_did.wav",
    "z": "case_z_zero.wav",
}


@dataclass
class Item:
    split: str
    word: str
    speaker: str
    labels: List[str]
    wav_path: Path
    phones: str


class MultiCnn(nn.Module):
    def __init__(self, n_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, n_labels),
        )

    def forward(self, x):
        return self.net(x)


def seed_all(seed: int = RNG):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path, labels: Sequence[str]) -> List[Item]:
    label_set = set(labels)
    rows: List[Item] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path_val = Path(row["wav_path"])
            if not path_val.is_absolute():
                path_val = path.parents[3] / path_val
            label_tokens = row.get("labels", "").split()
            if not label_tokens and row.get("label"):
                label_tokens = [row["label"].strip()]
            lab = sorted(label_set.intersection(label_tokens))
            rows.append(
                Item(
                    split=row["split"],
                    word=row["word"],
                    speaker=row["speaker"],
                    labels=lab,
                    wav_path=path_val,
                    phones=row["phones"],
                )
            )
    return rows


def load_wav(path: Path, target_sr: int = 16000) -> Tuple[int, np.ndarray]:
    x, sr = sf.read(path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    if len(x) < target_sr:
        x = np.pad(x, (0, target_sr - len(x)))
    elif len(x) > target_sr:
        x = x[:target_sr]
    return sr, x.astype(np.float32)


def hz_to_mel(f):
    return 2595 * np.log10(1 + np.asarray(f) / 700)


def mel_to_hz(m):
    return 700 * (10 ** (np.asarray(m) / 2595) - 1)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 50, fmax: float | None = None):
    if fmax is None:
        fmax = sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


def frame_signal(x: np.ndarray, sr: int, frame_ms: float = 25, hop_ms: float = 10) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, frame_len),
        strides=(x.strides[0] * hop, x.strides[0]),
    ).copy()
    frames *= get_window("hamming", frame_len, fftbins=True)
    return frames


def normalized_autocorrelation(x: np.ndarray, n_lags: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x)) + 1e-12
    corr = np.correlate(x, x, mode="full")
    center = len(x) - 1
    values = corr[center:center + min(n_lags, len(x))] / denom
    if len(values) < n_lags:
        values = np.pad(values, (0, n_lags - len(values)))
    return values.astype(np.float32)


def moving_average_1d(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if width <= 1 or len(x) == 0:
        return x.astype(np.float32)
    width = int(max(1, width))
    kernel = np.ones(width, dtype=np.float32) / float(width)
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def pad_or_trim_rows(x: np.ndarray, start: int, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    n_rows, n_cols = x.shape
    if n_rows == 0:
        return np.zeros((length, n_cols), dtype=np.float32)
    end = start + length
    left = max(0, -start)
    right = max(0, end - n_rows)
    s = max(0, start)
    e = min(n_rows, end)
    out = x[s:e]
    if left or right:
        out = np.pad(out, ((left, right), (0, 0)), mode="edge")
    if out.shape[0] != length:
        if out.shape[0] > length:
            out = out[:length]
        else:
            out = np.pad(out, ((0, length - out.shape[0]), (0, 0)), mode="edge")
    return out.astype(np.float32)


def frame_energy_from_logmel(logmel: np.ndarray) -> np.ndarray:
    logmel = np.asarray(logmel, dtype=np.float32)
    return np.log(np.mean(np.exp(logmel), axis=1) + 1e-12).astype(np.float32)


def segment_mean_spectrum(patch: np.ndarray, segments: int = COURSE_SEGMENTS) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float32)
    parts = np.array_split(patch, int(max(1, segments)), axis=0)
    return np.concatenate([part.mean(axis=0) for part in parts], axis=0).astype(np.float32)


def detect_course_onset(logmel: np.ndarray) -> int:
    energy = frame_energy_from_logmel(logmel)
    smoothed = moving_average_1d(energy, COURSE_ENERGY_SMOOTH_FRAMES)
    head = smoothed[: min(len(smoothed), max(6, COURSE_ENERGY_SMOOTH_FRAMES * 2))]
    baseline = float(np.median(head)) if len(head) else float(np.median(smoothed))
    cutoff = baseline + COURSE_ONSET_MARGIN
    hits = np.flatnonzero(smoothed >= cutoff)
    if len(hits) > 0:
        return int(hits[0])
    return int(np.argmax(smoothed))


def extract_course_patch(logmel: np.ndarray, onset_idx: int, shift: int = 0):
    start = onset_idx - COURSE_PRE_FRAMES + int(shift)
    patch = pad_or_trim_rows(logmel, start, COURSE_WINDOW_FRAMES)
    if COURSE_PRE_FRAMES > 0:
        baseline = patch[:COURSE_PRE_FRAMES].mean(axis=0, keepdims=True)
    else:
        baseline = patch[:1].mean(axis=0, keepdims=True)
    centered_patch = patch - baseline
    envelope = frame_energy_from_logmel(patch)
    raw_segment = segment_mean_spectrum(patch, COURSE_SEGMENTS)
    centered_segment = segment_mean_spectrum(centered_patch, COURSE_SEGMENTS)
    patch_vec = np.concatenate([raw_segment, centered_segment], axis=0).astype(np.float32)
    patch_auto = normalized_autocorrelation(envelope, COURSE_AUTOCORR_LAGS)
    return patch_vec, patch_auto


def extract_course_candidates(logmel: np.ndarray):
    onset_idx = detect_course_onset(logmel)
    patch_rows = []
    auto_rows = []
    for shift in COURSE_CANDIDATE_SHIFTS:
        patch_vec, patch_auto = extract_course_patch(logmel, onset_idx, shift=shift)
        patch_rows.append(patch_vec)
        auto_rows.append(patch_auto)
    return np.vstack(patch_rows), np.vstack(auto_rows)


def compute_representations(x: np.ndarray, sr: int):
    frames = frame_signal(x, sr)
    n_fft = 512
    mag = np.abs(np.fft.rfft(frames, n=n_fft)) + 1e-12
    power = (mag**2) / n_fft
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = mel_filterbank(sr, n_fft, FILTERBANK_MELS)
    logmel = np.log(np.maximum(power @ fb.T, 1e-12)).astype(np.float32)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :13].astype(np.float32)
    delta = np.gradient(mfcc, axis=0).astype(np.float32)
    bands = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        bands.append(np.log(np.mean(mag[:, mask] ** 2) + 1e-12))
    band_vec = np.asarray(bands, dtype=np.float32)
    stat = np.concatenate([
        mfcc.mean(0),
        mfcc.std(0),
        delta.mean(0),
        delta.std(0),
        band_vec,
    ]).astype(np.float32)
    mean_spectrum = np.log(np.mean(mag**2, axis=0) + 1e-12).astype(np.float32)
    filterbank_spectrum = logmel.mean(axis=0).astype(np.float32)
    filterbank_envelope = logmel.mean(axis=1).astype(np.float32)
    filterbank_autocorr = normalized_autocorrelation(filterbank_envelope, COURSE_AUTOCORR_LAGS)
    return {
        "band": band_vec,
        "stat": stat,
        "logmel": logmel,
        "mean_spectrum": mean_spectrum,
        "course_spectrum": filterbank_spectrum,
        "course_autocorr": filterbank_autocorr,
        "freqs": freqs.astype(np.float32),
    }


def strict_periodic_hamming(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(1, n), dtype=np.float32)
    k = np.arange(n, dtype=np.float32)
    return (0.54 - 0.46 * np.cos(2 * np.pi * k / n)).astype(np.float32)


def strict_frame_signal(x: np.ndarray, sr: int) -> np.ndarray:
    frame_len = max(1, round(sr * STRICT_FRAME_MS / 1000))
    hop = max(1, round(sr * STRICT_HOP_MS / 1000))
    x = np.asarray(x, dtype=np.float32)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = max(1, math.floor((len(x) - frame_len) / hop) + 1)
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    win = strict_periodic_hamming(frame_len)
    for i in range(n_frames):
        start = i * hop
        frames[i] = x[start : start + frame_len] * win
    return frames


def strict_block_descriptor(frames: np.ndarray, start_frame: int) -> np.ndarray:
    block = pad_or_trim_rows(frames, start_frame, STRICT_BLOCK_FRAMES)
    spec = np.abs(np.fft.fft(block, STRICT_NFFT, axis=1))[:, : STRICT_NFFT // 2 + 1]
    row_max = np.max(spec, axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    spec = spec / row_max
    descriptor = spec.mean(axis=0)
    descriptor = descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor.astype(np.float32)


def strict_select_max_energy_descriptor(x: np.ndarray, sr: int):
    frames = strict_frame_signal(x, sr)
    n_frames = frames.shape[0]
    n_starts = max(1, n_frames - STRICT_BLOCK_FRAMES + 1)
    best_energy = -np.inf
    best_start = 0
    best_descriptor = np.zeros(STRICT_NFFT // 2 + 1, dtype=np.float32)
    for start in range(n_starts):
        block = pad_or_trim_rows(frames, start, STRICT_BLOCK_FRAMES)
        block_energy = float(np.sum(block**2))
        if block_energy > best_energy:
            best_energy = block_energy
            best_start = start
            best_descriptor = strict_block_descriptor(frames, start)
    return best_descriptor, best_start


def strict_all_candidate_descriptors(x: np.ndarray, sr: int):
    frames = strict_frame_signal(x, sr)
    n_frames = frames.shape[0]
    n_starts = max(1, n_frames - STRICT_BLOCK_FRAMES + 1)
    matrix = np.zeros((n_starts, STRICT_NFFT // 2 + 1), dtype=np.float32)
    start_frames = np.arange(n_starts, dtype=np.int32)
    for start in range(n_starts):
        matrix[start] = strict_block_descriptor(frames, start)
    return matrix, start_frames


def build_strict_course_bank(case_dir: Path, labels: Sequence[str]):
    manifest_path = case_dir / "case_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    label_map = {str(row["label"]).strip(): row for _, row in manifest.iterrows()}

    template_paths: List[str] = []
    template_words: List[str] = []
    template_starts: List[int] = []
    templates: List[np.ndarray] = []

    for lab in labels:
        row = label_map[lab]
        wav_path = case_dir / str(row["filename"])
        sr, x = load_wav(wav_path)
        template, start_frame = strict_select_max_energy_descriptor(x, sr)
        template_paths.append(str(wav_path))
        template_words.append(str(row["word"]))
        template_starts.append(int(start_frame))
        templates.append(template)

    templates_arr = np.vstack(templates).astype(np.float32)
    return {
        "labels": list(labels),
        "template_paths": template_paths,
        "template_words": template_words,
        "template_start_frames": np.asarray(template_starts, dtype=np.int32),
        "templates": templates_arr,
        "freq_hz": np.arange(STRICT_NFFT // 2 + 1, dtype=np.float32)
        * (16000.0 / STRICT_NFFT),
        "cfg": {
            "target_sr": 16000,
            "frame_ms": STRICT_FRAME_MS,
            "hop_ms": STRICT_HOP_MS,
            "block_frames": STRICT_BLOCK_FRAMES,
            "nfft": STRICT_NFFT,
        },
    }


def score_strict_course_detector(paths: Sequence[str | Path], bank: Dict[str, np.ndarray]) -> np.ndarray:
    scores = np.zeros((len(paths), len(bank["labels"])), dtype=np.float32)
    templates = np.asarray(bank["templates"], dtype=np.float32)
    for i, wav_path in enumerate(paths):
        sr, x = load_wav(Path(wav_path))
        matrix, _ = strict_all_candidate_descriptors(x, sr)
        scores[i] = np.max(matrix @ templates.T, axis=0)
    return scores


def build_dataset(items: Sequence[Item], labels: Sequence[str], cache_path: Path):
    if cache_path.exists():
        return dict(np.load(cache_path, allow_pickle=True))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    X_band, X_course_spec, X_course_auto, X_stat, X_cnn, Y, words, speakers, splits, paths, phones = [], [], [], [], [], [], [], [], [], [], []
    for item in items:
        sr, x = load_wav(item.wav_path)
        rep = compute_representations(x, sr)
        y = np.zeros(len(labels), dtype=np.float32)
        for lab in item.labels:
            y[label_to_idx[lab]] = 1.0
        X_band.append(rep["band"])
        X_course_spec.append(rep["course_spectrum"])
        X_course_auto.append(rep["course_autocorr"])
        X_stat.append(rep["stat"])
        X_cnn.append(rep["logmel"])
        Y.append(y)
        words.append(item.word)
        speakers.append(item.speaker)
        splits.append(item.split)
        paths.append(str(item.wav_path))
        phones.append(item.phones)
    data = {
        "X_band": np.vstack(X_band),
        "X_course_spec": np.vstack(X_course_spec),
        "X_course_auto": np.vstack(X_course_auto),
        "X_stat": np.vstack(X_stat),
        "X_cnn": np.stack(X_cnn),
        "Y": np.vstack(Y).astype(np.float32),
        "words": np.asarray(words),
        "speakers": np.asarray(speakers),
        "splits": np.asarray(splits),
        "paths": np.asarray(paths),
        "phones": np.asarray(phones),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    return data


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    quantiles = np.quantile(scores, np.linspace(0.02, 0.98, 97))
    best_t = float(quantiles[0])
    best_f = -1.0
    for t in quantiles:
        pred = (scores >= float(t)).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return best_t


def max_normalized_cross_correlation_rows(X: np.ndarray, template: np.ndarray) -> np.ndarray:
    template = np.asarray(template, dtype=np.float32)
    template = template - float(np.mean(template))
    template_norm = float(np.linalg.norm(template)) + 1e-12
    scores = np.zeros(len(X), dtype=np.float32)
    for i, row in enumerate(np.asarray(X, dtype=np.float32)):
        centered = row - float(np.mean(row))
        row_norm = float(np.linalg.norm(centered)) + 1e-12
        corr = np.correlate(centered, template, mode="full") / (row_norm * template_norm)
        scores[i] = float(np.max(corr))
    return scores


def cosine_similarity_rows(X: np.ndarray, template: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)
    denom = (np.linalg.norm(X, axis=1) + 1e-12) * (float(np.linalg.norm(template)) + 1e-12)
    return ((X @ template) / denom).astype(np.float32)


def fit_course_detector(X_logmel_train: np.ndarray, Y_train: np.ndarray):
    sample_patch_features = []
    sample_autocorr_features = []
    for logmel in X_logmel_train:
        patch_vec, patch_auto = extract_course_patch(logmel, detect_course_onset(logmel), shift=0)
        sample_patch_features.append(patch_vec)
        sample_autocorr_features.append(patch_auto)
    sample_patch_features = np.vstack(sample_patch_features)
    sample_autocorr_features = np.vstack(sample_autocorr_features)

    patch_pos_templates = []
    patch_neg_templates = []
    autocorr_pos_templates = []
    autocorr_neg_templates = []
    for j in range(Y_train.shape[1]):
        pos = Y_train[:, j] == 1
        neg = ~pos
        if not np.any(pos):
            raise ValueError(f"No positive training samples found for label index {j}")
        if not np.any(neg):
            raise ValueError(f"No negative training samples found for label index {j}")
        patch_pos_templates.append(np.mean(sample_patch_features[pos], axis=0).astype(np.float32))
        patch_neg_templates.append(np.mean(sample_patch_features[neg], axis=0).astype(np.float32))
        autocorr_pos_templates.append(np.mean(sample_autocorr_features[pos], axis=0).astype(np.float32))
        autocorr_neg_templates.append(np.mean(sample_autocorr_features[neg], axis=0).astype(np.float32))
    return (
        np.vstack(patch_pos_templates),
        np.vstack(patch_neg_templates),
        np.vstack(autocorr_pos_templates),
        np.vstack(autocorr_neg_templates),
    )


def score_course_detector(
    X_logmel: np.ndarray,
    patch_pos_templates: np.ndarray,
    patch_neg_templates: np.ndarray,
    autocorr_pos_templates: np.ndarray,
    autocorr_neg_templates: np.ndarray,
) -> np.ndarray:
    scores = np.zeros((len(X_logmel), patch_pos_templates.shape[0]), dtype=np.float32)
    for i, logmel in enumerate(X_logmel):
        patch_rows, auto_rows = extract_course_candidates(logmel)
        for j in range(patch_pos_templates.shape[0]):
            patch_pos_scores = cosine_similarity_rows(patch_rows, patch_pos_templates[j])
            patch_neg_scores = cosine_similarity_rows(patch_rows, patch_neg_templates[j])
            auto_pos_scores = cosine_similarity_rows(auto_rows, autocorr_pos_templates[j])
            auto_neg_scores = cosine_similarity_rows(auto_rows, autocorr_neg_templates[j])
            candidate_scores = COURSE_CROSS_WEIGHT * (patch_pos_scores - patch_neg_scores) + COURSE_AUTOCORR_WEIGHT * (
                auto_pos_scores - auto_neg_scores
            )
            scores[i, j] = float(np.max(candidate_scores))
    return scores


def fit_mlp(X_train: np.ndarray, Y_train: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=RNG,
        ),
    )
    model.fit(X_train, Y_train.astype(int))
    return model


def get_mlp_scores(model, X: np.ndarray) -> np.ndarray:
    scores = model.predict_proba(X)
    if isinstance(scores, list):
        scores = np.column_stack([s[:, 1] for s in scores])
    return np.asarray(scores, dtype=np.float32)


def fit_thresholds(y_dev: np.ndarray, scores_dev: np.ndarray) -> np.ndarray:
    return np.asarray([choose_threshold(y_dev[:, j].astype(int), scores_dev[:, j]) for j in range(y_dev.shape[1])], dtype=np.float32)


def predict_with_thresholds(scores: np.ndarray, thresholds: np.ndarray, exclusive: bool = False) -> np.ndarray:
    scores = np.asarray(scores)
    if exclusive:
        pred = np.zeros_like(scores, dtype=int)
        pred[np.arange(len(scores)), np.argmax(scores, axis=1)] = 1
        return pred
    return (scores >= thresholds[None, :]).astype(int)


def label_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], method: str) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return pd.DataFrame(
        {
            "method": method,
            "label": list(labels),
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s.astype(int),
        }
    )


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], method: str) -> Dict[str, float | str]:
    return {
        "method": method,
        "label_wise_accuracy": float(1.0 - hamming_loss(y_true, y_pred)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "mean_positive_f1": float(np.mean(label_metrics(y_true, y_pred, labels, method)["f1"])),
    }


def build_example_rows(data: Dict[str, np.ndarray], labels: Sequence[str], preferred_words: Dict[str, str]):
    rows = []
    for j, lab in enumerate(labels):
        mask = data["Y"][:, j] == 1
        split_mask = data["splits"] == "test"
        word_mask = data["words"] == preferred_words.get(lab, "")
        idxs = np.where(mask & split_mask & word_mask)[0]
        if len(idxs) == 0:
            idxs = np.where(mask & split_mask)[0]
        if len(idxs) == 0:
            idxs = np.where(mask)[0]
        rows.append(int(idxs[0]))
    return rows


def plot_letter_examples(data: Dict[str, np.ndarray], labels: Sequence[str], preferred_words: Dict[str, str], out_path: Path):
    example_indices = build_example_rows(data, labels, preferred_words)
    fig, axes = plt.subplots(len(labels), 2, figsize=(10.5, 2.6 * len(labels)))
    if len(labels) == 1:
        axes = np.asarray([axes])
    for row_id, (lab, idx) in enumerate(zip(labels, example_indices)):
        path = Path(data["paths"][idx])
        sr, x = load_wav(path)
        t = np.arange(len(x)) / sr
        ax_wav = axes[row_id, 0]
        ax_wav.plot(t, x, linewidth=0.9, color="#1f77b4")
        ax_wav.set_xlim(0, t[-1] if len(t) else 1.0)
        ax_wav.set_title(f"/{lab}/ example: {data['words'][idx]}")
        ax_wav.set_xlabel("Time (s)")
        ax_wav.set_ylabel("Amplitude")
        f, tt, Z = stft(x, fs=sr, window="hann", nperseg=400, noverlap=240, nfft=512, boundary=None)
        ax_spec = axes[row_id, 1]
        im = ax_spec.pcolormesh(tt, f / 1000.0, 20 * np.log10(np.abs(Z) + 1e-6), shading="gouraud", cmap="magma", vmin=-80, vmax=0)
        ax_spec.set_ylim(0, 8)
        ax_spec.set_title(f"/{lab}/ spectrogram: {data['phones'][idx]}")
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (kHz)")
    fig.colorbar(im, ax=axes[:, 1], shrink=0.95, label="Magnitude (dB)")
    fig.subplots_adjust(wspace=0.28, hspace=0.6, right=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_template_grid(X_cnn: np.ndarray, Y: np.ndarray, labels: Sequence[str], out_path: Path):
    fig, axes = plt.subplots(2, math.ceil(len(labels) / 2), figsize=(10.5, 6.2))
    axes = np.asarray(axes).reshape(-1)
    im = None
    for ax, lab_idx, lab in zip(axes, range(len(labels)), labels):
        avg = X_cnn[Y[:, lab_idx] == 1].mean(0).T
        im = ax.imshow(avg, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"/{lab}/ average log-mel")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")
    for ax in axes[len(labels):]:
        ax.axis("off")
    if im is not None:
        cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
        fig.colorbar(im, cax=cax)
    fig.subplots_adjust(wspace=0.3, hspace=0.35, right=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metrics(summary_df: pd.DataFrame, out_path: Path):
    melt = summary_df.melt(
        id_vars="method",
        value_vars=["label_wise_accuracy", "exact_match_accuracy", "macro_f1", "micro_f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(9.6, 5.2))
    sns.barplot(data=melt, x="method", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("MSWC English four-letter detection summary")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_per_label_f1(metrics_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(9.2, 5.0))
    sns.barplot(data=metrics_df, x="label", y="f1", hue="method")
    plt.ylim(0, 1)
    plt.title("Per-label F1 on MSWC English test split")
    plt.xlabel("Voiced consonant")
    plt.ylabel("F1")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_score_distribution(scores: np.ndarray, y_true: np.ndarray, labels: Sequence[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, lab in enumerate(labels):
        plt.figure(figsize=(6.8, 4.2))
        pos = scores[y_true[:, j] == 1, j]
        neg = scores[y_true[:, j] == 0, j]
        plt.hist(neg, bins=30, alpha=0.55, label=f"not /{lab}/", color="#9ecae1", density=True)
        plt.hist(pos, bins=30, alpha=0.55, label=f"contains /{lab}/", color="#e6550d", density=True)
        plt.xlabel("template score")
        plt.ylabel("density")
        plt.title(f"Course-template score distribution for /{lab}/")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"score_distribution_{lab}.png", dpi=220)
        plt.close()


def plot_strict_template_construction(bank: Dict[str, np.ndarray], out_path: Path):
    fig, axes = plt.subplots(len(bank["labels"]), 2, figsize=(10.5, 8.5), constrained_layout=True)
    freq_khz = np.asarray(bank["freq_hz"], dtype=np.float32) / 1000.0
    for i, lab in enumerate(bank["labels"]):
        wav_path = Path(bank["template_paths"][i])
        _, x = load_wav(wav_path)
        frames = strict_frame_signal(x, 16000)
        block = pad_or_trim_rows(frames, int(bank["template_start_frames"][i]), STRICT_BLOCK_FRAMES)
        spec = np.abs(np.fft.fft(block, STRICT_NFFT, axis=1))[:, : STRICT_NFFT // 2 + 1]
        spec = spec / np.maximum(spec.max(axis=1, keepdims=True), 1e-12)
        mean_spec = np.asarray(bank["templates"][i], dtype=np.float32)

        ax0 = axes[i, 0]
        im = ax0.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[freq_khz[0], freq_khz[-1], 1, STRICT_BLOCK_FRAMES],
            cmap="magma",
        )
        ax0.set_title(f"/{lab}/ template block: {bank['template_words'][i]}")
        ax0.set_xlabel("Frequency (kHz)")
        ax0.set_ylabel("Frame in block")
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[i, 1]
        ax1.plot(freq_khz, mean_spec, color="#1f77b4", linewidth=2.0)
        ax1.fill_between(freq_khz, 0, mean_spec, color="#9ecae1", alpha=0.5)
        ax1.set_xlim(freq_khz[0], freq_khz[-1])
        ax1.set_ylim(0, max(1.02, float(mean_spec.max()) * 1.05))
        ax1.set_title(f"/{lab}/ normalized FFT template")
        ax1.set_xlabel("Frequency (kHz)")
        ax1.set_ylabel("Magnitude")
        ax1.grid(alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_strict_correlation_demo(example_path: Path, bank: Dict[str, np.ndarray], label: str, out_path: Path):
    sr, x = load_wav(example_path)
    frames = strict_frame_signal(x, sr)
    descriptors, start_frames = strict_all_candidate_descriptors(x, sr)
    score_traces = descriptors @ np.asarray(bank["templates"], dtype=np.float32).T
    label_idx = list(bank["labels"]).index(label)
    best_idx = int(np.argmax(score_traces[:, label_idx]))
    best_start = int(start_frames[best_idx])
    frame_len = max(1, round(sr * STRICT_FRAME_MS / 1000))
    hop = max(1, round(sr * STRICT_HOP_MS / 1000))
    start_sample = best_start * hop
    end_sample = min(len(x), start_sample + frame_len + hop * (STRICT_BLOCK_FRAMES - 1))
    time_axis = np.arange(len(x), dtype=np.float32) / float(sr)
    start_times = start_frames.astype(np.float32) * (STRICT_HOP_MS / 1000.0)

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(time_axis, x, color="black", linewidth=1.0)
    ax.axvspan(start_sample / sr, end_sample / sr, color="#fdae6b", alpha=0.35)
    ax.set_title(f"Strict course search on {example_path.stem} for /{label}/")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

    ax = axes[1]
    block = pad_or_trim_rows(frames, best_start, STRICT_BLOCK_FRAMES)
    block_spec = np.abs(np.fft.fft(block, STRICT_NFFT, axis=1))[:, : STRICT_NFFT // 2 + 1]
    block_spec = block_spec / np.maximum(block_spec.max(axis=1, keepdims=True), 1e-12)
    im = ax.imshow(
        block_spec,
        aspect="auto",
        origin="lower",
        extent=[0, STRICT_NFFT // 2, 1, STRICT_BLOCK_FRAMES],
        cmap="magma",
    )
    ax.set_title("Best-matching 4-frame block")
    ax.set_xlabel("FFT bin")
    ax.set_ylabel("Frame in block")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for j, lab in enumerate(bank["labels"]):
        ax.plot(start_times, score_traces[:, j], linewidth=2.0 if lab == label else 1.3, color=colors[j], label=f"/{lab}/")
    ax.axvline(start_times[best_idx], color="#111111", linestyle="--", linewidth=1.2)
    ax.scatter([start_times[best_idx]], [score_traces[best_idx, label_idx]], color="#111111", zorder=5)
    ax.set_title("Sliding normalized correlation by candidate block")
    ax.set_xlabel("Block start time (s)")
    ax.set_ylabel("Correlation score")
    ax.legend(ncol=4, frameon=False)
    ax.grid(alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
):
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_IMPORT_ERROR}")
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-6)
    X_train_n = ((X_train - mean) / std)[:, None, :, :].astype(np.float32)
    X_dev_n = ((X_dev - mean) / std)[:, None, :, :].astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = MultiCnn(y_train.shape[1]).to(device)
    pos = y_train.sum(0)
    neg = len(y_train) - pos
    pos_weight = np.clip(neg / np.maximum(pos, 1), 1, 10)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    best_dev_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    def predict_scores(Xn: np.ndarray):
        scores = []
        model.eval()
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xn)), batch_size=512):
                scores.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        return np.vstack(scores)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(yb)
            n += len(yb)
        dev_scores = predict_scores(X_dev_n)
        dev_thresholds = fit_thresholds(y_dev.astype(int), dev_scores)
        dev_pred = predict_with_thresholds(dev_scores, dev_thresholds)
        dev_macro_f1 = float(f1_score(y_dev, dev_pred, average="macro", zero_division=0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(n, 1),
                "dev_macro_f1": dev_macro_f1,
                "dev_label_wise_accuracy": float(1.0 - hamming_loss(y_dev, dev_pred)),
            }
        )
        if dev_macro_f1 > best_dev_macro_f1 + 1e-8:
            best_dev_macro_f1 = dev_macro_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_scores = predict_scores(X_train_n)
    dev_scores = predict_scores(X_dev_n)
    return model, train_scores, dev_scores, mean, std, pd.DataFrame(history), best_epoch, best_dev_macro_f1


def cnn_predict_scores(model, X: np.ndarray, mean: float, std: float, device: str) -> np.ndarray:
    Xn = ((X - mean) / std)[:, None, :, :].astype(np.float32)
    scores = []
    model.eval()
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xn)), batch_size=512):
            scores.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.vstack(scores)


def plot_cnn_history(history_df: pd.DataFrame, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train loss", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["dev_macro_f1"], marker="s", label="dev macro F1", color="#d62728")
    ax2.plot(history_df["epoch"], history_df["dev_label_wise_accuracy"], marker="^", label="dev label-wise acc", color="#2ca02c")
    ax2.set_ylabel("Dev score", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    ax1.set_title("CNN training trace on MSWC English")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    labels: Sequence[str],
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    example_rows: pd.DataFrame,
    dataset_counts: Dict[str, int],
    command_lines: Sequence[str],
    method_desc: str,
):
    lines = [
        "# MSWC English four-letter experiment results",
        "",
        "Generated by:",
        "",
        "```bash",
        *command_lines,
        "```",
        "",
        "## Scope",
        "",
        f"- Dataset: `MSWC English subset` with labels `{', '.join(labels)}`.",
        f"- Split sizes: train/dev/test = {dataset_counts['train']}/{dataset_counts['dev']}/{dataset_counts['test']}.",
        f"- Total files used: {dataset_counts['total']}.",
        f"- Methods: {method_desc}.",
        "",
        "## Summary metrics",
        "",
        summary_df.to_string(index=False),
        "",
        "## Per-label metrics",
        "",
        metrics_df.to_string(index=False),
        "",
        "## Example predictions",
        "",
        example_rows.to_string(index=False),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_course_template_bank(
    path: Path,
    labels: Sequence[str],
    patch_pos_templates: np.ndarray,
    patch_neg_templates: np.ndarray,
    autocorr_pos_templates: np.ndarray,
    autocorr_neg_templates: np.ndarray,
    thresholds: np.ndarray,
    *,
    exclusive_decode: bool,
):
    bank = {
        "labels": list(labels),
        "target_sr": 16000,
        "frame_ms": 25,
        "hop_ms": 10,
        "n_mels": FILTERBANK_MELS,
        "window_frames": COURSE_WINDOW_FRAMES,
        "pre_frames": COURSE_PRE_FRAMES,
        "segment_count": COURSE_SEGMENTS,
        "energy_smooth_frames": COURSE_ENERGY_SMOOTH_FRAMES,
        "onset_margin": COURSE_ONSET_MARGIN,
        "candidate_shifts": list(COURSE_CANDIDATE_SHIFTS),
        "autocorr_lags": COURSE_AUTOCORR_LAGS,
        "cross_correlation_weight": COURSE_CROSS_WEIGHT,
        "autocorrelation_weight": COURSE_AUTOCORR_WEIGHT,
        "exclusive_decode": bool(exclusive_decode),
        "patch_pos_templates": patch_pos_templates.astype(float).tolist(),
        "patch_neg_templates": patch_neg_templates.astype(float).tolist(),
        "autocorr_pos_templates": autocorr_pos_templates.astype(float).tolist(),
        "autocorr_neg_templates": autocorr_neg_templates.astype(float).tolist(),
        "patch_templates": patch_pos_templates.astype(float).tolist(),
        "spectrum_templates": patch_pos_templates.astype(float).tolist(),
        "autocorr_templates": autocorr_pos_templates.astype(float).tolist(),
        "thresholds": thresholds.astype(float).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/mswc_en_subset_600/manifest.csv"))
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--cache", type=Path, default=Path("build/mswc_course_600/features_v1.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("build/mswc_course_600"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/mswc_course_600"))
    parser.add_argument("--report-path", type=Path, default=Path("references/mswc-course-results-600.md"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cnn-batch-size", type=int, default=192)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-patience", type=int, default=5)
    parser.add_argument("--course-only", action="store_true")
    args = parser.parse_args()

    seed_all(RNG)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    items = load_manifest(args.manifest, args.labels)
    data = build_dataset(items, args.labels, args.cache)
    Y = data["Y"].astype(int)
    splits = data["splits"]
    train_idx = np.where(splits == "train")[0]
    dev_idx = np.where(splits == "dev")[0]
    test_idx = np.where(splits == "test")[0]

    X_course_tr = data["X_cnn"][train_idx]
    X_course_dev = data["X_cnn"][dev_idx]
    X_course_te = data["X_cnn"][test_idx]
    X_stat_tr, X_stat_dev, X_stat_te = data["X_stat"][train_idx], data["X_stat"][dev_idx], data["X_stat"][test_idx]
    X_cnn_tr, X_cnn_dev, X_cnn_te = data["X_cnn"][train_idx], data["X_cnn"][dev_idx], data["X_cnn"][test_idx]
    Y_tr, Y_dev, Y_te = Y[train_idx], Y[dev_idx], Y[test_idx]
    exclusive_decode = bool(np.all(Y.sum(axis=1) == 1))

    case_dir = Path(__file__).resolve().parents[1] / "cases"
    strict_bank = build_strict_course_bank(case_dir, args.labels)
    strict_dev_scores = score_strict_course_detector(data["paths"][dev_idx], strict_bank)
    strict_thresholds = fit_thresholds(Y_dev, strict_dev_scores)
    strict_te_scores = score_strict_course_detector(data["paths"][test_idx], strict_bank)
    strict_te_pred = predict_with_thresholds(strict_te_scores, strict_thresholds, exclusive=exclusive_decode)

    patch_pos_templates, patch_neg_templates, autocorr_pos_templates, autocorr_neg_templates = fit_course_detector(X_course_tr, Y_tr)
    course_dev_scores = score_course_detector(
        X_course_dev,
        patch_pos_templates,
        patch_neg_templates,
        autocorr_pos_templates,
        autocorr_neg_templates,
    )
    course_thresholds = fit_thresholds(Y_dev, course_dev_scores)
    course_te_scores = score_course_detector(
        X_course_te,
        patch_pos_templates,
        patch_neg_templates,
        autocorr_pos_templates,
        autocorr_neg_templates,
    )
    course_te_pred = predict_with_thresholds(course_te_scores, course_thresholds, exclusive=exclusive_decode)

    method_preds = {
        "strict_course_fft": strict_te_pred,
        "course_filterbank_corr": course_te_pred,
    }
    method_scores = {
        "strict_course_fft": strict_te_scores,
        "course_filterbank_corr": course_te_scores,
    }
    threshold_table = {
        "label": list(args.labels),
        "strict_course_threshold": strict_thresholds,
        "course_threshold": course_thresholds,
    }
    method_desc_parts = [
        "strict course FFT template detector",
        "course local-window contrastive template + autocorrelation detector",
    ]
    cnn_summary = None

    if not args.course_only:
        mlp_model = fit_mlp(X_stat_tr, Y_tr)
        mlp_dev_scores = get_mlp_scores(mlp_model, X_stat_dev)
        mlp_thresholds = fit_thresholds(Y_dev, mlp_dev_scores)
        mlp_te_scores = get_mlp_scores(mlp_model, X_stat_te)
        mlp_te_pred = predict_with_thresholds(mlp_te_scores, mlp_thresholds, exclusive=exclusive_decode)
        method_preds["mlp_ai"] = mlp_te_pred
        method_scores["mlp_ai"] = mlp_te_scores
        threshold_table["mlp_threshold"] = mlp_thresholds
        method_desc_parts.append("MLP on MFCC/statistics")

        if torch is None:
            cnn_summary = {
                "status": "unavailable",
                "reason": str(TORCH_IMPORT_ERROR),
            }
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cnn_model, _, cnn_dev_scores, cnn_mean, cnn_std, cnn_history_df, best_epoch, best_dev_macro_f1 = train_cnn(
                X_cnn_tr,
                Y_tr,
                X_cnn_dev,
                Y_dev,
                device=device,
                epochs=args.epochs,
                batch_size=args.cnn_batch_size,
                lr=args.cnn_lr,
                weight_decay=args.cnn_weight_decay,
                patience=args.cnn_patience,
            )
            cnn_thresholds = fit_thresholds(Y_dev, cnn_dev_scores)
            cnn_te_scores = cnn_predict_scores(cnn_model, X_cnn_te, cnn_mean, cnn_std, device=device)
            cnn_te_pred = predict_with_thresholds(cnn_te_scores, cnn_thresholds, exclusive=exclusive_decode)
            method_preds["cnn_ai"] = cnn_te_pred
            method_scores["cnn_ai"] = cnn_te_scores
            threshold_table["cnn_threshold"] = cnn_thresholds
            method_desc_parts.append("CNN on log-mel spectrograms")
            cnn_history_df.to_csv(args.out_dir / "cnn_history.csv", index=False)
            plot_cnn_history(cnn_history_df, args.fig_dir / "cnn_history.png")
            cnn_summary = {
                "status": "ok",
                "device": device,
                "epochs_requested": args.epochs,
                "epochs_completed": int(cnn_history_df["epoch"].max()),
                "best_epoch": int(best_epoch),
                "best_dev_macro_f1": float(best_dev_macro_f1),
                "batch_size": args.cnn_batch_size,
                "lr": args.cnn_lr,
                "weight_decay": args.cnn_weight_decay,
                "patience": args.cnn_patience,
                "mean": cnn_mean,
                "std": cnn_std,
            }

    summary_rows = [summary_metrics(Y_te, pred, args.labels, method) for method, pred in method_preds.items()]
    summary_df = pd.DataFrame(summary_rows)
    metrics_df = pd.concat([label_metrics(Y_te, pred, args.labels, method) for method, pred in method_preds.items()], ignore_index=True)

    example_words = [DEFAULT_EXAMPLE_WORDS[lab] for lab in args.labels if DEFAULT_EXAMPLE_WORDS.get(lab)]
    example_indices = []
    for word in example_words:
        idxs = test_idx[data["words"][test_idx] == word]
        if len(idxs) == 0:
            continue
        example_indices.append(int(idxs[0]))
    example_rows = []
    for idx in example_indices:
        row = {
            "word": str(data["words"][idx]),
            "phones": str(data["phones"][idx]),
            "true_labels": " ".join(l for l, v in zip(args.labels, Y[idx]) if v),
        }
        local_idx = int(np.where(test_idx == idx)[0][0])
        for method, pred in method_preds.items():
            row[f"{method}_pred"] = " ".join(l for l, v in zip(args.labels, pred[local_idx]) if v) or "-"
            row[f"{method}_scores"] = ", ".join(f"{lab}:{method_scores[method][local_idx, j]:.3f}" for j, lab in enumerate(args.labels))
        example_rows.append(row)
    example_df = pd.DataFrame(example_rows)

    summary = {
        "dataset": "MSWC English subset",
        "labels": list(args.labels),
        "manifest": str(args.manifest),
        "n_files": int(len(Y)),
        "train_files": int(len(train_idx)),
        "dev_files": int(len(dev_idx)),
        "test_files": int(len(test_idx)),
        "positive_counts_test": {lab: int(Y_te[:, j].sum()) for j, lab in enumerate(args.labels)},
        "strict_course_detector": {
            "frame_ms": STRICT_FRAME_MS,
            "hop_ms": STRICT_HOP_MS,
            "block_frames": STRICT_BLOCK_FRAMES,
            "nfft": STRICT_NFFT,
            "template_words": strict_bank["template_words"],
            "template_paths": strict_bank["template_paths"],
        },
        "course_detector": {
            "n_mels": FILTERBANK_MELS,
            "autocorr_lags": COURSE_AUTOCORR_LAGS,
            "cross_correlation_weight": COURSE_CROSS_WEIGHT,
            "autocorrelation_weight": COURSE_AUTOCORR_WEIGHT,
            "window_frames": COURSE_WINDOW_FRAMES,
            "pre_frames": COURSE_PRE_FRAMES,
            "segment_count": COURSE_SEGMENTS,
            "energy_smooth_frames": COURSE_ENERGY_SMOOTH_FRAMES,
            "onset_margin": COURSE_ONSET_MARGIN,
            "candidate_shifts": list(COURSE_CANDIDATE_SHIFTS),
            "exclusive_decode": exclusive_decode,
        },
        "decode_mode": "exclusive" if exclusive_decode else "threshold",
        "methods": summary_rows,
        "cnn": cnn_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_df.to_csv(args.out_dir / "summary_metrics.csv", index=False)
    metrics_df.to_csv(args.out_dir / "per_label_metrics.csv", index=False)
    example_df.to_csv(args.out_dir / "example_predictions.csv", index=False)
    pd.DataFrame(threshold_table).to_csv(args.out_dir / "thresholds.csv", index=False)
    strict_bank_json = {
        "labels": strict_bank["labels"],
        "template_paths": strict_bank["template_paths"],
        "template_words": strict_bank["template_words"],
        "template_start_frames": strict_bank["template_start_frames"].astype(int).tolist(),
        "templates": strict_bank["templates"].astype(float).tolist(),
        "freq_hz": strict_bank["freq_hz"].astype(float).tolist(),
        "cfg": strict_bank["cfg"],
    }
    (args.out_dir / "strict_course_template_bank.json").write_text(
        json.dumps(strict_bank_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_course_template_bank(
        args.out_dir / "course_template_bank.json",
        args.labels,
        patch_pos_templates,
        patch_neg_templates,
        autocorr_pos_templates,
        autocorr_neg_templates,
        course_thresholds,
        exclusive_decode=exclusive_decode,
    )
    command_lines = [
        "cd 216/project/topic2_speech",
        ".venv/bin/python src/run_mswc_course.py \\",
        f"  --manifest {args.manifest} \\",
        f"  --labels {' '.join(args.labels)} \\",
        f"  --cache {args.cache} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --fig-dir {args.fig_dir} \\",
        f"  --report-path {args.report_path} \\",
    ]
    if args.course_only:
        command_lines.append("  --course-only \\")
    command_lines.extend(
        [
            f"  --epochs {args.epochs} \\",
            f"  --cnn-batch-size {args.cnn_batch_size} \\",
            f"  --cnn-lr {args.cnn_lr} \\",
            f"  --cnn-weight-decay {args.cnn_weight_decay} \\",
            f"  --cnn-patience {args.cnn_patience}",
        ]
    )
    write_markdown_report(
        args.report_path,
        args.labels,
        summary_df,
        metrics_df,
        example_df,
        {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx), "total": len(Y)},
        command_lines,
        ", ".join(method_desc_parts),
    )

    plot_letter_examples(data, args.labels, DEFAULT_EXAMPLE_WORDS, args.fig_dir / "letter_examples.png")
    plot_strict_template_construction(strict_bank, args.fig_dir / "strict_course_template_construction.png")
    demo_idx = build_example_rows(data, args.labels, DEFAULT_EXAMPLE_WORDS)[0]
    plot_strict_correlation_demo(
        Path(data["paths"][demo_idx]),
        strict_bank,
        args.labels[0],
        args.fig_dir / "strict_course_correlation_demo.png",
    )
    plot_template_grid(data["X_cnn"][train_idx], Y_tr, args.labels, args.fig_dir / "template_grid.png")
    plot_metrics(summary_df, args.fig_dir / "metric_summary.png")
    plot_per_label_f1(metrics_df, args.fig_dir / "per_label_f1.png")
    plot_score_distribution(strict_te_scores, Y_te, args.labels, args.fig_dir / "strict_course")
    plot_score_distribution(course_te_scores, Y_te, args.labels, args.fig_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPer-label metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
