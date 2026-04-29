#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_mswc_course import (
    DEFAULT_LABELS,
    RNG,
    build_dataset,
    cnn_predict_scores,
    fit_thresholds,
    load_manifest,
    predict_with_thresholds,
    seed_all,
    summary_metrics,
    torch,
    train_cnn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep CNN max epochs for the MSWC course experiment.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--fig-path", type=Path, default=None)
    parser.add_argument("--epochs-list", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30, 40])
    parser.add_argument("--cnn-batch-size", type=int, default=192)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-patience", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch is None:
        raise RuntimeError("PyTorch is unavailable; CNN epoch sweep cannot run.")

    items = load_manifest(args.manifest, args.labels)
    data = build_dataset(items, args.labels, args.cache)
    y = data["Y"].astype(int)
    splits = data["splits"]
    train_idx = splits == "train"
    dev_idx = splits == "dev"
    test_idx = splits == "test"
    x_train, x_dev, x_test = data["X_cnn"][train_idx], data["X_cnn"][dev_idx], data["X_cnn"][test_idx]
    y_train, y_dev, y_test = y[train_idx], y[dev_idx], y[test_idx]
    exclusive_decode = bool((y.sum(axis=1) == 1).all())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for max_epochs in args.epochs_list:
        seed_all(RNG)
        model, _, dev_scores, mean, std, history_df, best_epoch, best_dev_macro_f1 = train_cnn(
            x_train,
            y_train,
            x_dev,
            y_dev,
            device=device,
            epochs=max_epochs,
            batch_size=args.cnn_batch_size,
            lr=args.cnn_lr,
            weight_decay=args.cnn_weight_decay,
            patience=args.cnn_patience,
        )
        thresholds = fit_thresholds(y_dev, dev_scores)
        test_scores = cnn_predict_scores(model, x_test, mean, std, device=device)
        test_pred = predict_with_thresholds(test_scores, thresholds, exclusive=exclusive_decode)
        metrics = summary_metrics(y_test, test_pred, args.labels, "cnn_ai")
        rows.append(
            {
                "max_epochs": int(max_epochs),
                "epochs_completed": int(history_df["epoch"].max()),
                "best_epoch": int(best_epoch),
                "best_dev_macro_f1": float(best_dev_macro_f1),
                "test_label_wise_accuracy": float(metrics["label_wise_accuracy"]),
                "test_exact_match_accuracy": float(metrics["exact_match_accuracy"]),
                "test_macro_f1": float(metrics["macro_f1"]),
                "test_micro_f1": float(metrics["micro_f1"]),
                "test_samples_f1": float(metrics["samples_f1"]),
                "device": device,
                "batch_size": args.cnn_batch_size,
                "lr": args.cnn_lr,
                "weight_decay": args.cnn_weight_decay,
                "patience": args.cnn_patience,
            }
        )

    result_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.out_csv, index=False)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fig_path:
        args.fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(result_df["max_epochs"], result_df["best_dev_macro_f1"], marker="o", label="best dev macro-F1")
        ax.plot(result_df["max_epochs"], result_df["test_macro_f1"], marker="s", label="test macro-F1")
        ax.plot(result_df["max_epochs"], result_df["test_exact_match_accuracy"], marker="^", label="test exact-match")
        for _, row in result_df.iterrows():
            ax.annotate(
                f"best {int(row['best_epoch'])}",
                (row["max_epochs"], row["best_dev_macro_f1"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel("Maximum epochs")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(args.fig_path, dpi=220)
        plt.close(fig)

    print(result_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
