#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pronouncing

RNG = 216
LABEL_MAP = {
    "B": "b",
    "D": "d",
    "DH": "dh",
    "G": "g",
    "JH": "jh",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "R": "r",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "zh",
}
LABELS = list(LABEL_MAP.values())


def load_metadata(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)["en"]


def labels_for_word(word: str):
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return None, None
    phone_str = phones[0]
    labels = []
    for token in phone_str.split():
        base = re.sub(r"\d", "", token)
        if base in LABEL_MAP:
            labels.append(LABEL_MAP[base])
    labels = sorted(set(labels))
    if not labels:
        return None, None
    return phone_str, labels


def load_split_rows(path: Path):
    counts = defaultdict(Counter)
    rows_by_word = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["VALID"] != "True":
                continue
            word = row["WORD"].strip().lower()
            split = row["SET"].strip().lower()
            item = {
                "split": split,
                "link": row["LINK"].strip(),
                "word": word,
                "speaker": row["SPEAKER"].strip(),
                "gender": row["GENDER"].strip(),
            }
            counts[word][split] += 1
            rows_by_word[word][split].append(item)
    return counts, rows_by_word


def build_candidates(metadata, counts, min_train, min_dev, min_test):
    candidates = []
    for raw_word, total in metadata["wordcounts"].items():
        word = raw_word.lower()
        if not re.fullmatch(r"[a-z]+", word):
            continue
        phone_str, labels = labels_for_word(word)
        if not labels:
            continue
        train_n = counts[word]["train"]
        dev_n = counts[word]["dev"]
        test_n = counts[word]["test"]
        if train_n < min_train or dev_n < min_dev or test_n < min_test:
            continue
        candidates.append(
            {
                "word": word,
                "phones": phone_str,
                "labels": labels,
                "total": int(train_n + dev_n + test_n),
                "train": int(train_n),
                "dev": int(dev_n),
                "test": int(test_n),
            }
        )
    candidates.sort(key=lambda x: (-x["total"], x["word"]))
    return candidates


def select_words(candidates, target_words, seed_words_per_label=4):
    by_label = defaultdict(list)
    for cand in candidates:
        for label in cand["labels"]:
            by_label[label].append(cand)
    for label in by_label:
        by_label[label].sort(key=lambda x: (-x["total"], x["word"]))

    selected = []
    seen = set()
    label_counts = Counter()

    for label in sorted(LABELS, key=lambda x: len(by_label[x])):
        need = seed_words_per_label
        for cand in by_label[label]:
            if cand["word"] in seen:
                continue
            selected.append(cand)
            seen.add(cand["word"])
            label_counts.update(cand["labels"])
            need -= 1
            if need == 0 or len(selected) >= target_words:
                break
        if len(selected) >= target_words:
            break

    while len(selected) < target_words:
        best = None
        best_score = None
        for cand in candidates:
            if cand["word"] in seen:
                continue
            balance = sum(1.0 / (1.0 + label_counts[label]) for label in cand["labels"])
            score = (
                math.log1p(cand["total"]) + 2.5 * balance + 0.2 * len(cand["labels"]),
                balance,
                len(cand["labels"]),
                -len(cand["word"]),
                cand["word"],
            )
            if best is None or score > best_score:
                best = cand
                best_score = score
        if best is None:
            break
        selected.append(best)
        seen.add(best["word"])
        label_counts.update(best["labels"])

    return selected


def choose_rows(selected_words, rows_by_word, train_per_word, dev_per_word, test_per_word, seed):
    rng = random.Random(seed)
    quotas = {"train": train_per_word, "dev": dev_per_word, "test": test_per_word}
    chosen = []
    for cand in selected_words:
        word = cand["word"]
        for split, quota in quotas.items():
            rows = list(rows_by_word[word][split])
            rng.shuffle(rows)
            if len(rows) < quota:
                raise RuntimeError(f"{word} has only {len(rows)} {split} rows, need {quota}")
            chosen.extend(rows[:quota])
    return chosen


def build_manifest(rows, selected_lookup, out_root: Path):
    manifest = []
    for row in rows:
        word = row["word"]
        stem = Path(row["link"]).stem
        speaker = row["speaker"]
        wav_name = f"{speaker}_{stem}.wav"
        rel_path = Path(word) / wav_name
        info = selected_lookup[word]
        manifest.append(
            {
                "split": row["split"],
                "word": word,
                "speaker": speaker,
                "gender": row["gender"],
                "link": row["link"],
                "phones": info["phones"],
                "labels": " ".join(info["labels"]),
                "wav_relpath": str(rel_path),
                "wav_path": str(out_root / rel_path),
            }
        )
    manifest.sort(key=lambda x: (x["word"], x["split"], x["link"]))
    return manifest


def member_key(name: str):
    parts = name.strip("/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return name.strip("/")


def convert_opus_to_wav(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def extract_and_convert(audio_tar: Path, manifest, tmp_root: Path):
    wanted = {item["link"]: item for item in manifest}
    pending = {link for link, item in wanted.items() if not Path(item["wav_path"]).exists()}
    if not pending:
        return 0

    extracted = 0
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(audio_tar, "r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key = member.name if member.name in wanted else member_key(member.name)
            if key not in pending:
                continue
            item = wanted[key]
            dst = Path(item["wav_path"])
            fileobj = tf.extractfile(member)
            if fileobj is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=".opus", dir=tmp_root, delete=False) as tmp:
                shutil.copyfileobj(fileobj, tmp)
                tmp_path = Path(tmp.name)
            try:
                convert_opus_to_wav(tmp_path, dst)
            finally:
                tmp_path.unlink(missing_ok=True)
            pending.remove(key)
            extracted += 1
            if not pending:
                break
    if pending:
        missing = sorted(pending)[:20]
        raise RuntimeError(f"Missing {len(pending)} selected members in tar, e.g. {missing}")
    return extracted


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/mswc_en"))
    ap.add_argument("--out-root", type=Path, default=Path("data/raw/mswc_en_subset_600"))
    ap.add_argument("--target-words", type=int, default=600)
    ap.add_argument("--train-per-word", type=int, default=40)
    ap.add_argument("--dev-per-word", type=int, default=5)
    ap.add_argument("--test-per-word", type=int, default=5)
    ap.add_argument("--seed-words-per-label", type=int, default=20)
    ap.add_argument("--seed", type=int, default=RNG)
    args = ap.parse_args()

    metadata_path = args.raw_root / "metadata.json.gz"
    splits_path = args.raw_root / "en_splits.csv"
    audio_tar = args.raw_root / "audio_en.tar.gz"
    if not metadata_path.exists() or not splits_path.exists() or not audio_tar.exists():
        raise FileNotFoundError("MSWC raw files are missing. Run src/download_mswc_en.sh first.")

    metadata = load_metadata(metadata_path)
    counts, rows_by_word = load_split_rows(splits_path)
    candidates = build_candidates(
        metadata,
        counts,
        min_train=args.train_per_word,
        min_dev=args.dev_per_word,
        min_test=args.test_per_word,
    )
    selected = select_words(
        candidates,
        args.target_words,
        seed_words_per_label=args.seed_words_per_label,
    )
    if len(selected) < args.target_words:
        raise RuntimeError(f"Only found {len(selected)} words, fewer than requested {args.target_words}")
    selected_lookup = {item["word"]: item for item in selected}
    chosen_rows = choose_rows(
        selected,
        rows_by_word,
        train_per_word=args.train_per_word,
        dev_per_word=args.dev_per_word,
        test_per_word=args.test_per_word,
        seed=args.seed,
    )
    manifest = build_manifest(chosen_rows, selected_lookup, args.out_root)

    args.out_root.mkdir(parents=True, exist_ok=True)
    word_rows = []
    label_counter = Counter()
    for item in selected:
        label_counter.update(item["labels"])
        word_rows.append(
            {
                "word": item["word"],
                "phones": item["phones"],
                "labels": " ".join(item["labels"]),
                "available_train": item["train"],
                "available_dev": item["dev"],
                "available_test": item["test"],
                "selected_train": args.train_per_word,
                "selected_dev": args.dev_per_word,
                "selected_test": args.test_per_word,
                "selected_total": args.train_per_word + args.dev_per_word + args.test_per_word,
            }
        )
    write_csv(
        args.out_root / "manifest.csv",
        manifest,
        ["split", "word", "speaker", "gender", "link", "phones", "labels", "wav_relpath", "wav_path"],
    )
    write_csv(
        args.out_root / "word_summary.csv",
        word_rows,
        [
            "word",
            "phones",
            "labels",
            "available_train",
            "available_dev",
            "available_test",
            "selected_train",
            "selected_dev",
            "selected_test",
            "selected_total",
        ],
    )

    summary = {
        "dataset": "MSWC English subset",
        "source": "https://mlcommons.org/datasets/multilingual-spoken-words/",
        "labels": LABELS,
        "n_words": len(selected),
        "n_files": len(manifest),
        "train_per_word": args.train_per_word,
        "dev_per_word": args.dev_per_word,
        "test_per_word": args.test_per_word,
        "seed_words_per_label": args.seed_words_per_label,
        "label_word_counts": dict(sorted(label_counter.items())),
    }
    (args.out_root / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    extracted = extract_and_convert(audio_tar, manifest, args.out_root / ".tmp")
    print(json.dumps({**summary, "newly_extracted": extracted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
