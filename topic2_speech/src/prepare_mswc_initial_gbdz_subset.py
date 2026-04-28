#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
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
TARGET_LABELS = ("g", "b", "d", "z")
VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
    "AX",
    "AXR",
    "IX",
    "UX",
}


def strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone)


def load_metadata(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)["en"]


def load_split_rows(path: Path):
    counts = defaultdict(Counter)
    rows_by_word = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
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


def first_consonant_label(word: str):
    if not word:
        return None, None
    first_letter = word[0].lower()
    if first_letter not in TARGET_LABELS:
        return None, None
    pronunciations = pronouncing.phones_for_word(word)
    if not pronunciations:
        return None, None
    phone_str = pronunciations[0]
    base_phones = [strip_stress(token) for token in phone_str.split()]
    for phone in base_phones:
        if phone not in VOWELS:
            if phone == first_letter.upper():
                return first_letter, phone_str
            return None, phone_str
    return None, phone_str


def build_candidates(metadata: dict, counts: dict, min_train: int, min_dev: int, min_test: int):
    initial_match_counts = Counter()
    quota_eligible_counts = Counter()
    candidates_by_label = defaultdict(list)
    for raw_word, _ in metadata["wordcounts"].items():
        word = raw_word.lower()
        if not re.fullmatch(r"[a-z]+", word):
            continue
        label, phone_str = first_consonant_label(word)
        if label is None:
            continue
        train_n = int(counts[word]["train"])
        dev_n = int(counts[word]["dev"])
        test_n = int(counts[word]["test"])
        initial_match_counts[label] += 1
        item = {
            "label": label,
            "word": word,
            "phones": phone_str,
            "available_train": train_n,
            "available_dev": dev_n,
            "available_test": test_n,
            "available_total": train_n + dev_n + test_n,
        }
        if train_n >= min_train and dev_n >= min_dev and test_n >= min_test:
            quota_eligible_counts[label] += 1
            candidates_by_label[label].append(item)
    for label in TARGET_LABELS:
        candidates_by_label[label].sort(key=lambda x: (-x["available_total"], x["word"]))
    return candidates_by_label, initial_match_counts, quota_eligible_counts


def select_words(candidates_by_label: dict, words_per_label: int):
    available = {label: len(candidates_by_label[label]) for label in TARGET_LABELS}
    if words_per_label <= 0:
        words_per_label = min(available.values())
    for label, count in available.items():
        if count < words_per_label:
            raise ValueError(
                f"Label '{label}' has only {count} eligible words, cannot select {words_per_label}."
            )
    selected = []
    for label in TARGET_LABELS:
        selected.extend(candidates_by_label[label][:words_per_label])
    return selected, words_per_label


def choose_rows(selected_words, rows_by_word, train_per_word: int, dev_per_word: int, test_per_word: int, seed: int):
    rng = random.Random(seed)
    quotas = {"train": train_per_word, "dev": dev_per_word, "test": test_per_word}
    chosen = []
    for item in selected_words:
        word = item["word"]
        for split, quota in quotas.items():
            pool = list(rows_by_word[word][split])
            if len(pool) < quota:
                raise ValueError(f"Word '{word}' has only {len(pool)} rows in split '{split}', need {quota}.")
            rng.shuffle(pool)
            for row in pool[:quota]:
                picked = dict(row)
                picked["label"] = item["label"]
                picked["phones"] = item["phones"]
                chosen.append(picked)
    return chosen


def build_manifest(chosen_rows, selected_lookup: dict, out_root: Path):
    manifest = []
    for row in chosen_rows:
        word = row["word"]
        stem = Path(row["link"]).stem
        speaker = row["speaker"]
        wav_name = f"{speaker}_{stem}.wav"
        rel_path = Path(word) / wav_name
        info = selected_lookup[word]
        manifest.append(
            {
                "split": row["split"],
                "label": info["label"],
                "word": word,
                "speaker": speaker,
                "gender": row["gender"],
                "link": row["link"],
                "phones": info["phones"],
                "wav_relpath": str(rel_path),
                "wav_path": str(out_root / rel_path),
            }
        )
    manifest.sort(key=lambda x: (x["label"], x["word"], x["split"], x["link"]))
    return manifest


def member_key(name: str) -> str:
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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build a balanced MSWC subset whose initial voiced consonant is one of g/b/d/z."
    )
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/mswc_en"))
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/raw/mswc_en_gbdz_initial_balanced_40_5_5"),
    )
    ap.add_argument("--train-per-word", type=int, default=40)
    ap.add_argument("--dev-per-word", type=int, default=5)
    ap.add_argument("--test-per-word", type=int, default=5)
    ap.add_argument(
        "--words-per-label",
        type=int,
        default=0,
        help="How many words to keep per label. Use 0 to auto-select the max balanced value.",
    )
    ap.add_argument("--seed", type=int, default=RNG)
    ap.add_argument("--tmp-root", type=Path, default=Path("data/tmp_extract_gbdz_initial"))
    return ap.parse_args()


def main():
    args = parse_args()
    metadata_path = args.raw_root / "metadata.json.gz"
    splits_path = args.raw_root / "en_splits.csv"
    audio_tar = args.raw_root / "audio_en.tar.gz"
    if not metadata_path.exists() or not splits_path.exists() or not audio_tar.exists():
        raise FileNotFoundError("MSWC raw files are missing. Run src/download_mswc_en.sh first.")

    metadata = load_metadata(metadata_path)
    counts, rows_by_word = load_split_rows(splits_path)
    candidates_by_label, initial_match_counts, quota_eligible_counts = build_candidates(
        metadata,
        counts,
        min_train=args.train_per_word,
        min_dev=args.dev_per_word,
        min_test=args.test_per_word,
    )
    selected_words, words_per_label = select_words(candidates_by_label, args.words_per_label)
    selected_lookup = {item["word"]: item for item in selected_words}
    chosen_rows = choose_rows(
        selected_words,
        rows_by_word,
        train_per_word=args.train_per_word,
        dev_per_word=args.dev_per_word,
        test_per_word=args.test_per_word,
        seed=args.seed,
    )
    manifest = build_manifest(chosen_rows, selected_lookup, args.out_root)

    args.out_root.mkdir(parents=True, exist_ok=True)

    pool_rows = []
    for label in TARGET_LABELS:
        for item in candidates_by_label[label]:
            pool_rows.append(item)

    word_rows = []
    label_summary_rows = []
    selected_files_by_label = Counter()
    selected_words_by_label = Counter()
    selected_words_list = defaultdict(list)
    for item in selected_words:
        selected_words_by_label[item["label"]] += 1
        selected_words_list[item["label"]].append(item["word"])
    for row in manifest:
        selected_files_by_label[row["label"]] += 1
    for label in TARGET_LABELS:
        for item in selected_words:
            if item["label"] != label:
                continue
            word_rows.append(
                {
                    "label": label,
                    "word": item["word"],
                    "phones": item["phones"],
                    "available_train": item["available_train"],
                    "available_dev": item["available_dev"],
                    "available_test": item["available_test"],
                    "available_total": item["available_total"],
                    "selected_train": args.train_per_word,
                    "selected_dev": args.dev_per_word,
                    "selected_test": args.test_per_word,
                    "selected_total": args.train_per_word + args.dev_per_word + args.test_per_word,
                }
            )
        label_summary_rows.append(
            {
                "label": label,
                "initial_match_words": int(initial_match_counts[label]),
                "quota_eligible_words": int(quota_eligible_counts[label]),
                "selected_words": int(selected_words_by_label[label]),
                "selected_train_samples": int(selected_words_by_label[label] * args.train_per_word),
                "selected_dev_samples": int(selected_words_by_label[label] * args.dev_per_word),
                "selected_test_samples": int(selected_words_by_label[label] * args.test_per_word),
                "selected_total_samples": int(selected_files_by_label[label]),
            }
        )

    write_csv(
        args.out_root / "eligible_word_pool.csv",
        pool_rows,
        [
            "label",
            "word",
            "phones",
            "available_train",
            "available_dev",
            "available_test",
            "available_total",
        ],
    )
    write_csv(
        args.out_root / "manifest.csv",
        manifest,
        ["split", "label", "word", "speaker", "gender", "link", "phones", "wav_relpath", "wav_path"],
    )
    write_csv(
        args.out_root / "word_summary.csv",
        word_rows,
        [
            "label",
            "word",
            "phones",
            "available_train",
            "available_dev",
            "available_test",
            "available_total",
            "selected_train",
            "selected_dev",
            "selected_test",
            "selected_total",
        ],
    )
    write_csv(
        args.out_root / "label_summary.csv",
        label_summary_rows,
        [
            "label",
            "initial_match_words",
            "quota_eligible_words",
            "selected_words",
            "selected_train_samples",
            "selected_dev_samples",
            "selected_test_samples",
            "selected_total_samples",
        ],
    )

    summary = {
        "dataset": "MSWC English initial-consonant subset",
        "criterion": {
            "type": "initial-consonant-match",
            "description": "Keep a word only if its first letter is g/b/d/z and the first non-vowel ARPABET phone in pronouncing/CMUdict is exactly G/B/D/Z.",
        },
        "labels": list(TARGET_LABELS),
        "raw_root": str(args.raw_root),
        "out_root": str(args.out_root),
        "train_per_word": args.train_per_word,
        "dev_per_word": args.dev_per_word,
        "test_per_word": args.test_per_word,
        "words_per_label": words_per_label,
        "seed": args.seed,
        "initial_match_words_by_label": {label: int(initial_match_counts[label]) for label in TARGET_LABELS},
        "quota_eligible_words_by_label": {label: int(quota_eligible_counts[label]) for label in TARGET_LABELS},
        "selected_words_by_label": {label: int(selected_words_by_label[label]) for label in TARGET_LABELS},
        "selected_files_by_label": {label: int(selected_files_by_label[label]) for label in TARGET_LABELS},
        "selected_words": {label: selected_words_list[label] for label in TARGET_LABELS},
        "n_words": len(selected_words),
        "n_files": len(manifest),
    }
    with (args.out_root / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    extracted = extract_and_convert(audio_tar, manifest, args.tmp_root)
    print(json.dumps({"out_root": str(args.out_root), "n_words": len(selected_words), "n_files": len(manifest), "extracted_now": extracted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
