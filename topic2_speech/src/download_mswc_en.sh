#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw/mswc_en"
mkdir -p "$RAW"

download() {
  local url="$1"
  local out="$2"
  local done_flag="${out}.complete"
  if [[ -f "$done_flag" ]]; then
    echo "[skip] $out"
    return 0
  fi
  echo "[download/resume] $url -> $out"
  curl -L --http1.1 -C - --fail --retry 10 "$url" -o "$out"
  touch "$done_flag"
}

download "https://mswc.mlcommons-storage.org/metadata.json.gz" "$RAW/metadata.json.gz"
download "https://mswc.mlcommons-storage.org/splits/en.tar.gz" "$RAW/splits_en.tar.gz"
download "https://mswc.mlcommons-storage.org/audio/en.tar.gz" "$RAW/audio_en.tar.gz"

if [[ ! -f "$RAW/en_splits.csv" ]]; then
  tar -xzf "$RAW/splits_en.tar.gz" -C "$RAW" \
    en_splits.csv en_train.csv en_dev.csv en_test.csv version.txt
fi

echo "[done] MSWC English raw files are under $RAW"
