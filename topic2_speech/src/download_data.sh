#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data/raw"
ZIP="$ROOT/data/raw/mini_speech_commands.zip"
URL="https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
if [[ ! -f "$ZIP" ]]; then
  curl -L --fail "$URL" -o "$ZIP"
fi
if [[ ! -d "$ROOT/data/raw/mini_speech_commands" ]]; then
  python - "$ZIP" "$ROOT/data/raw" <<'PY'
import sys, zipfile
zip_path, out_dir = sys.argv[1:3]
with zipfile.ZipFile(zip_path) as z:
    z.extractall(out_dir)
PY
fi
