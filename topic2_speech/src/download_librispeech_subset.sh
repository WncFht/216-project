#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw/librispeech"
mkdir -p "$RAW"
cd "$RAW"
# Mini LibriSpeech is enough for this course project and less brittle than
# short command words. It contains read-speech utterances with transcripts.
for name in dev-clean-2 train-clean-5; do
  if [[ ! -d "LibriSpeech/${name}" ]]; then
    if [[ ! -f "${name}.tar.gz" ]]; then
      curl -L --fail "https://www.openslr.org/resources/31/${name}.tar.gz" -o "${name}.tar.gz"
    fi
    tar -xzf "${name}.tar.gz"
  fi
done
