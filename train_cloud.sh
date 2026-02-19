#!/usr/bin/env bash
# train_cloud.sh — one-shot setup + training for a cloud GPU pod
# Usage: bash train_cloud.sh [options]
#   --epochs N        (default: 10)
#   --lr LR           (default: 0.005)
#   --voice-init V    (default: af_heart)
#   --lang L          (default: a)
#   --out-dir DIR     (default: logs/voice_loop)
#   --device DEV      (default: cuda)
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
EPOCHS=10
LR=0.005
VOICE_INIT="af_heart"
LANG="a"
OUT_DIR="logs/voice_loop"
DEVICE="cuda"

# ── CLI args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)     EPOCHS="$2";     shift 2 ;;
    --lr)         LR="$2";         shift 2 ;;
    --voice-init) VOICE_INIT="$2"; shift 2 ;;
    --lang)       LANG="$2";       shift 2 ;;
    --out-dir)    OUT_DIR="$2";    shift 2 ;;
    --device)     DEVICE="$2";     shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${REPO_DIR}/data"
LJSPEECH_DIR="${DATA_ROOT}/LJSpeech-1.1"
MANIFEST="${DATA_ROOT}/ljspeech_manifest.csv"

echo "==> repo:   ${REPO_DIR}"
echo "==> device: ${DEVICE}  epochs: ${EPOCHS}  lr: ${LR}"

# ── 1. Install uv ────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi
echo "==> uv $(uv --version)"

# ── 2. Install Python deps ───────────────────────────────────────────────────
cd "${REPO_DIR}"
echo "==> Installing dependencies..."
uv sync

# ── 3. Download LJSpeech-1.1 ────────────────────────────────────────────────
mkdir -p "${DATA_ROOT}"
if [ ! -d "${LJSPEECH_DIR}" ]; then
  echo "==> Downloading LJSpeech-1.1 (~2.6 GB)..."
  ARCHIVE="/tmp/LJSpeech-1.1.tar.bz2"
  wget -q --show-progress \
    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2" \
    -O "${ARCHIVE}"
  echo "==> Extracting..."
  tar -xjf "${ARCHIVE}" -C "${DATA_ROOT}"
  rm -f "${ARCHIVE}"
  echo "==> LJSpeech extracted to ${LJSPEECH_DIR}"
else
  echo "==> LJSpeech already present, skipping download"
fi

# ── 4. Build manifest CSV ────────────────────────────────────────────────────
if [ ! -f "${MANIFEST}" ]; then
  echo "==> Building manifest CSV..."
  uv run python3 - <<PYEOF
import csv, pathlib

data_dir = pathlib.Path("${LJSPEECH_DIR}")
out_path = pathlib.Path("${MANIFEST}")

rows = []
with (data_dir / "metadata.csv").open(encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 3:
            continue
        sample_id = parts[0].strip()
        # field 2 is the normalised transcript
        text = parts[2].strip() if parts[2].strip() else parts[1].strip()
        wav = str(data_dir / "wavs" / f"{sample_id}.wav")
        if pathlib.Path(wav).exists() and text:
            rows.append({"id": sample_id, "target_audio": wav, "text": text})

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "target_audio", "text"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {out_path}")
PYEOF
else
  echo "==> Manifest already exists, skipping"
fi

# ── 5. Kick off training ─────────────────────────────────────────────────────
echo "==> Starting voice pack training..."
echo "    epochs=${EPOCHS}  lr=${LR}  voice_init=${VOICE_INIT}  device=${DEVICE}"
echo "    out_dir=${OUT_DIR}"

uv run python3 kokoro/voice_loop.py \
  --manifest-csv "${MANIFEST}" \
  --lang         "${LANG}" \
  --voice-init   "${VOICE_INIT}" \
  --epochs       "${EPOCHS}" \
  --lr           "${LR}" \
  --device       "${DEVICE}" \
  --out-dir      "${OUT_DIR}" \
  --save-every-epoch \
  --log-every    50

echo ""
echo "==> Done! Trained pack: ${OUT_DIR}/voice_pack_trained.pt"
echo "==> History:            ${OUT_DIR}/history.jsonl"
echo "==> Metadata:           ${OUT_DIR}/run_metadata.json"
