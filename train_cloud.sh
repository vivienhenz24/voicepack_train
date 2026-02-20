#!/usr/bin/env bash
# train_cloud.sh — one-shot setup + training for a cloud GPU pod
# Usage: bash train_cloud.sh [options]
#   --epochs N        (default: 10)
#   --lr LR           (default: 0.003)
#   --voice-init V    (default: random)  use "random" for fresh LJ init or e.g. "af_heart"
#   --w-anchor W      (default: 0.0)     0 = free to move anywhere from init
#   --w-splitnorm W   (default: 0.0)     0 = don't preserve init norm profile
#   --w-dur W         (default: 0.1)     duration loss weight
#   --lang L          (default: a)
#   --out-dir DIR     (default: logs/lj_voice)
#   --device DEV      (default: cuda)
set -euo pipefail

# RunPod sets this but hf_transfer isn't in our venv
export HF_HUB_ENABLE_HF_TRANSFER=0

# ── Defaults ────────────────────────────────────────────────────────────────
EPOCHS=10
LR=0.003
VOICE_INIT="random"
W_ANCHOR=0.0
W_SPLITNORM=0.0
W_DUR=0.1
LANG="a"
OUT_DIR="logs/lj_voice"
DEVICE="cuda"

# ── CLI args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)     EPOCHS="$2";     shift 2 ;;
    --lr)         LR="$2";         shift 2 ;;
    --voice-init) VOICE_INIT="$2"; shift 2 ;;
    --w-anchor)   W_ANCHOR="$2";   shift 2 ;;
    --w-splitnorm) W_SPLITNORM="$2"; shift 2 ;;
    --w-dur)      W_DUR="$2";      shift 2 ;;
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
# seed pip into the venv so phonemizer (misaki dep) can call `python -m pip`
uv pip install pip

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
import csv, pathlib, wave

MAX_SECONDS = 10.0
data_dir = pathlib.Path("${LJSPEECH_DIR}")
out_path = pathlib.Path("${MANIFEST}")

def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()

rows = []
skipped = 0
with (data_dir / "metadata.csv").open(encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 3:
            continue
        sample_id = parts[0].strip()
        text = parts[2].strip() if parts[2].strip() else parts[1].strip()
        wav = str(data_dir / "wavs" / f"{sample_id}.wav")
        if not pathlib.Path(wav).exists() or not text:
            continue
        if wav_duration(wav) > MAX_SECONDS:
            skipped += 1
            continue
        rows.append({"id": sample_id, "target_audio": wav, "text": text})

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "target_audio", "text"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows (skipped {skipped} over {MAX_SECONDS}s) -> {out_path}")
PYEOF
else
  echo "==> Manifest already exists, skipping"
fi

# ── 5. Kick off training ─────────────────────────────────────────────────────
echo "==> Starting voice pack training..."
echo "    epochs=${EPOCHS}  lr=${LR}  voice_init=${VOICE_INIT}  device=${DEVICE}"
echo "    w_anchor=${W_ANCHOR}  w_splitnorm=${W_SPLITNORM}  w_dur=${W_DUR}"
echo "    out_dir=${OUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python3 kokoro/voice_loop.py \
  --manifest-csv  "${MANIFEST}" \
  --lang          "${LANG}" \
  --voice-init    "${VOICE_INIT}" \
  --epochs        "${EPOCHS}" \
  --lr            "${LR}" \
  --device        "${DEVICE}" \
  --out-dir       "${OUT_DIR}" \
  --w-anchor      "${W_ANCHOR}" \
  --w-splitnorm   "${W_SPLITNORM}" \
  --w-dur         "${W_DUR}" \
  --save-every-epoch \
  --log-every     50

echo ""
echo "==> Done! Trained pack: ${OUT_DIR}/voice_pack_trained.pt"
echo "==> History:            ${OUT_DIR}/history.jsonl"
echo "==> Metadata:           ${OUT_DIR}/run_metadata.json"
