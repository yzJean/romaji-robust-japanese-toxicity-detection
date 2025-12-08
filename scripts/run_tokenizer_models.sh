#!/usr/bin/env bash
set -eu

# Launcher for tokenizer-model experiments (BERT Japanese, mDeBERTa)
# By default this runs in quick-test mode to verify the pipeline quickly.
# To run full experiments set FULL=1 in the environment.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="python3"
TRAIN_SCRIPT="$ROOT_DIR/src/train.py"

# Default behavior: quick-test (fast, 50 samples, 1 epoch)
FULL=${FULL:-0}
DETERMINISTIC=${DETERMINISTIC:-0}
SEED=${SEED:-42}

if [ "$FULL" -eq 1 ]; then
  QUICK_FLAG=""
  EPOCHS_FLAG="--epochs 10"
  BATCH_FLAG="--batch-size 16"
else
  QUICK_FLAG="--quick-test"
  EPOCHS_FLAG=""
  BATCH_FLAG=""
fi

# Data path used in your examples
DATA="data/processed/paired_native_romaji_llmjp_binary.csv"

runs=(
  "--model-type mdeberta --data-path $DATA"
  "--model-type mdeberta --data-path $DATA --use-romaji"
  "--model-type bert-japanese --data-path $DATA"
  "--model-type bert-japanese --data-path $DATA --use-romaji"
)

OUTDIR_BASE="$ROOT_DIR/outputs/tokenizer_runs"
mkdir -p "$OUTDIR_BASE"

echo "Starting tokenizer model runs (FULL=$FULL)"

i=0
for cfg in "${runs[@]}"; do
  i=$((i+1))
  # Build a safe name for the run
  safe=$(echo "$cfg" | sed -E 's/[^a-zA-Z0-9]+/_/g' | sed -E 's/__+/_/g' | sed -E 's/^_+|_+$//g')
  outdir="$OUTDIR_BASE/run_${i}_${safe}"
  mkdir -p "$outdir"

  cmd=("$PYTHON" "$TRAIN_SCRIPT")
  # Split cfg into array
  read -r -a parts <<< "$cfg"
  for p in "${parts[@]}"; do
    cmd+=("$p")
  done
  # Append quick/full flags
  if [ -n "$QUICK_FLAG" ]; then
    # split QUICK_FLAG into tokens (safe even if it's a single token)
    read -r -a qparts <<< "$QUICK_FLAG"
    for qp in "${qparts[@]}"; do
      cmd+=("$qp")
    done
  fi
  if [ -n "$EPOCHS_FLAG" ]; then
    read -r -a eparts <<< "$EPOCHS_FLAG"
    for ep in "${eparts[@]}"; do
      cmd+=("$ep")
    done
  fi
  if [ -n "$BATCH_FLAG" ]; then
    read -r -a bparts <<< "$BATCH_FLAG"
    for bp in "${bparts[@]}"; do
      cmd+=("$bp")
    done
  fi
    # Append deterministic/seed flags if requested
    if [ "$DETERMINISTIC" -eq 1 ]; then
      # Ensure cuBLAS workspace is configured for deterministic ops
      export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-":4096:8"}
      cmd+=("--deterministic")
    fi
    if [ -n "$SEED" ]; then
      cmd+=("--seed" "$SEED")
    fi
  cmd+=("--output-dir" "$outdir")

  echo "\n=== Run $i: ${cmd[*]} ==="
  # Run and stream output to both console and logfile
  logfile="$outdir/run.log"
  ("${cmd[@]}" 2>&1) | tee "$logfile"
  echo "Run $i finished, logs: $logfile"
done

echo "All runs finished. Outputs in $OUTDIR_BASE"
