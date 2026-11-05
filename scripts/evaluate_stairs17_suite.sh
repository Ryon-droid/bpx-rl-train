#!/usr/bin/env bash
# Evaluate one existing 17 cm teacher or blind checkpoint; never starts training.
set -euo pipefail
if [[ $# != 3 ]]; then
  printf 'Usage: %s CHECKPOINT RUN_NAME OUTPUT_DIRECTORY\n' "$0" >&2
  exit 2
fi
BPX_CHECKPOINT="$(realpath "$1")"
BPX_RUN_NAME="$2"
BPX_OUTPUT="$(realpath -m "$3")"
BPX_PYTHON="${BPX_PYTHON:-python}"
BPX_TASK="${BPX_TASK:-bpx_stairs17_teacher}"
case "$BPX_TASK" in
  bpx_stairs17_teacher|bpx_stairs17_blind) ;;
  *) printf 'Task must be bpx_stairs17_teacher or bpx_stairs17_blind\n' >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
test -s "$BPX_CHECKPOINT"
mkdir -p "$BPX_OUTPUT"
"$BPX_PYTHON" - "$BPX_OUTPUT" "$BPX_RUN_NAME" <<'PYCHECK'
from pathlib import Path
import sys
output, name = Path(sys.argv[1]), sys.argv[2]
if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
    raise ValueError('Run name must contain only letters, digits, underscores or hyphens')
if list(output.glob(f'{name}_*')):
    raise RuntimeError('Evaluation prefix already exists; choose a new name or output directory')
PYCHECK
export PYTHONUNBUFFERED=1
printf 'PHASE: 17 cm nominal and robust suite, three seeds, both directions\n'
for bpx_condition in nominal robust; do
  for bpx_direction in forward backward; do
    bpx_flags=()
    if [[ "$bpx_direction" == backward ]]; then bpx_flags+=(--backward); fi
    for bpx_seed in 2026 2027 2028; do
      bpx_prefix="$BPX_OUTPUT/${BPX_RUN_NAME}_${bpx_condition}_${bpx_direction}_${bpx_seed}"
      "$BPX_PYTHON" scripts/evaluate_stairs.py --task "$BPX_TASK" --checkpoint "$BPX_CHECKPOINT" \
        --step_height 0.17 --num_envs 64 --speed 0.30 --seed "$bpx_seed" --condition "$bpx_condition" \
        --output "${bpx_prefix}.json" --device cuda:0 "${bpx_flags[@]}" > "${bpx_prefix}.log" 2>&1
      test -s "${bpx_prefix}.json"
      tail -n 2 "${bpx_prefix}.log"
    done
  done
done
printf 'PHASE: 8/12/15 cm regression, both directions\n'
for bpx_height in 0.08 0.12 0.15; do
  for bpx_direction in forward backward; do
    bpx_flags=()
    if [[ "$bpx_direction" == backward ]]; then bpx_flags+=(--backward); fi
    bpx_prefix="$BPX_OUTPUT/${BPX_RUN_NAME}_regression_${bpx_height}_${bpx_direction}"
    "$BPX_PYTHON" scripts/evaluate_stairs.py --task "$BPX_TASK" --checkpoint "$BPX_CHECKPOINT" \
      --step_height "$bpx_height" --num_envs 64 --speed 0.30 --seed 2026 --condition nominal \
      --output "${bpx_prefix}.json" --device cuda:0 "${bpx_flags[@]}" > "${bpx_prefix}.log" 2>&1
    test -s "${bpx_prefix}.json"
    tail -n 2 "${bpx_prefix}.log"
  done
done
"$BPX_PYTHON" scripts/summarize_stair_suite.py --directory "$BPX_OUTPUT" --prefix "$BPX_RUN_NAME" \
  --height 0.17 --output "$BPX_OUTPUT/${BPX_RUN_NAME}_qualification.json"
printf 'PHASE: evaluations complete; inspect qualification and regression\n'
