#!/usr/bin/env bash
# One bounded teacher trial, followed by complete first-attempt evaluations.
set -euo pipefail
if [[ $# != 3 ]]; then
  printf 'Usage: %s INITIAL_TEACHER RUN_NAME OUTPUT_DIRECTORY\n' "$0" >&2
  exit 2
fi
BPX_INITIAL_TEACHER="$(realpath "$1")"
BPX_RUN_NAME="$2"
BPX_OUTPUT="$(realpath -m "$3")"
BPX_PYTHON="${BPX_PYTHON:-python}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
test -s "$BPX_INITIAL_TEACHER"
mkdir -p "$BPX_OUTPUT"
"$BPX_PYTHON" - "$BPX_RUN_NAME" <<'PY'
from pathlib import Path
import sys
name=sys.argv[1]
if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
    raise ValueError('Run name must contain only letters, digits, underscores or hyphens')
if list(Path('logs/bpx_stairs17_teacher').glob(f'*_{name}')):
    raise RuntimeError('Run name already exists; choose a new name rather than duplicating a trial')
PY
printf 'PHASE: 1500 iteration teacher trial; estimated training 35 minutes plus 10-15 minutes evaluation\n'
"$BPX_PYTHON" scripts/train.py --task bpx_stairs17_teacher --num_envs 512 \
  --init_checkpoint "$BPX_INITIAL_TEACHER" --max_iterations 1500 --seed 2026 \
  --run_name "$BPX_RUN_NAME" --headless --device cuda:0
BPX_CHECKPOINT="$("$BPX_PYTHON" - "$BPX_RUN_NAME" <<'PY'
from pathlib import Path
import sys,torch
matches=list(Path('logs/bpx_stairs17_teacher').glob(f'*_{sys.argv[1]}/model_1499.pt'))
if len(matches)!=1:
    raise RuntimeError(f'Expected exactly one final checkpoint, got {matches}')
saved=torch.load(matches[0],map_location='cpu',weights_only=False)
if saved['iter']!=1499 or 'model_state_dict' not in saved:
    raise RuntimeError('Final teacher checkpoint is invalid')
print(matches[0].resolve())
PY
)"
printf 'PHASE: 17 cm nominal and robust suite, three seeds, both directions\n'
for bpx_condition in nominal robust; do
  for bpx_direction in forward backward; do
    bpx_flags=()
    if [[ "$bpx_direction" == backward ]]; then bpx_flags+=(--backward); fi
    for bpx_seed in 2026 2027 2028; do
      bpx_prefix="$BPX_OUTPUT/${BPX_RUN_NAME}_${bpx_condition}_${bpx_direction}_${bpx_seed}"
      "$BPX_PYTHON" scripts/evaluate_stairs.py --task bpx_stairs17_teacher --checkpoint "$BPX_CHECKPOINT" \
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
    "$BPX_PYTHON" scripts/evaluate_stairs.py --task bpx_stairs17_teacher --checkpoint "$BPX_CHECKPOINT" \
      --step_height "$bpx_height" --num_envs 64 --speed 0.30 --seed 2026 --condition nominal \
      --output "${bpx_prefix}.json" --device cuda:0 "${bpx_flags[@]}" > "${bpx_prefix}.log" 2>&1
    test -s "${bpx_prefix}.json"
    tail -n 2 "${bpx_prefix}.log"
  done
done
"$BPX_PYTHON" scripts/summarize_stair_suite.py --directory "$BPX_OUTPUT" --prefix "$BPX_RUN_NAME" \
  --height 0.17 --output "$BPX_OUTPUT/${BPX_RUN_NAME}_qualification.json"
printf 'PHASE: trial and evaluations complete; inspect qualification and regression before blind distillation\n'
