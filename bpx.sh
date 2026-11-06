#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  ./bpx.sh setup [environment-name] [--skip-contract]
  ./bpx.sh install
  ./bpx.sh contract [Isaac Lab args]
  ./bpx.sh train [train args]
  ./bpx.sh play --checkpoint PATH [play args]
  ./bpx.sh mujoco --policy PATH [MuJoCo args]
  ./bpx.sh export --checkpoint PATH [export args]
  ./bpx.sh stairs-eval --checkpoint PATH [evaluation args]
  ./bpx.sh stairs-suite CHECKPOINT RUN_NAME OUTPUT_DIRECTORY
  ./bpx.sh sysid [plan|synthetic|collect-readonly|prepare|inspect|fit|replay] [args]
  ./bpx.sh sysid-isaac [PhysX identification args]
EOF
}

command="${1:-}"
[[ $# -gt 0 ]] && shift || true
case "$command" in
  setup)
    bash ./scripts/setup_environment.sh "$@"
    ;;
  install)
    python -m pip install -e source/mirrorme_rl_train
    ;;
  contract)
    python scripts/inspect_contract.py "$@"
    ;;
  train)
    python scripts/train.py "$@"
    ;;
  play)
    python scripts/play.py "$@"
    ;;
  mujoco)
    python scripts/play_mujoco.py "$@"
    ;;
  export)
    python scripts/export_policy.py "$@"
    ;;
  stairs-eval)
    python scripts/evaluate_stairs.py "$@"
    ;;
  sysid)
    python scripts/sysid.py "$@"
    ;;
  sysid-isaac)
    python scripts/sysid_isaac.py "$@"
    ;;
  stairs-suite)
    bash scripts/evaluate_stairs17_suite.sh "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
