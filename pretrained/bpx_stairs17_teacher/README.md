# 17 cm privileged teacher — iteration 1499

[项目首页](../../README.zh-CN.md) · [实验报告](../../docs/stairs17-results.zh-CN.md)

This is a **235-D PPO teacher**, not a distilled blind student. It completed the 1,500-iteration 17 cm trial on 2026-11-05, but robust descent remains below the 95% per-mode qualification target. Formal blind-student training and real-robot validation are pending.

- `model_1499.pt`: complete RSL-RL teacher checkpoint, including optimizer and iteration counter; 12 actions.
- `params/`: unmodified historical environment/PPO YAML snapshots. They contain original machine paths and are not loaded automatically. `max_iterations: 1400` is the additional budget after resuming iteration 100.
- `reports/`: 12 fixed-17cm evaluations, 6 lower-stair regression evaluations, and the aggregate qualification report. Every evaluation retains all 64 first-attempt outcomes. Large trajectories, initial states and physics arrays are omitted; checkpoint paths are rewritten relative to the repository.
- `artifact-manifest.json`: SHA-256 checksums of every bundled file, original full-report checksums and model provenance.

To inspect aggregate evidence without Isaac Sim, run from the repository root:

```bash
python scripts/summarize_stair_suite.py \
  --directory pretrained/bpx_stairs17_teacher/reports \
  --prefix rtx2080_stairs17_balanced_resume_20261105 \
  --height 0.17 --output /tmp/bpx-stairs17-qualification.json
```

Expected result: `teacher_passed: false`. For a fresh simulation evaluation, use the `stairs-eval` or `stairs-suite` commands in the project README. A clone contains the latest teacher; it does not reproduce the entire earlier curriculum/training history automatically.
