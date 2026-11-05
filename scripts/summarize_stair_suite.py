#!/usr/bin/env python3
"""Fail closed on incomplete stair suites and report every traversal mode."""

import argparse
import json
import math
from pathlib import Path


def summarize(directory, prefix, *, height=0.15, seeds=(2026, 2027, 2028), threshold=0.95):
    rows = []
    checkpoint = None
    for condition in ("nominal", "robust"):
        for direction in ("forward", "backward"):
            reports = []
            for seed in seeds:
                path = Path(directory) / f"{prefix}_{condition}_{direction}_{seed}.json"
                report = json.loads(path.read_text())
                if checkpoint is None:
                    checkpoint = report["checkpoint"]
                expected_speed = 0.30 if direction == "forward" else -0.30
                if (report["checkpoint"] != checkpoint or report["seed"] != seed or
                        report["condition"] != condition or
                        not math.isclose(report["height_m"], height) or
                        not math.isclose(report["speed_m_s"], expected_speed) or
                        report["num_envs"] != 64):
                    raise ValueError(f"Mismatched evaluation settings: {path}")
                attempts = report["attempts"]
                if len(attempts) != 64 or {x["env"] for x in attempts} != set(range(64)):
                    raise ValueError(f"Missing or duplicated first attempts: {path}")
                if any(x["success"] and x["physical_failure"] for x in attempts):
                    raise ValueError(f"A physical failure was counted as success: {path}")
                reports.append(report)
            for terrain in ("up", "down"):
                groups = [[x for x in r["attempts"] if x["terrain"] == terrain] for r in reports]
                if any(len(group) != 32 for group in groups):
                    raise ValueError("Each seed must contain 32 up and 32 down attempts")
                attempts = [x for group in groups for x in group]
                successes = sum(x["success"] for x in attempts)
                rows.append({"condition": condition, "direction": direction, "terrain": terrain,
                             "attempts": len(attempts), "successes": successes,
                             "success_rate": successes / len(attempts),
                             "physical_failures": sum(x["physical_failure"] for x in attempts),
                             "lateral_misses": sum(abs(x["lateral_offset_m"]) >= 0.75 for x in attempts),
                             "passed": successes / len(attempts) >= threshold})
    return {"checkpoint": checkpoint, "height_m": height, "seeds": list(seeds),
            "threshold": threshold, "rows": rows,
            "teacher_passed": all(row["passed"] for row in rows),
            "note": "Simulation teacher gate only; low-stair regression and blind student/real robot remain separate checks."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=float, default=0.15)
    args = parser.parse_args()
    report = summarize(args.directory, args.prefix, height=args.height)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
