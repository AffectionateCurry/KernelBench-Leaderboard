#!/usr/bin/env python3
"""
Converts a leaderboard submission JSON into KernelBench run format.

Creates separate run directories for each level to avoid eval_results.json
collisions when running eval_from_generations.py multiple times.

Usage:
    python scripts/submission_to_run.py --submission submissions/model_name.json

Output:
    KernelBench/runs/{run_name}_level1/level_1_problem_*_kernel.py
    KernelBench/runs/{run_name}_level2/level_2_problem_*_kernel.py
    KernelBench/runs/{run_name}_level3/level_3_problem_*_kernel.py
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert submission JSON to KernelBench run format"
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to submission JSON file"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="KernelBench runs directory (default: KernelBench/runs)"
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Override run name (default: from submission metadata or filename)"
    )
    args = parser.parse_args()

    # Load submission
    submission_path = Path(args.submission)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_path}")

    with open(submission_path, 'r') as f:
        submission = json.load(f)

    kernels = submission.get("kernels", {})
    metadata = submission.get("metadata", {})

    # Determine run name (prefer filename stem for cleaner directory names)
    if args.run_name:
        run_name = args.run_name
    else:
        # Use filename stem by default (e.g., "gemini_2_5_flash" from "gemini_2_5_flash.json")
        run_name = submission_path.stem

    # Determine output directory
    if args.output_dir:
        runs_dir = Path(args.output_dir)
    else:
        repo_root = Path(__file__).parent.parent
        runs_dir = repo_root / "KernelBench" / "runs"

    print(f"Converting submission: {submission_path}")
    print(f"Run name: {run_name}")
    print(f"Output directory: {runs_dir}")

    # Group kernels by level
    kernels_by_level = {1: {}, 2: {}, 3: {}}
    parse_errors = []

    for key, code in kernels.items():
        try:
            # Parse key: level_1_problem_1_sample_0
            parts = key.split('_')
            level = int(parts[1])
            problem_id = int(parts[3])
            sample_id = int(parts[5])

            if level not in [1, 2, 3]:
                parse_errors.append(f"Invalid level {level} in key: {key}")
                continue

            kernels_by_level[level][(problem_id, sample_id)] = code
        except (IndexError, ValueError) as e:
            parse_errors.append(f"Failed to parse key '{key}': {e}")

    if parse_errors:
        print(f"\nWarning: {len(parse_errors)} parse errors:")
        for err in parse_errors[:5]:
            print(f"  - {err}")
        if len(parse_errors) > 5:
            print(f"  ... and {len(parse_errors) - 5} more")

    # Write to separate directories per level
    expected_counts = {1: 100, 2: 100, 3: 50}
    total_written = 0

    for level in [1, 2, 3]:
        level_run_dir = runs_dir / f"{run_name}_level{level}"
        level_run_dir.mkdir(parents=True, exist_ok=True)

        level_kernels = kernels_by_level[level]
        written = 0

        for (problem_id, sample_id), code in level_kernels.items():
            filename = f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
            filepath = level_run_dir / filename
            with open(filepath, 'w') as f:
                f.write(code)
            written += 1

        total_written += written
        expected = expected_counts[level]
        status = "OK" if written == expected else "INCOMPLETE"
        print(f"  Level {level}: {written}/{expected} kernels written [{status}]")
        print(f"    → {level_run_dir}")

    print(f"\nTotal: {total_written}/250 kernels written")
    print(f"\nTo evaluate, run:")
    print(f"  cd KernelBench")
    for level in [1, 2, 3]:
        print(f"  python scripts/eval_from_generations.py run_name={run_name}_level{level} level={level} eval_mode=modal gpu=H100 dataset_src=local")


if __name__ == "__main__":
    main()
