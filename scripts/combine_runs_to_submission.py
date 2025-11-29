#!/usr/bin/env python3
"""
Combines kernel files from multiple KernelBench generation runs into a single submission JSON.

Usage:
    python scripts/combine_runs_to_submission.py \
        --level1_run opus_4_5_level1 \
        --level2_run opus_4_5_level2 \
        --level3_run opus_4_5_level3 \
        --output submissions/opus_4_5.json \
        --display_name "Opus 4.5" \
        --organization "Anthropic"
"""

import argparse
import json
import os
import re
from pathlib import Path


def find_kernel_files(runs_dir: Path, run_name: str, level: int) -> dict:
    """Find all kernel files for a given run and level."""
    run_path = runs_dir / run_name
    if not run_path.exists():
        print(f"Warning: Run directory not found: {run_path}")
        return {}

    kernels = {}
    pattern = re.compile(rf"level_{level}_problem_(\d+)_sample_(\d+)_kernel\.py")

    for file in run_path.iterdir():
        if file.is_file() and file.suffix == ".py":
            match = pattern.match(file.name)
            if match:
                problem_id = int(match.group(1))
                sample_id = int(match.group(2))

                # Only include sample_0 for leaderboard
                if sample_id == 0:
                    key = f"level_{level}_problem_{problem_id}_sample_0"
                    with open(file, 'r') as f:
                        kernels[key] = f.read()

    return kernels


def main():
    parser = argparse.ArgumentParser(description="Combine KernelBench runs into submission format")
    parser.add_argument("--level1_run", required=True, help="Run name for level 1")
    parser.add_argument("--level2_run", required=True, help="Run name for level 2")
    parser.add_argument("--level3_run", required=True, help="Run name for level 3")
    parser.add_argument("--output", required=True, help="Output submission JSON path")
    parser.add_argument("--display_name", required=True, help="Display name for leaderboard")
    parser.add_argument("--organization", default="", help="Organization name")
    parser.add_argument("--notes", default="", help="Additional notes")
    parser.add_argument("--runs_dir", default=None, help="Path to runs directory (default: KernelBench/runs)")
    args = parser.parse_args()

    # Determine runs directory
    repo_root = Path(__file__).parent.parent
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
    else:
        runs_dir = repo_root / "KernelBench" / "runs"

    print(f"Looking for runs in: {runs_dir}")

    # Collect kernels from all levels
    all_kernels = {}

    print(f"\nCollecting Level 1 kernels from: {args.level1_run}")
    level1_kernels = find_kernel_files(runs_dir, args.level1_run, 1)
    all_kernels.update(level1_kernels)
    print(f"  Found {len(level1_kernels)} kernels")

    print(f"\nCollecting Level 2 kernels from: {args.level2_run}")
    level2_kernels = find_kernel_files(runs_dir, args.level2_run, 2)
    all_kernels.update(level2_kernels)
    print(f"  Found {len(level2_kernels)} kernels")

    print(f"\nCollecting Level 3 kernels from: {args.level3_run}")
    level3_kernels = find_kernel_files(runs_dir, args.level3_run, 3)
    all_kernels.update(level3_kernels)
    print(f"  Found {len(level3_kernels)} kernels")

    # Validate completeness
    expected_l1 = 100
    expected_l2 = 100
    expected_l3 = 50
    total_expected = expected_l1 + expected_l2 + expected_l3

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Level 1: {len(level1_kernels)}/{expected_l1}")
    print(f"  Level 2: {len(level2_kernels)}/{expected_l2}")
    print(f"  Level 3: {len(level3_kernels)}/{expected_l3}")
    print(f"  Total: {len(all_kernels)}/{total_expected}")

    if len(all_kernels) < total_expected:
        print(f"\nWarning: Missing {total_expected - len(all_kernels)} kernels!")

        # Show which ones are missing
        for level, expected, found in [(1, expected_l1, level1_kernels),
                                        (2, expected_l2, level2_kernels),
                                        (3, expected_l3, level3_kernels)]:
            missing = []
            for i in range(1, expected + 1):
                key = f"level_{level}_problem_{i}_sample_0"
                if key not in found:
                    missing.append(i)
            if missing:
                print(f"  Level {level} missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Create submission
    # Use output filename stem as run_name (e.g., submissions/gemini_2_5_flash.json -> gemini_2_5_flash)
    output_path = Path(args.output)
    run_name = output_path.stem

    submission = {
        "metadata": {
            "display_name": args.display_name,
            "organization": args.organization,
            "run_name": run_name,
            "notes": args.notes
        },
        "kernels": all_kernels
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\n✅ Submission saved to: {output_path}")
    print(f"   Total kernels: {len(all_kernels)}")


if __name__ == "__main__":
    main()
