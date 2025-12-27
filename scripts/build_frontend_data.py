#!/usr/bin/env python3
"""
Builds kernels.json for the frontend by combining baseline times with eval results.

This script generates a pre-computed JSON file containing per-kernel performance
data across all models, enabling fast frontend loading without fetching multiple files.

Usage:
    python scripts/build_frontend_data.py
"""

import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
BASELINES_DIR = REPO_ROOT / "baselines"
RUNS_DIR = REPO_ROOT / "KernelBench" / "runs"


def load_baselines(hardware: str = "H100") -> dict:
    """Load baseline timing data for given hardware."""
    baseline_path = BASELINES_DIR / f"{hardware}.json"
    if not baseline_path.exists():
        print(f"Warning: Baseline file not found: {baseline_path}")
        return {}

    with open(baseline_path, 'r') as f:
        return json.load(f)


def load_metadata() -> list:
    """Load leaderboard metadata."""
    metadata_path = DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        return []

    with open(metadata_path, 'r') as f:
        return json.load(f)


def find_eval_results(model_id: str) -> dict:
    """Find and load eval results for a model across all levels."""
    results = {}

    for level in [1, 2, 3]:
        # Try different naming patterns
        patterns = [
            f"{model_id}_level{level}",
            f"{model_id.replace('_H100', '')}_level{level}",
        ]

        for pattern in patterns:
            eval_path = RUNS_DIR / pattern / "eval_results.json"
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    level_results = json.load(f)
                    results[level] = level_results
                break

    return results


def clean_kernel_name(filename: str) -> str:
    """Extract clean kernel name from filename."""
    name = filename.replace('.py', '')
    name = re.sub(r'^(\d+)_', '', name)
    name = name.replace('_', ' ').strip()
    return name


def build_kernels_data(baselines: dict, metadata: list) -> dict:
    """Build comprehensive kernels data structure."""
    kernels = []
    model_names = [entry.get('unique_id', entry.get('id')) for entry in metadata]

    # Load eval results for all models
    model_eval_results = {}
    for entry in metadata:
        model_id = entry.get('id', '')
        unique_id = entry.get('unique_id', model_id)
        eval_results = find_eval_results(model_id)
        if eval_results:
            model_eval_results[unique_id] = eval_results

    # Process each level
    for level in [1, 2, 3]:
        level_key = f"level{level}"
        if level_key not in baselines:
            continue

        baseline_data = baselines[level_key]

        # Sort by problem number
        sorted_kernels = sorted(
            baseline_data.items(),
            key=lambda x: int(re.match(r'^(\d+)', x[0]).group(1)) if re.match(r'^(\d+)', x[0]) else 0
        )

        for filename, baseline_info in sorted_kernels:
            # Extract problem ID from filename
            match = re.match(r'^(\d+)', filename)
            if not match:
                continue

            problem_id = int(match.group(1))
            baseline_time = baseline_info.get('mean', 0)

            kernel_entry = {
                "id": f"level_{level}_problem_{problem_id}",
                "level": level,
                "problem_id": problem_id,
                "name": clean_kernel_name(filename),
                "filename": filename,
                "baseline_time": baseline_time,
                "models": {}
            }

            # Add results for each model
            for unique_id, eval_results in model_eval_results.items():
                if level not in eval_results:
                    continue

                level_results = eval_results[level]
                problem_key = str(problem_id)

                if problem_key in level_results:
                    result_list = level_results[problem_key]
                    # Get sample_id=0 result
                    result = next(
                        (r for r in result_list if r.get('sample_id', 0) == 0),
                        result_list[0] if result_list else None
                    )

                    if result:
                        runtime = result.get('runtime', -1)
                        compiled = result.get('compiled', False)
                        correct = result.get('correctness', False)

                        speedup = 0
                        if correct and runtime > 0 and baseline_time > 0:
                            speedup = baseline_time / runtime

                        kernel_entry["models"][unique_id] = {
                            "compiled": compiled,
                            "correct": correct,
                            "runtime": runtime,
                            "speedup": round(speedup, 4)
                        }

            kernels.append(kernel_entry)

    return {
        "kernels": kernels,
        "model_names": model_names,
        "total_kernels": len(kernels),
        "levels": {
            1: sum(1 for k in kernels if k["level"] == 1),
            2: sum(1 for k in kernels if k["level"] == 2),
            3: sum(1 for k in kernels if k["level"] == 3),
        }
    }


def main():
    print("Building frontend data...")

    # Load baselines
    baselines = load_baselines("H100")
    if not baselines:
        print("Error: No baseline data found")
        return

    print(f"Loaded baselines for {sum(len(v) for v in baselines.values())} kernels")

    # Load metadata
    metadata = load_metadata()
    print(f"Found {len(metadata)} models in metadata")

    # Build kernels data
    kernels_data = build_kernels_data(baselines, metadata)
    print(f"Built data for {kernels_data['total_kernels']} kernels")
    print(f"  Level 1: {kernels_data['levels'][1]}")
    print(f"  Level 2: {kernels_data['levels'][2]}")
    print(f"  Level 3: {kernels_data['levels'][3]}")

    # Write output
    output_path = DATA_DIR / "kernels.json"
    DATA_DIR.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(kernels_data, f, indent=2)

    print(f"\nWritten to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
