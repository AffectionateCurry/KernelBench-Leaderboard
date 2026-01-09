#!/usr/bin/env python3
"""
Builds kernels.json for the frontend by combining baseline times with eval results.

This script generates a pre-computed JSON file containing per-kernel performance
data across all models, enabling fast frontend loading without fetching multiple files.

Supports dual baselines:
- torch compile (inductor) baseline: H100.json
- torch eager baseline: H100_eager.json

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
PROBLEMS_DIR = REPO_ROOT / "KernelBench" / "KernelBench"


def load_baselines(hardware: str = "H100") -> tuple[dict, dict]:
    """Load both baseline timing data for given hardware.

    Returns:
        tuple: (compile_baselines, eager_baselines)
    """
    # Torch compile (inductor) baseline
    compile_path = BASELINES_DIR / f"{hardware}.json"
    compile_baselines = {}
    if compile_path.exists():
        with open(compile_path, 'r') as f:
            compile_baselines = json.load(f)
    else:
        print(f"Warning: Compile baseline file not found: {compile_path}")

    # Torch eager baseline
    eager_path = BASELINES_DIR / f"{hardware}_eager.json"
    eager_baselines = {}
    if eager_path.exists():
        with open(eager_path, 'r') as f:
            eager_baselines = json.load(f)
    else:
        print(f"Warning: Eager baseline file not found: {eager_path}")

    return compile_baselines, eager_baselines


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


def load_reference_code(level: int, filename: str) -> str:
    """Load the reference implementation code for a kernel."""
    ref_path = PROBLEMS_DIR / f"level{level}" / filename
    if ref_path.exists():
        try:
            with open(ref_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read {ref_path}: {e}")
    return ""


def load_model_solution_code(model_id: str, level: int, problem_id: int) -> tuple:
    """Load model's generated kernel code and return (code, error)."""
    # Try different naming patterns
    patterns = [
        f"{model_id}_level{level}",
        f"{model_id.replace('_H100', '')}_level{level}",
    ]

    for pattern in patterns:
        solution_path = RUNS_DIR / pattern / f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
        if solution_path.exists():
            try:
                with open(solution_path, 'r') as f:
                    return f.read(), None
            except Exception as e:
                return "", str(e)

    return "", None


def get_compilation_error(result: dict) -> str:
    """Extract compilation error from result."""
    metadata = result.get('metadata', {})
    error = metadata.get('compilation_error', '')
    if error:
        # Truncate very long errors
        if len(error) > 2000:
            error = error[:2000] + "\n... (truncated)"
    return error


def build_kernels_data(compile_baselines: dict, eager_baselines: dict, metadata: list) -> dict:
    """Build comprehensive kernels data structure with dual baselines.

    Args:
        compile_baselines: Torch compile (inductor) baseline times
        eager_baselines: Torch eager baseline times
        metadata: Leaderboard metadata entries
    """
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
        if level_key not in compile_baselines:
            continue

        compile_data = compile_baselines[level_key]
        eager_data = eager_baselines.get(level_key, {})

        # Sort by problem number
        sorted_kernels = sorted(
            compile_data.items(),
            key=lambda x: int(re.match(r'^(\d+)', x[0]).group(1)) if re.match(r'^(\d+)', x[0]) else 0
        )

        for filename, compile_info in sorted_kernels:
            # Extract problem ID from filename
            match = re.match(r'^(\d+)', filename)
            if not match:
                continue

            problem_id = int(match.group(1))
            compile_time = compile_info.get('mean', 0)
            eager_info = eager_data.get(filename, {})
            eager_time = eager_info.get('mean', 0)

            # Load reference implementation code
            ref_code = load_reference_code(level, filename)

            kernel_entry = {
                "id": f"level_{level}_problem_{problem_id}",
                "level": level,
                "problem_id": problem_id,
                "name": clean_kernel_name(filename),
                "filename": filename,
                # Dual baseline times
                "baseline_time_compile": compile_time,
                "baseline_time_eager": eager_time,
                # Keep backward compat alias (compile baseline is primary)
                "baseline_time": compile_time,
                "reference_code": ref_code,
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

                        # Compute speedups against both baselines
                        speedup_compile = 0
                        speedup_eager = 0
                        if correct and runtime > 0:
                            if compile_time > 0:
                                speedup_compile = compile_time / runtime
                            if eager_time > 0:
                                speedup_eager = eager_time / runtime

                        # Get compilation error if failed
                        error = get_compilation_error(result) if not compiled else ""

                        # Load model's solution code
                        model_id_for_code = unique_id.replace('_H100', '')
                        solution_code, _ = load_model_solution_code(model_id_for_code, level, problem_id)

                        kernel_entry["models"][unique_id] = {
                            "compiled": compiled,
                            "correct": correct,
                            "runtime": runtime,
                            # Dual speedups
                            "speedup_vs_compile": round(speedup_compile, 4),
                            "speedup_vs_eager": round(speedup_eager, 4),
                            # Keep backward compat alias (vs compile is primary)
                            "speedup": round(speedup_compile, 4),
                            "error": error,
                            "code": solution_code
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

    # Load both baselines
    compile_baselines, eager_baselines = load_baselines("H100")
    if not compile_baselines:
        print("Error: No compile baseline data found")
        return

    compile_count = sum(len(v) for v in compile_baselines.values())
    eager_count = sum(len(v) for v in eager_baselines.values())
    print(f"Loaded compile baselines for {compile_count} kernels")
    print(f"Loaded eager baselines for {eager_count} kernels")

    # Load metadata
    metadata = load_metadata()
    print(f"Found {len(metadata)} models in metadata")

    # Build kernels data with dual baselines
    kernels_data = build_kernels_data(compile_baselines, eager_baselines, metadata)
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
