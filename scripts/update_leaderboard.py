#!/usr/bin/env python3
"""
Updates leaderboard by reading benchmark_eval_analysis.py JSON outputs.

This script contains NO scoring logic - it just reads JSON files produced
by KernelBench's benchmark_eval_analysis.py, aggregates across levels,
and updates data/metadata.json.

Usage:
    # First, run benchmark_eval_analysis.py for each level (in KernelBench):
    python scripts/benchmark_eval_analysis.py \\
        run_name=model_level1 level=1 hardware=H100 baseline=baseline \\
        baseline_file=/path/to/baselines/H100.json \\
        output_file=level1_results.json

    # Then, update leaderboard with the JSON outputs:
    python scripts/update_leaderboard.py \\
        --level1 /path/to/level1_results.json \\
        --level2 /path/to/level2_results.json \\
        --level3 /path/to/level3_results.json \\
        --submission submissions/model.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime


def aggregate_levels(level_results: list[dict]) -> dict:
    """
    Combine metrics from 3 levels into aggregate metrics.

    Uses weighted geometric mean for geo_mean_speedup.
    """
    # Sum counts
    total_count = sum(r["total_count"] for r in level_results)
    total_compiled = sum(r["compiled_count"] for r in level_results)
    total_correct = sum(r["correct_count"] for r in level_results)

    # Weighted geometric mean of speedups
    # Weight by number of correct samples per level
    log_speedup_sum = 0.0
    correct_weight_sum = 0

    for r in level_results:
        if r["correct_count"] > 0 and r["geo_mean_speedup"] > 0:
            # geo_mean = exp(mean(log(speedups)))
            # So we weight the log(geo_mean) by correct_count
            log_speedup_sum += math.log(r["geo_mean_speedup"]) * r["correct_count"]
            correct_weight_sum += r["correct_count"]

    if correct_weight_sum > 0:
        geo_mean = math.exp(log_speedup_sum / correct_weight_sum)
    else:
        geo_mean = 0.0

    # Aggregate fast_p scores (weighted by total_count per level)
    fast_p_aggregate = {}
    for p in ["0.0", "0.5", "0.8", "1.0", "1.5", "2.0"]:
        # fast_p is already a rate (count / n), so we need to compute total fast / total n
        fast_count = sum(r["fast_p"].get(p, 0) * r["total_count"] for r in level_results)
        fast_p_aggregate[p] = fast_count / total_count if total_count > 0 else 0.0

    return {
        "geo_mean": geo_mean,
        "fast_p_1_0": fast_p_aggregate.get("1.0", 0.0),
        "fast_p_1_5": fast_p_aggregate.get("1.5", 0.0),
        "fast_p_2_0": fast_p_aggregate.get("2.0", 0.0),
        "total_submitted": total_count,
        "total_compiled": total_compiled,
        "total_correct": total_correct,
        "compile_rate": total_compiled / total_count if total_count > 0 else 0.0,
        "correct_rate": total_correct / total_count if total_count > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Update leaderboard from benchmark_eval_analysis.py JSON outputs"
    )
    parser.add_argument(
        "--level1",
        required=True,
        help="Path to level 1 analysis JSON (from benchmark_eval_analysis.py)"
    )
    parser.add_argument(
        "--level2",
        required=True,
        help="Path to level 2 analysis JSON (from benchmark_eval_analysis.py)"
    )
    parser.add_argument(
        "--level3",
        required=True,
        help="Path to level 3 analysis JSON (from benchmark_eval_analysis.py)"
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to submission JSON (for metadata)"
    )
    parser.add_argument(
        "--hardware",
        default="H100",
        help="Hardware configuration (default: H100)"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data"

    # Load level results
    level_files = [args.level1, args.level2, args.level3]
    level_results = []

    for i, level_file in enumerate(level_files, start=1):
        level_path = Path(level_file)
        if not level_path.exists():
            print(f"Error: Level {i} results not found: {level_file}")
            sys.exit(1)

        with open(level_path, 'r') as f:
            level_results.append(json.load(f))

    # Load submission metadata
    submission_path = Path(args.submission)
    if not submission_path.exists():
        print(f"Error: Submission not found: {submission_path}")
        sys.exit(1)

    with open(submission_path, 'r') as f:
        submission = json.load(f)
    metadata = submission.get("metadata", {})

    print(f"Updating leaderboard for: {metadata.get('display_name', 'Unknown')}")
    print(f"Hardware: {args.hardware}")

    # Aggregate across levels
    print(f"\n{'='*60}")
    print("Aggregating across all levels...")
    print(f"{'='*60}")

    metrics = aggregate_levels(level_results)

    print(f"\nAggregate Metrics:")
    print(f"  Geo Mean Speedup: {metrics['geo_mean']:.4f}")
    print(f"  Fast@1.0: {metrics['fast_p_1_0']:.2%}")
    print(f"  Fast@1.5: {metrics['fast_p_1_5']:.2%}")
    print(f"  Fast@2.0: {metrics['fast_p_2_0']:.2%}")
    print(f"  Compiled: {metrics['total_compiled']}/{metrics['total_submitted']} ({metrics['compile_rate']:.1%})")
    print(f"  Correct: {metrics['total_correct']}/{metrics['total_submitted']} ({metrics['correct_rate']:.1%})")

    # Build level stats
    level_stats = {}
    for r in level_results:
        level_stats[f"level{r['level']}"] = {
            "evaluated": r.get("total_eval", r["total_count"]),
            "expected": r["total_count"],
            "compiled": r["compiled_count"],
            "correct": r["correct_count"]
        }

    # Create leaderboard entry
    run_name = metadata.get("run_name", submission_path.stem)
    unique_id = f"{run_name}_{args.hardware}"

    entry = {
        "id": run_name,
        "unique_id": unique_id,
        "display_name": metadata.get("display_name", run_name),
        "organization": metadata.get("organization", ""),
        "hardware": args.hardware,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "level_stats": level_stats,
        "notes": metadata.get("notes", "")
    }

    # Update metadata.json
    data_dir.mkdir(exist_ok=True)
    metadata_path = data_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []

    # Remove existing entry with same unique_id (update case)
    leaderboard = [e for e in leaderboard if e.get("unique_id") != unique_id]
    leaderboard.append(entry)

    # Sort by geo_mean descending
    leaderboard.sort(key=lambda x: x.get("metrics", {}).get("geo_mean", 0), reverse=True)

    with open(metadata_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"\nUpdated: {metadata_path}")

    # Save detailed results
    results_dir = data_dir / "results"
    results_dir.mkdir(exist_ok=True)
    detailed_path = results_dir / f"{unique_id}.json"

    detailed = {
        "entry": entry,
        "level_results": level_results
    }
    with open(detailed_path, 'w') as f:
        json.dump(detailed, f, indent=2)

    print(f"Saved detailed results: {detailed_path}")

    # Print rank
    rank = next((i + 1 for i, e in enumerate(leaderboard) if e["unique_id"] == unique_id), 0)
    print(f"\nLeaderboard rank: #{rank} of {len(leaderboard)}")


if __name__ == "__main__":
    main()
