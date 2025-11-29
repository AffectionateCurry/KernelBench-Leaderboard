import json
import os
import re

import pydra
from pydra import Config, REQUIRED


def get_kernel_content(filepath):
    with open(filepath, "r") as f:
        return f.read()


class SubmissionConfig(Config):
    def __init__(self):
        # Required inputs
        self.run_name = REQUIRED
        self.submission_name = REQUIRED

        # Optional metadata
        self.organization = ""
        self.notes = ""

        # Output controls
        self.submission_file = "submission.json"
        self.allow_incomplete = False


@pydra.main(base=SubmissionConfig)
def main(config: SubmissionConfig):
    """
    Bundle kernels for leaderboard submission.
    Mirrors the `key=value` CLI style used throughout the repo.
    """

    # Locate the run directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(repo_root, "runs", config.run_name)

    if not os.path.exists(run_dir):
        print(f"❌ Error: Run directory not found at: {run_dir}")
        return

    # Prepare data structure
    submission = {
        "metadata": {
            "display_name": config.submission_name,
            "organization": config.organization,
            "run_name": config.run_name,
            "notes": config.notes,
        },
        "kernels": {},
    }

    # Regex to parse filenames reliably
    filename_pattern = re.compile(r"level_(\d+)_problem_(\d+)_sample_(\d+)_kernel\.py")

    files_processed = 0
    expected_counts = {1: 100, 2: 100, 3: 50}
    found_counts = {1: 0, 2: 0, 3: 0}

    print(f"📂 Scanning directory: {run_dir} ...")

    for filename in os.listdir(run_dir):
        if not filename.endswith(".py"):
            continue

        match = filename_pattern.match(filename)
        if match:
            level = int(match.group(1))
            problem_id = int(match.group(2))
            sample_id = int(match.group(3))

            if sample_id != 0:
                continue

            key = f"level_{level}_problem_{problem_id}_sample_{sample_id}"
            
            filepath = os.path.join(run_dir, filename)
            code = get_kernel_content(filepath)
            
            submission["kernels"][key] = code
            
            if level in found_counts:
                found_counts[level] += 1
            files_processed += 1

    # Status Report
    print("-" * 40)
    print(f"✅ Processed {files_processed} kernels.")
    
    if config.allow_incomplete:
        print("ℹ️  Test Mode: Ignoring completeness checks.")
    else:
        all_complete = True
        for level, expected in expected_counts.items():
            found = found_counts[level]
            if found < expected:
                print(f"⚠️  Level {level}: Found {found}/{expected} (Missing {expected - found})")
                all_complete = False
            else:
                print(f"✓ Level {level}: {found}/{expected} - Complete")
        
        if not all_complete:
            print("\n⚠️  Warning: This submission is incomplete.")
            print("    For the official leaderboard, missing problems score 0.")

    with open(config.submission_file, "w") as f:
        json.dump(submission, f, indent=2)

    print("-" * 40)
    print(f"📦 Submission file saved to: {config.submission_file}")
    print(f"   Leaderboard Entry: '{config.submission_name}' by '{config.organization}'")

if __name__ == "__main__":
    main()