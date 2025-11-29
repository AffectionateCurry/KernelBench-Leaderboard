import json
import os
import argparse
from datetime import datetime
import shutil

# Paths configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
PROBLEMS_DIR = os.path.join(DATA_DIR, "problems")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

def ensure_structure():
    os.makedirs(PROBLEMS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump([], f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_output", required=True, help="Path to the JSON output from evaluate_submission.py")
    args = parser.parse_args()

    ensure_structure()

    # 1. Load the Evaluation Result
    with open(args.eval_output, 'r') as f:
        new_data = json.load(f)

    # Create a clean ID for filenames (e.g., "Stanford Agent v1" -> "stanford_agent_v1")
    # This ID links the metadata, the full dump, and the problem entries together.
    submission_id = new_data["metadata"]["display_name"].lower().replace(" ", "_").replace("-", "_")

    # Extract hardware configuration for composite key
    hardware_tag = new_data.get("config", {}).get("hardware", "unknown")
    unique_entry_id = f"{submission_id}_{hardware_tag}"

    print(f"🔄 Updating Database for: {new_data['metadata']['display_name']} (ID: {unique_entry_id})")

    # -------------------------------------------------------------------------
    # PART A: Update Main Leaderboard (metadata.json)
    # -------------------------------------------------------------------------
    with open(METADATA_FILE, 'r') as f:
        metadata_list = json.load(f)

    # Remove existing entry if updating a model+hardware combo (to prevent duplicates)
    metadata_list = [m for m in metadata_list if m.get("unique_id") != unique_entry_id]

    # Add new entry with nested metrics structure
    new_entry = {
        "id": submission_id,
        "unique_id": unique_entry_id,
        "hardware": hardware_tag,
        "name": new_data["metadata"]["display_name"],
        "organization": new_data["metadata"].get("organization", ""),
        "notes": new_data["metadata"].get("notes", ""),
        "metrics": {
            "geo_mean": new_data["metrics"]["geometric_mean_speedup"],
            "fast_p": new_data["metrics"]["fast_p_1_5"],
            "total_submitted": new_data["metrics"].get("total_submitted", 0),
            "total_correct": new_data["metrics"].get("total_correct", 0)
        },
        "date": datetime.today().strftime('%Y-%m-%d')
    }
    metadata_list.append(new_entry)
    
    # Save Main Leaderboard
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"✅ Updated Main Leaderboard ({len(metadata_list)} entries)")

    # -------------------------------------------------------------------------
    # PART B: Save Full Result Dump
    # -------------------------------------------------------------------------
    # This allows users to download the raw data if they want
    # Use unique_entry_id to support same model on different hardware
    dump_path = os.path.join(RESULTS_DIR, f"{unique_entry_id}.json")
    with open(dump_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    print(f"✅ Saved full result dump to data/results/{unique_entry_id}.json")

    # -------------------------------------------------------------------------
    # PART C: Update Individual Problem Files
    # -------------------------------------------------------------------------
    # We iterate over every kernel result and update specific problem JSONs
    
    updated_problems_count = 0
    
    # Assuming evaluate_submission.py outputs a "details" or "kernels" dictionary
    # formatted like: "level_1_problem_1_sample_0": { "speedup": 2.0, "code": "...", "correct": true }
    
    kernels = new_data.get("details", {}) 
    
    for key, kernel_info in kernels.items():
        # Only leaderboard correct solutions
        if not kernel_info.get("correct", False):
            continue

        # Parse key: level_1_problem_15_sample_0
        parts = key.split('_')
        level = parts[1]
        prob_id = parts[3]
        
        problem_filename = f"level_{level}_problem_{prob_id}.json"
        problem_path = os.path.join(PROBLEMS_DIR, problem_filename)

        # Load existing problem data
        if os.path.exists(problem_path):
            with open(problem_path, 'r') as f:
                prob_data = json.load(f)
        else:
            prob_data = []

        # Remove previous entry from this specific model+hardware combo
        prob_data = [p for p in prob_data if p.get("unique_id") != unique_entry_id]

        # Add new entry with hardware tag
        prob_data.append({
            "model_id": submission_id,
            "unique_id": unique_entry_id,
            "model_name": new_data["metadata"]["display_name"],
            "hardware": hardware_tag,
            "speedup": kernel_info["speedup"],
            "runtime": kernel_info.get("runtime_ms", 0),
            "code": kernel_info.get("code", "")
        })

        # Sort by Speedup (High to Low)
        prob_data.sort(key=lambda x: x["speedup"], reverse=True)

        # Save back
        with open(problem_path, 'w') as f:
            json.dump(prob_data, f, indent=2)
        
        updated_problems_count += 1

    print(f"✅ Updated {updated_problems_count} problem files.")

if __name__ == "__main__":
    main()