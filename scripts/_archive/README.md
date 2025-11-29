# Archived Scripts

These scripts have been deprecated in favor of reusing KernelBench's evaluation infrastructure directly.

## Why Archived?

The original approach duplicated KernelBench's evaluation logic. The new approach:
1. Converts submissions to KernelBench run format (`submission_to_run.py`)
2. Calls KernelBench's `eval_from_generations.py` directly
3. Imports results to leaderboard (`import_results.py`)

## Archived Files

- `evaluate_submission.py` - Duplicated KernelBench's Modal evaluation logic
- `prepare_submission.py` - Replaced by `combine_runs_to_submission.py` (more flexible)
- `update_database.py` - Functionality merged into `import_results.py`

## New Workflow

See the main README for the current workflow using KernelBench scripts directly.
