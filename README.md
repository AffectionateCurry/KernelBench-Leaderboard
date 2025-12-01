# KernelBench Leaderboard

A leaderboard for tracking CUDA kernel optimization performance on the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) benchmark.

## Overview

KernelBench evaluates the ability of AI systems to optimize CUDA kernels across three difficulty levels:
- **Level 1**: 100 basic CUDA operations (matrix multiplication, activations, etc.)
- **Level 2**: 100 intermediate fused operations
- **Level 3**: 50 advanced architectures (ResNet, Transformers, etc.)

## Metrics

| Metric | Description |
|--------|-------------|
| **GeoMean Speedup** | Geometric mean of speedup ratios across all correctly solved problems. 1.0x = baseline PyTorch performance. |
| **Fast@1.0** | Percentage of kernels matching or exceeding baseline speed |
| **Fast@1.5** | Percentage of kernels achieving ≥1.5x speedup |
| **Fast@2.0** | Percentage of kernels achieving ≥2.0x speedup |
| **Correct Rate** | Percentage of kernels that compile and produce correct outputs |

## How to Submit

### Step 1: Prepare Your Submission

Create a JSON file with this exact structure:

```json
{
  "metadata": {
    "display_name": "Your Model Name",
    "organization": "Your Organization",
    "notes": "Optional notes about your submission"
  },
  "kernels": {
    "level_1_problem_1_sample_0": "import torch\nimport ...\n\nclass ModelNew(nn.Module):\n    ...",
    "level_1_problem_2_sample_0": "...",
    "level_2_problem_1_sample_0": "...",
    "level_3_problem_1_sample_0": "..."
  }
}
```

**Required fields:**
- `metadata.display_name`: Name shown on the leaderboard
- `kernels`: Dictionary with kernel code for each problem

**Kernel key format:** `level_{L}_problem_{P}_sample_0`
- Level 1: problems 1-100
- Level 2: problems 1-100
- Level 3: problems 1-50

**Total: 250 kernels expected**

### Step 2: Submit via Pull Request

1. Fork this repository
2. Add your submission: `submissions/your_model_name.json`
3. Open a Pull Request
4. Wait for a maintainer to review and add the `evaluate` label

### Step 3: Automated Evaluation

Once a maintainer adds the **`evaluate`** label to your PR:

1. **Validation** - Checks your JSON format and kernel counts
2. **Evaluation** - Runs all 250 kernels on H100 GPUs via [Modal](https://modal.com/)
3. **Scoring** - Computes metrics using KernelBench's scoring
4. **Results** - Posts a comment with your scores on the PR

Evaluation typically takes 30-60 minutes depending on queue.

## Hardware

All submissions are evaluated on **NVIDIA H100** GPUs.

## Local Development

### Prerequisites
- Python 3.10+
- [Modal](https://modal.com/) account and CLI configured

### Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/ScalingIntelligence/KernelBench-Leaderboard.git
cd KernelBench-Leaderboard

# Install dependencies
pip install modal pydra tqdm numpy tabulate datasets
pip install -r KernelBench/requirements.txt

# Configure Modal
modal token new
```

### Generate Kernels with Your Model

```bash
cd KernelBench

# Generate for each level
python scripts/generate_samples.py \
    dataset_src=local level=1 run_name=my_model_level1 \
    server_type=google model_name=gemini/gemini-2.5-flash

python scripts/generate_samples.py \
    dataset_src=local level=2 run_name=my_model_level2 \
    server_type=google model_name=gemini/gemini-2.5-flash

python scripts/generate_samples.py \
    dataset_src=local level=3 run_name=my_model_level3 \
    server_type=google model_name=gemini/gemini-2.5-flash
```

### Evaluate Locally

```bash
# Evaluate each level
python scripts/eval_from_generations.py \
    run_name=my_model_level1 level=1 eval_mode=modal gpu=H100 dataset_src=local

python scripts/eval_from_generations.py \
    run_name=my_model_level2 level=2 eval_mode=modal gpu=H100 dataset_src=local

python scripts/eval_from_generations.py \
    run_name=my_model_level3 level=3 eval_mode=modal gpu=H100 dataset_src=local

# Analyze results
python scripts/benchmark_eval_analysis.py \
    run_name=my_model_level1 level=1 hardware=H100 baseline=baseline \
    baseline_file=../baselines/H100.json output_file=../data/level1_results.json

python scripts/benchmark_eval_analysis.py \
    run_name=my_model_level2 level=2 hardware=H100 baseline=baseline \
    baseline_file=../baselines/H100.json output_file=../data/level2_results.json

python scripts/benchmark_eval_analysis.py \
    run_name=my_model_level3 level=3 hardware=H100 baseline=baseline \
    baseline_file=../baselines/H100.json output_file=../data/level3_results.json
```

### Create Submission & Update Leaderboard

```bash
cd ..

# Bundle kernels into submission JSON
python scripts/combine_runs_to_submission.py \
    --level1_run my_model_level1 \
    --level2_run my_model_level2 \
    --level3_run my_model_level3 \
    --output submissions/my_model.json \
    --display_name "My Model" \
    --organization "My Organization"

# Update leaderboard
python scripts/update_leaderboard.py \
    --level1 data/level1_results.json \
    --level2 data/level2_results.json \
    --level3 data/level3_results.json \
    --submission submissions/my_model.json
```

## Architecture

This leaderboard is a thin orchestration layer. All scoring logic lives in KernelBench:

```
┌─────────────────────────────────────────────────────────────────┐
│                    KernelBench Leaderboard                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ submission_to_   │  │ update_          │  │ combine_runs_ │  │
│  │ run.py           │  │ leaderboard.py   │  │ to_submission │  │
│  │ (JSON→files)     │  │ (aggregation)    │  │ .py           │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────────┘  │
│           │                     │                               │
└───────────┼─────────────────────┼───────────────────────────────┘
            │                     │
┌───────────▼─────────────────────▼───────────────────────────────┐
│                       KernelBench (submodule)                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ eval_from_       │  │ benchmark_eval_  │  │ src/score.py  │  │
│  │ generations.py   │  │ analysis.py      │  │ (all metrics) │  │
│  │ (GPU eval)       │  │ (scoring+JSON)   │  │               │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
KernelBench-Leaderboard/
├── baselines/              # Baseline timing data
│   └── H100.json
├── data/
│   ├── metadata.json       # Leaderboard entries
│   └── results/            # Detailed per-submission results
├── scripts/
│   ├── submission_to_run.py
│   ├── update_leaderboard.py
│   └── combine_runs_to_submission.py
├── submissions/            # Submission JSON files
├── KernelBench/            # Submodule: benchmark & scoring
├── index.html              # Leaderboard frontend
└── .github/workflows/      # CI/CD automation
```

## Links

- [KernelBench Repository](https://github.com/ScalingIntelligence/KernelBench)
- [KernelBench Paper](https://arxiv.org/abs/2502.10517)
- [Leaderboard Website](https://scalingintelligence.github.io/KernelBench-Leaderboard/)

## License

This project is part of the KernelBench benchmark suite. See the [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench) for licensing information.
