import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import modal
import pydra
from pydra import Config, REQUIRED

# =============================================================================
# MODAL CONFIGURATION
# =============================================================================

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kb_path_local = os.path.join(repo_root, "KernelBench")
kb_requirements = os.path.join(kb_path_local, "requirements.txt")

if not os.path.exists(kb_path_local):
    raise FileNotFoundError("KernelBench directory not found. Please initialize the submodule.")
if not os.path.exists(kb_requirements):
    raise FileNotFoundError("KernelBench requirements.txt not found. Please pull the submodule contents.")

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install_from_requirements(kb_requirements)
    .env({"PYTHONPATH": "/root/KernelBench"})
    .add_local_dir(kb_path_local, remote_path="/root/KernelBench")
)

app = modal.App("kernelbench-leaderboard-eval")

class EvalSubmissionConfig(Config):
    def __init__(self):
        self.submission_file = REQUIRED
        self.output_file = None
        self.hardware = "L40S"
        self.gpu_arch = "Ada"
        self.baseline_file = "baseline_time_torch_compile_inductor_default"

    def __repr__(self):
        return f"EvalSubmissionConfig({self.to_dict()})"

# =============================================================================
# REMOTE EXECUTION LOGIC (CLASS BASED)
# =============================================================================

@app.cls(image=image, gpu="L40S", timeout=600)
class ModelEvaluator:
    @modal.method()
    def evaluate_kernel(self, level: int, problem_id: int, kernel_code: str, gpu_arch: str):
        """
        Runs inside the container.
        """
        import sys
        # Ensure imports work
        sys.path.append("/root/KernelBench")
        
        from src.dataset import construct_kernelbench_dataset
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        
        # Configure Environment
        set_gpu_arch([gpu_arch])
        
        # 1. Fetch Reference Source
        dataset = construct_kernelbench_dataset(level)
        prefix = f"{problem_id}_"
        ref_src = None
        
        for f_path in dataset:
            f_name = os.path.basename(f_path)
            if f_name.startswith(prefix):
                with open(f_path, 'r') as f:
                    ref_src = f.read()
                break
                
        if not ref_src:
            return {
                "error": f"Reference not found for L{level} P{problem_id}",
                "compiled": False,
                "correct": False,
                "runtime": -1
            }

        # 2. Run Evaluation
        try:
            result = eval_kernel_against_ref(
                original_model_src=ref_src,
                custom_model_src=kernel_code,
                num_correct_trials=5,
                num_perf_trials=100,
                measure_performance=True,
                verbose=False
            )
            
            return {
                "compiled": result.compiled,
                "correct": result.correctness,
                "runtime": result.runtime,
                "error": None
            }
        except Exception as e:
            return {
                "error": str(e),
                "compiled": False,
                "correct": False,
                "runtime": -1
            }

# =============================================================================
# LOCAL ORCHESTRATION LOGIC
# =============================================================================

def resolve_hardware_results_dir(hardware: str) -> str:
    alias_map = {
        "H100": "H100_PCIe_LambdaLabs",
        "H100_PCIe": "H100_PCIe_LambdaLabs",
    }
    target = alias_map.get(hardware, hardware)
    timing_root = os.path.join(kb_path_local, "results", "timing")
    candidate_path = os.path.join(timing_root, target)
    if os.path.isdir(candidate_path):
        return target
    if os.path.isdir(timing_root):
        for entry in os.listdir(timing_root):
            entry_path = os.path.join(timing_root, entry)
            if os.path.isdir(entry_path) and hardware.lower() in entry.lower():
                return entry
    return target

def load_local_baselines(hardware, baseline_file):
    hardware_dir = resolve_hardware_results_dir(hardware)
    path = os.path.join(kb_path_local, "results", "timing", hardware_dir, f"{baseline_file}.json")
    if not os.path.exists(path):
        print(f"⚠️  Baseline file not found locally at: {path}")
        print("    Run KernelBench/scripts/generate_baseline_time.py first if you need accurate speedups.")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def get_ref_name(level, problem_id):
    dataset_path = os.path.join(kb_path_local, f"level{level}")
    for f in os.listdir(dataset_path):
        if f.startswith(f"{problem_id}_") and f.endswith(".py"):
            return f
    return None

@pydra.main(base=EvalSubmissionConfig)
def main(config: EvalSubmissionConfig):
    sys.path.append(kb_path_local)
    from src.score import geometric_mean_speed_ratio_correct_only, fastp

    print(f"🚀 Launching evaluation on Modal ({config.hardware})...")
    
    with open(config.submission_file, 'r') as f:
        data = json.load(f)
    kernels = data["kernels"]
    
    baselines = load_local_baselines(config.hardware, config.baseline_file)

    work_items = []
    keys = []
    
    for key, code in kernels.items():
        try:
            parts = key.split('_')
            level = int(parts[1])
            problem_id = int(parts[3])
            
            keys.append(key)
            work_items.append((level, problem_id, code, config.gpu_arch))
        except Exception as e:
            print(f"Skipping malformed key {key}: {e}")

    print(f"☁️  Spinning up containers for {len(work_items)} kernels...")
    
    results_map = {}
    
    with app.run():
        # --- CLASS BASED CONFIGURATION ---
        # 1. Configure the Class with options (Hardware override)
        ConfiguredEvaluator = ModelEvaluator.with_options(gpu=config.hardware)
        
        # 2. Instantiate the class
        evaluator_instance = ConfiguredEvaluator()
        
        # 3. Call the method using starmap
        results_iterator = evaluator_instance.evaluate_kernel.starmap(work_items)
        
        # 4. Iterate results
        for key, res in zip(keys, tqdm(results_iterator, total=len(keys))):
            results_map[key] = res

    # --- Processing Results ---
    final_details = {}
    all_is_correct = []
    all_baseline_times = []
    all_actual_times = []

    for key, result in results_map.items():
        parts = key.split('_')
        level = int(parts[1])
        problem_id = int(parts[3])
        
        ref_name = get_ref_name(level, problem_id)
        baseline_ms = 0.0
        if ref_name and f"level{level}" in baselines:
            baseline_ms = baselines[f"level{level}"].get(ref_name, {}).get("mean", 0.0)
            
        runtime = result["runtime"]
        is_correct = result["correct"]
        
        speedup = 0.0
        if is_correct and runtime > 0 and baseline_ms > 0:
            speedup = baseline_ms / runtime
            
        final_details[key] = {
            "compiled": result["compiled"],
            "correct": is_correct,
            "runtime_ms": runtime,
            "baseline_ms": baseline_ms,
            "speedup": speedup,
            "code": kernels[key],
            "error": result["error"]
        }

        all_is_correct.append(is_correct)
        all_baseline_times.append(baseline_ms if baseline_ms > 0 else 1.0)
        all_actual_times.append(runtime if runtime > 0 else float('inf'))

    n = len(all_is_correct)
    if n > 0:
        gmsr = geometric_mean_speed_ratio_correct_only(
            np.array(all_is_correct), 
            np.array(all_baseline_times), 
            np.array(all_actual_times), 
            n
        )
        fast_p_15 = fastp(
            np.array(all_is_correct), 
            np.array(all_baseline_times), 
            np.array(all_actual_times), 
            n, 
            1.5
        )
    else:
        gmsr = 0.0
        fast_p_15 = 0.0

    output_data = {
        "metadata": data.get("metadata", {}),
        "metrics": {
            "geometric_mean_speedup": gmsr,
            "fast_p_1_5": fast_p_15,
            "total_submitted": n,
            "total_correct": sum(all_is_correct)
        },
        "details": final_details
    }

    print("\n" + "="*50)
    print(f"📊 Evaluation Complete")
    print(f"Geometric Mean Speedup: {gmsr:.4f}")
    print(f"Fast_p (1.5x): {fast_p_15:.2f}")
    print("="*50)

    output_file = config.output_file
    if output_file is None:
        run_name = data.get("metadata", {}).get("run_name", "unknown")
        output_file = f"eval_results_{run_name}.json"
        
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved results to {output_file}")

if __name__ == "__main__":
    main()