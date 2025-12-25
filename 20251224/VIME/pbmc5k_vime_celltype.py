"""Run VIME pipeline on PBMC 5k data with cell type labels."""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# Add base/ directory for new models
base_models_dir = base_dir / "base"
if str(base_models_dir) not in sys.path:
    sys.path.insert(0, str(base_models_dir))

from pbmc5k_data_celltype import load_pbmc5k_data_with_celltype


def _setup_vime_paths():
    base_dir = Path(__file__).resolve().parent
    # 指向 tem/20251210/VIME
    vime_dir = base_dir.parent.parent.parent / "20251210" / "VIME"
    vime_dir_str = str(vime_dir)
    if vime_dir_str not in sys.path:
        sys.path.insert(0, vime_dir_str)


def save_results_summary(summary_path, method_name, method_results, metadata=None):
    """Save experiment summary to JSON file."""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    else:
        summary_data = {"metadata": {}, "methods": {}}

    if metadata:
        summary_data["metadata"].update(metadata)

    summary_data["methods"][method_name] = method_results

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"  Summary results saved: {summary_path}")


def run_experiments_with_save(args, vime_train):
    """Run experiments with multiple iterations and save results."""
    import random
    import torch

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    results = np.zeros([args.iterations, 4])
    methods_full = ["Supervised", "Autoencoder", "VIME-Self", "VIME"]

    SEP_HEADER = "#" * 70
    SEP_MINOR = "-" * 70

    for i in range(args.iterations):
        set_seed(args.seed + i)
        print(f"\n{SEP_HEADER}\n# ITERATION {i+1}/{args.iterations}\n{SEP_HEADER}")

        results[i, :] = vime_train.train_vime(
            args.label_data_rate,
            [args.model_name],
            args.label_no,
            args.p_m,
            args.alpha,
            args.K,
            args.beta,
            args.dataset,
            args.n_noise_features,
            args.hidden_dim,
            args.patience,
        )

        print(f"\n{SEP_MINOR}")
        print(
            f"Iteration {i+1} Results: Supervised={results[i,0]:.4f}, "
            f"Autoencoder={results[i,1]:.4f}, VIME-Self={results[i,2]:.4f}, VIME={results[i,3]:.4f}"
        )
        print(SEP_MINOR)

    # Print final results
    print(f"\n{SEP_HEADER}\n# FINAL RESULTS (Average over {args.iterations} iterations)\n{SEP_HEADER}")
    for j, name in enumerate(methods_full):
        print(f"{name}: Avg={np.mean(results[:, j]):.4f}, Std={np.std(results[:, j]):.4f}")
    print(SEP_HEADER)

    # Save results
    per_iteration = []
    for i in range(args.iterations):
        per_iteration.append({
            "iteration": i + 1,
            "seed": args.seed + i,
            "accuracies": {
                methods_full[j]: float(results[i, j]) for j in range(4)
            },
        })

    summary_stats = {
        methods_full[j]: {
            "mean": float(np.mean(results[:, j])),
            "std": float(np.std(results[:, j])),
            "min": float(np.min(results[:, j])),
            "max": float(np.max(results[:, j])),
            "all_values": [float(val) for val in results[:, j]],
        }
        for j in range(4)
    }

    method_results = {
        "per_iteration": per_iteration,
        "summary": summary_stats,
    }

    metadata = {
        "dataset": "PBMC_5k_celltype",
        "label_type": "Cell type (6 classes)",
        "cell_types": [
            "Monocytes", "B cells", "NK cells",
            "CD4 T cells", "CD8 T cells", "Other"
        ],
        "n_noise_features": args.n_noise_features,
        "label_no": args.label_no,
        "label_data_rate": args.label_data_rate,
        "iterations": args.iterations,
        "seed_start": args.seed,
        "model_name": args.model_name,
        "hidden_dim": args.hidden_dim,
        "patience": args.patience,
        "p_m": args.p_m,
        "alpha": args.alpha,
        "K": args.K,
        "beta": args.beta,
    }

    os.makedirs("results_celltype", exist_ok=True)
    summary_filename = (
        f"./results_celltype/summary_celltype_"
        f"labels{args.label_no}_iters{args.iterations}.json"
    )
    save_results_summary(summary_filename, "VIME", method_results, metadata)


def main():
    _setup_vime_paths()

    def load_mnist_data_for_pbmc(label_data_rate: float):
        return load_pbmc5k_data_with_celltype(label_data_rate)

    import data as vime_data
    vime_data.load_mnist_data = load_mnist_data_for_pbmc

    # Monkey-patch the models with new 256-128-64 architecture
    import self_supervised as vime_ss
    import autoencoder as vime_ae
    from self_supervised_vime import vime_self as vime_self_new
    from autoencoder import train_autoencoder as train_autoencoder_new
    vime_ss.vime_self = vime_self_new
    vime_ae.train_autoencoder = train_autoencoder_new

    if 'train' in sys.modules:
        del sys.modules['train']

    import train as vime_train
    vime_train.load_mnist_data = load_mnist_data_for_pbmc

    parser = argparse.ArgumentParser(
        description="Run VIME pipeline on PBMC 5k data with cell type labels"
    )

    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--model_name", choices=["logit", "xgboost", "mlp"], default="mlp")
    parser.add_argument("--label_no", type=int, default=1000)
    parser.add_argument("--p_m", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--label_data_rate", type=float, default=0.1)
    parser.add_argument("--dataset", choices=["mnist", "synthetic"], default="mnist")
    parser.add_argument("--n_noise_features", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    print("="*70)
    print("Running VIME with Cell Type Labels")
    print("="*70)
    print("Cell types: Monocytes, B cells, NK cells, CD4 T cells, CD8 T cells, Other")
    print("="*70)

    os.makedirs("save_model_celltype", exist_ok=True)
    run_experiments_with_save(args, vime_train)


if __name__ == "__main__":
    main()
