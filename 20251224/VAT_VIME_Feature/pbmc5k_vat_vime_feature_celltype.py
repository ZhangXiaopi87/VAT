"""Run VAT_VIME_Feature pipeline on PBMC 5k data with cell type labels."""

import argparse
import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# Add base/ directory for new models
base_models_dir = base_dir / "base"
if str(base_models_dir) not in sys.path:
    sys.path.insert(0, str(base_models_dir))

from pbmc5k_data_celltype import load_pbmc5k_data_with_celltype


def _setup_vat_vime_paths():
    base_dir = Path(__file__).resolve().parent
    # 指向 tem/20251210/VAT_VIME_Feature
    method_dir = base_dir.parent.parent.parent / "20251210" / "VAT_VIME_Feature"
    method_dir_str = str(method_dir)
    if method_dir_str not in sys.path:
        sys.path.insert(0, method_dir_str)


def main():
    _setup_vat_vime_paths()

    def load_mnist_data_for_pbmc(label_data_rate: float):
        return load_pbmc5k_data_with_celltype(label_data_rate)

    import data as method_data
    method_data.load_mnist_data = load_mnist_data_for_pbmc

    # Monkey-patch the models with new 256-128-64 architecture
    import self_supervised_multilayer as method_ssml
    from self_supervised_multilayer import (
        vime_self_multilayer as vime_self_multilayer_new,
        get_encoders_from_model as get_encoders_new
    )
    method_ssml.vime_self_multilayer = vime_self_multilayer_new
    method_ssml.get_encoders_from_model = get_encoders_new

    if 'train' in sys.modules:
        del sys.modules['train']

    import train as method_train
    method_train.load_mnist_data = load_mnist_data_for_pbmc

    parser = argparse.ArgumentParser(
        description="Run VAT_VIME_Feature pipeline on PBMC 5k data with cell type labels"
    )

    parser.add_argument("--method", choices=["vat", "vat_mask"], default="vat")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--model_name", choices=["logit", "xgboost", "mlp"], default="mlp")
    parser.add_argument("--label_no", type=int, default=1000)
    parser.add_argument("--label_data_rate", type=float, default=0.1)
    parser.add_argument("--dataset", choices=["mnist", "synthetic"], default="mnist")
    parser.add_argument("--n_noise_features", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--p_m", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    args = parser.parse_args()

    print("="*70)
    print("Running VAT_VIME_Feature with Cell Type Labels")
    print("="*70)
    print("Cell types: Monocytes, B cells, NK cells, CD4 T cells, CD8 T cells, Other")
    print("="*70)

    os.makedirs("results_celltype", exist_ok=True)
    method_train.run_experiments(args)


if __name__ == "__main__":
    main()
