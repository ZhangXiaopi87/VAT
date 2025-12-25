"""Run VAT pipeline on PBMC 5k data with cell type labels.

This script uses biologically meaningful cell type labels instead of n_counts bins:
- Monocytes (CD14+)
- B cells (CD19+)
- NK cells (CD56+ CD3-)
- CD4 T cells (CD3+ CD4+)
- CD8 T cells (CD3+ CD8+)
- Other
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# Add base/ directory for new models
base_models_dir = base_dir / "base"
if str(base_models_dir) not in sys.path:
    sys.path.insert(0, str(base_models_dir))

from pbmc5k_data_celltype import load_pbmc5k_data_with_celltype


def _setup_vat_paths():
    """Add 20251210/VAT to sys.path."""
    base_dir = Path(__file__).resolve().parent
    # 需要指向 tem/20251210/VAT（同级目录下的旧版 VAT 代码）
    vat_dir = base_dir.parent.parent.parent / "20251210" / "VAT"
    vat_dir_str = str(vat_dir)
    if vat_dir_str not in sys.path:
        sys.path.insert(0, vat_dir_str)


def main():
    _setup_vat_paths()

    # Replace load_mnist_data with our cell type version
    def load_mnist_data_for_pbmc(label_data_rate: float):
        return load_pbmc5k_data_with_celltype(label_data_rate)

    # Monkey-patch the data loading function
    import data as vat_data
    vat_data.load_mnist_data = load_mnist_data_for_pbmc

    # Monkey-patch the models with new 256-128-64 architecture
    import self_supervised as vat_ss
    import autoencoder as vat_ae
    from self_supervised_vime import vime_self as vime_self_new
    from autoencoder import train_autoencoder as train_autoencoder_new
    vat_ss.vime_self = vime_self_new
    vat_ae.train_autoencoder = train_autoencoder_new

    # Remove train module from cache if it exists
    if 'train' in sys.modules:
        del sys.modules['train']

    # Import train module (now with replaced data loader and models)
    import train as vat_train
    vat_train.load_mnist_data = load_mnist_data_for_pbmc

    parser = argparse.ArgumentParser(
        description="Run VAT pipeline on PBMC 5k data with cell type labels"
    )

    parser.add_argument(
        "--method",
        choices=["vat", "vat_mask"],
        default="vat",
        help="Semi-supervised method",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--model_name",
        choices=["logit", "xgboost", "mlp"],
        default="mlp",
    )
    parser.add_argument("--label_no", type=int, default=1000)
    parser.add_argument("--label_data_rate", type=float, default=0.1)
    parser.add_argument(
        "--dataset",
        choices=["mnist", "synthetic"],
        default="mnist",
        help="This parameter only affects file naming; actual data is PBMC 5k with cell types.",
    )
    parser.add_argument("--n_noise_features", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--p_m", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    args = parser.parse_args()

    # Create results directory
    os.makedirs("results_celltype", exist_ok=True)

    print("="*70)
    print("Running VAT with Cell Type Labels")
    print("="*70)
    print("Cell types:")
    print("  0: Monocytes (CD14+)")
    print("  1: B cells (CD19+)")
    print("  2: NK cells (CD56+ CD3-)")
    print("  3: CD4 T cells (CD3+ CD4+)")
    print("  4: CD8 T cells (CD3+ CD8+)")
    print("  5: Other")
    print("="*70)

    vat_train.run_experiments(args)


if __name__ == "__main__":
    main()
