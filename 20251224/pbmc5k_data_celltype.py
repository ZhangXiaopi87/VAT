"""Load PBMC 5k data with biologically meaningful cell type labels.

This module generates cell type labels based on protein markers:
- B cells: CD19+
- CD4 T cells: CD3+ CD4+
- CD8 T cells: CD3+ CD8+
- NK cells: CD56+ CD3-
- Monocytes: CD14+
- Other: remaining cells

References:
    10X Genomics PBMC datasets typically contain these major immune cell types.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py


def split_labeled_unlabeled(X, y, label_data_rate, seed=42):
    """Split data into labeled and unlabeled sets.

    This is a standalone implementation to avoid importing from VAT code.
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_labeled = int(n_samples * label_data_rate)

    # Random shuffle
    indices = np.random.permutation(n_samples)
    labeled_idx = indices[:n_labeled]
    unlabeled_idx = indices[n_labeled:]

    X_label = X[labeled_idx]
    y_label = y[labeled_idx]
    X_unlab = X[unlabeled_idx]

    return X_label, y_label, X_unlab


def generate_celltype_labels(protein_expression, protein_names):
    """Generate cell type labels based on protein markers.

    Args:
        protein_expression: (n_cells, n_proteins) array
        protein_names: list of protein names

    Returns:
        cell_types: (n_cells,) array with integer labels
            0 = Monocytes (CD14+)
            1 = B cells (CD19+)
            2 = NK cells (CD56+ CD3-)
            3 = CD4 T cells (CD3+ CD4+)
            4 = CD8 T cells (CD3+ CD8+)
            5 = Other
    """
    # Decode protein names if needed
    if isinstance(protein_names[0], bytes):
        protein_names = [p.decode('utf-8') for p in protein_names]

    # Find marker indices
    markers = {}
    for marker in ['CD3', 'CD4', 'CD8a', 'CD14', 'CD19', 'CD56']:
        for i, name in enumerate(protein_names):
            if marker in name:
                markers[marker] = i
                break

    print(f"Found markers: {list(markers.keys())}")

    # Get marker expression
    cd3_expr = protein_expression[:, markers['CD3']]
    cd4_expr = protein_expression[:, markers['CD4']]
    cd8_expr = protein_expression[:, markers['CD8a']]
    cd14_expr = protein_expression[:, markers['CD14']]
    cd19_expr = protein_expression[:, markers['CD19']]
    cd56_expr = protein_expression[:, markers['CD56']]

    # Use percentile thresholds to define positive cells
    # Using 60th percentile as a relatively permissive threshold
    threshold_pct = 60
    cd3_pos = cd3_expr > np.percentile(cd3_expr, threshold_pct)
    cd4_pos = cd4_expr > np.percentile(cd4_expr, threshold_pct)
    cd8_pos = cd8_expr > np.percentile(cd8_expr, threshold_pct)
    cd14_pos = cd14_expr > np.percentile(cd14_expr, threshold_pct)
    cd19_pos = cd19_expr > np.percentile(cd19_expr, threshold_pct)
    cd56_pos = cd56_expr > np.percentile(cd56_expr, threshold_pct)

    # Assign cell types (order matters - more specific first)
    cell_types = np.full(len(cd3_expr), 5, dtype=np.int32)  # Default: Other

    # Priority order (to avoid conflicts):
    cell_types[cd14_pos] = 0  # Monocytes
    cell_types[cd19_pos] = 1  # B cells
    cell_types[cd56_pos & ~cd3_pos] = 2  # NK cells
    cell_types[cd3_pos & cd4_pos & ~cd8_pos] = 3  # CD4 T cells
    cell_types[cd3_pos & cd8_pos & ~cd4_pos] = 4  # CD8 T cells

    # Print cell type distribution
    cell_type_names = [
        'Monocytes', 'B cells', 'NK cells',
        'CD4 T cells', 'CD8 T cells', 'Other'
    ]

    print("\nCell type distribution:")
    for ct in range(6):
        count = np.sum(cell_types == ct)
        pct = count / len(cell_types) * 100
        print(f"  {cell_type_names[ct]}: {count} cells ({pct:.1f}%)")

    return cell_types, cell_type_names


def load_pbmc5k_data_with_celltype(label_data_rate: float, seed: int = 42):
    """Load PBMC 5k CITE-seq data with cell type labels.

    Args:
        label_data_rate: Fraction of training data to use as labeled
        seed: Random seed for reproducibility

    Returns:
        X_label, y_label, X_unlab, X_valid, y_valid, X_test, y_test
        where y_* are one-hot encoded cell type labels (6 classes)
    """
    data_path = (
        Path(__file__)
        .resolve()
        .parent.parent.parent
        / "totalVI_reproducibility"
        / "data"
        / "pbmc_5k_protein_v3.h5ad"
    )

    # Fallback to absolute path
    if not data_path.exists():
        data_path = Path(
            "/userhome/home/zhangqirui/project/self-play/totalVI_reproducibility/data/pbmc_5k_protein_v3.h5ad"
        )

    print(f"Loading data from: {data_path}")

    with h5py.File(data_path, 'r') as f:
        # Load RNA expression
        X = f['X'][()]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        # Load protein expression
        protein_expression = f['obsm']['protein_expression'][()]
        protein_names = f['uns']['protein_names'][()]

        print(f"Data shape: {X.shape}")
        print(f"Protein shape: {protein_expression.shape}")

    # Generate cell type labels
    cell_types, cell_type_names = generate_celltype_labels(
        protein_expression, protein_names
    )

    # Convert to one-hot encoding
    y_onehot = np.zeros((len(cell_types), 6), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        y_onehot[i, ct] = 1.0

    # Split into train/valid/test
    n_samples = X.shape[0]
    n_train = int(n_samples * 0.64)
    n_valid = int(n_samples * 0.16)

    X_train = X[:n_train]
    y_train = y_onehot[:n_train]

    X_valid = X[n_train : n_train + n_valid]
    y_valid = y_onehot[n_train : n_train + n_valid]

    X_test = X[n_train + n_valid :]
    y_test = y_onehot[n_train + n_valid :]

    # Split labeled/unlabeled
    X_label, y_label, X_unlab = split_labeled_unlabeled(
        X_train, y_train, label_data_rate, seed=seed
    )

    print(f"\nData splits:")
    print(f"  Labeled: {X_label.shape[0]} samples")
    print(f"  Unlabeled: {X_unlab.shape[0]} samples")
    print(f"  Validation: {X_valid.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    return X_label, y_label, X_unlab, X_valid, y_valid, X_test, y_test


__all__ = ["load_pbmc5k_data_with_celltype"]


if __name__ == "__main__":
    # Test the data loader
    print("="*70)
    print("Testing PBMC 5k cell type label generation")
    print("="*70)

    X_label, y_label, X_unlab, X_valid, y_valid, X_test, y_test = \
        load_pbmc5k_data_with_celltype(label_data_rate=0.1, seed=42)

    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)
