#!/bin/bash

# 运行三个方法在PBMC 5k数据集上，使用细胞类型标签
# Cell types: Monocytes, B cells, NK cells, CD4 T cells, CD8 T cells, Other

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "PBMC 5k 数据集实验 - 细胞类型标签"
echo "=========================================="
echo ""
echo "标签类型: 基于蛋白质标记的细胞类型 (6类)"
echo "  - Monocytes (CD14+)"
echo "  - B cells (CD19+)"
echo "  - NK cells (CD56+ CD3-)"
echo "  - CD4 T cells (CD3+ CD4+)"
echo "  - CD8 T cells (CD3+ CD8+)"
echo "  - Other"
echo ""
echo "=========================================="
echo ""

# 1. 运行 VAT
echo "=========================================="
echo "[1/3] 运行 VAT 方法..."
echo "=========================================="
cd "$SCRIPT_DIR/VAT"
CUDA_VISIBLE_DEVICES=1 python3 pbmc5k_vat_celltype.py
echo "VAT 运行完成"
echo ""

# 2. 运行 VAT_VIME_Feature
echo "=========================================="
echo "[2/3] 运行 VAT_VIME_Feature 方法..."
echo "=========================================="
cd "$SCRIPT_DIR/VAT_VIME_Feature"
CUDA_VISIBLE_DEVICES=1 python3 pbmc5k_vat_vime_feature_celltype.py
echo "VAT_VIME_Feature 运行完成"
echo ""

# 3. 运行 VIME
echo "=========================================="
echo "[3/3] 运行 VIME 方法..."
echo "=========================================="
cd "$SCRIPT_DIR/VIME"
CUDA_VISIBLE_DEVICES=1 python3 pbmc5k_vime_celltype.py
echo "VIME 运行完成"
echo ""

echo "=========================================="
echo "所有实验运行完成！"
echo "=========================================="
echo ""
echo "结果保存位置:"
echo "  - VAT/results_celltype/"
echo "  - VAT_VIME_Feature/results_celltype/"
echo "  - VIME/results_celltype/"
echo ""
