#!/bin/bash
# Run VAT with VIME Feature comparison experiments

echo "=================================================="
echo "VAT with VIME Feature Comparison Experiments"
echo "=================================================="

# Experiment 1: Synthetic data, 0 noise features
echo ""
echo "Experiment 1: Synthetic data, 0 noise features (0% noise)"
echo "----------------------------------------------------------"
python train.py --method vat --dataset synthetic --n_noise_features 0 --iterations 5 --seed 42
python train.py --method vat_mask --dataset synthetic --n_noise_features 0 --iterations 5 --seed 42

# Experiment 2: Synthetic data, 50 noise features
echo ""
echo "Experiment 2: Synthetic data, 50 noise features (25% noise)"
echo "----------------------------------------------------------"
python train.py --method vat --dataset synthetic --n_noise_features 50 --iterations 5 --seed 42
python train.py --method vat_mask --dataset synthetic --n_noise_features 50 --iterations 5 --seed 42

# Experiment 3: Synthetic data, 100 noise features
echo ""
echo "Experiment 3: Synthetic data, 100 noise features (50% noise)"
echo "----------------------------------------------------------"
python train.py --method vat --dataset synthetic --n_noise_features 100 --iterations 5 --seed 42
python train.py --method vat_mask --dataset synthetic --n_noise_features 100 --iterations 5 --seed 42

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "Results saved in ./results/ directory"
echo "=================================================="
