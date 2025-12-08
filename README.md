# VAT with VIME Feature Comparison

This project compares VAT (Virtual Adversarial Training) with different input types:
1. **VAT on X**: Original VAT with input as raw samples (baseline)
2. **VAT on Layer 1 Feature**: VAT with input as VIME Layer 1 features
3. **VAT on Layer 2 Feature**: VAT with input as VIME Layer 2 features
4. **VAT on Layer 3 Feature**: VAT with input as VIME Layer 3 features
5. **VAT on X+Feature (Concat)**: VAT with input as concatenation of X and Layer 3 features

## Key Differences from Original VAT

- **Original VAT**: Applies adversarial perturbation in the original input space X
- **VAT on Feature**: Applies adversarial perturbation in the VIME feature space
- **VAT on Concat**: Applies adversarial perturbation in the combined [X, Feature] space

## Multi-Layer VIME Encoder

The VIME encoder has been extended to 3 layers:
```
Layer 1: input_dim -> hidden_dim (ReLU)
Layer 2: hidden_dim -> hidden_dim (ReLU)
Layer 3: hidden_dim -> hidden_dim (ReLU)
```

This allows comparison of features from different depths of the encoder.

## Project Structure

```
VAT_VIME_Feature/
├── train.py              # Main training script
├── data.py               # Data loading utilities
├── baselines.py          # Supervised baselines (MLP, Logit, XGBoost)
├── autoencoder.py        # Autoencoder baseline
├── self_supervised.py    # Multi-layer VIME encoder
├── semi_supervised.py    # VAT/VAT-Mask with different input types
├── vat.py                # VAT core algorithm
├── utils.py              # Utility functions
├── run_experiments.sh    # Experiment runner script
└── results/              # Experiment results
```

## Usage

### Basic Usage

```bash
# VAT with different input types (default: synthetic data)
python train.py --method vat --dataset synthetic --iterations 5

# VAT-Mask with different input types
python train.py --method vat_mask --dataset synthetic --iterations 5

# MNIST dataset
python train.py --method vat --dataset mnist --iterations 5
```

### Run All Experiments

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--method` | vat | Semi-supervised method: vat or vat_mask |
| `--dataset` | synthetic | Dataset: synthetic or mnist |
| `--n_noise_features` | 0 | Number of noise features (synthetic only) |
| `--label_no` | 1000 | Number of labeled samples |
| `--iterations` | 10 | Number of independent runs |
| `--epsilon` | 1.0 | VAT perturbation magnitude |
| `--beta` | 1.0 | Unsupervised loss weight |
| `--p_m` | 0.3 | Mask probability (VIME and VAT-Mask) |
| `--alpha` | 2.0 | Feature reconstruction weight (VIME) |
| `--hidden_dim` | 100 | Hidden layer dimension |
| `--seed` | 42 | Random seed |

## Comparison Dimensions

The experiments compare:

1. **Perturbation Space**:
   - Input space X (original)
   - Feature space (Layer 1/2/3)
   - Concatenated space [X, Feature]

2. **Perturbation Method** (VAT vs VAT-Mask):
   - VAT: Gradient-based adversarial perturbation
   - VAT-Mask: Random mask + replacement

3. **Feature Depth**:
   - Layer 1: First hidden layer
   - Layer 2: Second hidden layer
   - Layer 3: Third hidden layer (full encoder)

## Output

Results are saved in `./results/` directory as JSON files containing:
- Per-iteration accuracy for each method
- Mean, std, min, max statistics
- Experiment metadata
