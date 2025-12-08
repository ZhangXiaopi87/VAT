"""VAT with VIME Feature training script.

This script compares VAT with different input types:
1. VAT on X: Original VAT with input as raw samples
2. VAT on Layer 1 Feature: VAT on VIME Layer 1 features
3. VAT on Layer 2 Feature: VAT on VIME Layer 2 features
4. VAT on Layer 3 Feature: VAT on VIME Layer 3 features
5. VAT on X+Feature: VAT on concatenation of X and Layer 3 features

Training pipeline:
1. Train supervised baseline (MLP)
2. Train Autoencoder baseline
3. Train multi-layer VIME encoder
4. Run VAT/VAT-Mask with different input types
"""

import argparse
import numpy as np
import os
import torch
import random

from data import load_mnist_data, load_synthetic_data
from baselines import logit, xgb_model, mlp
from self_supervised import (
    vime_self_multilayer,
    Layer1Encoder,
    Layer2Encoder,
    Layer3Encoder,
    ConcatEncoder,
)
from semi_supervised import (
    vat_semi_on_x,
    vat_semi_on_feature,
    vat_semi_on_concat,
    vat_mask_semi_on_x,
    vat_mask_semi_on_feature,
    vat_mask_semi_on_concat,
    random_noise_semi_on_x,
)
from autoencoder import train_autoencoder
from utils import evaluate, save_results_summary

# Print separators
SEP_MAJOR = "=" * 70
SEP_MINOR = "-" * 70
SEP_HEADER = "#" * 70


def transform_with_encoder(encoder, X_train, X_val, X_test):
    """Transform data using a trained encoder."""
    device = next(encoder.parameters()).device
    with torch.no_grad():
        X_train_enc = (
            encoder(torch.from_numpy(X_train).float().to(device)).cpu().numpy()
        )
        X_val_enc = (
            encoder(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
        )
        X_test_enc = (
            encoder(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
        )
    return X_train_enc, X_val_enc, X_test_enc


def set_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_supervised(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model,
    metric,
    hidden_dim=100,
    patience=50,
):
    """Train supervised models."""
    if model in ("logit", "xgboost"):
        X_full = np.vstack([X_train, X_val])
        y_full = np.vstack([y_train, y_val])
        y_pred = (
            logit(X_full, y_full, X_test)
            if model == "logit"
            else xgb_model(X_full, y_full, X_test)
        )
    elif model == "mlp":
        params = {
            "hidden_dim": hidden_dim,
            "epochs": 1000,
            "activation": "relu",
            "batch_size": 100,
            "patience": patience,
        }
        y_pred = mlp(X_train, y_train, X_val, y_val, X_test, params)
    else:
        raise ValueError(f"Unknown model: {model}")

    return evaluate(metric, y_test, y_pred)


def train_vat_feature_pipeline(
    label_rate,
    models,
    n_labeled,
    p_mask,
    alpha,
    epsilon,
    beta,
    method="vat",
    dataset="mnist",
    n_noise=0,
    hidden_dim=100,
    patience=50,
):
    """VAT with VIME Feature main training pipeline.

    Training steps:
    1. Supervised baseline
    2. Autoencoder baseline
    3. Multi-layer VIME encoder training
    4. VAT/VAT-Mask with different input types

    Returns:
        results: Dictionary of accuracies for each method
    """
    results = {}

    # Load data
    if dataset == "mnist":
        X_train, y_train, X_unlabeled, X_val, y_val, X_test, y_test = load_mnist_data(
            label_rate
        )
    else:
        X_train, y_train, X_unlabeled, X_val, y_val, X_test, y_test = (
            load_synthetic_data(
                label_rate, total_features=200, n_noise_features=n_noise
            )
        )

    X_train, y_train = X_train[:n_labeled], y_train[:n_labeled]
    input_dim = X_train.shape[1]

    # =========================================================================
    # STEP 1: Train supervised models
    # =========================================================================
    print(f"\n{SEP_MAJOR}\nSTEP 1: Training Supervised Baseline ({models[0].upper()})\n{SEP_MAJOR}")
    results["Supervised"] = train_supervised(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        models[0],
        "acc",
        hidden_dim,
        patience,
    )
    print(f"Supervised baseline accuracy: {results['Supervised']:.4f}")

    # =========================================================================
    # STEP 2: Train Autoencoder
    # =========================================================================
    print(f"\n{SEP_MAJOR}\nSTEP 2: Training Autoencoder\n{SEP_MAJOR}")
    X_all = np.vstack([X_train, X_unlabeled])
    encoder_params = {
        "batch_size": 128,
        "epochs": 1000,
        "hidden_dim": hidden_dim,
        "patience": patience,
    }
    encoder_ae, _ = train_autoencoder(X_all, X_val, encoder_params)

    os.makedirs("./results/save_model", exist_ok=True)
    torch.save(encoder_ae, "./results/save_model/autoencoder_encoder.pth")

    # Test Autoencoder
    print(f"\n{SEP_MINOR}\nTesting Autoencoder + MLP classifier\n{SEP_MINOR}")
    X_train_ae, X_val_ae, X_test_ae = transform_with_encoder(
        encoder_ae, X_train, X_val, X_test
    )

    results["Autoencoder"] = train_supervised(
        X_train_ae,
        y_train,
        X_val_ae,
        y_val,
        X_test_ae,
        y_test,
        "mlp",
        "acc",
        hidden_dim,
        patience,
    )
    print(f"Autoencoder + MLP accuracy: {results['Autoencoder']:.4f}")

    # =========================================================================
    # STEP 3: Train Multi-Layer VIME Encoder
    # =========================================================================
    print(f"\n{SEP_MAJOR}\nSTEP 3: Training Multi-Layer VIME Encoder\n{SEP_MAJOR}")
    vime_params = {
        "batch_size": 128,
        "epochs": 1000,
        "hidden_dim": hidden_dim,
        "patience": patience,
    }
    vime_model = vime_self_multilayer(X_all, X_val, p_mask, alpha, vime_params)

    # Save the full model
    torch.save(vime_model, "./results/save_model/vime_multilayer_model.pth")

    # Create and save layer-specific encoders
    device = next(vime_model.parameters()).device
    encoder_layer1 = Layer1Encoder(vime_model).to(device).eval()
    encoder_layer2 = Layer2Encoder(vime_model).to(device).eval()
    encoder_layer3 = Layer3Encoder(vime_model).to(device).eval()
    encoder_concat = ConcatEncoder(vime_model).to(device).eval()

    torch.save(encoder_layer1, "./results/save_model/encoder_layer1.pth")
    torch.save(encoder_layer2, "./results/save_model/encoder_layer2.pth")
    torch.save(encoder_layer3, "./results/save_model/encoder_layer3.pth")
    torch.save(encoder_concat, "./results/save_model/encoder_concat.pth")

    # Test VIME Layer 3 (full encoder) with MLP
    print(f"\n{SEP_MINOR}\nTesting VIME-Layer3 + MLP classifier\n{SEP_MINOR}")
    X_train_l3, X_val_l3, X_test_l3 = transform_with_encoder(
        encoder_layer3, X_train, X_val, X_test
    )

    results["VIME-Layer3"] = train_supervised(
        X_train_l3,
        y_train,
        X_val_l3,
        y_val,
        X_test_l3,
        y_test,
        "mlp",
        "acc",
        hidden_dim,
        patience,
    )
    print(f"VIME-Layer3 + MLP accuracy: {results['VIME-Layer3']:.4f}")

    # =========================================================================
    # STEP 4: VAT/VAT-Mask with Different Input Types
    # =========================================================================
    semi_params = {
        "hidden_dim": hidden_dim,
        "batch_size": 128,
        "iterations": 1000,
        "patience": 100,
    }

    if method == "vat":
        print(f"\n{SEP_MAJOR}\nSTEP 4: Training VAT with Different Input Types\n{SEP_MAJOR}")
        print(f"  Epsilon: {epsilon}, Beta: {beta}")

        # VAT on X (Original) - Adversarial Perturbation
        print(f"\n{SEP_MINOR}\n4.1 VAT on X (Adversarial Perturbation)\n{SEP_MINOR}")
        y_pred = vat_semi_on_x(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, epsilon, beta,
            "./results/save_model/encoder_layer3.pth",
        )
        results["VAT-on-X"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-on-X accuracy: {results['VAT-on-X']:.4f}")

        # Random Noise on X - Baseline for VAT (Random Perturbation)
        print(f"\n{SEP_MINOR}\n4.2 Random Noise on X (Random Perturbation - Baseline)\n{SEP_MINOR}")
        y_pred = random_noise_semi_on_x(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, epsilon, beta,
            "./results/save_model/encoder_layer3.pth",
        )
        results["RandomNoise-on-X"] = evaluate("acc", y_test, y_pred)
        print(f"RandomNoise-on-X accuracy: {results['RandomNoise-on-X']:.4f}")

        # VAT on Layer 1 Feature
        print(f"\n{SEP_MINOR}\n4.3 VAT on Layer 1 Feature\n{SEP_MINOR}")
        y_pred = vat_semi_on_feature(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, epsilon, beta,
            "./results/save_model/encoder_layer1.pth",
        )
        results["VAT-on-Layer1"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-on-Layer1 accuracy: {results['VAT-on-Layer1']:.4f}")

        # VAT on Layer 2 Feature
        print(f"\n{SEP_MINOR}\n4.4 VAT on Layer 2 Feature\n{SEP_MINOR}")
        y_pred = vat_semi_on_feature(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, epsilon, beta,
            "./results/save_model/encoder_layer2.pth",
        )
        results["VAT-on-Layer2"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-on-Layer2 accuracy: {results['VAT-on-Layer2']:.4f}")

        # VAT on Layer 3 Feature
        print(f"\n{SEP_MINOR}\n4.5 VAT on Layer 3 Feature\n{SEP_MINOR}")
        y_pred = vat_semi_on_feature(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, epsilon, beta,
            "./results/save_model/encoder_layer3.pth",
        )
        results["VAT-on-Layer3"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-on-Layer3 accuracy: {results['VAT-on-Layer3']:.4f}")

        # VAT on X + Layer 3 Feature (Concat)
        print(f"\n{SEP_MINOR}\n4.6 VAT on X + Layer3 Feature (Concat)\n{SEP_MINOR}")
        y_pred = vat_semi_on_concat(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, epsilon, beta,
            "./results/save_model/encoder_layer3.pth",
        )
        results["VAT-on-Concat"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-on-Concat accuracy: {results['VAT-on-Concat']:.4f}")

    else:  # vat_mask
        print(f"\n{SEP_MAJOR}\nSTEP 4: Training VAT-Mask with Different Input Types\n{SEP_MAJOR}")
        print(f"  p_mask: {p_mask}, Beta: {beta}")

        # VAT-Mask on X (Original)
        print(f"\n{SEP_MINOR}\n4.1 VAT-Mask on X (Original)\n{SEP_MINOR}")
        y_pred = vat_mask_semi_on_x(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, p_mask, beta,
            "./results/save_model/encoder_layer3.pth",
        )
        results["VAT-Mask-on-X"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-Mask-on-X accuracy: {results['VAT-Mask-on-X']:.4f}")

        # VAT-Mask on Layer 1 Feature
        print(f"\n{SEP_MINOR}\n4.2 VAT-Mask on Layer 1 Feature\n{SEP_MINOR}")
        y_pred = vat_mask_semi_on_feature(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, p_mask, beta,
            "./results/save_model/encoder_layer1.pth",
        )
        results["VAT-Mask-on-Layer1"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-Mask-on-Layer1 accuracy: {results['VAT-Mask-on-Layer1']:.4f}")

        # VAT-Mask on Layer 2 Feature
        print(f"\n{SEP_MINOR}\n4.3 VAT-Mask on Layer 2 Feature\n{SEP_MINOR}")
        y_pred = vat_mask_semi_on_feature(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, p_mask, beta,
            "./results/save_model/encoder_layer2.pth",
        )
        results["VAT-Mask-on-Layer2"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-Mask-on-Layer2 accuracy: {results['VAT-Mask-on-Layer2']:.4f}")

        # VAT-Mask on Layer 3 Feature
        print(f"\n{SEP_MINOR}\n4.4 VAT-Mask on Layer 3 Feature\n{SEP_MINOR}")
        y_pred = vat_mask_semi_on_feature(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, p_mask, beta,
            "./results/save_model/encoder_layer3.pth",
        )
        results["VAT-Mask-on-Layer3"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-Mask-on-Layer3 accuracy: {results['VAT-Mask-on-Layer3']:.4f}")

        # VAT-Mask on X + Layer 3 Feature (Concat)
        print(f"\n{SEP_MINOR}\n4.5 VAT-Mask on X + Layer3 Feature (Concat)\n{SEP_MINOR}")
        y_pred = vat_mask_semi_on_concat(
            X_train, y_train, X_val, y_val, X_unlabeled, X_test,
            semi_params, p_mask, beta,
            "./results/save_model/encoder_layer3.pth",
        )
        results["VAT-Mask-on-Concat"] = evaluate("acc", y_test, y_pred)
        print(f"VAT-Mask-on-Concat accuracy: {results['VAT-Mask-on-Concat']:.4f}")

    return results


def run_experiments(args):
    """Run experiments with multiple iterations."""
    all_results = []
    method_name = "VAT" if args.method == "vat" else "VAT-Mask"

    for i in range(args.iterations):
        set_seed(args.seed + i)
        print(f"\n{SEP_HEADER}\n# ITERATION {i+1}/{args.iterations} ({method_name})\n{SEP_HEADER}")

        results = train_vat_feature_pipeline(
            args.label_data_rate,
            [args.model_name],
            args.label_no,
            args.p_m,
            args.alpha,
            args.epsilon,
            args.beta,
            args.method,
            args.dataset,
            args.n_noise_features,
            args.hidden_dim,
            args.patience,
        )
        all_results.append(results)

        print(f"\n{SEP_MINOR}")
        print(f"Iteration {i+1} Results:")
        for name, acc in results.items():
            print(f"  {name}: {acc:.4f}")
        print(SEP_MINOR)

    # Aggregate results
    print(f"\n{SEP_HEADER}\n# FINAL RESULTS (Average over {args.iterations} iterations)\n{SEP_HEADER}")

    # Get all method names from the first result
    method_names = list(all_results[0].keys())

    aggregated = {}
    for name in method_names:
        values = [r[name] for r in all_results]
        aggregated[name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "all_values": values,
        }
        print(f"{name}: Avg={aggregated[name]['mean']:.4f}, Std={aggregated[name]['std']:.4f}")

    print(SEP_HEADER)

    # Save results
    per_iteration = []
    for i, result in enumerate(all_results):
        per_iteration.append({
            "iteration": i + 1,
            "seed": args.seed + i,
            "accuracies": result,
        })

    method_results = {
        "per_iteration": per_iteration,
        "summary": aggregated,
    }

    metadata = {
        "dataset": args.dataset,
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
        "beta": args.beta,
        "epsilon": args.epsilon,
        "method": args.method,
    }

    summary_filename = (
        f"./results/summary_{args.dataset}_noise{args.n_noise_features}_"
        f"labels{args.label_no}_{method_name}_iters{args.iterations}.json"
    )
    save_results_summary(summary_filename, method_name, method_results, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAT with VIME Feature Training Script")

    # Method selection
    parser.add_argument(
        "--method", choices=["vat", "vat_mask"], default="vat",
        help="Semi-supervised method: vat (adversarial) or vat_mask (random mask + KL)"
    )

    # Experiment settings
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of independent runs")
    parser.add_argument(
        "--model_name", choices=["logit", "xgboost", "mlp"], default="mlp",
        help="Baseline model for supervised learning"
    )
    parser.add_argument("--label_no", type=int, default=1000,
                        help="Number of labeled samples")
    parser.add_argument("--label_data_rate", type=float, default=0.1,
                        help="Fraction of data to use as labeled")
    parser.add_argument(
        "--dataset", choices=["mnist", "synthetic"], default="synthetic",
        help="Dataset to use"
    )
    parser.add_argument("--n_noise_features", type=int, default=0,
                        help="Number of noise features (for synthetic data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--hidden_dim", type=int, default=100,
                        help="Hidden dimension for all models")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")

    # VIME parameters
    parser.add_argument("--p_m", type=float, default=0.3,
                        help="Mask probability for VIME and VAT-Mask")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Weight for feature reconstruction in VIME")

    # VAT parameters
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="VAT perturbation magnitude (for VAT method)")

    # Semi-supervised loss weight
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for unsupervised loss")

    run_experiments(parser.parse_args())
