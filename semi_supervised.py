"""VAT semi-supervised learning with different input types.

This module implements semi-supervised learning using VAT with:
1. VAT on X: Original VAT with perturbation in input space
2. VAT on Feature: VAT with perturbation in feature space (VIME Layer 1/2/3)
3. VAT on X+Feature: VAT with perturbation in concatenated space

Each input type has its own separate function to keep the code simple
and avoid complex abstractions.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils import (
    generate_mask,
    corrupt_samples,
    EarlyStopping,
    log_iteration_progress,
    save_checkpoint,
    load_checkpoint,
)
from vat import (
    compute_vat_loss_with_encoder,
    compute_vat_loss_feature_space,
    compute_random_noise_loss_with_encoder,
)


class Predictor(nn.Module):
    """Predictor network."""

    def __init__(self, input_dim: int, hidden_dim: int, label_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, label_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.network(x)
        return logit, F.softmax(logit, dim=-1)


# =============================================================================
# VAT on X (Original): Perturbation in input space, same as original VAT
# =============================================================================


def vat_semi_on_x(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    epsilon: float,
    beta: float,
    encoder_path: str,
):
    """VAT semi-supervised learning on original input X.

    Standard VAT: Perturbation in input space, then encode.
    This is the original VAT algorithm.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode data for supervised loss
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device))
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device))
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device))

    predictor = Predictor(
        X_train_enc.shape[1], params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./results/save_model", exist_ok=True)
    model_path = "./results/save_model/vat_on_x_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch
        idx = np.random.permutation(len(X_train))[: params["batch_size"]]
        X_batch, y_batch = X_train_enc[idx], y_train_t[idx]

        # Sample unlabeled batch (original input space)
        X_u = X_unlabeled[
            np.random.permutation(len(X_unlabeled))[: params["batch_size"]]
        ]
        X_u_t = torch.from_numpy(X_u).float().to(device)

        predictor.train()

        # Supervised loss
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        # VAT unsupervised loss: Perturbation in input space
        unsup_loss = compute_vat_loss_with_encoder(
            encoder, predictor, X_u_t, epsilon=epsilon
        )

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_enc)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_enc)[1].cpu().numpy()


# =============================================================================
# Random Noise on X: Random perturbation in input space (baseline for VAT)
# =============================================================================


def random_noise_semi_on_x(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    epsilon: float,
    beta: float,
    encoder_path: str,
):
    """Random Noise semi-supervised learning on original input X.

    Baseline for VAT: Uses random noise instead of adversarial noise.
    Compares: VAT (adversarial perturbation) vs Random Noise (random perturbation).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode data for supervised loss
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device))
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device))
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device))

    predictor = Predictor(
        X_train_enc.shape[1], params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./results/save_model", exist_ok=True)
    model_path = "./results/save_model/random_noise_on_x_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch
        idx = np.random.permutation(len(X_train))[: params["batch_size"]]
        X_batch, y_batch = X_train_enc[idx], y_train_t[idx]

        # Sample unlabeled batch (original input space)
        X_u = X_unlabeled[
            np.random.permutation(len(X_unlabeled))[: params["batch_size"]]
        ]
        X_u_t = torch.from_numpy(X_u).float().to(device)

        predictor.train()

        # Supervised loss
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        # Random Noise unsupervised loss: Random perturbation in input space
        unsup_loss = compute_random_noise_loss_with_encoder(
            encoder, predictor, X_u_t, epsilon=epsilon
        )

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_enc)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_enc)[1].cpu().numpy()


# =============================================================================
# VAT on Feature: Perturbation in feature space
# =============================================================================


def vat_semi_on_feature(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    epsilon: float,
    beta: float,
    encoder_path: str,
):
    """VAT semi-supervised learning on VIME features.

    Perturbation happens in feature space (after VIME encoding).
    The encoder is used to transform data, then VAT is applied to features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode all data
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device))
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device))
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device))
        X_unlabeled_enc = encoder(torch.from_numpy(X_unlabeled).float().to(device))

    predictor = Predictor(
        X_train_enc.shape[1], params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./results/save_model", exist_ok=True)
    model_path = "./results/save_model/vat_on_feature_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch (in feature space)
        idx = np.random.permutation(len(X_train))[: params["batch_size"]]
        X_batch, y_batch = X_train_enc[idx], y_train_t[idx]

        # Sample unlabeled batch (in feature space)
        u_idx = np.random.permutation(len(X_unlabeled))[: params["batch_size"]]
        X_u_enc = X_unlabeled_enc[u_idx]

        predictor.train()

        # Supervised loss
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        # VAT unsupervised loss: Perturbation in feature space
        unsup_loss = compute_vat_loss_feature_space(
            predictor, X_u_enc, epsilon=epsilon
        )

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_enc)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_enc)[1].cpu().numpy()


# =============================================================================
# VAT on X+Feature (Concat): Perturbation in concatenated space
# =============================================================================


def vat_semi_on_concat(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    epsilon: float,
    beta: float,
    encoder_path: str,
):
    """VAT semi-supervised learning on concatenation of X and features.

    Input is [X, Feature], perturbation happens in this concatenated space.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode and concatenate all data
    with torch.no_grad():
        X_train_t = torch.from_numpy(X_train).float().to(device)
        X_val_t = torch.from_numpy(X_val).float().to(device)
        X_test_t = torch.from_numpy(X_test).float().to(device)
        X_unlabeled_t = torch.from_numpy(X_unlabeled).float().to(device)

        X_train_enc = encoder(X_train_t)
        X_val_enc = encoder(X_val_t)
        X_test_enc = encoder(X_test_t)
        X_unlabeled_enc = encoder(X_unlabeled_t)

        # Concatenate original X with features
        X_train_concat = torch.cat([X_train_t, X_train_enc], dim=1)
        X_val_concat = torch.cat([X_val_t, X_val_enc], dim=1)
        X_test_concat = torch.cat([X_test_t, X_test_enc], dim=1)
        X_unlabeled_concat = torch.cat([X_unlabeled_t, X_unlabeled_enc], dim=1)

    input_dim = X_train_concat.shape[1]
    predictor = Predictor(
        input_dim, params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./results/save_model", exist_ok=True)
    model_path = "./results/save_model/vat_on_concat_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch (in concat space)
        idx = np.random.permutation(len(X_train))[: params["batch_size"]]
        X_batch, y_batch = X_train_concat[idx], y_train_t[idx]

        # Sample unlabeled batch (in concat space)
        u_idx = np.random.permutation(len(X_unlabeled))[: params["batch_size"]]
        X_u_concat = X_unlabeled_concat[u_idx]

        predictor.train()

        # Supervised loss
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        # VAT unsupervised loss: Perturbation in concat space
        unsup_loss = compute_vat_loss_feature_space(
            predictor, X_u_concat, epsilon=epsilon
        )

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_concat)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_concat)[1].cpu().numpy()


# =============================================================================
# VAT-Mask Variants (for comparison)
# =============================================================================


def vat_mask_semi_on_x(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    p_mask: float,
    beta: float,
    encoder_path: str,
):
    """VAT-Mask semi-supervised learning on original input X.

    Uses random mask + replacement in input space + KL divergence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode data for supervised loss
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device))
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device))
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device))

    predictor = Predictor(
        X_train_enc.shape[1], params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./results/save_model", exist_ok=True)
    model_path = "./results/save_model/vat_mask_on_x_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch
        idx = np.random.permutation(len(X_train))[: params["batch_size"]]
        X_batch, y_batch = X_train_enc[idx], y_train_t[idx]

        # Sample unlabeled batch
        X_u = X_unlabeled[
            np.random.permutation(len(X_unlabeled))[: params["batch_size"]]
        ]

        # Generate corrupted samples in input space
        _, X_u_corrupt = corrupt_samples(generate_mask(p_mask, X_u), X_u)

        # Encode original and corrupted samples
        X_u_t = torch.from_numpy(X_u).float().to(device)
        X_u_corrupt_t = torch.from_numpy(X_u_corrupt).float().to(device)

        X_u_enc = encoder(X_u_t)
        X_u_corrupt_enc = encoder(X_u_corrupt_t)

        predictor.train()

        # Supervised loss
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        # VAT-Mask unsupervised loss: KL divergence with random mask
        with torch.no_grad():
            _, p_original = predictor(X_u_enc)

        _, p_corrupt = predictor(X_u_corrupt_enc)

        unsup_loss = F.kl_div(
            torch.log(p_corrupt + 1e-8),
            p_original.detach(),
            reduction="batchmean",
        )

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_enc)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_enc)[1].cpu().numpy()


def vat_mask_semi_on_feature(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    p_mask: float,
    beta: float,
    encoder_path: str,
):
    """VAT-Mask semi-supervised learning on VIME features.

    Uses random mask + replacement in feature space + KL divergence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode all data
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device)).cpu().numpy()
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
        X_unlabeled_enc = encoder(torch.from_numpy(X_unlabeled).float().to(device)).cpu().numpy()

    # Convert back to tensor
    X_train_enc_t = torch.from_numpy(X_train_enc).float().to(device)
    X_val_enc_t = torch.from_numpy(X_val_enc).float().to(device)
    X_test_enc_t = torch.from_numpy(X_test_enc).float().to(device)

    predictor = Predictor(
        X_train_enc.shape[1], params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./results/save_model", exist_ok=True)
    model_path = "./results/save_model/vat_mask_on_feature_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch (in feature space)
        idx = np.random.permutation(len(X_train_enc))[: params["batch_size"]]
        X_batch, y_batch = X_train_enc_t[idx], y_train_t[idx]

        # Sample unlabeled batch (in feature space, numpy for mask generation)
        u_idx = np.random.permutation(len(X_unlabeled_enc))[: params["batch_size"]]
        X_u_enc = X_unlabeled_enc[u_idx]

        # Generate corrupted samples in feature space
        _, X_u_enc_corrupt = corrupt_samples(generate_mask(p_mask, X_u_enc), X_u_enc)

        X_u_enc_t = torch.from_numpy(X_u_enc).float().to(device)
        X_u_enc_corrupt_t = torch.from_numpy(X_u_enc_corrupt).float().to(device)

        predictor.train()

        # Supervised loss
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        # VAT-Mask unsupervised loss: KL divergence with random mask in feature space
        with torch.no_grad():
            _, p_original = predictor(X_u_enc_t)

        _, p_corrupt = predictor(X_u_enc_corrupt_t)

        unsup_loss = F.kl_div(
            torch.log(p_corrupt + 1e-8),
            p_original.detach(),
            reduction="batchmean",
        )

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_enc_t)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_enc_t)[1].cpu().numpy()


def vat_mask_semi_on_concat(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    p_mask: float,
    beta: float,
    encoder_path: str,
):
    """VAT-Mask semi-supervised learning on concatenation of X and features.

    Uses random mask + replacement in concat space + KL divergence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode and concatenate all data (numpy for mask generation)
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device)).cpu().numpy()
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
        X_unlabeled_enc = encoder(torch.from_numpy(X_unlabeled).float().to(device)).cpu().numpy()

    # Concatenate
    X_train_concat = np.hstack([X_train, X_train_enc])
    X_val_concat = np.hstack([X_val, X_val_enc])
    X_test_concat = np.hstack([X_test, X_test_enc])
    X_unlabeled_concat = np.hstack([X_unlabeled, X_unlabeled_enc])

    X_train_concat_t = torch.from_numpy(X_train_concat).float().to(device)
    X_val_concat_t = torch.from_numpy(X_val_concat).float().to(device)
    X_test_concat_t = torch.from_numpy(X_test_concat).float().to(device)

    input_dim = X_train_concat.shape[1]
    predictor = Predictor(
        input_dim, params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./results/save_model", exist_ok=True)
    model_path = "./results/save_model/vat_mask_on_concat_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch (in concat space)
        idx = np.random.permutation(len(X_train_concat))[: params["batch_size"]]
        X_batch, y_batch = X_train_concat_t[idx], y_train_t[idx]

        # Sample unlabeled batch (in concat space, numpy for mask generation)
        u_idx = np.random.permutation(len(X_unlabeled_concat))[: params["batch_size"]]
        X_u_concat = X_unlabeled_concat[u_idx]

        # Generate corrupted samples in concat space
        _, X_u_concat_corrupt = corrupt_samples(generate_mask(p_mask, X_u_concat), X_u_concat)

        X_u_concat_t = torch.from_numpy(X_u_concat).float().to(device)
        X_u_concat_corrupt_t = torch.from_numpy(X_u_concat_corrupt).float().to(device)

        predictor.train()

        # Supervised loss
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        # VAT-Mask unsupervised loss
        with torch.no_grad():
            _, p_original = predictor(X_u_concat_t)

        _, p_corrupt = predictor(X_u_concat_corrupt_t)

        unsup_loss = F.kl_div(
            torch.log(p_corrupt + 1e-8),
            p_original.detach(),
            reduction="batchmean",
        )

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_concat_t)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_concat_t)[1].cpu().numpy()
