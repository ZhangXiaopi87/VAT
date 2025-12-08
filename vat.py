"""Virtual Adversarial Training (VAT) implementation.

This module implements VAT with support for different input types:
1. VAT on X: Original VAT with input as raw samples (existing VAT)
2. VAT on Feature: VAT with input as VIME extracted features
3. VAT on X+Feature: VAT with input as concatenation of X and features

Key formulas:
- VAT: r_adv = argmax_{r; ||r|| <= epsilon} D_KL[p(y|x, theta_hat) || p(y|x+r, theta)]
"""

import numpy as np
import torch
import torch.nn.functional as F


def get_normalized_vector(d):
    """Normalize vector to unit norm along dim=1."""
    d_norm = torch.norm(d, p=2, dim=1, keepdim=True)
    return d / (d_norm + 1e-12)


# =============================================================================
# VAT Loss (Perturbation in feature space)
# =============================================================================


def compute_vat_loss_feature_space(
    predictor,
    x_feature,
    epsilon=1.0,
    xi=1e-6,
    num_iterations=1,
):
    """Compute VAT loss with perturbation in FEATURE space.

    This version applies adversarial perturbation directly to the features
    (not in the original input space). Used when VAT is applied on VIME features.

    Args:
        predictor: Predictor network f(z) -> (logits, probs)
        x_feature: Feature tensor (already encoded by VIME)
        epsilon: Maximum perturbation magnitude
        xi: Small constant for finite difference
        num_iterations: Number of power iteration steps

    Returns:
        VAT loss (scalar tensor)
    """
    # Get original prediction (detached)
    with torch.no_grad():
        _, p_original = predictor(x_feature)

    # Initialize random unit vector in feature space
    d = torch.randn_like(x_feature)
    d = get_normalized_vector(d)

    # Power iteration
    for _ in range(num_iterations):
        d.requires_grad_(True)

        # Perturb in feature space
        x_perturbed = x_feature + xi * d
        _, p_perturbed = predictor(x_perturbed)

        # KL divergence
        kl_div = F.kl_div(
            torch.log(p_perturbed + 1e-8),
            p_original,
            reduction="batchmean",
        )

        kl_div.backward()
        d_grad = d.grad.detach()
        d = get_normalized_vector(d_grad).detach()

    # Generate final adversarial perturbation
    r_adv = epsilon * d

    # Compute VAT loss with adversarial perturbation
    # Note: We don't clamp here since features may have different range
    x_adv = x_feature + r_adv
    _, p_adv = predictor(x_adv)

    vat_loss = F.kl_div(
        torch.log(p_adv + 1e-8),
        p_original.detach(),
        reduction="batchmean",
    )

    return vat_loss


def compute_vat_loss_with_encoder(
    encoder,
    predictor,
    x,
    epsilon=1.0,
    xi=1e-6,
    num_iterations=1,
):
    """Compute VAT loss with perturbation in INPUT space, then encode.

    This is the standard VAT approach: perturbation in input space,
    then pass through encoder to get features for prediction.

    Args:
        encoder: Encoder network E(x) -> z
        predictor: Predictor network f(z) -> (logits, probs)
        x: Input tensor in original space
        epsilon: Maximum perturbation magnitude
        xi: Small constant for finite difference
        num_iterations: Number of power iteration steps

    Returns:
        VAT loss (scalar tensor)
    """
    # Get original prediction (detached)
    with torch.no_grad():
        x_enc = encoder(x)
        _, p_original = predictor(x_enc)

    # Initialize random unit vector in input space
    d = torch.randn_like(x)
    d = get_normalized_vector(d)

    # Power iteration
    for _ in range(num_iterations):
        d.requires_grad_(True)

        # Perturb in input space, then encode
        x_perturbed = x + xi * d
        x_perturbed_enc = encoder(x_perturbed)
        _, p_perturbed = predictor(x_perturbed_enc)

        # KL divergence
        kl_div = F.kl_div(
            torch.log(p_perturbed + 1e-8),
            p_original,
            reduction="batchmean",
        )

        kl_div.backward()
        d_grad = d.grad.detach()
        d = get_normalized_vector(d_grad).detach()

    # Generate final adversarial perturbation
    r_adv = epsilon * d

    # Compute VAT loss with adversarial perturbation
    x_adv = torch.clamp(x + r_adv, 0, 1)
    x_adv_enc = encoder(x_adv)
    _, p_adv = predictor(x_adv_enc)

    vat_loss = F.kl_div(
        torch.log(p_adv + 1e-8),
        p_original.detach(),
        reduction="batchmean",
    )

    return vat_loss


# =============================================================================
# VAT-Mask Loss (Random mask + KL Divergence)
# =============================================================================


def generate_mask(p, x):
    """Generate binary mask matrix."""
    return np.random.binomial(1, p, x.shape)


def corrupt_samples(mask, x):
    """Generate corrupted samples by column-wise shuffling.

    Core formula: tilde_x = m * bar_x + (1-m) * x
    Where bar_x is generated by column-wise shuffling.
    """
    n, dim = x.shape
    x_shuffled = np.array(
        [x[np.random.permutation(n), i] for i in range(dim)]
    ).T
    x_corrupt = x * (1 - mask) + x_shuffled * mask
    return (x != x_corrupt).astype(int), x_corrupt


def compute_vat_mask_loss_feature_space(predictor, x_feature, x_feature_corrupt):
    """Compute VAT-Mask consistency loss using KL divergence (feature space).

    Uses the mask+replacement perturbation but in feature space,
    measuring consistency using KL divergence.

    Args:
        predictor: Predictor network
        x_feature: Original feature tensor
        x_feature_corrupt: Corrupted feature tensor

    Returns:
        KL divergence loss (scalar tensor)
    """
    # Get original prediction (detached, no gradient)
    with torch.no_grad():
        _, p_original = predictor(x_feature)

    # Get prediction for corrupted sample
    _, p_corrupt = predictor(x_feature_corrupt)

    # Compute KL divergence loss
    vat_mask_loss = F.kl_div(
        torch.log(p_corrupt + 1e-8),
        p_original.detach(),
        reduction="batchmean",
    )

    return vat_mask_loss


def compute_vat_mask_loss_with_encoder(encoder, predictor, x_original, x_corrupt, device):
    """Compute VAT-Mask consistency loss using KL divergence (with encoder).

    Uses the same mask+replacement perturbation as VIME, but measures
    consistency using KL divergence instead of variance.

    Args:
        encoder: Encoder network
        predictor: Predictor network
        x_original: Original input (numpy array)
        x_corrupt: Corrupted input (numpy array)
        device: Torch device

    Returns:
        KL divergence loss (scalar tensor)
    """
    x_original_t = torch.from_numpy(x_original).float().to(device)
    x_corrupt_t = torch.from_numpy(x_corrupt).float().to(device)

    # Get original prediction (detached, no gradient)
    with torch.no_grad():
        x_enc = encoder(x_original_t)
        _, p_original = predictor(x_enc)

    # Get prediction for corrupted sample
    x_corrupt_enc = encoder(x_corrupt_t)
    _, p_corrupt = predictor(x_corrupt_enc)

    # Compute KL divergence loss
    vat_mask_loss = F.kl_div(
        torch.log(p_corrupt + 1e-8),
        p_original.detach(),
        reduction="batchmean",
    )

    return vat_mask_loss


# =============================================================================
# Random Noise Loss (Random perturbation + KL Divergence) - Baseline for VAT
# =============================================================================


def compute_random_noise_loss_with_encoder(
    encoder,
    predictor,
    x,
    epsilon=1.0,
):
    """Compute consistency loss with RANDOM perturbation in INPUT space.

    This is the baseline for VAT: uses random noise instead of adversarial noise.
    Compares: VAT (adversarial) vs Random Noise (random).

    Args:
        encoder: Encoder network E(x) -> z
        predictor: Predictor network f(z) -> (logits, probs)
        x: Input tensor in original space
        epsilon: Perturbation magnitude

    Returns:
        Random noise consistency loss (scalar tensor)
    """
    # Get original prediction (detached)
    with torch.no_grad():
        x_enc = encoder(x)
        _, p_original = predictor(x_enc)

    # Generate random unit vector and scale by epsilon
    r_random = torch.randn_like(x)
    r_random = get_normalized_vector(r_random) * epsilon

    # Apply random perturbation and clamp to valid range [0, 1]
    x_noisy = torch.clamp(x + r_random, 0, 1)
    x_noisy_enc = encoder(x_noisy)
    _, p_noisy = predictor(x_noisy_enc)

    # Compute KL divergence loss
    random_loss = F.kl_div(
        torch.log(p_noisy + 1e-8),
        p_original.detach(),
        reduction="batchmean",
    )

    return random_loss


def compute_random_noise_loss_feature_space(
    predictor,
    x_feature,
    epsilon=1.0,
):
    """Compute consistency loss with RANDOM perturbation in FEATURE space.

    This is the baseline for VAT on features: uses random noise instead of adversarial.

    Args:
        predictor: Predictor network f(z) -> (logits, probs)
        x_feature: Feature tensor (already encoded by VIME)
        epsilon: Perturbation magnitude

    Returns:
        Random noise consistency loss (scalar tensor)
    """
    # Get original prediction (detached)
    with torch.no_grad():
        _, p_original = predictor(x_feature)

    # Generate random unit vector and scale by epsilon
    r_random = torch.randn_like(x_feature)
    r_random = get_normalized_vector(r_random) * epsilon

    # Apply random perturbation
    x_noisy = x_feature + r_random
    _, p_noisy = predictor(x_noisy)

    # Compute KL divergence loss
    random_loss = F.kl_div(
        torch.log(p_noisy + 1e-8),
        p_original.detach(),
        reduction="batchmean",
    )

    return random_loss


class VATLoss(torch.nn.Module):
    """VAT Loss module for easy integration."""

    def __init__(self, epsilon=1.0, xi=1e-6, num_iterations=1):
        super().__init__()
        self.epsilon = epsilon
        self.xi = xi
        self.num_iterations = num_iterations

    def forward(self, predictor, x_feature):
        """Compute VAT loss in feature space."""
        return compute_vat_loss_feature_space(
            predictor, x_feature, self.epsilon, self.xi, self.num_iterations
        )
