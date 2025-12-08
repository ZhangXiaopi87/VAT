"""VIME self-supervised learning with multi-layer encoder.

This module implements a multi-layer VIME encoder that can extract features
from different layers for comparison experiments.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import generate_mask, corrupt_samples, EarlyStopping, log_progress


class VIMEMultiLayerModel(nn.Module):
    """VIME self-supervised model with multi-layer encoder.

    Encoder architecture:
        Layer 1: input_dim -> hidden_dim (ReLU)
        Layer 2: hidden_dim -> hidden_dim (ReLU)
        Layer 3: hidden_dim -> hidden_dim (ReLU)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Multi-layer encoder
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        # Mask estimator (from final layer)
        self.mask_estimator = nn.Sequential(
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )
        # Feature estimator (from final layer)
        self.feature_estimator = nn.Sequential(
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all layers."""
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        return self.mask_estimator(h3), self.feature_estimator(h3)

    def get_layer1_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from layer 1."""
        return self.layer1(x)

    def get_layer2_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from layer 2."""
        h1 = self.layer1(x)
        return self.layer2(h1)

    def get_layer3_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from layer 3."""
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return self.layer3(h2)

    def get_all_layer_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from all layers."""
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        return h1, h2, h3


class Layer1Encoder(nn.Module):
    """Wrapper to extract layer 1 features."""
    def __init__(self, model: VIMEMultiLayerModel):
        super().__init__()
        self.layer1 = model.layer1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(x)


class Layer2Encoder(nn.Module):
    """Wrapper to extract layer 2 features."""
    def __init__(self, model: VIMEMultiLayerModel):
        super().__init__()
        self.layer1 = model.layer1
        self.layer2 = model.layer2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)
        return self.layer2(h1)


class Layer3Encoder(nn.Module):
    """Wrapper to extract layer 3 features."""
    def __init__(self, model: VIMEMultiLayerModel):
        super().__init__()
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return self.layer3(h2)


class ConcatEncoder(nn.Module):
    """Wrapper that outputs concatenation of original input and layer 3 features."""
    def __init__(self, model: VIMEMultiLayerModel):
        super().__init__()
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        return torch.cat([x, h3], dim=1)


class IdentityEncoder(nn.Module):
    """Identity encoder that returns the input as-is (for VAT on X baseline)."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def vime_self_multilayer(
    X_train, X_val, p_mask: float, alpha: float, params: Dict[str, int]
) -> VIMEMultiLayerModel:
    """VIME self-supervised learning with multi-layer encoder.

    Returns:
        The trained multi-layer model (not just encoder)
    """
    _, dim = X_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VIMEMultiLayerModel(dim, params["hidden_dim"]).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.001, alpha=0.9, eps=1e-7
    )

    # Generate corrupted samples
    mask_train, mask_val = generate_mask(p_mask, X_train), generate_mask(
        p_mask, X_val
    )
    mask_label, X_corrupt = corrupt_samples(mask_train, X_train)
    mask_label_val, X_corrupt_val = corrupt_samples(mask_val, X_val)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_corrupt).float(),
            torch.from_numpy(mask_label).float(),
            torch.from_numpy(X_train).float(),
        ),
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    # Validation tensors
    X_corrupt_val_t = torch.from_numpy(X_corrupt_val).float().to(device)
    mask_label_val_t = torch.from_numpy(mask_label_val).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)

    early_stopping = EarlyStopping(patience=params.get("patience", 5))

    for epoch in range(params["epochs"]):
        model.train()
        total_loss, mask_loss_sum, feat_loss_sum, n_batches = 0.0, 0.0, 0.0, 0
        for X_corrupt_b, mask_label_b, X_orig_b in loader:
            X_corrupt_b = X_corrupt_b.to(device, non_blocking=True)
            mask_label_b = mask_label_b.to(device, non_blocking=True)
            X_orig_b = X_orig_b.to(device, non_blocking=True)

            mask_pred, feat_pred = model(X_corrupt_b)
            mask_loss = F.binary_cross_entropy(mask_pred, mask_label_b)
            feat_loss = F.mse_loss(feat_pred, X_orig_b)
            loss = mask_loss + alpha * feat_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mask_loss_sum += mask_loss.item()
            feat_loss_sum += feat_loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            mask_pred_val, feat_pred_val = model(X_corrupt_val_t)
            val_loss = (
                F.binary_cross_entropy(mask_pred_val, mask_label_val_t).item()
                + alpha * F.mse_loss(feat_pred_val, X_val_t).item()
            )

        log_progress(
            epoch,
            params["epochs"],
            {
                "train": total_loss / n_batches,
                "mask": mask_loss_sum / n_batches,
                "feat": feat_loss_sum / n_batches,
            },
            val_loss,
        )

        if early_stopping.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.load_best(model)
    model.eval()
    return model


def get_encoders_from_model(model: VIMEMultiLayerModel, input_dim: int, hidden_dim: int):
    """Extract different encoders from the trained multi-layer model.

    Returns:
        dict: Dictionary containing different encoder types
            - 'identity': Identity encoder (input X as-is)
            - 'layer1': Layer 1 encoder
            - 'layer2': Layer 2 encoder
            - 'layer3': Layer 3 encoder (full encoder)
            - 'concat': Concatenation of X and layer3 features
    """
    return {
        'identity': IdentityEncoder(),
        'layer1': Layer1Encoder(model),
        'layer2': Layer2Encoder(model),
        'layer3': Layer3Encoder(model),
        'concat': ConcatEncoder(model),
    }
