"""Carregamento padronizado dos classificadores ImageNet."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b0,
    resnet50,
    vit_b_16,
)

MODEL_NAMES = ("resnet50", "efficientnet", "vit")


def get_device() -> torch.device:
    """Retorna CUDA quando disponível; caso contrário, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name: str) -> tuple[nn.Module, Callable[[Any], torch.Tensor], list[str]]:
    """Carrega modelo, transformação oficial e categorias do ImageNet."""
    normalized_name = name.lower()
    if normalized_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    elif normalized_name == "efficientnet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    elif normalized_name == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
    else:
        valid_names = ", ".join(MODEL_NAMES)
        raise ValueError(f"Modelo desconhecido: {name}. Use: {valid_names}.")

    model = model.to(get_device())
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


def parameter_count(model: nn.Module) -> int:
    """Conta todos os parâmetros do modelo."""
    return sum(parameter.numel() for parameter in model.parameters())
