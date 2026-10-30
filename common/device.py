"""Dispositivo de execução e contagem de parâmetros."""

import torch
from torch import nn


def get_device() -> torch.device:
    """Retorna CUDA quando disponível; caso contrário, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parameter_count(model: nn.Module) -> int:
    """Conta todos os parâmetros do modelo."""
    return sum(parameter.numel() for parameter in model.parameters())
