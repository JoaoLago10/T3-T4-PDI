"""Tipos compartilhados pelas etapas de visão."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Uma instância localizada no quadro."""

    box: tuple[float, float, float, float]
    label: str
    score: float
    mask: np.ndarray | None = None
    track_id: int | None = None


@dataclass
class VideoInfo:
    """Metadados de um vídeo aberto pelo OpenCV."""

    width: int
    height: int
    fps: float
    total_frames: int
