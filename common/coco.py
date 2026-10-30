"""Vocabulário COCO e correspondência com os rótulos do ImageNet."""

from collections.abc import Sequence

import cv2
import numpy as np

from .types import Detection

SCENE_CLASSES = ("backpack", "bottle", "keyboard")

IMAGENET_TO_COCO = {
    "backpack": "backpack",
    "water bottle": "bottle",
    "computer keyboard": "keyboard",
}

SCENE_COLORS_BGR = {
    "backpack": (214, 92, 29),
    "bottle": (196, 52, 112),
    "keyboard": (125, 138, 12),
}


def map_imagenet_label(label: str) -> str:
    """Converte um rótulo ImageNet da anotação para a classe COCO equivalente."""
    return IMAGENET_TO_COCO.get(label, label)


def color_for_label(label: str) -> tuple[int, int, int]:
    """Devolve uma cor BGR estável para a classe."""
    if label in SCENE_COLORS_BGR:
        return SCENE_COLORS_BGR[label]
    rng = np.random.default_rng(abs(hash(label)) % (2**32))
    return int(rng.integers(40, 230)), int(rng.integers(40, 230)), int(rng.integers(40, 230))


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Devolve uma cor BGR determinística para o identificador de rastreio."""
    hue = int((track_id * 37) % 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def pack_detections(
    labels: Sequence[str],
    scores: Sequence[float],
    boxes: np.ndarray,
    masks: np.ndarray | None,
    score_threshold: float,
    scene_only: bool,
    width: int,
    height: int,
) -> list[Detection]:
    """Filtra, recorta e empacota as instâncias de um quadro."""
    kept: list[Detection] = []
    for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        if score < score_threshold:
            continue
        if scene_only and label not in SCENE_CLASSES:
            continue
        x1, y1, x2, y2 = (float(value) for value in boxes[index])
        x1 = min(max(x1, 0.0), width - 1.0)
        y1 = min(max(y1, 0.0), height - 1.0)
        x2 = min(max(x2, 0.0), width - 1.0)
        y2 = min(max(y2, 0.0), height - 1.0)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue
        mask = None
        if masks is not None:
            instance_mask = masks[index]
            if instance_mask.ndim == 3:
                instance_mask = instance_mask[0]
            mask = instance_mask
        kept.append(
            Detection(
                box=(x1, y1, x2, y2),
                label=str(label),
                score=float(score),
                mask=mask,
            )
        )
    return kept
