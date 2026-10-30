"""Desenho de caixas, máscaras e identificadores de rastreio."""

import cv2
import numpy as np

from .coco import color_for_id, color_for_label
from .types import Detection


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> None:
    """Pinta a máscara sobre o quadro, no lugar."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.float32),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    binary = mask > 0.5
    if not np.any(binary):
        return
    overlay = frame.copy()
    overlay[binary] = color
    blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
    frame[binary] = blended[binary]


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    draw_masks: bool = False,
    show_ids: bool = False,
) -> np.ndarray:
    """Desenha instâncias no quadro e devolve o próprio array anotado."""
    for item in detections:
        color = color_for_id(item.track_id) if show_ids and item.track_id else color_for_label(item.label)
        if draw_masks and item.mask is not None:
            overlay_mask(frame, item.mask, color)
        x1, y1, x2, y2 = (int(round(value)) for value in item.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        identity = f"#{item.track_id} " if show_ids and item.track_id is not None else ""
        text = f"{identity}{item.label} {item.score:.0%}"
        text_y = max(y1 - 8, 16)
        width = 10 + 9 * len(text)
        cv2.rectangle(frame, (x1, text_y - 16), (x1 + width, text_y + 4), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_status(frame: np.ndarray, lines: list[str]) -> None:
    """Escreve um bloco de status no canto superior esquerdo."""
    box_h = 16 + 26 * len(lines)
    box_w = min(frame.shape[1] - 16, max(280, 10 + 9 * max(len(line) for line in lines)))
    cv2.rectangle(frame, (8, 8), (8 + box_w, box_h), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 30 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
