"""Abertura de vídeo e criação do escritor de saída."""

from pathlib import Path

import cv2
import numpy as np

from .types import VideoInfo


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo, np.ndarray]:
    """Abre o vídeo e devolve captura, metadados e o primeiro quadro."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {path}")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"O vídeo não contém frames legíveis: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError("O vídeo não informa uma taxa de quadros válida.")

    info = VideoInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps,
        total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return capture, info, first_frame


def make_writer(
    path: Path,
    info: VideoInfo,
    codec: str,
) -> cv2.VideoWriter:
    """Cria um VideoWriter no mesmo tamanho e FPS do vídeo de entrada."""
    if len(codec) != 4:
        raise ValueError("--codec deve ter exatamente quatro caracteres.")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Não foi possível criar o vídeo de saída: {path}")
    return writer
