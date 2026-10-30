"""Loop comum de leitura, inferência espaçada e gravação anotada."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np
from tqdm import tqdm

from .types import VideoInfo
from .video import make_writer, open_video

State = TypeVar("State")
InferFn = Callable[[np.ndarray, int, float], tuple[State, dict[str, Any] | None]]
DrawFn = Callable[[np.ndarray, State], None]


def process_video(
    video_path: Path,
    output_video: Path,
    frame_step: int,
    codec: str,
    infer_fn: InferFn[State],
    draw_fn: DrawFn[State],
    desc: str,
) -> tuple[list[dict[str, Any]], VideoInfo]:
    """Percorre o vídeo, infere a cada N quadros e mantém o último resultado nos demais."""
    capture, info, _first_frame = open_video(video_path)
    writer = make_writer(output_video, info, codec)
    rows: list[dict[str, Any]] = []
    last_state: State | None = None
    frame_idx = 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        with tqdm(total=info.total_frames or None, desc=desc, unit="frame") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_idx % frame_step == 0:
                    last_state, row = infer_fn(frame, frame_idx, info.fps)
                    if row is not None:
                        rows.append(row)
                annotated = frame
                if last_state is not None:
                    draw_fn(annotated, last_state)
                writer.write(annotated)
                frame_idx += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()
    return rows, info
