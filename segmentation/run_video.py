"""Executa segmentação em um vídeo gravado."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common.device import get_device
from common.drawing import draw_detections
from common.engine import process_video
from common.video import open_video
from detection.models import detections_to_json
from segmentation.models import MODEL_NAMES, load_model


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--detector", default="ssdlite", help="Detector usado por GrabCut e Otsu.")
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--scene_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/segmentation"))
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def process_model(args: argparse.Namespace, model_name: str) -> None:
    """Processa o vídeo completo com um segmentador."""
    capture, _info, first_frame = open_video(args.video)
    capture.release()
    segmenter = load_model(
        model_name,
        detector_name=args.detector,
        score_threshold=args.score_threshold,
        scene_only=args.scene_only,
    )
    print(f"{model_name}: dispositivo={get_device()}, aquecimento={args.warmup}")
    segmenter.warm_up(first_frame, args.warmup)
    if get_device().type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def infer_fn(frame, frame_idx, fps):
        detections, latency_ms = segmenter.predict(frame)
        payload = detections_to_json(detections)
        mask_areas = [
            float(np.count_nonzero(item.mask > 0.5)) if item.mask is not None else 0.0
            for item in detections
        ]
        row = {
            "frame_idx": frame_idx,
            "timestamp": frame_idx / fps,
            "latency_ms": latency_ms,
            "mean_mask_area": float(sum(mask_areas) / len(mask_areas)) if mask_areas else 0.0,
            **{
                key: json.dumps(value, ensure_ascii=False) if key != "num_detections" else value
                for key, value in payload.items()
            },
        }
        return detections, row

    def draw_fn(frame, detections):
        draw_detections(frame, detections, draw_masks=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, _info = process_video(
        args.video,
        args.output_dir / f"annotated_{model_name}.mp4",
        args.frame_step,
        args.codec,
        infer_fn,
        draw_fn,
        model_name,
    )
    csv_path = args.output_dir / f"predictions_{model_name}.csv"
    fieldnames = [
        "frame_idx",
        "timestamp",
        "num_detections",
        "labels",
        "scores",
        "boxes",
        "mean_mask_area",
        "latency_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if get_device().type == "cuda"
        else None
    )
    metadata = {
        "model": model_name,
        "parameters": segmenter.n_parameters,
        "device": str(get_device()),
        "gpu_peak_memory_mb": peak_memory_mb,
        "processed_frames": len(rows),
        "score_threshold": args.score_threshold,
        "scene_only": args.scene_only,
        "detector": args.detector if model_name != "maskrcnn" else "maskrcnn",
    }
    (args.output_dir / f"metrics_{model_name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Salvos: {args.output_dir / f'annotated_{model_name}.mp4'} e {csv_path}")


def main() -> None:
    """Valida os argumentos e executa os segmentadores selecionados."""
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {args.video}")
    if args.frame_step < 1 or args.warmup < 0:
        raise ValueError("--frame_step deve ser positivo; --warmup não pode ser negativo.")
    if not 0 < args.score_threshold <= 1:
        raise ValueError("--score_threshold deve estar em (0, 1].")
    selected = MODEL_NAMES if args.model == "all" else (args.model,)
    for model_name in selected:
        process_model(args, model_name)


if __name__ == "__main__":
    main()
