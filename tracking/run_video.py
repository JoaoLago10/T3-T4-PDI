"""Executa rastreamento sobre detecções em um vídeo gravado."""

import argparse
import csv
import json
from pathlib import Path

import torch

from common.device import get_device, parameter_count
from common.drawing import draw_detections
from common.engine import process_video
from common.video import open_video
from detection.models import load_model as load_detector, predict, warm_up
from tracking.trackers import TRACKER_NAMES, load_tracker


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracker", choices=(*TRACKER_NAMES, "all"), default="all")
    parser.add_argument("--detector", default="ssdlite")
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--scene_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tracking"))
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def process_tracker(args: argparse.Namespace, tracker_name: str) -> None:
    """Processa o vídeo com um detector fixo e um rastreador."""
    capture, _info, first_frame = open_video(args.video)
    capture.release()
    detector = load_detector(args.detector)
    tracker = load_tracker(tracker_name)
    print(
        f"{tracker_name}: detector={args.detector}, dispositivo={get_device()}, "
        f"aquecimento={args.warmup}"
    )
    warm_up(detector, first_frame, args.warmup)
    if get_device().type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def infer_fn(frame, frame_idx, fps):
        detections, latency_ms = predict(
            detector, frame, args.score_threshold, args.scene_only
        )
        tracked = tracker.update(detections)
        row = {
            "frame_idx": frame_idx,
            "timestamp": frame_idx / fps,
            "num_tracks": len(tracked),
            "track_ids": json.dumps([item.track_id for item in tracked], ensure_ascii=False),
            "labels": json.dumps([item.label for item in tracked], ensure_ascii=False),
            "scores": json.dumps([item.score for item in tracked], ensure_ascii=False),
            "boxes": json.dumps([list(item.box) for item in tracked], ensure_ascii=False),
            "latency_ms": latency_ms,
        }
        return tracked, row

    def draw_fn(frame, detections):
        draw_detections(frame, detections, show_ids=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, _info = process_video(
        args.video,
        args.output_dir / f"annotated_{tracker_name}.mp4",
        args.frame_step,
        args.codec,
        infer_fn,
        draw_fn,
        tracker_name,
    )
    csv_path = args.output_dir / f"predictions_{tracker_name}.csv"
    fieldnames = [
        "frame_idx",
        "timestamp",
        "num_tracks",
        "track_ids",
        "labels",
        "scores",
        "boxes",
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
    unique_ids = set()
    for row in rows:
        unique_ids.update(json.loads(row["track_ids"]))
    metadata = {
        "tracker": tracker_name,
        "detector": args.detector,
        "parameters": parameter_count(detector[0]),
        "device": str(get_device()),
        "gpu_peak_memory_mb": peak_memory_mb,
        "processed_frames": len(rows),
        "unique_ids": len(unique_ids),
        "score_threshold": args.score_threshold,
        "scene_only": args.scene_only,
    }
    (args.output_dir / f"metrics_{tracker_name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Salvos: {args.output_dir / f'annotated_{tracker_name}.mp4'} e {csv_path}")


def main() -> None:
    """Valida os argumentos e executa os rastreadores selecionados."""
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {args.video}")
    if args.frame_step < 1 or args.warmup < 0:
        raise ValueError("--frame_step deve ser positivo; --warmup não pode ser negativo.")
    if not 0 < args.score_threshold <= 1:
        raise ValueError("--score_threshold deve estar em (0, 1].")
    selected = TRACKER_NAMES if args.tracker == "all" else (args.tracker,)
    for tracker_name in selected:
        process_tracker(args, tracker_name)


if __name__ == "__main__":
    main()
