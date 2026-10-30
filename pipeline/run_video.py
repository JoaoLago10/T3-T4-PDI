"""Executa detecção, segmentação e rastreamento em uma só passagem."""

import argparse
import csv
import json
from pathlib import Path

import torch

from common.device import get_device
from common.drawing import draw_detections, draw_status
from common.engine import process_video
from common.video import open_video
from segmentation.models import load_model as load_segmenter
from tracking.trackers import TRACKER_NAMES, load_tracker


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segmenter", default="grabcut")
    parser.add_argument("--tracker", choices=TRACKER_NAMES, default="kalman")
    parser.add_argument("--detector", default="ssdlite")
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--scene_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def main() -> None:
    """Roda o pipeline completo e grava vídeo, CSV e metadados."""
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {args.video}")
    if args.frame_step < 1 or args.warmup < 0:
        raise ValueError("--frame_step deve ser positivo; --warmup não pode ser negativo.")
    if not 0 < args.score_threshold <= 1:
        raise ValueError("--score_threshold deve estar em (0, 1].")

    capture, _info, first_frame = open_video(args.video)
    capture.release()
    segmenter = load_segmenter(
        args.segmenter,
        detector_name=args.detector,
        score_threshold=args.score_threshold,
        scene_only=args.scene_only,
    )
    tracker = load_tracker(args.tracker)
    print(
        f"pipeline: segmenter={args.segmenter}, tracker={args.tracker}, "
        f"dispositivo={get_device()}"
    )
    segmenter.warm_up(first_frame, args.warmup)
    if get_device().type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def infer_fn(frame, frame_idx, fps):
        detections, latency_ms = segmenter.predict(frame)
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
        draw_detections(frame, detections, draw_masks=True, show_ids=True)
        draw_status(frame, [f"{args.segmenter} + {args.tracker} | {len(detections)} IDs"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "annotated_pipeline.mp4"
    rows, _info = process_video(
        args.video,
        video_path,
        args.frame_step,
        args.codec,
        infer_fn,
        draw_fn,
        "pipeline",
    )
    csv_path = args.output_dir / "predictions_pipeline.csv"
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

    unique_ids = set()
    for row in rows:
        unique_ids.update(json.loads(row["track_ids"]))
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if get_device().type == "cuda"
        else None
    )
    metadata = {
        "segmenter": args.segmenter,
        "tracker": args.tracker,
        "detector": args.detector if args.segmenter != "maskrcnn" else "maskrcnn",
        "parameters": segmenter.n_parameters,
        "device": str(get_device()),
        "gpu_peak_memory_mb": peak_memory_mb,
        "processed_frames": len(rows),
        "unique_ids": len(unique_ids),
        "score_threshold": args.score_threshold,
        "scene_only": args.scene_only,
    }
    (args.output_dir / "metrics_pipeline.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Salvos: {video_path} e {csv_path}")


if __name__ == "__main__":
    main()
