"""Executa o pipeline completo em tempo real usando uma webcam."""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from common.device import get_device
from common.drawing import draw_detections, draw_status
from segmentation.models import load_model as load_segmenter
from tracking.trackers import TRACKER_NAMES, load_tracker


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmenter", default="threshold")
    parser.add_argument("--tracker", choices=TRACKER_NAMES, default="kalman")
    parser.add_argument("--detector", default="ssdlite")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--scene_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moving_window", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/pipeline"))
    return parser.parse_args()


def main() -> None:
    """Abre a webcam, segmenta, rastreia e grava o resumo ao sair."""
    args = parse_args()
    if args.moving_window < 1 or args.warmup < 0:
        raise ValueError("--moving_window deve ser positivo.")
    if not 0 < args.score_threshold <= 1:
        raise ValueError("--score_threshold deve estar em (0, 1].")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Webcam {args.camera} indisponível.")

    print(f"Carregando {args.segmenter} + {args.tracker} em {get_device()}...")
    segmenter = None
    tracker = load_tracker(args.tracker)
    latencies: list[float] = []
    rates: list[float] = []
    recent_durations: deque[float] = deque(maxlen=args.moving_window)
    previous_loop_time = time.perf_counter()
    print("Tecla q encerra.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("A webcam parou de fornecer frames.")
            if segmenter is None:
                segmenter = load_segmenter(
                    args.segmenter,
                    detector_name=args.detector,
                    score_threshold=args.score_threshold,
                    scene_only=args.scene_only,
                )
                segmenter.warm_up(frame, args.warmup)

            detections, latency_ms = segmenter.predict(frame)
            tracked = tracker.update(detections)
            now = time.perf_counter()
            recent_durations.append(now - previous_loop_time)
            previous_loop_time = now
            current_fps = len(recent_durations) / sum(recent_durations)
            latencies.append(latency_ms)
            rates.append(current_fps)

            draw_detections(frame, tracked, draw_masks=True, show_ids=True)
            draw_status(
                frame,
                [
                    f"{args.segmenter} + {args.tracker} | FPS: {current_fps:.1f}",
                    f"IDs: {len(tracked)}",
                ],
            )
            cv2.imshow("Pipeline completo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if latencies:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "segmenter": args.segmenter,
                "tracker": args.tracker,
                "frames": len(latencies),
                "avg_fps": float(np.mean(rates)),
                "avg_latency_ms": float(np.mean(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
            }
            output_path = args.output_dir / "webcam_pipeline.csv"
            with output_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            print(f"Resumo salvo em {output_path}")


if __name__ == "__main__":
    main()
