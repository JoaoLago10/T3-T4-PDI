"""Executa rastreamento em tempo real usando uma webcam."""

import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from common.device import get_device
from common.drawing import draw_detections, draw_status
from detection.models import DetectorBundle, load_model as load_detector, predict, warm_up
from tracking.trackers import TRACKER_NAMES, BaseTracker, load_tracker


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", choices=TRACKER_NAMES, default="kalman")
    parser.add_argument("--detector", default="ssdlite")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--scene_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moving_window", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tracking"))
    return parser.parse_args()


def save_summaries(
    output_dir: Path,
    latencies: dict[str, list[float]],
    frame_rates: dict[str, list[float]],
) -> None:
    """Salva um resumo separado para cada rastreador utilizado."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model_latencies in latencies.items():
        rates = frame_rates[name]
        row = {
            "tracker": name,
            "frames": len(model_latencies),
            "avg_fps": float(np.mean(rates)),
            "avg_latency_ms": float(np.mean(model_latencies)),
            "p95_latency_ms": float(np.percentile(model_latencies, 95)),
        }
        output_path = output_dir / f"webcam_{name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(f"Resumo salvo em {output_path}")


def main() -> None:
    """Executa o loop da webcam e permite troca de rastreador."""
    args = parse_args()
    if args.moving_window < 1 or args.warmup < 0:
        raise ValueError("--moving_window deve ser positivo.")
    if not 0 < args.score_threshold <= 1:
        raise ValueError("--score_threshold deve estar em (0, 1].")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Webcam {args.camera} indisponível.")

    print(f"Carregando detector {args.detector} em {get_device()}...")
    detector: DetectorBundle | None = None
    trackers: dict[str, BaseTracker] = {name: load_tracker(name) for name in TRACKER_NAMES}
    active = args.tracker
    latencies: dict[str, list[float]] = defaultdict(list)
    frame_rates: dict[str, list[float]] = defaultdict(list)
    recent_durations: deque[float] = deque(maxlen=args.moving_window)
    previous_loop_time = time.perf_counter()
    key_to_tracker = {ord("1"): "iou", ord("2"): "centroid", ord("3"): "kalman"}

    print("Teclas: 1=IoU, 2=Centróide, 3=Kalman, q=sair")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("A webcam parou de fornecer frames.")
            if detector is None:
                detector = load_detector(args.detector)
                warm_up(detector, frame, args.warmup)

            detections, latency_ms = predict(
                detector, frame, args.score_threshold, args.scene_only
            )
            tracked = trackers[active].update(detections)

            now = time.perf_counter()
            recent_durations.append(now - previous_loop_time)
            previous_loop_time = now
            current_fps = len(recent_durations) / sum(recent_durations)
            latencies[active].append(latency_ms)
            frame_rates[active].append(current_fps)

            draw_detections(frame, tracked, show_ids=True)
            draw_status(
                frame,
                [
                    f"Rastreio: {active} | FPS: {current_fps:.1f}",
                    f"IDs visíveis: {len(tracked)}",
                ],
            )
            cv2.imshow("Rastreamento", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in key_to_tracker:
                active = key_to_tracker[key]
                recent_durations.clear()
                previous_loop_time = time.perf_counter()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if latencies:
            save_summaries(args.output_dir, latencies, frame_rates)


if __name__ == "__main__":
    main()
