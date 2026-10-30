"""Executa detecção de objetos em tempo real usando uma webcam."""

import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from common.device import get_device
from common.drawing import draw_detections, draw_status
from detection.models import MODEL_NAMES, DetectorBundle, load_model, predict, warm_up


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="ssdlite")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--scene_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moving_window", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/detection"))
    return parser.parse_args()


def get_bundle(
    name: str,
    cache: dict[str, DetectorBundle],
    frame: np.ndarray,
    warmup_iterations: int,
) -> DetectorBundle:
    """Carrega e aquece um detector apenas na primeira utilização."""
    if name in cache:
        return cache[name]
    print(f"Carregando {name} em {get_device()}...")
    bundle = load_model(name)
    warm_up(bundle, frame, warmup_iterations)
    cache[name] = bundle
    return bundle


def save_summaries(
    output_dir: Path,
    latencies: dict[str, list[float]],
    frame_rates: dict[str, list[float]],
) -> None:
    """Salva um resumo separado para cada detector utilizado."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model_latencies in latencies.items():
        rates = frame_rates[model_name]
        row = {
            "model": model_name,
            "frames": len(model_latencies),
            "avg_fps": float(np.mean(rates)),
            "avg_latency_ms": float(np.mean(model_latencies)),
            "p95_latency_ms": float(np.percentile(model_latencies, 95)),
        }
        output_path = output_dir / f"webcam_{model_name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(f"Resumo salvo em {output_path}")


def main() -> None:
    """Executa o loop da webcam e permite troca de detector."""
    args = parse_args()
    if args.moving_window < 1 or args.warmup < 0:
        raise ValueError("--moving_window deve ser positivo.")
    if not 0 < args.score_threshold <= 1:
        raise ValueError("--score_threshold deve estar em (0, 1].")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Webcam {args.camera} indisponível.")

    active_model = args.model
    cache: dict[str, DetectorBundle] = {}
    latencies: dict[str, list[float]] = defaultdict(list)
    frame_rates: dict[str, list[float]] = defaultdict(list)
    recent_durations: deque[float] = deque(maxlen=args.moving_window)
    previous_loop_time = time.perf_counter()
    key_to_model = {ord("1"): "fasterrcnn", ord("2"): "mobile", ord("3"): "ssdlite"}

    print("Teclas: 1=Faster R-CNN, 2=MobileNet, 3=SSDLite, q=sair")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("A webcam parou de fornecer frames.")

            bundle = get_bundle(active_model, cache, frame, args.warmup)
            detections, latency_ms = predict(
                bundle, frame, args.score_threshold, args.scene_only
            )

            now = time.perf_counter()
            recent_durations.append(now - previous_loop_time)
            previous_loop_time = now
            current_fps = len(recent_durations) / sum(recent_durations)
            latencies[active_model].append(latency_ms)
            frame_rates[active_model].append(current_fps)

            draw_detections(frame, detections)
            draw_status(
                frame,
                [
                    f"Detector: {active_model} | FPS: {current_fps:.1f}",
                    f"Objetos: {len(detections)} | limiar {args.score_threshold:.2f}",
                ],
            )
            cv2.imshow("Detecção COCO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in key_to_model:
                active_model = key_to_model[key]
                recent_durations.clear()
                previous_loop_time = time.perf_counter()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if latencies:
            save_summaries(args.output_dir, latencies, frame_rates)


if __name__ == "__main__":
    main()
