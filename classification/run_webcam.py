"""Executa classificação em tempo real usando uma webcam."""

import argparse
import csv
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

try:
    from .models import MODEL_NAMES, get_device, load_model
except ImportError:
    from models import MODEL_NAMES, get_device, load_model

Preprocess = Callable[[Image.Image], torch.Tensor]
ModelBundle = tuple[nn.Module, Preprocess, list[str]]


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet50")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera.")
    parser.add_argument("--topk", type=int, default=3, help="Classes exibidas.")
    parser.add_argument("--moving_window", type=int, default=30, help="Janela móvel do FPS.")
    parser.add_argument("--warmup", type=int, default=10, help="Inferências de aquecimento.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def get_bundle(
    name: str,
    cache: dict[str, ModelBundle],
    frame: np.ndarray,
    warmup_iterations: int,
) -> ModelBundle:
    """Carrega e aquece um modelo apenas na primeira utilização."""
    if name in cache:
        return cache[name]

    print(f"Carregando {name} em {get_device()}...")
    bundle = load_model(name)
    model, preprocess, _ = bundle
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = preprocess(image).unsqueeze(0).to(get_device())
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(tensor)
    if get_device().type == "cuda":
        torch.cuda.synchronize()
    cache[name] = bundle
    return bundle


def infer(
    bundle: ModelBundle,
    frame: np.ndarray,
    topk: int,
) -> tuple[list[str], list[float], float]:
    """Classifica um frame e mede a latência da inferência."""
    model, preprocess, categories = bundle
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = preprocess(image).unsqueeze(0).to(get_device())

    if get_device().type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)
    if get_device().type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1_000

    count = min(topk, len(categories))
    values, indexes = probabilities.topk(count, dim=1)
    labels = [categories[index] for index in indexes[0].tolist()]
    return labels, values[0].tolist(), latency_ms


def draw_overlay(
    frame: np.ndarray,
    model_name: str,
    labels: list[str],
    confidences: list[float],
    fps: float,
) -> None:
    """Sobrepõe modelo, FPS e melhores classificações."""
    lines = [f"Modelo: {model_name} | FPS: {fps:.1f}"]
    lines.extend(
        f"{position}. {label}: {confidence:.1%}"
        for position, (label, confidence) in enumerate(
            zip(labels, confidences, strict=True), start=1
        )
    )
    height = 18 + 30 * len(lines)
    cv2.rectangle(frame, (8, 8), (620, height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 35 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def save_summaries(
    output_dir: Path,
    latencies: dict[str, list[float]],
    frame_rates: dict[str, list[float]],
) -> None:
    """Salva um resumo separado para cada modelo utilizado."""
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
    """Executa o loop da webcam e permite troca de modelo."""
    args = parse_args()
    if args.topk < 1 or args.moving_window < 1 or args.warmup < 0:
        raise ValueError("--topk e --moving_window devem ser positivos.")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Webcam {args.camera} indisponível.")

    active_model = args.model
    cache: dict[str, ModelBundle] = {}
    latencies: dict[str, list[float]] = defaultdict(list)
    frame_rates: dict[str, list[float]] = defaultdict(list)
    recent_durations: deque[float] = deque(maxlen=args.moving_window)
    previous_loop_time = time.perf_counter()
    key_to_model = {ord("1"): "resnet50", ord("2"): "efficientnet", ord("3"): "vit"}

    print("Teclas: 1=ResNet-50, 2=EfficientNet-B0, 3=ViT-B/16, q=sair")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("A webcam parou de fornecer frames.")

            bundle = get_bundle(active_model, cache, frame, args.warmup)
            labels, confidences, latency_ms = infer(bundle, frame, args.topk)

            now = time.perf_counter()
            recent_durations.append(now - previous_loop_time)
            previous_loop_time = now
            current_fps = len(recent_durations) / sum(recent_durations)
            latencies[active_model].append(latency_ms)
            frame_rates[active_model].append(current_fps)

            draw_overlay(frame, active_model, labels, confidences, current_fps)
            cv2.imshow("Classificação ImageNet", frame)
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
