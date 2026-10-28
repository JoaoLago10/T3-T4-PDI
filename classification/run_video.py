"""Executa classificação em um vídeo gravado."""

import argparse
import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

try:
    from .models import MODEL_NAMES, get_device, load_model, parameter_count
except ImportError:
    from models import MODEL_NAMES, get_device, load_model, parameter_count

Preprocess = Callable[[Image.Image], torch.Tensor]


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Vídeo de entrada.")
    parser.add_argument(
        "--model",
        choices=(*MODEL_NAMES, "all"),
        default="all",
        help="Modelo a executar.",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=5,
        help="Processa um frame a cada N.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Número de classes retornadas.")
    parser.add_argument("--warmup", type=int, default=10, help="Inferências de aquecimento.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--codec", default="mp4v", help="Codec FourCC do vídeo de saída.")
    return parser.parse_args()


def infer(
    model: nn.Module,
    preprocess: Preprocess,
    categories: list[str],
    frame: np.ndarray,
    topk: int,
) -> tuple[list[str], list[float], float]:
    """Executa uma inferência e devolve classes, confianças e latência."""
    device = get_device()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = preprocess(image).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1_000

    count = min(topk, len(categories))
    values, indexes = probabilities.topk(count, dim=1)
    labels = [categories[index] for index in indexes[0].tolist()]
    confidences = values[0].tolist()
    return labels, confidences, latency_ms


def warm_up(
    model: nn.Module,
    preprocess: Preprocess,
    frame: np.ndarray,
    iterations: int,
) -> None:
    """Aquece o modelo sem registrar as medições."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = preprocess(image).unsqueeze(0).to(get_device())
    with torch.inference_mode():
        for _ in range(iterations):
            model(tensor)
    if get_device().type == "cuda":
        torch.cuda.synchronize()


def annotate_frame(
    frame: np.ndarray,
    label: str,
    confidence: float,
) -> np.ndarray:
    """Desenha a última classificação disponível no frame."""
    text = f"{label}: {confidence:.1%}"
    cv2.rectangle(frame, (8, 8), (max(360, 14 + len(text) * 11), 48), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (16, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def process_video(args: argparse.Namespace, model_name: str) -> None:
    """Processa o vídeo completo com um único modelo."""
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {args.video}")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"O vídeo não contém frames legíveis: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        capture.release()
        raise RuntimeError("O vídeo não informa uma taxa de quadros válida.")

    model, preprocess, categories = load_model(model_name)
    print(f"{model_name}: dispositivo={get_device()}, aquecimento={args.warmup}")
    warm_up(model, preprocess, first_frame, args.warmup)
    if get_device().type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"annotated_{model_name}.mp4"
    csv_path = args.output_dir / f"predictions_{model_name}.csv"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Não foi possível criar o vídeo de saída: {video_path}")

    rows: list[dict[str, object]] = []
    frame_idx = 0
    last_label = "aguardando inferência"
    last_confidence = 0.0

    try:
        with tqdm(total=total_frames or None, desc=model_name, unit="frame") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_idx % args.frame_step == 0:
                    labels, confidences, latency_ms = infer(
                        model, preprocess, categories, frame, args.topk
                    )
                    last_label, last_confidence = labels[0], confidences[0]
                    rows.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp": frame_idx / fps,
                            "top1_label": last_label,
                            "top1_conf": last_confidence,
                            "top5_labels": json.dumps(labels, ensure_ascii=False),
                            "latency_ms": latency_ms,
                        }
                    )

                writer.write(annotate_frame(frame, last_label, last_confidence))
                frame_idx += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "frame_idx",
            "timestamp",
            "top1_label",
            "top1_conf",
            "top5_labels",
            "latency_ms",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if get_device().type == "cuda"
        else None
    )
    metadata = {
        "model": model_name,
        "parameters": parameter_count(model),
        "device": str(get_device()),
        "gpu_peak_memory_mb": peak_memory_mb,
        "processed_frames": len(rows),
    }
    (args.output_dir / f"metrics_{model_name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Salvos: {video_path} e {csv_path}")


def main() -> None:
    """Valida os argumentos e executa os modelos selecionados."""
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {args.video}")
    if args.frame_step < 1 or args.topk < 1 or args.warmup < 0:
        raise ValueError("--frame_step e --topk devem ser positivos; --warmup não pode ser negativo.")
    if len(args.codec) != 4:
        raise ValueError("--codec deve ter exatamente quatro caracteres.")

    selected_models = MODEL_NAMES if args.model == "all" else (args.model,)
    for model_name in selected_models:
        process_video(args, model_name)


if __name__ == "__main__":
    main()
