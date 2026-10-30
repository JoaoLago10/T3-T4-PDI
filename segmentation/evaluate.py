"""Avalia os segmentadores com recall de classe e área média das máscaras."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from common.coco import map_imagenet_label
from detection.evaluate import read_labels, read_predictions
from segmentation.models import MODEL_NAMES, load_model


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    parser.add_argument("--predictions_dir", type=Path, default=Path("outputs/segmentation"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/segmentation"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--figure_dpi", type=int, default=150)
    return parser.parse_args()


def load_efficiency(model_name: str, predictions_dir: Path) -> tuple[int, float | None]:
    """Obtém parâmetros e pico de memória medidos no experimento."""
    metadata_path = predictions_dir / f"metrics_{model_name}.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return int(metadata["parameters"]), metadata.get("gpu_peak_memory_mb")
    segmenter = load_model(model_name)
    parameters = segmenter.n_parameters
    del segmenter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Aviso: {metadata_path} ausente; pico de memória ficará vazio.")
    return parameters, None


def evaluate_model(
    model_name: str,
    labels: pd.DataFrame,
    predictions_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Calcula recall de presença da classe e estatísticas das máscaras."""
    predictions = read_predictions(predictions_dir / f"predictions_{model_name}.csv")
    merged = labels.merge(predictions, on="frame_idx", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"Nenhum frame anotado possui predição para {model_name}.")
    if "coco_label" not in merged.columns:
        merged = merged.copy()
        merged["coco_label"] = merged["true_label"].map(map_imagenet_label)

    hits = [
        coco_label in detected
        for coco_label, detected in zip(merged["coco_label"], merged["labels"], strict=True)
    ]
    errors = merged.loc[
        [not hit for hit in hits],
        ["frame_idx", "timestamp", "true_label", "coco_label", "labels", "num_detections"],
    ].copy()
    errors["labels"] = errors["labels"].apply(lambda items: json.dumps(items, ensure_ascii=False))
    errors.to_csv(output_dir / f"errors_{model_name}.csv", index=False)

    parameters, peak_memory_mb = load_efficiency(model_name, predictions_dir)
    mean_latency = float(predictions["latency_ms"].mean())
    mean_area = float(predictions["mean_mask_area"].mean()) if "mean_mask_area" in predictions else 0.0
    return {
        "model": model_name,
        "evaluated_frames": len(merged),
        "recall": float(np.mean(hits)),
        "mean_detections": float(predictions["num_detections"].mean()),
        "mean_mask_area": mean_area,
        "parameters": parameters,
        "latency_mean_ms": mean_latency,
        "fps": 1_000 / mean_latency if mean_latency else 0.0,
        "gpu_peak_memory_mb": peak_memory_mb,
        "error_count": int(sum(not hit for hit in hits)),
    }


def save_plots(comparison: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Salva gráficos de recall versus custo."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=comparison, x="model", y="recall", ax=axes[0])
    axes[0].set(title="Recall da classe anotada", xlabel="Modelo", ylabel="Recall")
    axes[0].set_ylim(0, 1)
    sns.barplot(data=comparison, x="model", y="fps", ax=axes[1])
    axes[1].set(title="FPS", xlabel="Modelo", ylabel="FPS")
    figure.tight_layout()
    figure.savefig(output_dir / "recall_vs_fps.png", dpi=dpi)
    plt.close(figure)


def main() -> None:
    """Avalia os segmentadores selecionados e gera a comparação."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.labels)
    rows = [
        evaluate_model(model_name, labels, args.predictions_dir, args.output_dir)
        for model_name in args.models
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    save_plots(comparison, args.output_dir, args.figure_dpi)
    print(comparison.to_string(index=False))
    print(f"Resultados salvos em {args.output_dir}")


if __name__ == "__main__":
    main()
