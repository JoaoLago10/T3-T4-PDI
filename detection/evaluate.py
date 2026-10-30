"""Avalia os detectores com os rótulos de quadro e as métricas de eficiência."""

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from common.coco import SCENE_CLASSES, map_imagenet_label
from common.device import parameter_count
from detection.models import MODEL_NAMES, load_model


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    parser.add_argument("--predictions_dir", type=Path, default=Path("outputs/detection"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/detection"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--figure_dpi", type=int, default=150)
    return parser.parse_args()


def parse_list(value: object) -> list:
    """Converte uma lista serializada em JSON ou literal Python."""
    if isinstance(value, list):
        return value
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Lista inválida: {value}")
    return parsed


def read_labels(path: Path) -> pd.DataFrame:
    """Carrega as anotações de classificação e mapeia para COCO."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {path}")
    labels = pd.read_csv(path)
    required = {"frame_idx", "true_label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{path} deve conter as colunas: {sorted(required)}")
    labels = labels.copy()
    labels["coco_label"] = labels["true_label"].map(map_imagenet_label)
    return labels


def read_predictions(path: Path) -> pd.DataFrame:
    """Carrega as predições de um detector."""
    if not path.is_file():
        raise FileNotFoundError(f"Predições não encontradas: {path}")
    predictions = pd.read_csv(path)
    required = {"frame_idx", "timestamp", "num_detections", "labels", "scores", "boxes", "latency_ms"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"{path} deve conter as colunas: {sorted(required)}")
    predictions = predictions.copy()
    predictions["labels"] = predictions["labels"].apply(parse_list)
    predictions["scores"] = predictions["scores"].apply(parse_list)
    predictions["boxes"] = predictions["boxes"].apply(parse_list)
    return predictions


def load_efficiency(model_name: str, predictions_dir: Path) -> tuple[int, float | None]:
    """Obtém parâmetros e pico de memória medidos no experimento."""
    metadata_path = predictions_dir / f"metrics_{model_name}.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return int(metadata["parameters"]), metadata.get("gpu_peak_memory_mb")
    model, _, _ = load_model(model_name)
    parameters = parameter_count(model)
    del model
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
    """Calcula recall de presença da classe e listas de erros."""
    predictions = read_predictions(predictions_dir / f"predictions_{model_name}.csv")
    merged = labels.merge(predictions, on="frame_idx", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"Nenhum frame anotado possui predição para {model_name}.")

    hits = [
        coco_label in detected
        for coco_label, detected in zip(merged["coco_label"], merged["labels"], strict=True)
    ]
    true_class_counts = []
    scene_counts = []
    for coco_label, detected in zip(merged["coco_label"], merged["labels"], strict=True):
        if detected:
            true_class_counts.append(sum(label == coco_label for label in detected) / len(detected))
            scene_counts.append(sum(label in SCENE_CLASSES for label in detected) / len(detected))
        else:
            true_class_counts.append(0.0)
            scene_counts.append(0.0)

    errors = merged.loc[
        [not hit for hit in hits],
        ["frame_idx", "timestamp", "true_label", "coco_label", "labels", "scores", "num_detections"],
    ].copy()
    errors["labels"] = errors["labels"].apply(lambda items: json.dumps(items, ensure_ascii=False))
    errors["scores"] = errors["scores"].apply(lambda items: json.dumps(items, ensure_ascii=False))
    errors.to_csv(output_dir / f"errors_{model_name}.csv", index=False)

    per_class = {}
    for coco_label in SCENE_CLASSES:
        subset = merged[merged["coco_label"] == coco_label]
        if subset.empty:
            per_class[coco_label] = None
            continue
        class_hits = [
            coco_label in detected
            for detected in subset["labels"]
        ]
        per_class[coco_label] = float(np.mean(class_hits))

    parameters, peak_memory_mb = load_efficiency(model_name, predictions_dir)
    mean_latency = float(predictions["latency_ms"].mean())
    return {
        "model": model_name,
        "evaluated_frames": len(merged),
        "recall": float(np.mean(hits)),
        "precision_true_class": float(np.mean(true_class_counts)),
        "precision_scene": float(np.mean(scene_counts)),
        "mean_detections": float(predictions["num_detections"].mean()),
        "recall_backpack": per_class["backpack"],
        "recall_bottle": per_class["bottle"],
        "recall_keyboard": per_class["keyboard"],
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
    axes[1].set(title="FPS de inferência", xlabel="Modelo", ylabel="FPS")
    figure.tight_layout()
    figure.savefig(output_dir / "recall_vs_fps.png", dpi=dpi)
    plt.close(figure)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=comparison,
        x="latency_mean_ms",
        y="recall",
        hue="model",
        s=140,
    )
    for row in comparison.itertuples():
        plt.annotate(
            row.model,
            (row.latency_mean_ms, row.recall),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("Latência média (ms)")
    plt.ylabel("Recall")
    plt.title("Recall × latência")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_vs_latency.png", dpi=dpi)
    plt.close()


def main() -> None:
    """Avalia os detectores selecionados e gera a comparação."""
    args = parse_args()
    if args.figure_dpi < 1:
        raise ValueError("--figure_dpi deve ser positivo.")
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
