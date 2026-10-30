"""Avalia os rastreadores: IDs únicos, duração e recall da classe anotada."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.coco import map_imagenet_label
from detection.evaluate import parse_list, read_labels
from tracking.trackers import TRACKER_NAMES


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    parser.add_argument("--predictions_dir", type=Path, default=Path("outputs/tracking"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tracking"))
    parser.add_argument("--trackers", nargs="+", choices=TRACKER_NAMES, default=list(TRACKER_NAMES))
    parser.add_argument("--figure_dpi", type=int, default=150)
    return parser.parse_args()


def read_predictions(path: Path) -> pd.DataFrame:
    """Carrega as predições de um rastreador."""
    if not path.is_file():
        raise FileNotFoundError(f"Predições não encontradas: {path}")
    predictions = pd.read_csv(path)
    required = {"frame_idx", "timestamp", "num_tracks", "track_ids", "labels", "latency_ms"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"{path} deve conter as colunas: {sorted(required)}")
    predictions = predictions.copy()
    predictions["track_ids"] = predictions["track_ids"].apply(parse_list)
    predictions["labels"] = predictions["labels"].apply(parse_list)
    return predictions


def evaluate_tracker(
    tracker_name: str,
    labels: pd.DataFrame,
    predictions_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Calcula métricas de identidade e recall fraco."""
    predictions = read_predictions(predictions_dir / f"predictions_{tracker_name}.csv")
    merged = labels.merge(predictions, on="frame_idx", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"Nenhum frame anotado possui predição para {tracker_name}.")
    if "coco_label" not in merged.columns:
        merged = merged.copy()
        merged["coco_label"] = merged["true_label"].map(map_imagenet_label)

    hits = [
        coco_label in detected
        for coco_label, detected in zip(merged["coco_label"], merged["labels"], strict=True)
    ]
    all_ids: list[int] = []
    id_to_frames: dict[int, int] = {}
    id_to_label: dict[int, set[str]] = {}
    for track_ids, labels_list in zip(predictions["track_ids"], predictions["labels"], strict=True):
        for track_id, label in zip(track_ids, labels_list, strict=True):
            track_id = int(track_id)
            all_ids.append(track_id)
            id_to_frames[track_id] = id_to_frames.get(track_id, 0) + 1
            id_to_label.setdefault(track_id, set()).add(str(label))

    impure = sum(1 for labels_set in id_to_label.values() if len(labels_set) > 1)
    mean_length = float(np.mean(list(id_to_frames.values()))) if id_to_frames else 0.0
    unique_ids = len(id_to_frames)
    errors = merged.loc[
        [not hit for hit in hits],
        ["frame_idx", "timestamp", "true_label", "coco_label", "labels", "track_ids"],
    ].copy()
    errors["labels"] = errors["labels"].apply(lambda items: json.dumps(items, ensure_ascii=False))
    errors["track_ids"] = errors["track_ids"].apply(lambda items: json.dumps(items, ensure_ascii=False))
    errors.to_csv(output_dir / f"errors_{tracker_name}.csv", index=False)

    mean_latency = float(predictions["latency_ms"].mean())
    metadata_path = predictions_dir / f"metrics_{tracker_name}.json"
    parameters = None
    if metadata_path.is_file():
        parameters = json.loads(metadata_path.read_text(encoding="utf-8")).get("parameters")

    return {
        "tracker": tracker_name,
        "evaluated_frames": len(merged),
        "recall": float(np.mean(hits)),
        "unique_ids": unique_ids,
        "mean_track_length": mean_length,
        "impure_tracks": impure,
        "mean_tracks_per_frame": float(predictions["num_tracks"].mean()),
        "parameters": parameters,
        "latency_mean_ms": mean_latency,
        "fps": 1_000 / mean_latency if mean_latency else 0.0,
        "error_count": int(sum(not hit for hit in hits)),
    }


def save_plots(comparison: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Salva gráficos de IDs e recall."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=comparison, x="tracker", y="unique_ids", ax=axes[0])
    axes[0].set(title="IDs únicos (menos é mais estável)", xlabel="Rastreador", ylabel="IDs")
    sns.barplot(data=comparison, x="tracker", y="mean_track_length", ax=axes[1])
    axes[1].set(title="Duração média da trajetória", xlabel="Rastreador", ylabel="Quadros processados")
    figure.tight_layout()
    figure.savefig(output_dir / "ids_vs_length.png", dpi=dpi)
    plt.close(figure)


def main() -> None:
    """Avalia os rastreadores selecionados e gera a comparação."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.labels)
    rows = [
        evaluate_tracker(name, labels, args.predictions_dir, args.output_dir)
        for name in args.trackers
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    save_plots(comparison, args.output_dir, args.figure_dpi)
    print(comparison.to_string(index=False))
    print(f"Resultados salvos em {args.output_dir}")


if __name__ == "__main__":
    main()
