"""Avalia acurácia e eficiência dos classificadores."""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.models import ResNet50_Weights

try:
    from .models import MODEL_NAMES, load_model, parameter_count
except ImportError:
    from models import MODEL_NAMES, load_model, parameter_count


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/labels.csv"),
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--figure_dpi", type=int, default=150)
    return parser.parse_args()


def read_labels(path: Path) -> pd.DataFrame:
    """Carrega e valida as anotações manuais."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {path}")
    labels = pd.read_csv(path)
    required = {"frame_idx", "true_label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{path} deve conter as colunas: {sorted(required)}")
    if labels["frame_idx"].duplicated().any():
        raise ValueError("labels.csv contém frame_idx duplicado.")
    if labels["true_label"].isna().any() or (labels["true_label"].str.strip() == "").any():
        raise ValueError("Preencha todos os valores de true_label antes de avaliar.")

    categories = set(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    invalid = sorted(set(labels["true_label"]) - categories)
    if invalid:
        raise ValueError(
            "Rótulos fora das categorias oficiais do ImageNet: " + ", ".join(invalid)
        )
    return labels


def parse_topk(value: object) -> list[str]:
    """Converte a lista serializada de classes para Python."""
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"top5_labels inválido: {value}")
    return [str(item) for item in parsed]


def read_predictions(path: Path) -> pd.DataFrame:
    """Carrega e valida as predições de um modelo."""
    if not path.is_file():
        raise FileNotFoundError(f"Predições não encontradas: {path}")
    predictions = pd.read_csv(path)
    required = {
        "frame_idx",
        "timestamp",
        "top1_label",
        "top1_conf",
        "top5_labels",
        "latency_ms",
    }
    if not required.issubset(predictions.columns):
        raise ValueError(f"{path} deve conter as colunas: {sorted(required)}")
    if predictions["frame_idx"].duplicated().any():
        raise ValueError(f"{path} contém frame_idx duplicado.")
    predictions["top5_labels"] = predictions["top5_labels"].apply(parse_topk)
    return predictions


def load_efficiency(
    model_name: str,
    predictions_dir: Path,
) -> tuple[int, float | None]:
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


def save_confusion_matrix(
    merged: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Gera a matriz de confusão para as classes observadas."""
    observed_labels = sorted(
        set(merged["true_label"]).union(set(merged["top1_label"]))
    )
    matrix = confusion_matrix(
        merged["true_label"],
        merged["top1_label"],
        labels=observed_labels,
    )
    figure_size = max(7.0, min(20.0, len(observed_labels) * 0.8))
    plt.figure(figsize=(figure_size, figure_size))
    sns.heatmap(
        matrix,
        annot=len(observed_labels) <= 20,
        fmt="d",
        cmap="Blues",
        xticklabels=observed_labels,
        yticklabels=observed_labels,
    )
    plt.xlabel("Classe predita")
    plt.ylabel("Classe verdadeira")
    plt.title(f"Matriz de confusão — {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=dpi)
    plt.close()


def evaluate_model(
    model_name: str,
    labels: pd.DataFrame,
    predictions_dir: Path,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    """Calcula métricas, matriz de confusão e lista de erros."""
    predictions = read_predictions(
        predictions_dir / f"predictions_{model_name}.csv"
    )
    merged = labels.merge(predictions, on="frame_idx", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"Nenhum frame anotado possui predição para {model_name}.")
    missing_count = len(labels) - len(merged)
    if missing_count:
        print(f"Aviso: {model_name} não possui predição para {missing_count} anotações.")

    true_values = merged["true_label"]
    predicted_values = merged["top1_label"]
    top5_hits = [
        true_label in top_labels
        for true_label, top_labels in zip(
            true_values, merged["top5_labels"], strict=True
        )
    ]
    parameters, peak_memory_mb = load_efficiency(model_name, predictions_dir)
    mean_latency = float(predictions["latency_ms"].mean())

    errors = merged.loc[
        true_values != predicted_values,
        [
            "frame_idx",
            "timestamp",
            "true_label",
            "top1_label",
            "top1_conf",
            "top5_labels",
        ],
    ].copy()
    errors["true_label_in_top5"] = [
        row.true_label in row.top5_labels for row in errors.itertuples()
    ]
    errors["top5_labels"] = errors["top5_labels"].apply(
        lambda labels_list: json.dumps(labels_list, ensure_ascii=False)
    )
    errors.to_csv(output_dir / f"errors_{model_name}.csv", index=False)
    save_confusion_matrix(merged, model_name, output_dir, dpi)

    return {
        "model": model_name,
        "evaluated_frames": len(merged),
        "top1_accuracy": accuracy_score(true_values, predicted_values),
        "top5_accuracy": float(np.mean(top5_hits)),
        "precision_macro": precision_score(
            true_values, predicted_values, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            true_values, predicted_values, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            true_values, predicted_values, average="macro", zero_division=0
        ),
        "parameters": parameters,
        "latency_mean_ms": mean_latency,
        "fps": 1_000 / mean_latency,
        "gpu_peak_memory_mb": peak_memory_mb,
        "error_count": len(errors),
    }


def save_comparison_plots(
    comparison: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Salva gráficos de barras e de dispersão."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=comparison, x="model", y="top1_accuracy", ax=axes[0])
    axes[0].set(title="Acurácia top-1", xlabel="Modelo", ylabel="Acurácia")
    axes[0].set_ylim(0, 1)
    sns.barplot(data=comparison, x="model", y="fps", ax=axes[1])
    axes[1].set(title="FPS de inferência", xlabel="Modelo", ylabel="FPS")
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_vs_fps.png", dpi=dpi)
    plt.close(figure)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=comparison,
        x="latency_mean_ms",
        y="top1_accuracy",
        hue="model",
        s=140,
    )
    for row in comparison.itertuples():
        plt.annotate(
            row.model,
            (row.latency_mean_ms, row.top1_accuracy),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("Latência média (ms)")
    plt.ylabel("Acurácia top-1")
    plt.title("Acurácia × latência")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_latency.png", dpi=dpi)
    plt.close()


def main() -> None:
    """Avalia os modelos selecionados e gera a comparação."""
    args = parse_args()
    if args.figure_dpi < 1:
        raise ValueError("--figure_dpi deve ser positivo.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.labels)

    rows = [
        evaluate_model(
            model_name,
            labels,
            args.predictions_dir,
            args.output_dir,
            args.figure_dpi,
        )
        for model_name in args.models
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    save_comparison_plots(comparison, args.output_dir, args.figure_dpi)
    print(comparison.to_string(index=False))
    print(f"Resultados salvos em {args.output_dir}")


if __name__ == "__main__":
    main()
