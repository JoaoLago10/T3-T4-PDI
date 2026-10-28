"""Extrai frames e cria um CSV para anotação manual."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Vídeo de entrada.")
    parser.add_argument("--num_frames", type=int, default=50, help="Quantidade de frames.")
    parser.add_argument(
        "--frame_step",
        type=int,
        default=5,
        help="Passo usado posteriormente no run_video.py.",
    )
    parser.add_argument(
        "--annotations_dir",
        type=Path,
        default=Path("data/annotations"),
    )
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescrever labels.csv existente.",
    )
    return parser.parse_args()


def main() -> None:
    """Seleciona frames uniformes e gera labels.csv vazio."""
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {args.video}")
    if args.num_frames < 1 or args.frame_step < 1:
        raise ValueError("--num_frames e --frame_step devem ser positivos.")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality deve estar entre 0 e 100.")

    labels_path = args.annotations_dir / "labels.csv"
    if labels_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{labels_path} já existe. Use --overwrite para substituí-lo."
        )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {args.video}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        capture.release()
        raise RuntimeError("O vídeo não informa uma quantidade válida de frames.")

    processable = np.arange(0, total_frames, args.frame_step, dtype=int)
    sample_count = min(args.num_frames, len(processable))
    positions = np.linspace(0, len(processable) - 1, sample_count, dtype=int)
    selected_frames = np.unique(processable[positions]).tolist()

    frames_dir = args.annotations_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    try:
        for frame_idx in selected_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                print(f"Aviso: não foi possível extrair o frame {frame_idx}.")
                continue
            output_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            saved = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not saved:
                raise RuntimeError(f"Falha ao salvar {output_path}")
            rows.append({"frame_idx": frame_idx, "true_label": ""})
    finally:
        capture.release()

    args.annotations_dir.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["frame_idx", "true_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} frames extraídos em {frames_dir}")
    print(f"Preencha a coluna true_label em {labels_path}")


if __name__ == "__main__":
    main()
