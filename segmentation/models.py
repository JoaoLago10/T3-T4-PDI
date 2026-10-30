"""Segmentadores: Mask R-CNN, GrabCut e limiar de Otsu nas caixas."""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from common.coco import pack_detections
from common.device import get_device, parameter_count
from common.types import Detection
from detection.models import DetectorBundle, load_model as load_detector, predict as detect, warm_up as warm_detector

MODEL_NAMES = ("maskrcnn", "grabcut", "threshold")


def _box_crop_mask(frame: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Recorta o quadro na caixa inteira, com limites válidos."""
    height, width = frame.shape[:2]
    x1 = int(np.clip(round(box[0]), 0, width - 1))
    y1 = int(np.clip(round(box[1]), 0, height - 1))
    x2 = int(np.clip(round(box[2]), 0, width - 1))
    y2 = int(np.clip(round(box[3]), 0, height - 1))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def grabcut_mask(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Segmenta o objeto dentro da caixa com GrabCut."""
    crop, (x1, y1, x2, y2) = _box_crop_mask(frame, box)
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (2, 2, max(crop.shape[1] - 4, 1), max(crop.shape[0] - 4, 1))
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1.0,
        0.0,
    ).astype(np.float32)
    full = np.zeros(frame.shape[:2], np.float32)
    full[y1:y2, x1:x2] = binary
    return full


def threshold_mask(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Segmenta o objeto dentro da caixa com Otsu e limpeza morfológica."""
    crop, (x1, y1, x2, y2) = _box_crop_mask(frame, box)
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    if border.mean() > 127:
        binary = cv2.bitwise_not(binary)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    full = np.zeros(frame.shape[:2], np.float32)
    full[y1:y2, x1:x2] = (binary > 127).astype(np.float32)
    return full


@dataclass
class Segmenter:
    """Empacota um segmentador com a mesma interface de predição."""

    name: str
    predict_fn: object
    warm_up_fn: object
    n_parameters: int

    def predict(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Devolve instâncias com máscara e a latência em milissegundos."""
        return self.predict_fn(frame)  # type: ignore[operator]

    def warm_up(self, frame: np.ndarray, iterations: int) -> None:
        """Aquece o modelo interno, se houver."""
        self.warm_up_fn(frame, iterations)  # type: ignore[operator]


def _load_maskrcnn() -> tuple[object, object, list[str], int]:
    """Carrega Mask R-CNN ResNet-50 FPN V2."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model = model.to(get_device())
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"]), parameter_count(model)


def load_model(
    name: str,
    detector_name: str = "ssdlite",
    score_threshold: float = 0.4,
    scene_only: bool = True,
) -> Segmenter:
    """Carrega o segmentador pedido."""
    normalized = name.lower()
    if normalized == "maskrcnn":
        model, preprocess, categories, n_parameters = _load_maskrcnn()

        def predict_fn(frame: np.ndarray) -> tuple[list[Detection], float]:
            device = get_device()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = preprocess(image).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model([tensor])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1_000
            labels = [
                categories[int(index)] if 0 <= int(index) < len(categories) else str(int(index))
                for index in output["labels"].detach().cpu().tolist()
            ]
            scores = output["scores"].detach().cpu().numpy()
            boxes = output["boxes"].detach().cpu().numpy()
            masks = output["masks"].detach().cpu().numpy()
            height, width = frame.shape[:2]
            detections = pack_detections(
                labels, scores, boxes, masks, score_threshold, scene_only, width, height
            )
            return detections, latency_ms

        def warm_up_fn(frame: np.ndarray, iterations: int) -> None:
            if iterations <= 0:
                return
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = preprocess(image).to(get_device())
            with torch.inference_mode():
                for _ in range(iterations):
                    model([tensor])
            if get_device().type == "cuda":
                torch.cuda.synchronize()

        return Segmenter(normalized, predict_fn, warm_up_fn, n_parameters)

    if normalized in {"grabcut", "threshold"}:
        detector: DetectorBundle = load_detector(detector_name)
        mask_fn = grabcut_mask if normalized == "grabcut" else threshold_mask

        def predict_fn(frame: np.ndarray) -> tuple[list[Detection], float]:
            detections, detect_ms = detect(detector, frame, score_threshold, scene_only)
            start = time.perf_counter()
            for item in detections:
                item.mask = mask_fn(frame, item.box)
            extra_ms = (time.perf_counter() - start) * 1_000
            return detections, detect_ms + extra_ms

        def warm_up_fn(frame: np.ndarray, iterations: int) -> None:
            warm_detector(detector, frame, iterations)

        return Segmenter(normalized, predict_fn, warm_up_fn, parameter_count(detector[0]))

    valid_names = ", ".join(MODEL_NAMES)
    raise ValueError(f"Modelo desconhecido: {name}. Use: {valid_names}.")
