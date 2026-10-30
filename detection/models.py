"""Carregamento e inferência dos detectores COCO."""

import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)

from common.coco import pack_detections
from common.device import get_device
from common.types import Detection

MODEL_NAMES = ("fasterrcnn", "mobile", "ssdlite")
Preprocess = Callable[[Image.Image], torch.Tensor]
DetectorBundle = tuple[nn.Module, Preprocess, list[str]]


def load_model(name: str) -> DetectorBundle:
    """Carrega detector, transformação oficial e categorias COCO."""
    normalized = name.lower()
    if normalized == "fasterrcnn":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    elif normalized == "mobile":
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    elif normalized == "ssdlite":
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights)
    else:
        valid_names = ", ".join(MODEL_NAMES)
        raise ValueError(f"Modelo desconhecido: {name}. Use: {valid_names}.")

    model = model.to(get_device())
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


def predict(
    bundle: DetectorBundle,
    frame: np.ndarray,
    score_threshold: float,
    scene_only: bool,
) -> tuple[list[Detection], float]:
    """Detecta objetos em um quadro BGR e mede a latência da inferência."""
    model, preprocess, categories = bundle
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
    masks = output["masks"].detach().cpu().numpy() if "masks" in output else None
    height, width = frame.shape[:2]
    detections = pack_detections(
        labels,
        scores,
        boxes,
        masks,
        score_threshold,
        scene_only,
        width,
        height,
    )
    return detections, latency_ms


def warm_up(bundle: DetectorBundle, frame: np.ndarray, iterations: int) -> None:
    """Aquece o detector sem registrar as medições."""
    if iterations <= 0:
        return
    model, preprocess, _categories = bundle
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = preprocess(image).to(get_device())
    with torch.inference_mode():
        for _ in range(iterations):
            model([tensor])
    if get_device().type == "cuda":
        torch.cuda.synchronize()


def detections_to_json(detections: list[Detection]) -> dict[str, Any]:
    """Serializa as instâncias de um quadro para CSV/JSON."""
    return {
        "num_detections": len(detections),
        "labels": [item.label for item in detections],
        "scores": [item.score for item in detections],
        "boxes": [list(item.box) for item in detections],
    }
