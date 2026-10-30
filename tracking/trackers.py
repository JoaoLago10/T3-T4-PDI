"""Rastreadores por associação: IoU, centróide e Kalman."""

from dataclasses import dataclass, field

import numpy as np

from common.types import Detection

TRACKER_NAMES = ("iou", "centroid", "kalman")


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Calcula a interseção sobre união de duas caixas xyxy."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Devolve o centro da caixa."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _greedy_match(score_matrix: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Associa pares de maior pontuação sem reutilizar linhas ou colunas."""
    if score_matrix.size == 0:
        return []
    pairs = [
        (float(score_matrix[row, col]), row, col)
        for row in range(score_matrix.shape[0])
        for col in range(score_matrix.shape[1])
        if score_matrix[row, col] >= threshold
    ]
    pairs.sort(reverse=True)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, row, col in pairs:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        matches.append((row, col))
    return matches


class KalmanBox:
    """Filtro de Kalman de velocidade constante sobre (cx, cy, w, h)."""

    def __init__(self, box: tuple[float, float, float, float]) -> None:
        cx, cy = box_center(box)
        width = max(box[2] - box[0], 1.0)
        height = max(box[3] - box[1], 1.0)
        self.state = np.array([cx, cy, width, height, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64) * 10.0
        self.transition = np.eye(8, dtype=np.float64)
        for index in range(4):
            self.transition[index, index + 4] = 1.0
        self.observation = np.eye(4, 8, dtype=np.float64)
        self.process_noise = np.eye(8, dtype=np.float64)
        self.process_noise[4:, 4:] *= 0.1
        self.measure_noise = np.eye(4, dtype=np.float64) * 1.0

    def predict(self) -> tuple[float, float, float, float]:
        """Projeta o estado um passo à frente."""
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + self.process_noise
        )
        self.state[2] = max(self.state[2], 1.0)
        self.state[3] = max(self.state[3], 1.0)
        return self.as_box()

    def update(self, box: tuple[float, float, float, float]) -> None:
        """Corrige o estado com a caixa observada."""
        cx, cy = box_center(box)
        measurement = np.array(
            [cx, cy, max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)],
            dtype=np.float64,
        )
        projected = self.observation @ self.state
        residual = measurement - projected
        innovation = self.observation @ self.covariance @ self.observation.T + self.measure_noise
        gain = self.covariance @ self.observation.T @ np.linalg.pinv(innovation)
        self.state = self.state + gain @ residual
        identity = np.eye(8)
        self.covariance = (identity - gain @ self.observation) @ self.covariance

    def as_box(self) -> tuple[float, float, float, float]:
        """Converte o estado em caixa xyxy."""
        cx, cy, width, height = self.state[:4]
        return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


@dataclass
class Track:
    """Uma trajetória ativa."""

    track_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    age: int = 0
    hits: int = 1
    kalman: KalmanBox | None = field(default=None, repr=False)


class BaseTracker:
    """Rastreador com associação gulosa e tempo de vida limitado."""

    def __init__(self, match_threshold: float, max_age: int) -> None:
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: list[Track] = []

    def reset(self) -> None:
        """Remove todas as trajetórias."""
        self.next_id = 1
        self.tracks = []

    def _score(self, track: Track, detection: Detection) -> float:
        raise NotImplementedError

    def _predicted_box(self, track: Track) -> tuple[float, float, float, float]:
        return track.box

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Associa detecções às trajetórias e devolve instâncias com ID."""
        for track in self.tracks:
            track.box = self._predicted_box(track)
            track.age += 1

        score_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
        for row, track in enumerate(self.tracks):
            for col, detection in enumerate(detections):
                if track.label != detection.label:
                    score_matrix[row, col] = -1.0
                    continue
                score_matrix[row, col] = self._score(track, detection)

        matches = _greedy_match(score_matrix, self.match_threshold)
        matched_tracks = {row for row, _col in matches}
        matched_detections = {col for _row, col in matches}

        for row, col in matches:
            detection = detections[col]
            track = self.tracks[row]
            track.box = detection.box
            track.score = detection.score
            track.mask = detection.mask
            track.label = detection.label
            track.age = 0
            track.hits += 1
            if track.kalman is not None:
                track.kalman.update(detection.box)

        for col, detection in enumerate(detections):
            if col in matched_detections:
                continue
            kalman = KalmanBox(detection.box) if isinstance(self, KalmanTracker) else None
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    label=detection.label,
                    score=detection.score,
                    box=detection.box,
                    mask=detection.mask,
                    kalman=kalman,
                )
            )
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track.age <= self.max_age]
        results: list[Detection] = []
        for track in self.tracks:
            if track.age > 0:
                continue
            results.append(
                Detection(
                    box=track.box,
                    label=track.label,
                    score=track.score,
                    mask=track.mask,
                    track_id=track.track_id,
                )
            )
        return results


class IoUTracker(BaseTracker):
    """Associa pela maior IoU entre caixas da mesma classe."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 8) -> None:
        super().__init__(iou_threshold, max_age)

    def _score(self, track: Track, detection: Detection) -> float:
        return iou(track.box, detection.box)


class CentroidTracker(BaseTracker):
    """Associa pelo menor deslocamento do centro, convertido em pontuação."""

    def __init__(self, max_distance: float = 80.0, max_age: int = 8) -> None:
        super().__init__(0.0, max_age)
        self.max_distance = max_distance

    def _score(self, track: Track, detection: Detection) -> float:
        tx, ty = box_center(track.box)
        dx, dy = box_center(detection.box)
        distance = float(np.hypot(tx - dx, ty - dy))
        if distance > self.max_distance:
            return -1.0
        return 1.0 - distance / self.max_distance


class KalmanTracker(BaseTracker):
    """Prevê a caixa com Kalman e associa pela IoU prevista."""

    def __init__(self, iou_threshold: float = 0.2, max_age: int = 10) -> None:
        super().__init__(iou_threshold, max_age)

    def _predicted_box(self, track: Track) -> tuple[float, float, float, float]:
        if track.kalman is None:
            track.kalman = KalmanBox(track.box)
        return track.kalman.predict()

    def _score(self, track: Track, detection: Detection) -> float:
        return iou(track.box, detection.box)


def load_tracker(name: str) -> BaseTracker:
    """Instancia o rastreador pedido."""
    normalized = name.lower()
    if normalized == "iou":
        return IoUTracker()
    if normalized == "centroid":
        return CentroidTracker()
    if normalized == "kalman":
        return KalmanTracker()
    valid_names = ", ".join(TRACKER_NAMES)
    raise ValueError(f"Rastreador desconhecido: {name}. Use: {valid_names}.")
