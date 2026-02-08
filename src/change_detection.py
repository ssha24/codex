"""Core utilities for multitemporal satellite image change detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage import filters

try:
    import rasterio
except ImportError:  # pragma: no cover - optional dependency for GeoTIFF
    rasterio = None
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for deep learning
    torch = None

from PIL import Image


@dataclass
class ChangeDetectionResult:
    """Container for change detection outputs."""

    change_score: np.ndarray
    change_map: np.ndarray


def _read_with_rasterio(path: Path, band: int) -> np.ndarray:
    if rasterio is None:
        raise RuntimeError("rasterio is required to read GeoTIFF files.")
    with rasterio.open(path) as dataset:
        data = dataset.read(band).astype(np.float32)
    return data


def _read_with_pillow(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        data = np.array(img).astype(np.float32)
    if data.ndim == 3:
        data = data[:, :, 0]
    return data


def load_image(path: str | Path, band: int = 1) -> np.ndarray:
    """Load a single-band image into a float32 NumPy array."""

    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return _read_with_rasterio(path, band)
    return _read_with_pillow(path)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range, handling constant arrays."""

    image = image.astype(np.float32)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))
    if np.isclose(max_val, min_val):
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def compute_change_score(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Compute absolute difference change score between two images."""

    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same shape.")
    return np.abs(image_a - image_b)


def threshold_change(change_score: np.ndarray, threshold: float | None = None) -> Tuple[np.ndarray, float]:
    """Threshold change score using Otsu's method or a fixed threshold."""

    if threshold is None:
        threshold = float(filters.threshold_otsu(change_score))
    change_map = (change_score >= threshold).astype(np.uint8)
    return change_map, threshold


def detect_change(
    image_a: np.ndarray,
    image_b: np.ndarray,
    threshold: float | None = None,
) -> ChangeDetectionResult:
    """Full change detection pipeline returning score and binary map."""

    normalized_a = normalize(image_a)
    normalized_b = normalize(image_b)
    change_score = compute_change_score(normalized_a, normalized_b)
    change_map, _ = threshold_change(change_score, threshold)
    return ChangeDetectionResult(change_score=change_score, change_map=change_map)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised in optional test
        raise RuntimeError("PyTorch is required for deep learning change detection.")


def _prepare_deeplearning_input(image_a: np.ndarray, image_b: np.ndarray) -> "torch.Tensor":
    normalized_a = normalize(image_a)
    normalized_b = normalize(image_b)
    stacked = np.stack([normalized_a, normalized_b], axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    return tensor


def _extract_change_score(output: "torch.Tensor") -> np.ndarray:
    if output.ndim == 4:
        output = output.squeeze(0)
    if output.ndim == 3 and output.shape[0] > 1:
        output = torch.softmax(output, dim=0)[1]
    elif output.ndim == 3:
        output = output[0]
    elif output.ndim != 2:
        raise ValueError("Model output must be 2D or 3D with channel dimension.")
    return output.detach().cpu().numpy().astype(np.float32)


def detect_change_deeplearning(
    image_a: np.ndarray,
    image_b: np.ndarray,
    model_path: str | Path,
    *,
    device: str = "cpu",
    threshold: float = 0.5,
) -> ChangeDetectionResult:
    """Run change detection using a TorchScript model."""

    _require_torch()
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    tensor = _prepare_deeplearning_input(image_a, image_b).to(device)
    with torch.no_grad():
        output = model(tensor)
    change_score = _extract_change_score(output)
    change_map = (change_score >= threshold).astype(np.uint8)
    return ChangeDetectionResult(change_score=change_score, change_map=change_map)
