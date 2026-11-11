import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from fp.core import to_gray, to_uint8, rescale

log = logging.getLogger(__name__)

SUPPORTED = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".pgm"}
MIN_SIDE = 32


class ImageLoadError(Exception):
    """Зображення неможливо обробити (порожній, пошкоджений або завеликий файл)."""


def load_gray(path):
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Файл не знайдено: {path}")
    if path.stat().st_size == 0:
        raise ImageLoadError(f"Порожній файл: {path}")
    try:
        with Image.open(path) as im:
            im.load()
            arr = np.asarray(im)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageLoadError(f"Пошкоджений або непідтримуваний файл {path}: {exc}") from exc

    if arr.dtype == bool:
        arr = arr.astype(np.float64) * 255.0
    gray = to_gray(arr)
    if gray.max() > 255:
        gray = rescale(gray)
    if min(gray.shape) < MIN_SIDE:
        raise ImageLoadError(f"Зображення замале ({gray.shape}), мінімум {MIN_SIDE}px: {path}")
    if gray.max() - gray.min() < 1:
        log.warning("Однотонне зображення (без папілярних ліній): %s", path)
    return gray


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype == bool:
        out = np.where(img, 0, 255).astype(np.uint8)
    else:
        out = to_uint8(img)
    Image.fromarray(out).save(path)
