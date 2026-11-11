## Використання ШІ.
## Інструменти
Claude (Anthropic), claude.ai.
## Використовувала для пошуку джерел основної вибірки.
## Допомога у вирішенні проблем з кодом, частичне виправлення.
## Також використовувала для написання тестів test_methods.py, зокрема методи CLAHE, Оцу і Sauvola, test_block_stats_by_hand(), 
У test_pipeline.py. Мала запити для пояснення основної суті пайплайнів, а також я брала такий код тестів за основу до своїх: import requests
from typing import Any, Dict, List
import logging
import sys

def get_exchange_data() -> List[Dict[str, Any]]:
    url = "https://api.coincap.io/v2/exchanges" # Note this API is not longer active
    try:
        r = requests.get(url)
    except requests.ConnectionError as ce:
        logging.error(f"There was an error with the request, {ce}")
        sys.exit(1)
    return r.json().get("data", [])
##Згенеровані методи для розбору roi.py, sauvola.py, otsu.py, pipelines.py(частково:
"""Два пайплайни обробки відбитка. Кожна стадія - метод з іншої категорії.

П1 "Габор":  CLAHE -> фільтр Габора за полем орієнтацій -> Оцу -> Zhang-Suen -> мінуції
П2 "Фур'є":  ROI за дисперсією + нормалізація -> FFT-підсилення -> Sauvola
             -> відкриття + скелетування -> мінуції

Кожен пайплайн повертає PipelineResult зі словником проміжних результатів,
щоб програма могла зберегти вихід кожної стадії.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from fp.core import rescale
from fp.methods.clahe import clahe
from fp.methods.orientation import orientation_field, block_frequency
from fp.methods.gabor import gabor_filter
from fp.methods.otsu import otsu_binarize
from fp.methods.thinning import zhang_suen
from fp.methods.minutiae import extract_minutiae
from fp.methods.morphology import opening, square, cross, closing
from fp.methods.roi import variance_roi, normalize
from fp.methods.fft_enhance import fft_enhance
from fp.methods.sauvola import sauvola_binarize


@dataclass
class PipelineResult:
    """Результат запуску пайплайна."""
    name: str
    stages: dict = field(default_factory=dict)   # назва стадії -> зображення/маска
    enhanced: np.ndarray = None                   # сіре підсилене зображення (для метрик)
    binary: np.ndarray = None                     # маска ліній
    skeleton: np.ndarray = None
    roi: np.ndarray = None
    theta: np.ndarray = None
    minutiae: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)   # назва стадії -> секунди


def _timed(res, name, func, *args, **kwargs):
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    res.timings[name] = time.perf_counter() - t0
    return out


def _gabor_stage(img, cfg):
    """Стадія 2 П1: поле орієнтацій, частота, фільтр Габора, маска надійних ділянок."""
    theta, coherence, energy = orientation_field(img, cfg["orient_window"], cfg["orient_smooth"])
    freq = cfg["freq"]
    if cfg.get("estimate_freq", True):
        est = block_frequency(img, theta)
        if est is not None:
            freq = est
    resp = gabor_filter(img, theta, freq, cfg["sigma_x"], cfg["sigma_y"], cfg["n_angles"])
    # ROI П1: ділянки з вираженою орієнтацією та достатньою енергією градієнта
    lo, hi = np.percentile(energy, [5, 95])
    roi = (coherence > cfg["coherence_min"]) & (energy > lo + cfg["energy_rel"] * (hi - lo))
    roi = opening(closing(roi, square(15)), square(15))
    enhanced = np.full(img.shape, 255.0)
    if roi.any():
        lo, hi = np.percentile(resp[roi], [1, 99])
        enhanced[roi] = np.clip((resp[roi] - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255)
    return enhanced, roi, theta, freq
)

## Першу структуру з файлами без коду згенерував ШІ, далі для цих файлів я писала код, або ж вставляла згенерований коди, що були вказані попередньо вище.