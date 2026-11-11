"""Локальна бінаризація методом Sauvola (Sauvola & Pietikainen, 2000).

Категорія: сегментація (стадія 3 пайплайна П2).

Для кожного пікселя поріг рахується за вікном window x window:
    T = m * (1 + k * (s / R - 1)),
де m і s - локальні середнє та стандартне відхилення, R - динамічний діапазон
std (128 для 8-бітних зображень). На відміну від глобального порогу Оцу, метод
пристосовується до нерівного освітлення та контрасту.
Локальні m і s рахуються через інтегральні зображення за O(1) на піксель.
"""
import numpy as np

from fp.core import box_sum


def sauvola_threshold(img, window=15, k=0.2, r=128.0):
    """Повертає карту локальних порогів."""
    img = np.asarray(img, dtype=np.float64)
    n = window * window
    mean = box_sum(img, window) / n
    sq = box_sum(img * img, window) / n
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    return mean * (1 + k * (std / r - 1))


def sauvola_binarize(img, roi=None, window=15, k=0.2, r=128.0):
    """Маска ліній (True = піксель темніший за локальний поріг), обмежена ROI."""
    img = np.asarray(img, dtype=np.float64)
    if roi is not None and roi.any():
        # фон поза ROI замінюємо медіаною ROI, щоб білий фон не завищував поріг біля межі
        img = np.where(roi, img, np.median(img[roi]))
    mask = img <= sauvola_threshold(img, window, k, r)
    if roi is not None:
        mask &= roi
    return mask
