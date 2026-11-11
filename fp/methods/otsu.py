"""Бінаризація методом Оцу (Otsu, 1979).

Категорія: сегментація (стадія 3 пайплайна П1).

Поріг t обирається так, щоб максимізувати міжкласову дисперсію
    sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2,
де w0, w1 - частки пікселів темнішого та світлішого класу, mu0, mu1 - їх середні.
Перебираються всі 256 порогів за один прохід через кумулятивні суми.
"""
import numpy as np


def otsu_threshold(values):
    """Повертає поріг Оцу для набору значень 0..255 (клас 0: значення <= t).

    Для однорідного набору (одне значення) повертає це значення.
    """
    v = np.clip(np.rint(np.asarray(values)), 0, 255).astype(np.int64).ravel()
    if v.size == 0:
        return 0
    hist = np.bincount(v, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    levels = np.arange(256)
    w0 = p.cumsum()
    m = (p * levels).cumsum()
    mt = m[-1]
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / (w0 * w1)
    sb[~np.isfinite(sb)] = 0.0
    if sb.max() <= 0:
        return int(v[0])
    return int(np.argmax(sb))


def otsu_binarize(img, roi=None):
    """Повертає маску ліній (True = темний піксель лінії).

    Поріг рахується лише за пікселями ROI; поза ROI маска False.
    Однотонне зображення дає порожню маску.
    """
    roi = np.ones(img.shape, dtype=bool) if roi is None else roi
    vals = img[roi]
    if vals.size == 0 or vals.max() - vals.min() < 1:
        return np.zeros(img.shape, dtype=bool), 0
    t = otsu_threshold(vals)
    return (np.rint(img) <= t) & roi, t
