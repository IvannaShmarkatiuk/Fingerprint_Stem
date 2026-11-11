"""Виділення області відбитка (ROI) за локальною дисперсією та нормалізація.

Категорія: аналіз текстур, статистичні методи (стадія 1 пайплайна П2).

Текстура папілярних ліній має великий локальний розкид яскравості,
а фон - малий. Зображення ділиться на блоки block x block, для кожного
рахується робастна оцінка розкиду (IQR / 1.349). Блок належить відбитку,
якщо розкид більший за lo + rel_threshold * (hi - lo), де lo і hi - 5-й і 95-й
процентилі розкиду по всіх блоках. Маска очищується морфологічним закриттям
і відкриттям на рівні блоків.

Потім пікселі ROI нормалізуються до заданих середнього M0 і дисперсії V0
(Hong, Wan, Jain, 1998), щоб різні сенсори мали однаковий контраст.
"""
import numpy as np

from fp.methods.morphology import closing, opening, square


def block_stats(img, block):
    """Для кожного блоку: середнє, std і робастна оцінка розкиду IQR / 1.349.

    IQR (різниця 75-го і 25-го процентилів) майже не реагує на поодинокі
    викиди (імпульсний шум), тоді як звичайне std різко зростає.
    """
    h, w = img.shape
    hb, wb = h // block, w // block
    blocks = img[:hb * block, :wb * block].reshape(hb, block, wb, block)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(hb, wb, block * block)
    q1, q3 = np.percentile(blocks, [25, 75], axis=2)
    return blocks.mean(axis=2), blocks.std(axis=2), (q3 - q1) / 1.349


def variance_roi(img, block=16, rel_threshold=0.35):
    """Повертає булеву маску ROI розміру зображення."""
    h, w = img.shape
    _, _, std = block_stats(img, block)
    if img.std() < 1e-6:
        return np.zeros((h, w), dtype=bool)
    # поріг між "рівнем фону" (5-й процентиль std блоків) і "рівнем ліній" (95-й):
    # однаковий шум піднімає обидва рівні, тому маска майже не змінюється від шуму
    lo, hi = np.percentile(std, [5, 95])
    bm = std > lo + rel_threshold * (hi - lo)
    bm = closing(bm, square(3))
    bm = opening(bm, square(3))
    full = np.zeros((h, w), dtype=bool)
    up = np.repeat(np.repeat(bm, block, axis=0), block, axis=1)
    full[:up.shape[0], :up.shape[1]] = up
    return full


def normalize(img, roi, m0=128.0, v0=2500.0):
    """Нормалізація до середнього m0 і дисперсії v0 у межах ROI; поза ROI - білий фон."""
    out = np.full(img.shape, 255.0)
    if not roi.any():
        return out
    vals = img[roi]
    m, v = vals.mean(), vals.var()
    if v < 1e-9:
        out[roi] = m0
        return out
    dev = np.sqrt(v0 * (img - m) ** 2 / v)
    norm = np.where(img > m, m0 + dev, m0 - dev)
    out[roi] = np.clip(norm[roi], 0, 255)
    return out
