"""Генератор синтетичних відбитків з відомою правильною відповіддю (еталони для C1).

Ідея як у SFinGe (Cappelli, Maio, Maltoni): поле орієнтацій задається моделлю
"нулів і полюсів" (Sherlock & Monro, 1993) з одним ядром (core) і дельтою,
потім випадковий шум багато разів фільтрується орієнтованим фільтром Габора
з насиченням. З шуму виростають папілярні лінії з природними закінченнями
та розгалуженнями. Оскільки орієнтація і "чиста" маска ліній відомі заздалегідь,
ними можна перевіряти, чи правильно працюють методи.
"""
import numpy as np

from fp.core import conv2d, gaussian_kernel
from fp.methods.gabor import gabor_filter


def zero_pole_orientation(h, w, cores, deltas, base=0.0):
    """Напрям ліній theta(x, y) за моделлю Sherlock-Monro, радіани [0, pi)."""
    y, x = np.mgrid[0:h, 0:w]
    z = x + 1j * y
    theta = np.full((h, w), base)
    for cy, cx in cores:
        theta += 0.5 * np.angle(z - (cx + 1j * cy))
    for dy, dx in deltas:
        theta -= 0.5 * np.angle(z - (dx + 1j * dy))
    return np.mod(theta, np.pi)


def ellipse_mask(h, w, margin=0.08):
    """Еліптична маска форми подушечки пальця."""
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    ry, rx = h * (0.5 - margin), w * (0.5 - margin)
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0


def synth_fingerprint(h=256, w=224, freq=1 / 9, seed=0, iterations=12):
    """Генерує синтетичний відбиток.

    Returns:
        dict з ключами:
            image - сіре зображення 0..255 (лінії темні, фон білий);
            theta - правильне поле орієнтацій;
            ridges - правильна маска ліній (True = лінія);
            roi - маска області відбитка;
            freq - частота ліній.
    """
    rng = np.random.default_rng(seed)
    core = (h * rng.uniform(0.35, 0.45), w * rng.uniform(0.4, 0.6))
    delta = (h * rng.uniform(0.75, 0.85), w * rng.uniform(0.2, 0.8))
    theta = zero_pole_orientation(h, w, [core], [delta], base=rng.uniform(-0.2, 0.2))

    field = rng.normal(0, 1, (h, w))
    for _ in range(iterations):
        field = gabor_filter(field, theta, freq, sx=3.5, sy=3.5)
        field = np.clip(field / (np.abs(field).std() + 1e-9) * 1.5, -1, 1)

    roi = ellipse_mask(h, w)
    ridges = (field > 0) & roi
    ink = 0.5 + 0.5 * np.tanh(2.0 * field)
    # реалістичність: нерівний тиск пальця, розмиття сенсора, шум і текстура фону
    pressure = 0.55 + 0.45 * smooth_field(h, w, 5, rng)
    img = 235.0 - 160.0 * ink * pressure
    img = np.where(roi, img, 235.0)
    img = conv2d(img, gaussian_kernel(0.8))
    img = img + rng.normal(0, 4.0, (h, w))
    return {"image": np.clip(img, 0, 255), "theta": theta, "ridges": ridges,
            "roi": roi, "freq": freq}


def smooth_field(h, w, cells, rng):
    """Плавне випадкове поле 0..1: випадкова сітка cells x cells з білінійною інтерполяцією."""
    grid = rng.random((cells, cells))
    gy = np.linspace(0, cells - 1, h)
    gx = np.linspace(0, cells - 1, w)
    rows = np.array([np.interp(gx, np.arange(cells), grid[i]) for i in range(cells)])
    return np.array([np.interp(gy, np.arange(cells), rows[:, j]) for j in range(w)]).T


def stripes(h, w, angle, freq):
    """Паралельні смуги з напрямом ліній angle (для простих тестів орієнтації)."""
    y, x = np.mgrid[0:h, 0:w]
    a = angle - np.pi / 2
    u = x * np.cos(a) + y * np.sin(a)
    return 127.5 + 100 * np.cos(2 * np.pi * freq * u)
