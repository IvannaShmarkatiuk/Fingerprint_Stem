"""Поле орієнтацій та частота папілярних ліній.

Допоміжні обчислення для фільтра Габора (стадія 2 пайплайна П1).

Орієнтація оцінюється через структурний тензор градієнта (Kass & Witkin, 1987;
Bazen & Gerez, 2002). Градієнт береться центральними різницями [-1, 0, 1],
а не оператором Собела, бо Собел реалізовувався в лабораторних роботах.

Домовленість про координати: x = стовпець (вправо), y = рядок (вниз).
theta - напрям ЛІНІЇ в [0, pi). Напрям градієнта (нормалі) = theta - pi/2.
"""
import numpy as np

from fp.core import box_sum


def gradients(img):
    """Центральні різниці: gx = d/dx (по стовпцях), gy = d/dy (по рядках)."""
    p = np.pad(np.asarray(img, dtype=np.float64), 1, mode="edge")
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) / 2.0
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2.0
    return gx, gy


def orientation_field(img, window=17, smooth=33):
    """Обчислює поле орієнтацій ліній, когерентність і енергію градієнта.

    Args:
        window: розмір вікна для сумування структурного тензора.
        smooth: розмір вікна для згладжування поля орієнтацій у подвоєних кутах.
    Returns:
        theta: напрям ліній у кожному пікселі, радіани [0, pi).
        coherence: 0..1, наскільки чітко виражений один напрям.
        energy: середня енергія градієнта у вікні.
    """
    gx, gy = gradients(img)
    gxx = box_sum(gx * gx, window)
    gyy = box_sum(gy * gy, window)
    gxy = box_sum(gx * gy, window)

    # подвоєний кут нормалі; усереднення векторів (cos 2a, sin 2a) коректне для кутів mod pi
    c2 = gxx - gyy
    s2 = 2 * gxy
    c2s = box_sum(c2, smooth)
    s2s = box_sum(s2, smooth)
    normal = 0.5 * np.arctan2(s2s, c2s)
    theta = np.mod(normal + np.pi / 2, np.pi)

    # когерентність рахуємо за згладженим тензором: так вона стабільніша при шумі
    denom = box_sum(gxx + gyy, smooth)
    coherence = np.where(denom > 1e-9, np.hypot(c2s, s2s) / np.maximum(denom, 1e-9), 0.0)
    energy = (gxx + gyy) / (window * window)
    return theta, coherence, energy


def block_frequency(img, theta, block=32, fmin=1 / 15, fmax=1 / 4):
    """Оцінює частоту ліній методом x-сигнатури (Hong, Wan, Jain, 1998).

    Для кожного блоку значення проєктуються на нормаль до ліній; відстань між
    сусідніми мінімумами сигнатури дає період. Повертає медіану по блоках
    (стійка до блоків, де оцінка не вдалася) або None.
    """
    h, w = img.shape
    freqs = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = img[y:y + block, x:x + block]
            if patch.std() < 1e-6:
                continue
            t = theta[y + block // 2, x + block // 2]
            nx, ny = np.cos(t - np.pi / 2), np.sin(t - np.pi / 2)
            yy, xx = np.mgrid[0:block, 0:block]
            proj = (xx - block / 2) * nx + (yy - block / 2) * ny
            bins = np.rint(proj + block).astype(int)
            sums = np.bincount(bins.ravel(), patch.ravel(), minlength=2 * block + 1)
            cnt = np.bincount(bins.ravel(), minlength=2 * block + 1)
            sig = sums[cnt > 0] / cnt[cnt > 0]
            mins = np.where((sig[1:-1] < sig[:-2]) & (sig[1:-1] <= sig[2:]))[0]
            if len(mins) >= 2:
                f = (len(mins) - 1) / float(mins[-1] - mins[0])
                if fmin <= f <= fmax:
                    freqs.append(f)
    if not freqs:
        return None
    return float(np.median(freqs))
