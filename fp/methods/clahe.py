"""CLAHE - адаптивне вирівнювання гістограми з обмеженням контрасту.

Категорія: гістограмна корекція (стадія 1 пайплайна П1).

Алгоритм:
1. Зображення ділиться на сітку плиток tiles x tiles.
2. Для кожної плитки рахується гістограма (256 рівнів).
3. Стовпці гістограми обрізаються на рівні clip_limit * (середня висота стовпця),
   надлишок рівномірно розподіляється між усіма рівнями. Це не дає шуму
   у рівних ділянках підсилюватися надто сильно.
4. З обрізаної гістограми будується таблиця перетворення (LUT) через CDF.
5. Значення кожного пікселя отримується білінійною інтерполяцією LUT
   чотирьох найближчих плиток, щоб не було видно меж між плитками.
"""
import numpy as np


def tile_lut(tile, clip_limit):
    """Будує таблицю перетворення (256 значень) для однієї плитки."""
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    n = tile.size
    limit = max(1.0, clip_limit * n / 256.0)
    excess = np.maximum(hist - limit, 0).sum()
    hist = np.minimum(hist, limit) + excess / 256.0
    cdf = hist.cumsum()
    cdf_min = cdf[hist > 0][0]
    if n - cdf_min <= 0:
        return np.arange(256, dtype=np.float64)
    return np.clip((cdf - cdf_min) / (n - cdf_min) * 255.0, 0, 255)


def clahe(img, tiles=8, clip_limit=2.0):
    """Застосовує CLAHE до сірого зображення [0, 255].

    Args:
        img: двовимірний масив.
        tiles: кількість плиток уздовж кожної осі.
        clip_limit: поріг обрізання гістограми у разах від середньої висоти стовпця.
            Велике значення дає звичайне вирівнювання гістограми.
    Returns:
        Масив float64 того ж розміру.
    """
    levels = np.clip(np.rint(img), 0, 255).astype(np.int64)
    h, w = levels.shape
    tiles = max(1, min(tiles, h, w))
    ys = np.linspace(0, h, tiles + 1).astype(int)
    xs = np.linspace(0, w, tiles + 1).astype(int)

    luts = np.empty((tiles, tiles, 256))
    for i in range(tiles):
        for j in range(tiles):
            luts[i, j] = tile_lut(levels[ys[i]:ys[i + 1], xs[j]:xs[j + 1]], clip_limit)

    y0, y1, wy = _neighbour_tiles(ys, h)
    x0, x1, wx = _neighbour_tiles(xs, w)
    y0, y1, wy = y0[:, None], y1[:, None], wy[:, None]
    x0, x1, wx = x0[None, :], x1[None, :], wx[None, :]
    top = (1 - wx) * luts[y0, x0, levels] + wx * luts[y0, x1, levels]
    bottom = (1 - wx) * luts[y1, x0, levels] + wx * luts[y1, x1, levels]
    return (1 - wy) * top + wy * bottom


def _neighbour_tiles(edges, n):
    """Для кожного пікселя осі: індекси двох сусідніх плиток і вага другої з них.

    Координата пікселя переводиться в систему центрів плиток; пікселі до
    першого центру та після останнього беруть значення крайньої плитки.
    """
    tiles = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2.0
    f = np.interp(np.arange(n) + 0.5, centers, np.arange(tiles))
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, tiles - 1)
    return i0, i1, f - i0
