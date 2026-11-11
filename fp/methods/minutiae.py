"""Пошук мінуцій методом crossing number (Rutovitz, 1966).

Категорія: виділення ключових точок (стадія 5 обох пайплайнів).

Для пікселя скелета CN = 0.5 * sum |P_i - P_(i+1)| по 8 сусідах по колу.
CN == 1 - закінчення лінії (ending), CN == 3 - розгалуження (bifurcation).

Фільтрація хибних мінуцій:
- мінуції біля краю області відбитка відкидаються (край - не справжнє закінчення);
- пари закінчень ближче за min_dist відкидаються (розрив лінії);
- пари закінчення + розгалуження ближче за min_dist відкидаються (короткий відросток).
"""
import numpy as np

from fp.methods.morphology import erode, square

ENDING, BIFURCATION = "E", "B"


def crossing_number(skel):
    """Повертає масив CN для кожного пікселя скелета (0 поза скелетом)."""
    s = np.pad(skel.astype(np.int8), 1)
    nb = [s[:-2, 1:-1], s[:-2, 2:], s[1:-1, 2:], s[2:, 2:],
          s[2:, 1:-1], s[2:, :-2], s[1:-1, :-2], s[:-2, :-2]]
    cn = sum(np.abs(nb[i] - nb[(i + 1) % 8]) for i in range(8)) // 2
    return np.where(skel, cn, 0)


def extract_minutiae(skel, roi=None, border=12, min_dist=8.0):
    """Знаходить і фільтрує мінуції.

    Returns:
        список кортежів (row, col, type), type = "E" або "B".
    """
    cn = crossing_number(skel)
    raw = [(int(r), int(c), ENDING) for r, c in zip(*np.nonzero(cn == 1))]
    raw += [(int(r), int(c), BIFURCATION) for r, c in zip(*np.nonzero(cn == 3))]

    if roi is not None and border > 0:
        # ерозія з фоном за межами зображення відкидає і точки біля краю кадру
        inner = erode(roi, square(2 * border + 1))
        raw = [m for m in raw if inner[m[0], m[1]]]

    if not raw:
        return []
    pts = np.array([(m[0], m[1]) for m in raw], dtype=np.float64)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    drop = np.zeros(len(raw), dtype=bool)
    close_i, close_j = np.nonzero(d < min_dist)
    for i, j in zip(close_i, close_j):
        if i < j and ENDING in (raw[i][2], raw[j][2]):
            drop[i] = drop[j] = True
    return [m for m, x in zip(raw, drop) if not x]


def draw_minutiae(gray, skel, minutiae):
    """Кольорова візуалізація: сіре зображення, скелет синім, E червоним, B зеленим."""
    base = np.clip(gray, 0, 255).astype(np.float64)
    rgb = np.stack([base * 0.6 + 100] * 3, axis=-1)
    rgb[skel] = (40, 40, 220)
    h, w = skel.shape
    for r, c, t in minutiae:
        color = (230, 30, 30) if t == ENDING else (20, 180, 40)
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if max(abs(dr), abs(dc)) == 3 and 0 <= r + dr < h and 0 <= c + dc < w:
                    rgb[r + dr, c + dc] = color
    return rgb
