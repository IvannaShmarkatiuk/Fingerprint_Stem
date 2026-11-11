"""Морфологічне скелетування методом Чжана-Суня (Zhang & Suen, 1984).

Категорія: морфологічна обробка (стадія 4 пайплайна П1, частина стадії 4 П2).

Сусіди пікселя P1 нумеруються за годинниковою стрілкою, починаючи зверху:
    P9 P2 P3
    P8 P1 P4
    P7 P6 P5
B = кількість ненульових сусідів, A = кількість переходів 0->1 у послідовності
P2..P9,P2. Піксель видаляється, якщо 2 <= B <= 6, A == 1 та:
    підітерація 1: P2*P4*P6 == 0 і P4*P6*P8 == 0
    підітерація 2: P2*P4*P8 == 0 і P2*P6*P8 == 0
Ітерації повторюються, доки є що видаляти. Результат - лінії товщиною 1 піксель.
"""
import numpy as np


def _neighbours(img):
    """Повертає масиви сусідів P2..P9 для кожного пікселя (нулі за краєм)."""
    p = np.pad(img, 1, constant_values=0)
    return (p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2])


def _step(img, first):
    p2, p3, p4, p5, p6, p7, p8, p9 = _neighbours(img)
    seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
    b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
    if first:
        c1 = (p2 * p4 * p6) == 0
        c2 = (p4 * p6 * p8) == 0
    else:
        c1 = (p2 * p4 * p8) == 0
        c2 = (p2 * p6 * p8) == 0
    remove = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
    return remove


def zhang_suen(mask, max_iter=100):
    """Векторизоване скелетування бінарної маски (True = лінія)."""
    img = mask.astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for first in (True, False):
            rem = _step(img, first)
            if rem.any():
                img[rem] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def zhang_suen_naive(mask, max_iter=100):
    """Наївна версія з циклами по пікселях (для бенчмарку)."""
    img = np.pad(mask.astype(np.uint8), 1)
    h, w = img.shape
    for _ in range(max_iter):
        changed = False
        for first in (True, False):
            to_remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img[y, x] != 1:
                        continue
                    n = [img[y - 1, x], img[y - 1, x + 1], img[y, x + 1], img[y + 1, x + 1],
                         img[y + 1, x], img[y + 1, x - 1], img[y, x - 1], img[y - 1, x - 1]]
                    b = sum(n)
                    a = sum(1 for i in range(8) if n[i] == 0 and n[(i + 1) % 8] == 1)
                    p2, p4, p6, p8 = n[0], n[2], n[4], n[6]
                    if first:
                        ok = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                    else:
                        ok = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                    if 2 <= b <= 6 and a == 1 and ok:
                        to_remove.append((y, x))
            for y, x in to_remove:
                img[y, x] = 0
            changed = changed or bool(to_remove)
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)
