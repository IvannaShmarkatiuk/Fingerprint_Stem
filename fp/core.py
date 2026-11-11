"""Базові операції над масивами: згортка, нормалізація, перетворення у відтінки сірого.

Усі функції реалізовано вручну на numpy, без бібліотек комп'ютерного зору.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_gray(img):
    """Перетворює RGB або RGBA зображення у відтінки сірого (рекомендація ITU-R BT.601).

    Сіре зображення повертається без змін (лише як float64).
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0]
    raise ValueError(f"Непідтримувана форма зображення: {img.shape}")


def to_uint8(img):
    """Обрізає значення до діапазону [0, 255] і переводить у uint8."""
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def rescale(img, lo=0.0, hi=255.0):
    """Лінійно розтягує значення масиву в діапазон [lo, hi].

    Для сталого зображення (max == min) повертає масив, заповнений lo.
    """
    img = np.asarray(img, dtype=np.float64)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-12:
        return np.full_like(img, lo)
    return lo + (img - mn) * (hi - lo) / (mx - mn)


def conv2d(img, kernel, mode="reflect"):
    """Двовимірна згортка (кореляція з перевернутим ядром) з доповненням країв.

    Результат має той самий розмір, що й вхід. Реалізація векторизована:
    будуються всі вікна розміру ядра, після чого виконується одне скалярне
    множення через einsum.
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)[::-1, ::-1]
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, kh - 1 - ph), (pw, kw - 1 - pw)), mode=mode)
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def box_sum(img, size):
    """Сума значень у квадратному вікні size x size навколо кожного пікселя.

    Використовує інтегральне зображення, тому час не залежить від розміру вікна.
    """
    img = np.asarray(img, dtype=np.float64)
    r = size // 2
    padded = np.pad(img, ((r + 1, r), (r + 1, r)), mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    h, w = img.shape
    s = size
    return (integral[s:s + h, s:s + w] - integral[0:h, s:s + w]
            - integral[s:s + h, 0:w] + integral[0:h, 0:w])


def gaussian_kernel(sigma):
    """Нормоване гаусове ядро розміром 2*ceil(3*sigma)+1."""
    r = max(1, int(np.ceil(3 * sigma)))
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()
