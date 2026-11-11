"""Фільтр Габора, налаштований на локальне поле орієнтацій.

Категорія: фільтрація (стадія 2 пайплайна П1). Метод: Hong, Wan, Jain (1998).

Парне ядро Габора:
    g(u, v) = exp(-(u^2/sx^2 + v^2/sy^2) / 2) * cos(2*pi*f*u),
де u - координата вздовж нормалі до ліній, v - уздовж лінії.
Фільтр підсилює коливання з частотою ліній f саме впоперек ліній і згладжує
уздовж них, тому розриви ліній заповнюються, а шум пригнічується.

Оптимізація: орієнтації квантуються на n_angles значень. Кожен піксель
згортається лише з одним ядром своєї орієнтації (векторизована вибірка вікон),
тому загальна вартість дорівнює одній звичайній згортці, а не n_angles згорткам.
Для порівняння є дві повільніші реалізації (див. benchmark.py).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def gabor_kernel(theta, freq, sx=4.0, sy=4.0):
    """Будує ядро Габора для напряму лінії theta і частоти freq.

    Середнє ядра віднімається, щоб рівномірний фон давав нульовий відгук.
    """
    r = int(np.ceil(3 * max(sx, sy)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    a = theta - np.pi / 2  # напрям нормалі
    u = x * np.cos(a) + y * np.sin(a)
    v = -x * np.sin(a) + y * np.cos(a)
    k = np.exp(-0.5 * (u ** 2 / sx ** 2 + v ** 2 / sy ** 2)) * np.cos(2 * np.pi * freq * u)
    return k - k.mean()


def quantize(theta, n_angles):
    """Індекс найближчого квантованого кута для кожного пікселя."""
    return np.rint(np.mod(theta, np.pi) / np.pi * n_angles).astype(int) % n_angles


def gabor_filter(img, theta, freq, sx=4.0, sy=4.0, n_angles=16):
    """Оптимізована фільтрація: кожен піксель - зі своїм ядром (вибірка вікон)."""
    img = np.asarray(img, dtype=np.float64)
    img = img - img.mean()
    kernels = [gabor_kernel(i * np.pi / n_angles, freq, sx, sy) for i in range(n_angles)]
    r = kernels[0].shape[0] // 2
    padded = np.pad(img, r, mode="reflect")
    windows = sliding_window_view(padded, kernels[0].shape)
    idx = quantize(theta, n_angles)
    out = np.zeros_like(img)
    for i, k in enumerate(kernels):
        sel = idx == i
        if sel.any():
            # ядро симетричне відносно центру, тому кореляція = згортка
            out[sel] = np.einsum("nkl,kl->n", windows[sel], k)
    return out


def gabor_filter_full(img, theta, freq, sx=4.0, sy=4.0, n_angles=16):
    """Проста векторизована версія: n_angles повних згорток, потім вибір за орієнтацією."""
    img = np.asarray(img, dtype=np.float64)
    img = img - img.mean()
    idx = quantize(theta, n_angles)
    out = np.zeros_like(img)
    for i in range(n_angles):
        k = gabor_kernel(i * np.pi / n_angles, freq, sx, sy)
        r = k.shape[0] // 2
        windows = sliding_window_view(np.pad(img, r, mode="reflect"), k.shape)
        full = np.einsum("ijkl,kl->ij", windows, k)
        out[idx == i] = full[idx == i]
    return out


def gabor_filter_naive(img, theta, freq, sx=4.0, sy=4.0, n_angles=16):
    """Наївна версія циклами Python (для бенчмарку на малих зображеннях)."""
    img = np.asarray(img, dtype=np.float64)
    img = img - img.mean()
    kernels = [gabor_kernel(i * np.pi / n_angles, freq, sx, sy) for i in range(n_angles)]
    r = kernels[0].shape[0] // 2
    padded = np.pad(img, r, mode="reflect")
    idx = quantize(theta, n_angles)
    h, w = img.shape
    out = np.zeros_like(img)
    size = 2 * r + 1
    for y in range(h):
        for x in range(w):
            k = kernels[idx[y, x]]
            acc = 0.0
            for i in range(size):
                for j in range(size):
                    acc += padded[y + i, x + j] * k[i, j]
            out[y, x] = acc
    return out
