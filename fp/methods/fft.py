"""Власна реалізація швидкого перетворення Фур'є (Cooley-Tukey, radix-2).

Використовується стадією частотного підсилення пайплайна П2.
Перетворення обчислюється одночасно для багатьох блоків (уздовж останніх осей),
тому всі блоки зображення обробляються одним викликом.
Довжина осі має бути степенем двійки.
"""
import numpy as np


def _bit_reverse(n):
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=int)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def fft1(x, inverse=False):
    """Ітеративне FFT уздовж останньої осі масиву.

    Спочатку елементи переставляються в бітово-інверсному порядку, потім
    log2(n) разів виконуються "метелики": пари (a, b) замінюються на
    (a + w*b, a - w*b), де w - поворотний множник exp(-2*pi*i*k/len).
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Довжина {n} не є степенем двійки")
    y = x[..., _bit_reverse(n)].copy()
    sign = 1.0 if inverse else -1.0
    size = 2
    while size <= n:
        half = size // 2
        w = np.exp(sign * 2j * np.pi * np.arange(half) / size)
        y = y.reshape(*x.shape[:-1], n // size, size)
        a = y[..., :half].copy()
        b = y[..., half:] * w
        y[..., :half] = a + b
        y[..., half:] = a - b
        y = y.reshape(*x.shape[:-1], n)
        size *= 2
    if inverse:
        y /= n
    return y


def fft2(x):
    """Двовимірне FFT уздовж двох останніх осей."""
    y = fft1(x)
    return np.swapaxes(fft1(np.swapaxes(y, -1, -2)), -1, -2)


def ifft2(x):
    """Обернене двовимірне FFT уздовж двох останніх осей."""
    y = fft1(x, inverse=True)
    return np.swapaxes(fft1(np.swapaxes(y, -1, -2), inverse=True), -1, -2)


def freq_grid(n):
    """Частоти (цикли на піксель) для осі довжини n у порядку виходу FFT."""
    k = np.arange(n)
    k[k >= (n + 1) // 2] -= n
    return k / n
