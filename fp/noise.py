"""Контрольовані спотворення зображень для експериментів зі стійкості."""
import numpy as np


def gaussian_noise(img, sigma, rng):
    """Адитивний гаусів шум із середнім 0 та стандартним відхиленням sigma."""
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0, 255)


def salt_pepper(img, amount, rng):
    """Імпульсний шум: частка amount пікселів стає 0 або 255 з рівною ймовірністю."""
    out = img.copy()
    u = rng.random(img.shape)
    out[u < amount / 2] = 0
    out[(u >= amount / 2) & (u < amount)] = 255
    return out


def uneven_light(img, strength):
    """Нерівне освітлення: лінійний градієнт яскравості зліва направо."""
    w = img.shape[1]
    ramp = np.linspace(-strength, strength, w)[None, :]
    return np.clip(img + ramp, 0, 255)


def degrade(img, spec, seed):
    """Застосовує спотворення за описом spec = {"type": ..., "level": ...}.

    Типи: gaussian (level = sigma), saltpepper (level = частка), light (level = амплітуда).
    """
    rng = np.random.default_rng(seed)
    kind, level = spec["type"], float(spec["level"])
    if kind == "gaussian":
        return gaussian_noise(img, level, rng)
    if kind == "saltpepper":
        return salt_pepper(img, level, rng)
    if kind == "light":
        return uneven_light(img, level)
    raise ValueError(f"Невідомий тип спотворення: {kind}")
