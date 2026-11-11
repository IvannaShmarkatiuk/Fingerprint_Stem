import numpy as np


def gaussian_noise(img, sigma, rng):
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0, 255)


def salt_pepper(img, amount, rng):
    out = img.copy()
    u = rng.random(img.shape)
    out[u < amount / 2] = 0
    out[(u >= amount / 2) & (u < amount)] = 255
    return out


def uneven_light(img, strength):
    w = img.shape[1]
    ramp = np.linspace(-strength, strength, w)[None, :]
    return np.clip(img + ramp, 0, 255)


def degrade(img, spec, seed):
    rng = np.random.default_rng(seed)
    kind, level = spec["type"], float(spec["level"])
    if kind == "gaussian":
        return gaussian_noise(img, level, rng)
    if kind == "saltpepper":
        return salt_pepper(img, level, rng)
    if kind == "light":
        return uneven_light(img, level)
    raise ValueError(f"Невідомий тип спотворення: {kind}")
