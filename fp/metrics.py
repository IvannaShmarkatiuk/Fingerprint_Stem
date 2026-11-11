import numpy as np

from fp.core import conv2d, gaussian_kernel


def _as_float(img):
    if img.dtype == bool:
        return np.where(img, 0.0, 255.0)
    return np.asarray(img, dtype=np.float64)


def mse(a, b):
    a, b = _as_float(a), _as_float(b)
    if a.shape != b.shape:
        raise ValueError("Розміри зображень не збігаються")
    return float(np.mean((a - b) ** 2))


def psnr(a, b, max_val=255.0):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / m))


def ssim(a, b, max_val=255.0, sigma=1.5):
    a, b = _as_float(a), _as_float(b)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    k = gaussian_kernel(sigma)
    mu_a, mu_b = conv2d(a, k), conv2d(b, k)
    var_a = conv2d(a * a, k) - mu_a ** 2
    var_b = conv2d(b * b, k) - mu_b ** 2
    cov = conv2d(a * b, k) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def pixel_agreement(mask_a, mask_b, roi=None):
    eq = mask_a == mask_b
    if roi is not None:
        if not roi.any():
            return 0.0
        return float(eq[roi].mean())
    return float(eq.mean())


def minutiae_retention(reference, found, tol=8.0):
    if not reference:
        return 1.0 if not found else 0.0
    used = np.zeros(len(found), dtype=bool)
    hits = 0
    for r, c, t in reference:
        best, best_d = -1, tol
        for j, (r2, c2, t2) in enumerate(found):
            if used[j] or t2 != t:
                continue
            d = np.hypot(r - r2, c - c2)
            if d <= best_d:
                best, best_d = j, d
        if best >= 0:
            used[best] = True
            hits += 1
    return hits / len(reference)


def angle_error_deg(theta_a, theta_b, mask=None):
    d = np.abs(np.angle(np.exp(2j * (theta_a - theta_b)))) / 2
    if mask is not None:
        d = d[mask]
    return float(np.degrees(d.mean()))
