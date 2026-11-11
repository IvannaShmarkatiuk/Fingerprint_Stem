"""Порівняння швидкості наївних і оптимізованих реалізацій (ознака "нетривіальна оптимізація").

    python benchmark.py

Порівнюються:
- фільтр Габора: цикли Python / 16 повних згорток / вибірка вікон по орієнтації;
- скелетування Zhang-Suen: цикли по пікселях / векторизовані зсуви масивів.
Перед вимірюванням кожна функція запускається один раз (прогрів).
"""
import csv
import time

from fp.config import project_path
from fp.methods.gabor import gabor_filter, gabor_filter_full, gabor_filter_naive
from fp.methods.orientation import orientation_field
from fp.methods.otsu import otsu_binarize
from fp.methods.thinning import zhang_suen, zhang_suen_naive
from fp.synthetic import synth_fingerprint


def best_time(func, *args, repeat=3):
    """Мінімальний час із repeat запусків після одного прогріву, секунди."""
    func(*args)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    """Запускає вимірювання і зберігає results/benchmark.csv."""
    img = synth_fingerprint(256, 224, seed=7)["image"]
    theta, _, _ = orientation_field(img)
    binary, _ = otsu_binarize(img)
    rows = []

    small = (slice(0, 48), slice(0, 48))
    t_naive = best_time(gabor_filter_naive, img[small], theta[small], 1 / 9, 4, 3, 16, repeat=1)
    t_full_s = best_time(gabor_filter_full, img[small], theta[small], 1 / 9, 4, 3)
    t_mask_s = best_time(gabor_filter, img[small], theta[small], 1 / 9, 4, 3)
    rows += [
        {"task": "Габор 48x48", "method": "наївні цикли", "seconds": t_naive},
        {"task": "Габор 48x48", "method": "16 повних згорток", "seconds": t_full_s},
        {"task": "Габор 48x48", "method": "вибірка вікон (наш)", "seconds": t_mask_s},
    ]
    t_full = best_time(gabor_filter_full, img, theta, 1 / 9, 4, 3)
    t_mask = best_time(gabor_filter, img, theta, 1 / 9, 4, 3)
    rows += [
        {"task": "Габор 256x224", "method": "16 повних згорток", "seconds": t_full},
        {"task": "Габор 256x224", "method": "вибірка вікон (наш)", "seconds": t_mask},
    ]

    t_zs_naive = best_time(zhang_suen_naive, binary, repeat=1)
    t_zs = best_time(zhang_suen, binary)
    rows += [
        {"task": "Zhang-Suen 256x224", "method": "наївні цикли", "seconds": t_zs_naive},
        {"task": "Zhang-Suen 256x224", "method": "векторизовано (наш)", "seconds": t_zs},
    ]

    for r in rows:
        r["seconds"] = round(r["seconds"], 5)
    out = project_path("results")
    out.mkdir(exist_ok=True)
    with open(out / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "method", "seconds"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['task']:<28} {r['method']:<22} {r['seconds']:>10.4f} с")
    print(f"\nПрискорення Габора (наївний -> наш, 48x48): {t_naive / t_mask_s:.0f}x")
    print(f"Прискорення Габора (16 згорток -> наш, 256x224): {t_full / t_mask:.1f}x")
    print(f"Прискорення Zhang-Suen: {t_zs_naive / t_zs:.0f}x")


if __name__ == "__main__":
    main()
