"""Експерименти для порівняння пайплайнів.

    python experiment.py robustness   # стійкість до шуму (C1, B2), будь-яка вибірка
    python experiment.py groundtruth  # точність на синтетичних еталонах (C1)
    python experiment.py sweep        # підбір параметрів (обґрунтування в звіті)
    python experiment.py figures      # ілюстрації стадій для звіту

Результати: CSV, JSON і PNG у папці results/experiments.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fp.config import load_config, project_path, set_by_path
from fp.io_utils import ImageLoadError, list_images, load_gray
from fp.methods.minutiae import extract_minutiae, draw_minutiae
from fp.methods.thinning import zhang_suen
from fp.metrics import (angle_error_deg, minutiae_retention, mse, pixel_agreement,
                        psnr, ssim)
from fp.noise import degrade
from fp.pipelines import PIPELINES, run_pipeline

plt.switch_backend("Agg")  # малювання у файли без вікон
log = logging.getLogger("experiment")
NAMES = list(PIPELINES)
LABELS = {"p1": "П1 Габор", "p2": "П2 Фур'є"}


def scenario_name(spec):
    """Назва сценарію спотворення, напр. gaussian_20."""
    level = spec["level"]
    return f"{spec['type']}_{int(level) if float(level).is_integer() else level}"


def write_csv(rows, path):
    """Зберігає список словників у CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def robustness(cfg, files, out):
    """Запускає кожен пайплайн на чистому і спотворених зображеннях та порівнює результати."""
    exp = cfg["experiment"]
    thr = exp["thresholds"]
    rows = []
    for idx, path in enumerate(files):
        try:
            gray = load_gray(path)
        except ImageLoadError as exc:
            log.warning("Пропущено: %s", exc)
            continue
        for name in NAMES:
            clean = run_pipeline(name, gray, cfg)
            for spec in exp["degradations"]:
                noisy = run_pipeline(name, degrade(gray, spec, cfg["seed"] + idx), cfg)
                s = ssim(noisy.enhanced, clean.enhanced)
                p = psnr(noisy.enhanced, clean.enhanced)
                r = minutiae_retention(clean.minutiae, noisy.minutiae)
                rows.append({
                    "image": path.name, "pipeline": name, "scenario": scenario_name(spec),
                    "type": spec["type"], "level": spec["level"],
                    "ssim": round(s, 4), "psnr": round(min(p, 99.0), 3),
                    "mse": round(mse(noisy.enhanced, clean.enhanced), 2),
                    "retention": round(r, 4),
                    "pass": int(s >= thr["ssim"] and p >= thr["psnr"] and r >= thr["retention"]),
                })
        log.info("[%d/%d] %s", idx + 1, len(files), path.name)
    write_csv(rows, out / "robustness.csv")

    summary = []
    for name in NAMES:
        for spec in exp["degradations"]:
            sc = scenario_name(spec)
            sub = [r for r in rows if r["pipeline"] == name and r["scenario"] == sc]
            if not sub:
                continue
            summary.append({
                "pipeline": name, "scenario": sc,
                **{f"{k}_mean": round(float(np.mean([r[k] for r in sub])), 4)
                   for k in ("ssim", "psnr", "retention")},
                **{f"{k}_std": round(float(np.std([r[k] for r in sub])), 4)
                   for k in ("ssim", "psnr", "retention")},
                "pass_rate": round(float(np.mean([r["pass"] for r in sub])), 4),
                "n": len(sub),
            })
    write_csv(summary, out / "robustness_summary.csv")

    target = set(exp["threshold_scenarios"])
    c1 = {}
    for name in NAMES:
        sub = [r for r in rows if r["pipeline"] == name and r["scenario"] in target]
        c1[name] = {"cases": len(sub), "passed": int(sum(r["pass"] for r in sub)),
                    "share": round(float(np.mean([r["pass"] for r in sub])), 4) if sub else None}
    with open(out / "c1_robustness.json", "w", encoding="utf-8") as f:
        json.dump({"thresholds": thr, "scenarios": sorted(target), "result": c1}, f,
                  ensure_ascii=False, indent=2)
    plot_robustness(rows, exp["degradations"], out / "robustness.png")
    return summary, c1


def plot_robustness(rows, degradations, path):
    """Графіки SSIM і збереження мінуцій від рівня спотворення для кожного типу."""
    types = sorted({d["type"] for d in degradations})
    fig, axes = plt.subplots(2, len(types), figsize=(4.2 * len(types), 7), squeeze=False)
    titles = {"gaussian": "Гаусів шум, σ", "saltpepper": "Імпульсний шум, частка",
              "light": "Нерівне освітлення, амплітуда"}
    for col, t in enumerate(types):
        levels = sorted({d["level"] for d in degradations if d["type"] == t})
        for row, (metric, ylabel) in enumerate((("ssim", "SSIM"), ("retention", "Збережено мінуцій"))):
            ax = axes[row][col]
            for name in NAMES:
                means, stds = [], []
                for lv in levels:
                    vals = [r[metric] for r in rows
                            if r["pipeline"] == name and r["type"] == t and r["level"] == lv]
                    means.append(np.mean(vals) if vals else np.nan)
                    stds.append(np.std(vals) if vals else 0)
                ax.errorbar(levels, means, yerr=stds, marker="o", capsize=3, label=LABELS[name])
            ax.set_xlabel(titles.get(t, t))
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def groundtruth(cfg, gt_dir, out):
    """Порівнює результати з відомими еталонами синтетичних відбитків."""
    thr = cfg["experiment"]["thresholds"]
    rows = []
    for gt_path in sorted(Path(gt_dir).glob("*_gt.npz")):
        img_path = gt_path.with_name(gt_path.name.replace("_gt.npz", ".png"))
        gt = np.load(gt_path)
        gray = load_gray(img_path)
        ref_min = extract_minutiae(zhang_suen(gt["ridges"]), gt["roi"])
        for name in NAMES:
            res = run_pipeline(name, gray, cfg)
            area = gt["roi"] & res.roi
            agree = pixel_agreement(res.binary, gt["ridges"], area)
            row = {"image": img_path.name, "pipeline": name,
                   "binary_agreement": round(agree, 4),
                   "minutiae_found": len(res.minutiae), "minutiae_gt": len(ref_min),
                   "minutiae_recall": round(minutiae_retention(ref_min, res.minutiae), 4),
                   "orientation_err_deg": ""}
            ok = agree >= thr["binary_agreement"]
            if res.theta is not None:
                err = angle_error_deg(res.theta, gt["theta"], area)
                row["orientation_err_deg"] = round(err, 3)
                ok = ok and err <= thr["orientation_deg"]
            row["pass"] = int(ok)
            rows.append(row)
    write_csv(rows, out / "groundtruth.csv")
    res = {n: round(float(np.mean([r["pass"] for r in rows if r["pipeline"] == n])), 4)
           for n in NAMES if any(r["pipeline"] == n for r in rows)}
    with open(out / "c1_groundtruth.json", "w", encoding="utf-8") as f:
        json.dump({"thresholds": {k: thr[k] for k in ("orientation_deg", "binary_agreement")},
                   "pass_share": res, "cases": len(rows)}, f, ensure_ascii=False, indent=2)
    return rows


SWEEPS = {
    "p1.gabor.sigma_y": [2.0, 3.0, 4.0, 5.0, 6.0],
    "p1.clahe.clip_limit": [1.0, 2.0, 3.0, 4.0],
    "p2.fft.k": [0.2, 0.35, 0.45, 0.6, 0.8],
    "p2.sauvola.k": [0.05, 0.1, 0.2, 0.3, 0.4],
}


def sweep(cfg, gt_dir, out):
    """Перебирає значення одного параметра за раз; оцінка - збіг з еталоном при шумі σ=20.

    Для підбору використовуються окремі синтетичні відбитки (не ті, що в groundtruth),
    щоб параметри не "підганялися" під тестові дані.
    """
    rows = []
    gts = sorted(Path(gt_dir).glob("*_gt.npz"))
    for key, values in SWEEPS.items():
        name = key.split(".", maxsplit=1)[0]
        for v in values:
            local = json.loads(json.dumps(cfg))
            set_by_path(local, key, v)
            scores = []
            for i, gt_path in enumerate(gts):
                gt = np.load(gt_path)
                gray = load_gray(gt_path.with_name(gt_path.name.replace("_gt.npz", ".png")))
                noisy = degrade(gray, {"type": "gaussian", "level": 20}, cfg["seed"] + i)
                res = run_pipeline(name, noisy, local)
                scores.append(pixel_agreement(res.binary, gt["ridges"], gt["roi"] & res.roi))
            rows.append({"parameter": key, "value": v,
                         "agreement_mean": round(float(np.mean(scores)), 4),
                         "agreement_std": round(float(np.std(scores)), 4)})
            log.info("%s=%s -> %.4f", key, v, np.mean(scores))
    write_csv(rows, out / "sweep.csv")

    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(4 * len(SWEEPS), 3.4))
    for ax, key in zip(axes, SWEEPS):
        sub = [r for r in rows if r["parameter"] == key]
        ax.errorbar([r["value"] for r in sub], [r["agreement_mean"] for r in sub],
                    yerr=[r["agreement_std"] for r in sub], marker="o", capsize=3)
        ax.axvline(_get(cfg, key), color="gray", ls="--", lw=1)
        ax.set_title(key, fontsize=10)
        ax.set_ylabel("Збіг з еталоном")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=130)
    plt.close(fig)
    return rows


def _get(cfg, dotted):
    node = cfg
    for p in dotted.split("."):
        node = node[p]
    return node


def figures(cfg, files, out):
    """Зберігає для кожного з перших файлів рисунок з усіма стадіями обох пайплайнів."""
    for path in files:
        gray = load_gray(path)
        fig, axes = plt.subplots(len(NAMES), 6, figsize=(16, 3 * len(NAMES)))
        for row, name in enumerate(NAMES):
            res = run_pipeline(name, gray, cfg)
            panels = [("Вхід", gray)] + [(k, v) for k, v in res.stages.items()
                                         if k != "4_open"][:4]
            panels.append((f"мінуції: {len(res.minutiae)}",
                           draw_minutiae(gray, res.skeleton, res.minutiae) / 255.0))
            for ax, (title, img) in zip(axes[row], panels):
                if img.ndim == 3:
                    ax.imshow(np.clip(img, 0, 1))
                elif img.dtype == bool:
                    ax.imshow(~img, cmap="gray")
                else:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{LABELS[name]}: {title}", fontsize=9)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"stages_{path.stem}.png", dpi=110)
        plt.close(fig)


def main(argv=None):
    """Точка входу."""
    ap = argparse.ArgumentParser(description="Експерименти порівняння пайплайнів")
    ap.add_argument("mode", choices=["robustness", "groundtruth", "sweep", "figures"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", help="папка зображень (robustness, figures)")
    ap.add_argument("--gt", default="data/synthetic", help="папка з еталонами *_gt.npz")
    ap.add_argument("--tune", default="data/synthetic_tune",
                    help="окремі еталони для підбору параметрів (sweep)")
    ap.add_argument("--out", default="results/experiments")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕННЯ")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config, args.set)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_images(project_path(args.input or cfg["paths"]["input"]))
    if args.limit:
        files = files[:args.limit]

    if args.mode == "robustness":
        if not files:
            log.error("Немає зображень")
            return 1
        _, c1 = robustness(cfg, files, out)
        print(json.dumps(c1, ensure_ascii=False, indent=2))
    elif args.mode == "groundtruth":
        groundtruth(cfg, project_path(args.gt), out)
        print((out / "c1_groundtruth.json").read_text(encoding="utf-8"))
    elif args.mode == "sweep":
        sweep(cfg, project_path(args.tune), out)
    else:
        figures(cfg, files[:args.limit or 3], out)
    log.info("Результати в %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
