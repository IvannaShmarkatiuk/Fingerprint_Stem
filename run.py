"""Головна програма: обробка вибірки обраним пайплайном.

Приклади:
    python run.py --pipeline all
    python run.py --pipeline p1 --input data/fvc2002 --limit 20
    python run.py --pipeline p2 --set p2.sauvola.k=0.3
"""
import argparse
import csv
import json
import logging
import sys

import numpy as np

from fp.config import load_config, project_path
from fp.io_utils import ImageLoadError, list_images, load_gray, save_image
from fp.methods.minutiae import draw_minutiae
from fp.metrics import mse, psnr, ssim
from fp.pipelines import PIPELINES, run_pipeline

log = logging.getLogger("run")


def parse_args(argv=None):
    """Аргументи командного рядка."""
    ap = argparse.ArgumentParser(description="Обробка відбитків пальців пайплайнами П1/П2")
    ap.add_argument("--pipeline", choices=[*PIPELINES, "all"], default="all")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", help="папка із зображеннями (за замовчуванням з конфігу)")
    ap.add_argument("--output", help="папка результатів (за замовчуванням з конфігу)")
    ap.add_argument("--limit", type=int, default=0, help="обробити лише перші N зображень")
    ap.add_argument("--no-stages", action="store_true", help="не зберігати проміжні зображення")
    ap.add_argument("--set", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕННЯ",
                    help="перевизначити параметр, напр. p1.gabor.sigma_y=5")
    return ap.parse_args(argv)


def process_image(path, names, cfg, out_dir, save_stages):
    """Обробляє одне зображення всіма обраними пайплайнами, повертає рядки метрик."""
    gray = load_gray(path)
    rows = []
    for name in names:
        res = run_pipeline(name, gray, cfg)
        if save_stages:
            folder = out_dir / name / path.stem
            save_image(gray, folder / "0_input.png")
            for stage, img in res.stages.items():
                save_image(img, folder / f"{stage}.png")
            save_image(draw_minutiae(gray, res.skeleton, res.minutiae), folder / "5_minutiae.png")
        endings = sum(1 for m in res.minutiae if m[2] == "E")
        rows.append({
            "image": path.name,
            "pipeline": name,
            "minutiae": len(res.minutiae),
            "endings": endings,
            "bifurcations": len(res.minutiae) - endings,
            "roi_fraction": round(float(res.roi.mean()), 4),
            # порівняння оригіналу з підсиленим зображенням
            "mse_vs_input": round(mse(gray, res.enhanced), 2),
            "psnr_vs_input": round(psnr(gray, res.enhanced), 3),
            "ssim_vs_input": round(ssim(gray, res.enhanced), 4),
            "time_s": round(sum(v for k, v in res.timings.items() if k != "freq"), 4),
        })
    return rows


def main(argv=None):
    """Точка входу."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(args.config, args.set)
    except (OSError, KeyError, ValueError) as exc:
        log.error("Помилка конфігурації: %s", exc)
        return 2

    in_dir = project_path(args.input or cfg["paths"]["input"])
    out_dir = project_path(args.output or cfg["paths"]["output"])
    names = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]
    files = list_images(in_dir)
    if args.limit:
        files = files[:args.limit]
    if not files:
        log.error("У папці %s не знайдено зображень", in_dir)
        return 1

    rows, errors = [], []
    for i, path in enumerate(files, 1):
        try:
            rows += process_image(path, names, cfg, out_dir, not args.no_stages)
            log.info("[%d/%d] %s", i, len(files), path.name)
        except ImageLoadError as exc:
            log.warning("Пропущено: %s", exc)
            errors.append({"image": path.name, "error": str(exc)})

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    summary = {}
    for name in names:
        sub = [r for r in rows if r["pipeline"] == name]
        if sub:
            summary[name] = {k: round(float(np.mean([r[k] for r in sub])), 4)
                             for k in ("minutiae", "ssim_vs_input", "psnr_vs_input", "time_s")}
            summary[name]["images"] = len(sub)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": args.config, "overrides": args.set, "pipelines": summary,
                   "skipped": len(errors)}, f, ensure_ascii=False, indent=2)
    log.info("Готово: оброблено %d, пропущено %d. Результати в %s",
             len(files) - len(errors), len(errors), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
