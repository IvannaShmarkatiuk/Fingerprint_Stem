import time
from dataclasses import dataclass, field

import numpy as np

from fp.core import rescale
from fp.methods.clahe import clahe
from fp.methods.orientation import orientation_field, block_frequency
from fp.methods.gabor import gabor_filter
from fp.methods.otsu import otsu_binarize
from fp.methods.thinning import zhang_suen
from fp.methods.minutiae import extract_minutiae
from fp.methods.morphology import opening, square, cross, closing
from fp.methods.roi import variance_roi, normalize
from fp.methods.fft_enhance import fft_enhance
from fp.methods.sauvola import sauvola_binarize


@dataclass
class PipelineResult:
    name: str
    stages: dict = field(default_factory=dict)   # назва стадії -> зображення/маска
    enhanced: np.ndarray = None                   # сіре підсилене зображення (для метрик)
    binary: np.ndarray = None                     # маска ліній
    skeleton: np.ndarray = None
    roi: np.ndarray = None
    theta: np.ndarray = None
    minutiae: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)   # назва стадії -> секунди


def _timed(res, name, func, *args, **kwargs):
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    res.timings[name] = time.perf_counter() - t0
    return out


def _gabor_stage(img, cfg):
    theta, coherence, energy = orientation_field(img, cfg["orient_window"], cfg["orient_smooth"])
    freq = cfg["freq"]
    if cfg.get("estimate_freq", True):
        est = block_frequency(img, theta)
        if est is not None:
            freq = est
    resp = gabor_filter(img, theta, freq, cfg["sigma_x"], cfg["sigma_y"], cfg["n_angles"])
    # ROI П1: ділянки з вираженою орієнтацією та достатньою енергією градієнта
    lo, hi = np.percentile(energy, [5, 95])
    roi = (coherence > cfg["coherence_min"]) & (energy > lo + cfg["energy_rel"] * (hi - lo))
    roi = opening(closing(roi, square(15)), square(15))
    enhanced = np.full(img.shape, 255.0)
    if roi.any():
        lo, hi = np.percentile(resp[roi], [1, 99])
        enhanced[roi] = np.clip((resp[roi] - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255)
    return enhanced, roi, theta, freq


def run_p1(gray, cfg):
    res = PipelineResult("P1")
    c = cfg["p1"]
    eq = _timed(res, "1_clahe", clahe, gray, c["clahe"]["tiles"], c["clahe"]["clip_limit"])
    res.stages["1_clahe"] = eq

    enhanced, roi, theta, freq = _timed(res, "2_gabor", _gabor_stage, eq, c["gabor"])
    res.stages["2_gabor"] = enhanced
    res.enhanced, res.roi, res.theta = enhanced, roi, theta
    res.timings["freq"] = freq

    binary, _ = _timed(res, "3_otsu", otsu_binarize, enhanced, roi)
    res.stages["3_otsu"] = binary
    res.binary = binary

    skel = _timed(res, "4_skeleton", zhang_suen, binary)
    res.stages["4_skeleton"] = skel
    res.skeleton = skel

    m = c["minutiae"]
    res.minutiae = _timed(res, "5_minutiae", extract_minutiae, skel, roi, m["border"], m["min_dist"])
    return res


def run_p2(gray, cfg):
    res = PipelineResult("P2")
    c = cfg["p2"]
    r = c["roi"]
    roi = _timed(res, "1_roi", variance_roi, gray, r["block"], r["rel_threshold"])
    norm = normalize(gray, roi, r["mean"], r["var"])
    res.stages["1_roi"] = norm
    res.roi = roi

    f = c["fft"]
    enhanced = _timed(res, "2_fft", fft_enhance, norm, roi, f["block"], f["step"],
                      f["k"], f["fmin"], f["fmax"])
    res.stages["2_fft"] = enhanced
    res.enhanced = enhanced

    s = c["sauvola"]
    binary = _timed(res, "3_sauvola", sauvola_binarize, enhanced, roi, s["window"], s["k"], s["r"])
    res.stages["3_sauvola"] = binary

    def morph(mask):
        se = cross(3) if c["morph"]["element"] == "cross" else square(c["morph"]["size"])
        cleaned = opening(mask, se)
        return cleaned, zhang_suen(cleaned)

    cleaned, skel = _timed(res, "4_morph", morph, binary)
    res.stages["4_open"] = cleaned
    res.stages["4_skeleton"] = skel
    res.binary, res.skeleton = cleaned, skel

    m = c["minutiae"]
    res.minutiae = _timed(res, "5_minutiae", extract_minutiae, skel, roi, m["border"], m["min_dist"])
    return res


PIPELINES = {"p1": run_p1, "p2": run_p2}


def run_pipeline(name, gray, cfg):
    if name not in PIPELINES:
        raise ValueError(f"Невідомий пайплайн: {name}. Доступні: {', '.join(PIPELINES)}")
    return PIPELINES[name](np.asarray(gray, dtype=np.float64), cfg)


def enhanced_view(res):
    return rescale(res.enhanced)
