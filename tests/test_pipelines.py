"""Тести пайплайнів, метрик, спотворень і крайових випадків (C1, C2, C3)."""
# pylint: disable=missing-function-docstring,redefined-outer-name
import json

import numpy as np
import pytest
import yaml
from PIL import Image

import run
from fp.config import load_config
from fp.io_utils import ImageLoadError, load_gray
from fp.metrics import minutiae_retention, mse, pixel_agreement, psnr, ssim
from fp.noise import degrade
from fp.pipelines import run_pipeline
from fp.synthetic import stripes, synth_fingerprint


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- метрики ----------

def test_mse_psnr_by_hand():
    a = np.zeros((2, 2))
    b = np.array([[0, 0], [0, 255.0]])
    assert mse(a, b) == pytest.approx(255 ** 2 / 4)
    assert psnr(a, b) == pytest.approx(10 * np.log10(4))
    assert psnr(a, a) == float("inf")


def test_ssim_identical_and_inverted():
    img = stripes(32, 32, 0.4, 1 / 6)
    assert ssim(img, img) == pytest.approx(1.0)
    assert ssim(img, 255 - img) < 0


def test_minutiae_retention_by_hand():
    ref = [(10, 10, "E"), (30, 30, "B")]
    found = [(12, 11, "E"), (30, 30, "E")]  # друга - не того типу
    assert minutiae_retention(ref, found, tol=5) == 0.5


def test_degrade_reproducible_with_seed():
    img = np.full((40, 40), 128.0)
    a = degrade(img, {"type": "gaussian", "level": 10}, 7)
    b = degrade(img, {"type": "gaussian", "level": 10}, 7)
    assert np.array_equal(a, b)
    sp = degrade(img, {"type": "saltpepper", "level": 0.1}, 1)
    assert 0.07 < np.mean((sp == 0) | (sp == 255)) < 0.13


#пайплайни

@pytest.mark.parametrize("name", ["p1", "p2"])
def test_pipeline_parallel_lines_have_no_minutiae(name, cfg):
    # паралельні нескінченні лінії не мають ні закінчень, ні розгалужень,
    # а скелет дає по одній лінії на період: 96 / 8 = 12 ліній
    img = stripes(96, 96, np.pi / 2, 1 / 8)
    res = run_pipeline(name, img, cfg)
    assert res.minutiae == []
    row = res.skeleton[48, 16:80]
    assert 7 <= row.sum() <= 9  # 64 px / 8 = 8 ліній у центральній смузі


@pytest.mark.parametrize("name", ["p1", "p2"])
def test_pipeline_matches_synthetic_ground_truth(name, cfg):
    d = synth_fingerprint(160, 144, seed=5)
    res = run_pipeline(name, d["image"], cfg)
    area = d["roi"] & res.roi
    assert pixel_agreement(res.binary, d["ridges"], area) > 0.93
    assert set(res.stages)  # усі проміжні результати збережено
    assert res.enhanced.shape == d["image"].shape


def test_unknown_pipeline_raises(cfg):
    with pytest.raises(ValueError):
        run_pipeline("p9", np.zeros((40, 40)), cfg)


@pytest.mark.parametrize("name", ["p1", "p2"])
def test_pipeline_survives_blank_image(name, cfg):
    res = run_pipeline(name, np.full((64, 64), 200.0), cfg)
    assert res.minutiae == []


# ---------- крайові випадки введення (C2) ----------

def test_load_edge_cases(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image at all")
    tiny = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tiny)
    for p in (empty, broken, tiny, tmp_path / "missing.png"):
        with pytest.raises(ImageLoadError):
            load_gray(p)

    color = tmp_path / "color.png"
    Image.fromarray(np.full((40, 40, 3), (100, 0, 0), np.uint8)).save(color)
    assert load_gray(color)[0, 0] == pytest.approx(29.9)

    deep = tmp_path / "deep.tif"
    Image.fromarray(np.linspace(0, 65535, 1600).reshape(40, 40).astype(np.uint16)).save(deep)
    g = load_gray(deep)
    assert g.min() == 0 and g.max() == pytest.approx(255)

    mono = tmp_path / "mono.bmp"
    Image.fromarray(np.full((40, 40), 255, np.uint8)).convert("1").save(mono)
    assert load_gray(mono).shape == (40, 40)


def test_cli_skips_bad_files(tmp_path):
    data = tmp_path / "in"
    data.mkdir()
    Image.fromarray(stripes(64, 64, 0.5, 1 / 8).astype(np.uint8)).save(data / "good.png")
    (data / "bad.png").write_bytes(b"\x89PNG broken")
    out = tmp_path / "out"
    code = run.main(["--input", str(data), "--output", str(out), "--pipeline", "all"])
    assert code == 0
    errors = json.loads((out / "errors.json").read_text(encoding="utf-8"))
    assert [e["image"] for e in errors] == ["bad.png"]
    assert (out / "p1" / "good" / "5_minutiae.png").exists()
    assert (out / "metrics.csv").exists()


def test_config_override():
    cfg = load_config("config.yaml", ["p2.sauvola.k=0.3"])
    assert cfg["p2"]["sauvola"]["k"] == 0.3
    with pytest.raises(KeyError):
        load_config("config.yaml", ["p2.nothing.k=1"])
