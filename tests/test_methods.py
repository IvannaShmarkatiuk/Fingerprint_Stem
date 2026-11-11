
import numpy as np
import pytest

from fp.core import box_sum, conv2d, to_gray
from fp.methods.clahe import clahe
from fp.methods.fft import fft1, fft2, ifft2
from fp.methods.fft_enhance import fft_enhance
from fp.methods.gabor import gabor_filter, gabor_filter_full, gabor_filter_naive, gabor_kernel
from fp.methods.minutiae import crossing_number, extract_minutiae
from fp.methods.morphology import closing, cross, dilate, erode, opening, square
from fp.methods.orientation import gradients, orientation_field
from fp.methods.otsu import otsu_binarize, otsu_threshold
from fp.methods.roi import block_stats, normalize, variance_roi
from fp.methods.sauvola import sauvola_threshold
from fp.methods.thinning import zhang_suen, zhang_suen_naive
from fp.synthetic import stripes


# базові операції 

def test_conv2d_identity_and_sum():
    img = np.arange(9, dtype=float).reshape(3, 3)
    ident = np.zeros((3, 3))
    ident[1, 1] = 1
    assert np.allclose(conv2d(img, ident), img)
    # центральний піксель: сума всіх 9 значень 0+1+...+8 = 36
    assert conv2d(img, np.ones((3, 3)))[1, 1] == pytest.approx(36)


def test_box_sum_center():
    img = np.arange(25, dtype=float).reshape(5, 5)
    # вікно 3x3 навколо (2,2): 6+7+8+11+12+13+16+17+18 = 108
    assert box_sum(img, 3)[2, 2] == pytest.approx(108)


def test_to_gray_weights():
    rgb = np.zeros((2, 2, 3))
    rgb[..., 0] = 100
    assert np.allclose(to_gray(rgb), 29.9)  # 0.299 * 100


# CLAHE (гістограмна корекція) 

def test_clahe_one_tile_equals_histogram_equalization():
    # 4 рівні по 4 пікселі: CDF = 4, 8, 12, 16; LUT = (CDF-4)/(16-4)*255
    img = np.array([[0] * 4, [100] * 4, [200] * 4, [255] * 4], dtype=float)
    out = clahe(img, tiles=1, clip_limit=1000)
    expected = np.array([[0] * 4, [85] * 4, [170] * 4, [255] * 4], dtype=float)
    assert np.allclose(out, expected)


def test_clahe_constant_image_stays_constant():
    out = clahe(np.full((40, 40), 90.0), tiles=4, clip_limit=2)
    assert np.ptp(out) < 1e-9


# поле орієнтацій і Габор (фільтрація)

def test_gradients_ramp():
    img = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=float)
    gx, gy = gradients(img)
    # центральна різниця (2 - 0) / 2 = 1 по x, 0 по y
    assert gx[1, 1] == pytest.approx(1.0)
    assert gy[1, 1] == pytest.approx(0.0)


def test_orientation_vertical_stripes():
    # яскравість змінюється вздовж x -> лінії вертикальні -> theta = pi/2
    img = stripes(64, 64, np.pi / 2, 1 / 8)
    theta, coh, _ = orientation_field(img, 9, 9)
    assert abs(theta[32, 32] - np.pi / 2) < np.radians(2)
    assert coh[32, 32] > 0.9


def test_gabor_kernel_zero_mean_and_symmetric():
    k = gabor_kernel(0.3, 1 / 9, 4, 4)
    assert abs(k.sum()) < 1e-9
    assert np.allclose(k, k[::-1, ::-1])


def test_gabor_three_implementations_agree():
    rng = np.random.default_rng(1)
    img = rng.random((9, 9)) * 255
    theta = rng.random((9, 9)) * np.pi
    a = gabor_filter(img, theta, 1 / 9, 2, 2, 8)
    b = gabor_filter_full(img, theta, 1 / 9, 2, 2, 8)
    c = gabor_filter_naive(img, theta, 1 / 9, 2, 2, 8)
    assert np.allclose(a, b) and np.allclose(a, c)


def test_gabor_prefers_matching_orientation():
    img = stripes(48, 48, 0.0, 1 / 8)
    good = gabor_filter(img, np.zeros((48, 48)), 1 / 8)
    bad = gabor_filter(img, np.full((48, 48), np.pi / 2), 1 / 8)
    assert np.abs(good[16:32, 16:32]).mean() > 5 * np.abs(bad[16:32, 16:32]).mean()


# Оцу і Sauvola (сегментація) 

def test_otsu_two_levels():
    # будь-який поріг 10..199 розділяє ідеально; argmax повертає перший - 10
    vals = np.array([10, 10, 10, 200, 200, 200])
    assert otsu_threshold(vals) == 10
    mask, _ = otsu_binarize(vals.reshape(2, 3).astype(float))
    assert mask.tolist() == [[True, True, True], [False, False, False]]


def test_otsu_constant_image_empty_mask():
    mask, _ = otsu_binarize(np.full((5, 5), 77.0))
    assert not mask.any()


def test_sauvola_threshold_by_hand():
    img = np.full((5, 5), 100.0)
    img[2, 2] = 0
    # вікно 3x3: m = 800/9 = 88.89, s = sqrt(80000/9 - m^2) = 31.43
    # T = m * (1 + 0.2 * (s/128 - 1)) = 75.47
    t = sauvola_threshold(img, window=3, k=0.2, r=128)
    assert t[2, 2] == pytest.approx(75.47, abs=0.05)
    assert t[0, 0] == pytest.approx(80.0)  # однорідна ділянка: T = 100 * (1 - 0.2)


# морфологія 
def test_erode_dilate_cross():
    m = np.zeros((5, 5), dtype=bool)
    m[1:4, 1:4] = True
    er = erode(m, cross(3))
    assert er.sum() == 1 and er[2, 2]
    one = np.zeros((5, 5), dtype=bool)
    one[2, 2] = True
    assert (dilate(one, cross(3)) == cross(5) & square(5) & _cross_mask()).all()


def _cross_mask():
    c = np.zeros((5, 5), dtype=bool)
    c[2, 1:4] = True
    c[1:4, 2] = True
    return c


def test_opening_removes_dot_closing_fills_hole():
    m = np.zeros((7, 7), dtype=bool)
    m[3, 3] = True
    assert not opening(m, square(3)).any()
    h = np.ones((7, 7), dtype=bool)
    h[3, 3] = False
    assert closing(h, square(3))[3, 3]


def test_zhang_suen_square_to_point():
    m = np.zeros((5, 5), dtype=bool)
    m[1:4, 1:4] = True
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    assert (zhang_suen(m) == expected).all()
    assert (zhang_suen_naive(m) == expected).all()


def test_zhang_suen_thick_bar_to_line():
    m = np.zeros((5, 9), dtype=bool)
    m[1:4, 1:8] = True
    skel = zhang_suen(m)
    assert skel.sum() == 4 and skel[2, 2:6].all()
    assert (zhang_suen_naive(m) == skel).all()


#  мінуції (ключові точки) 

def test_crossing_number_y_shape():
    s = np.zeros((7, 7), dtype=bool)
    s[3, 0:4] = True
    for i in range(3):
        s[2 - i, 4 + i] = True
        s[4 + i, 4 + i] = True
    cn = crossing_number(s)
    assert cn[3, 3] == 3          # розгалуження
    assert cn[3, 0] == 1          # закінчення
    assert cn[0, 6] == 1 and cn[6, 6] == 1
    assert cn[3, 1] == 2          # звичайна точка лінії


def test_extract_minutiae_line_ending():
    s = np.zeros((40, 40), dtype=bool)
    s[20, 5:20] = True
    roi = np.zeros((40, 40), dtype=bool)
    roi[2:38, 2:38] = True
    found = extract_minutiae(s, roi, border=4, min_dist=3)
    # закінчення (20,5) лежить біля краю ROI і відкидається, лишається (20,19)
    assert found == [(20, 19, "E")]


#ROI і нормалізація (аналіз текстур) 

def test_block_stats_by_hand():
    img = np.array([[0, 2, 5, 5], [2, 0, 5, 5], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=float)
    mean, std, _ = block_stats(img, 2)
    assert np.allclose(mean, [[1, 5], [1, 1]])
    assert np.allclose(std, [[1, 0], [0, 0]])


def test_variance_roi_finds_textured_half():
    # блоки 16x16: ліва половина (4 блоки) - смуги, права - однорідна
    img = np.full((128, 128), 128.0)
    img[:, :64] = stripes(128, 64, 0.0, 1 / 6)
    roi = variance_roi(img, block=16, rel_threshold=0.3)
    assert roi[:, :64].all() and not roi[:, 64:].any()


def test_normalize_mean_and_variance():
    img = np.array([[0, 10], [20, 30]], dtype=float)
    out = normalize(img, np.ones((2, 2), dtype=bool), m0=100, v0=400)
    assert out.mean() == pytest.approx(100)
    assert out.var() == pytest.approx(400)


#  FFT (перетворення Фур'є) 

def test_fft_by_hand():
    assert np.allclose(fft1([1, 0, 0, 0]), [1, 1, 1, 1])
    assert np.allclose(fft1([1, 1, 1, 1]), [4, 0, 0, 0])
    assert np.allclose(fft1([0, 1, 0, 0]), [1, -1j, -1, 1j])


def test_fft2_matches_numpy_and_inverts():
    x = np.random.default_rng(3).random((2, 8, 16))
    assert np.allclose(fft2(x), np.fft.fft2(x))
    assert np.allclose(ifft2(fft2(x)).real, x)


def test_fft1_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fft1(np.ones(6))


def test_fft_enhance_keeps_stripes_and_background():
    img = stripes(64, 64, 0.0, 1 / 8)
    roi = np.zeros((64, 64), dtype=bool)
    roi[8:56, 8:56] = True
    out = fft_enhance(img, roi)
    assert (out[~roi] == 255).all()
    # напрям ліній зберігається: рядки однакові, вздовж стовпця - коливання
    inner = out[16:48, 16:48]
    assert inner.std(axis=1).mean() < 0.2 * inner.std(axis=0).mean()
