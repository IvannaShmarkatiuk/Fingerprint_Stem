"""Блокове підсилення відбитка в частотній області.

Категорія: перетворення Фур'є, частотна фільтрація (стадія 2 пайплайна П2).
Метод: Watson, Candela, Grother (1994), з додатковим смуговим фільтром.

Для кожного блоку block x block (з перекриттям step):
1. блок множиться на вікно Ганна, щоб краї блоку не давали артефактів;
2. обчислюється спектр F = FFT2(блок);
3. залишаються лише частоти в смузі [fmin, fmax] (типовий діапазон частот ліній);
4. спектр підсилюється: G = F * |F|^k. Домінантна частота ліній у блоці
   підсилюється сильніше за шум, тому лінії стають чіткішими, а розриви заповнюються;
5. блок повертається через обернене FFT, блоки складаються з вагами вікна.
"""
import numpy as np

from fp.core import rescale
from fp.methods.fft import fft2, ifft2, freq_grid


def fft_enhance(img, roi=None, block=32, step=16, k=0.45, fmin=1 / 16, fmax=1 / 4):
    """Повертає підсилене зображення [0, 255]; лінії темні, фон поза ROI білий."""
    img = np.asarray(img, dtype=np.float64)
    if roi is not None and roi.any():
        # фон заповнюємо середнім ROI, щоб різкий перехід на межі не давав "дзвону" в спектрі
        img = np.where(roi, img, img[roi].mean())
    h, w = img.shape
    pad_h = (-(h - block) % step) if h > block else block - h
    pad_w = (-(w - block) % step) if w > block else block - w
    src = np.pad(img - img.mean(), ((0, pad_h), (0, pad_w)), mode="reflect")
    hp, wp = src.shape

    iy, ix = _block_index(hp, wp, block, step)
    win1 = np.hanning(block + 2)[1:-1]
    win = np.outer(win1, win1)
    out_blocks = _enhance_blocks(src[iy, ix] * win, k, fmin, fmax) * win

    acc = np.zeros((hp, wp))
    wsum = np.zeros((hp, wp))
    np.add.at(acc, (iy, ix), out_blocks)
    np.add.at(wsum, (iy, ix), np.broadcast_to(win ** 2, out_blocks.shape))
    res = (acc / np.maximum(wsum, 1e-9))[:h, :w]

    return _scale_to_roi(res, roi)


def _enhance_blocks(blocks, k, fmin, fmax):
    """Смуговий фільтр і підсилення спектра G = F * |F|^k для всіх блоків одразу."""
    f = freq_grid(blocks.shape[-1])
    rad = np.hypot(f[:, None], f[None, :])
    band = (rad >= fmin) & (rad <= fmax)
    spec = fft2(blocks) * band
    spec = spec * np.abs(spec) ** k
    return np.real(ifft2(spec))


def _block_index(hp, wp, block, step):
    """Масиви індексів рядків і стовпців усіх блоків: форма (ny, nx, block, block)."""
    ys = np.arange(0, hp - block + 1, step)
    xs = np.arange(0, wp - block + 1, step)
    iy = ys[:, None, None, None] + np.arange(block)[None, None, :, None]
    ix = xs[None, :, None, None] + np.arange(block)[None, None, None, :]
    return iy, ix


def _scale_to_roi(res, roi):
    """Розтягує результат до 0..255 за 1-99 процентилями ROI; фон поза ROI білий."""
    h, w = res.shape
    if roi is None:
        return rescale(res)
    out = np.full((h, w), 255.0)
    if roi.any():
        lo, hi = np.percentile(res[roi], [1, 99])
        out[roi] = np.clip((res[roi] - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255)
    return out
