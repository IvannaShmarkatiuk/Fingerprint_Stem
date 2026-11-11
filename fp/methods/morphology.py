"""Бінарна морфологія: ерозія, дилатація, відкриття, закриття.

Категорія: морфологічна обробка (стадія 4 пайплайна П2, допоміжно в ROI).
Структурний елемент - булева матриця непарного розміру.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def square(size):
    """Квадратний структурний елемент size x size."""
    return np.ones((size, size), dtype=bool)


def cross(size=3):
    """Хрестоподібний структурний елемент."""
    se = np.zeros((size, size), dtype=bool)
    se[size // 2, :] = True
    se[:, size // 2] = True
    return se


def dilate(mask, se):
    """Дилатація: піксель стає True, якщо хоч один піксель під елементом True."""
    r0, r1 = se.shape[0] // 2, se.shape[1] // 2
    p = np.pad(mask, ((r0, r0), (r1, r1)), constant_values=False)
    win = sliding_window_view(p, se.shape)
    return (win & se[::-1, ::-1]).any(axis=(2, 3))


def erode(mask, se, border=False):
    """Ерозія: піксель лишається True, лише якщо всі пікселі під елементом True.

    border - значення, яким вважаються пікселі за межами зображення
    (False: краї зображення "з'їдаються", як фон).
    """
    r0, r1 = se.shape[0] // 2, se.shape[1] // 2
    p = np.pad(mask, ((r0, r0), (r1, r1)), constant_values=border)
    win = sliding_window_view(p, se.shape)
    return (win | ~se).all(axis=(2, 3))


def opening(mask, se):
    """Відкриття = ерозія, потім дилатація. Прибирає дрібні острівці та виступи."""
    return dilate(erode(mask, se), se)


def closing(mask, se):
    """Закриття = дилатація, потім ерозія. Заповнює дрібні дірки та розриви.

    Маска спочатку доповнюється повторенням крайніх пікселів: об'єкт, що
    торкається краю кадру, "продовжується" за межу і не з'їдається ерозією,
    а фон біля краю не прилипає до об'єкта.
    """
    r0, r1 = se.shape[0] // 2, se.shape[1] // 2
    p = np.pad(mask, ((r0, r0), (r1, r1)), mode="edge")
    out = erode(dilate(p, se), se)
    return out[r0:r0 + mask.shape[0], r1:r1 + mask.shape[1]]
