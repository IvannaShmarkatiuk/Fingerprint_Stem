"""Завантаження конфігурації з YAML та перевизначення параметрів з командного рядка."""
import copy

import yaml


def load_config(path, overrides=None):
    """Читає YAML і застосовує перевизначення виду "p1.gabor.sigma_y=5".

    Значення після "=" розбирається як YAML (числа, true/false, списки).
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Очікується ключ=значення, отримано: {item}")
        key, value = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def set_by_path(cfg, dotted, value):
    """Встановлює вкладений параметр за шляхом через крапку."""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Немає секції '{p}' у шляху {dotted}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Немає параметра '{parts[-1]}' у шляху {dotted}")
    node[parts[-1]] = value
