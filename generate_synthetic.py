import argparse
from pathlib import Path

import numpy as np

from fp.io_utils import save_image
from fp.synthetic import synth_fingerprint


def main(argv=None):
    ap = argparse.ArgumentParser(description="Генерація синтетичних відбитків-еталонів")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0, help="перший seed (далі seed+1, ...)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        seed = args.seed + i
        # частота ліній від 1/10 до 1/8, як у сканах 500 dpi
        freq = 1 / (8 + (seed % 3))
        d = synth_fingerprint(freq=freq, seed=seed)
        save_image(d["image"], out / f"synth_{seed:03d}.png")
        np.savez_compressed(out / f"synth_{seed:03d}_gt.npz", theta=d["theta"],
                            ridges=d["ridges"], roi=d["roi"], freq=d["freq"])
        print(f"synth_{seed:03d}.png  freq=1/{1 / freq:.0f}")


if __name__ == "__main__":
    main()
