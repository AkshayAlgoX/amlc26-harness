"""Generate a fake 2025-style dataset (catalog text -> price) to smoke-test the harness.
Includes product families with pack-size variants split across train/test, like real catalogs.

  python scripts/make_synthetic.py            # -> data/synthetic/{train,test,sample_test_out}.csv
Never mix this with real data: it lives in data/synthetic and is used via config.synthetic.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BRANDS = ["Acme", "Zenith", "Nova", "Orbit", "Kora", "Veda", "Tulsi", "Maple", "Crest", "Indus",
          "Lumo", "Pixel", "Sora", "Banyan", "Quill", "Rivo", "Saffron", "Delta", "Mango", "Opal"]
TYPES = {"green tea": 6.0, "coffee beans": 14.0, "almonds": 11.0, "dark chocolate": 4.5, "olive oil": 12.0,
         "protein powder": 30.0, "basmati rice": 9.0, "honey": 8.0, "shampoo": 7.0, "dish soap": 3.5,
         "hand cream": 6.5, "multivitamin": 16.0, "pasta": 2.5, "peanut butter": 5.5, "granola": 5.0}
ADJ = ["Organic", "Classic", "Premium", "Low Sugar", "Gluten-Free", "Family Size", "Natural", "Extra Virgin", "Roasted"]
UNITS = [("Ounce", 1.0), ("Gram", 0.035), ("Fl Oz", 1.0), ("Count", 0.4)]


def make(out_dir: Path, n_train: int = 4000, n_test: int = 2000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    brand_f = {b: rng.lognormal(0, 0.35) for b in BRANDS}
    n_fam = (n_train + n_test) // 3
    fams = [(rng.choice(BRANDS), rng.choice(list(TYPES)), rng.choice(ADJ), UNITS[rng.integers(len(UNITS))])
            for _ in range(n_fam)]
    rows = []
    for i in range(n_train + n_test):
        b, t, a, (unit, oz_per_unit) = fams[rng.integers(n_fam)]
        size = float(rng.choice([1.8, 3.5, 5, 8, 12, 16, 24, 32, 100, 250, 500]))
        pack = int(rng.choice([1, 1, 1, 2, 3, 4, 6, 12]))
        oz = size * oz_per_unit
        price = TYPES[t] * brand_f[b] * (1 + 0.25 * (a in ("Organic", "Premium"))) * (oz ** 0.55) * (pack ** 0.85)
        price *= rng.lognormal(0, 0.18)
        text = (f"Item Name: {b} {a} {t.title()}, {size:g} {unit} (Pack of {pack})\n"
                f"Bullet Point 1: {a} {t} from {b}.\nValue: {size * pack:g}\nUnit: {unit}")
        rows.append({"sample_id": str(100000 + i), "catalog_content": text,
                     "image_link": f"http://127.0.0.1:8765/img/{100000 + i}.jpg", "price": round(price, 2)})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    train.to_csv(out_dir / "train.csv", index=False)
    test.drop(columns=["price"]).to_csv(out_dir / "test.csv", index=False)
    # like 2025: the sample output only covers a few rows; we use it for column names/order
    test[["sample_id", "price"]].head(100).assign(price=1.0).to_csv(out_dir / "sample_test_out.csv", index=False)
    print(f"wrote {out_dir} ({n_train} train / {n_test} test)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    args = ap.parse_args()
    make(Path(args.out), args.n_train, args.n_test)
