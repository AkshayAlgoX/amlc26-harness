"""Resumable, parallel image downloader (the Day-1 hour-0 job on the GPU box).

  python scripts/download_images.py --split train --workers 64 --size 512
  python scripts/download_images.py --split test  --workers 64 --size 512

- reads id + image URL columns from config.yaml (columns.id, columns.image_url)
- saves <out>/<split>/<id>.jpg, resized so the long side <= --size (0 = keep original)
- re-running skips files that already exist, so it resumes after a crash/disconnect
- failures after retries -> <out>/failures_<split>.csv (re-run to retry them)
Images come from the organisers' own links, which is what they told teams to use.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.config import ROOT, col  # noqa: E402
from src.data import load_test, load_train  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def fetch(url: str, dest: Path, size: int, retries: int, timeout: float) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    if not isinstance(url, str) or not url.startswith("http"):
        return "fail:no-url"
    err = "unknown"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if size:
                img.thumbnail((size, size))
            tmp = dest.with_suffix(".part")
            img.save(tmp, "JPEG", quality=92)
            os.replace(tmp, dest)
            return "ok"
        except Exception as e:  # network, HTTP, decode errors: back off and retry
            err = type(e).__name__
            time.sleep(min(0.5 * 2 ** attempt, 8))
    return f"fail:{err}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "images"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (for testing)")
    args = ap.parse_args(argv)

    url_col, id_col = col("image_url"), col("id")
    if not url_col:
        sys.exit("config columns.image_url is null — this task has no images")
    df = load_train() if args.split == "train" else load_test()
    if args.limit:
        df = df.head(args.limit)
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    jobs = list(zip(df[id_col].astype(str), df[url_col]))
    t0, counts, failures = time.time(), {"ok": 0, "skip": 0, "fail": 0}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, u, out / f"{i}.jpg", args.size, args.retries, args.timeout): (i, u) for i, u in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            key = "fail" if res.startswith("fail") else res
            counts[key] += 1
            if key == "fail":
                failures.append((*futs[fut], res))
            if n % 2000 == 0 or n == len(jobs):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"{n}/{len(jobs)}  ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}  {rate:.0f}/s")
    fail_path = Path(args.out) / f"failures_{args.split}.csv"
    with open(fail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "url", "error"])
        w.writerows(failures)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}  (failures: {len(failures)} listed in {fail_path})")


if __name__ == "__main__":
    main()
