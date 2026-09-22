"""Local leaderboard built from artifacts/logs/*.json (one file per experiment, so no git conflicts).

  python -m src.board              # top 20 full runs, sorted by CV
  python -m src.board --all        # include QUICK runs
  python -m src.board --md         # markdown table for docs/APPROACH.md
Also writes artifacts/experiments.csv (derived file — never edit by hand).
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import artifacts_dir
from .metric import greater_is_better
from .registry import load_logs

COLS = ["exp_id", "cv", "cv_std", "kind", "owner", "track", "minutes", "test_mode", "git_sha", "quick", "notes"]


def collect(include_quick: bool = False) -> pd.DataFrame:
    logs = load_logs(include_quick)
    if not logs:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(logs)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df.sort_values("cv", ascending=not greater_is_better()).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["exp_id", "cv", "cv_std", "minutes", "notes"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include QUICK runs")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--track", default=None)
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args(argv)
    df = collect(args.all)
    if df.empty:
        print("no experiments logged yet")
        return
    out = artifacts_dir() / "experiments.csv"
    df.to_csv(out, index=False)
    if args.track:
        df = df[df["track"] == args.track]
    view = df.head(args.top)
    if args.md:
        print(to_markdown(view))
        return
    arrow = "higher=better" if greater_is_better() else "lower=better"
    print(f"local leaderboard ({arrow}) — {len(df)} runs, full table in {out}")
    show = view[["exp_id", "cv", "cv_std", "kind", "minutes", "notes"]].copy()
    show["notes"] = show["notes"].fillna("").astype(str).str.slice(0, 48)
    print(show.to_string(index=True))


if __name__ == "__main__":
    main()
