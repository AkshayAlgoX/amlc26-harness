"""Build <team>_submission.zip in the organiser layout from an explicit ALLOWLIST (nothing else can enter).

<team>_submission.zip
  output/matching_results.tsv, output/candidate_pairs.tsv        <- the merge step's validated outputs
  code/business_entity_resolution/src/...                         <- pipeline code (repo-relative paths preserved)
  code/business_entity_resolution/src/export/...                  <- the H4-validated fold models + decision + manifest
  code/business_entity_resolution/README.md, requirements.txt     <- packaging/README.md, packaging/requirements.txt
  Documentation_template.md                                       <- the filled-in methodology document

Never included: data/, expected/, exclude_ids/, scratch/, output/ (except the two TSVs), artifacts/, logs, notebooks,
label-derived files, romanization/attack benchmarks. Holdout ID lists are excluded; the H1 training-sample ID list
is added only with --include-training-sample (operator decision: needed to re-train, not to re-run inference).
usage: python scripts/build_submission_package.py --team NAME --outputs DIR --export DIR --doc FILE --zip OUT.zip
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ALLOWLIST = [
    "experiments/__init__.py",
    "experiments/agy2_pipeline.py", "experiments/agy2_pipeline_merge.py", "experiments/agy2_pipeline_freeze.py",
    "experiments/agy2_exp4_fresh_holdout_validation.py", "experiments/agy2_exp5_corrected_holdout.py",
    "experiments/agy2_exp6_integrated.py", "experiments/agy2_exp7_prep_h3.py", "experiments/agy2_exp7_freeze.py",
    "experiments/agy2_exp7_h3.py", "experiments/agy2_exp8_store.py", "experiments/agy2_exp9_features.py",
    "experiments/agy2_exp9_h4_freeze.py", "experiments/configs/h4_frozen.json", "experiments/configs/test_shards.json",
    "src/__init__.py", "src/er/__init__.py", "src/er/blocking.py", "src/er/features.py", "src/er/generation.py",
    "src/er/ids.py", "src/er/loader.py", "src/er/metric.py", "src/er/normalize.py", "src/er/pipeline.py",
    "src/er/singletons.py", "src/er/split.py", "src/er/validator.py",
]
TRAINING_SAMPLE = "exclude_ids/prev_val_s1_50k.ids"
FORBIDDEN_PARTS = ("data/", "expected/", "scratch/", "artifacts/", "output/", "__pycache__", ".ipynb")
FORBIDDEN_NAMES = ("romanization", "retrieval_attack", "top_misses", "baseline_misses", "baseline_positives", "patch",
                   "kaggle_run", "retrieval_report", "ablation", "prompts/", "contributions/", "third_party", "exclude_ids",
                   "h1_", "h3_", "h4_s1", "exp5_holdout", "prev_val")


def plan(team: str, outputs: Path, export: Path, doc: Path, include_training_sample=False) -> dict:
    """Return {zip_member: source_path}. Raises on any missing or forbidden item."""
    base = "code/business_entity_resolution"
    m = {"output/matching_results.tsv": outputs / "matching_results.tsv",
         "output/candidate_pairs.tsv": outputs / "candidate_pairs.tsv",
         f"{base}/README.md": ROOT / "packaging" / "README.md",
         f"{base}/requirements.txt": ROOT / "packaging" / "requirements.txt",
         "Documentation_template.md": doc}
    for rel in CODE_ALLOWLIST:
        m[f"{base}/src/{rel}"] = ROOT / rel
    for f in sorted(export.iterdir()):
        if f.is_file():
            m[f"{base}/src/export/{f.name}"] = f
    if include_training_sample:
        m[f"{base}/src/{TRAINING_SAMPLE}"] = ROOT / TRAINING_SAMPLE
    for member, src in m.items():
        if not Path(src).is_file():
            raise SystemExit(f"missing package input: {src}")
        inner = member.split(f"{base}/src/", 1)[-1] if member.startswith(f"{base}/src/") else ""
        if inner and (any(p in inner for p in FORBIDDEN_PARTS) and inner != TRAINING_SAMPLE):
            raise SystemExit(f"forbidden path in package: {member}")
        if any(n in member.lower() for n in FORBIDDEN_NAMES) and not member.endswith(TRAINING_SAMPLE):
            raise SystemExit(f"forbidden file in package: {member}")
        if inner.endswith(".ids") and inner != TRAINING_SAMPLE:
            raise SystemExit(f"holdout id list in package: {member}")
    return m


def build(team, outputs, export, doc, out_zip, include_training_sample=False) -> dict:
    m = plan(team, Path(outputs), Path(export), Path(doc), include_training_sample)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for member, src in sorted(m.items()):
            z.write(src, member)
    return {"zip": str(out_zip), "members": len(m),
            "sha256": hashlib.sha256(Path(out_zip).read_bytes()).hexdigest()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--include-training-sample", action="store_true")
    a = ap.parse_args()
    print(build(a.team, a.outputs, a.export, a.doc, a.zip, a.include_training_sample))
