"""Draw the judgments for the hand check of the study A dataset.

The hand check asks, for a random sample of the judgments citing the WGBO
(the 703 in caselaw_citations.csv, before any are dropped):

    - does the script find the facts section, and is it the right text?
    - is each article counted as an answer applied in substance, or only
      cited in passing or for procedure?

Sampling from all 703, not only the 391 dataset items, also shows whether
the 254 judgments dropped for "no facts section found" really have none.

The sample is drawn once with a fixed seed and written to
data/validation/handcheck_sample.csv (committed), so every annotator checks
the same judgments in the same order. Re-running refuses to overwrite it
unless --force is given. The checking itself happens in handcheck.py:

    python scripts/sample_handcheck.py
    streamlit run handcheck.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CITATIONS = PROJECT_ROOT / "data" / "processed" / "caselaw_citations.csv"
SPLIT = PROJECT_ROOT / "data" / "processed" / "study_a" / "split.csv"
SAMPLE = PROJECT_ROOT / "data" / "validation" / "handcheck_sample.csv"

SEED = 20261021


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--force", action="store_true", help="overwrite an existing sample")
    args = p.parse_args()
    if SAMPLE.exists() and not args.force:
        sys.exit(f"{SAMPLE.relative_to(PROJECT_ROOT)} exists; the sample is fixed. "
                 "Use --force only if no one has started checking.")

    with CITATIONS.open(encoding="utf-8") as f:
        eclis = sorted(r["ecli"] for r in csv.DictReader(f)
                       if r["set"] == "candidate" and int(r["n_wgbo_articles"]) > 0)
    with SPLIT.open(encoding="utf-8") as f:
        split = {r["ecli"]: r["split"] for r in csv.DictReader(f)}

    sample = random.Random(SEED).sample(eclis, args.n)
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "ecli", "in_dataset"])
        for i, ecli in enumerate(sample, 1):
            w.writerow([i, ecli, int(ecli in split)])

    n_in = sum(e in split for e in sample)
    print(f"{args.n} of {len(eclis)} judgments ({n_in} in the dataset, "
          f"{args.n - n_in} dropped by the build) -> {SAMPLE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
